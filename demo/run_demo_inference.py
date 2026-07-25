#!/usr/bin/env python3
"""READ-ONLY inference producing demo_data.json for an interactive SMPL-IK visualization.

Mirrors ops/diagnose_single_rank_gradient.py for config composition / datamodule /
model instantiation, and smplik/models/smpl_model.py shared_step for the
FK-refined prediction path. CPU only. Writes only into this demo dir.

PUBLIC RELEASE EDIT (only change vs. the original cluster script): the
hard-coded absolute cluster paths were replaced by environment variables.
  SMPLIK_REPO  path to the upstream smpl-ik checkout (default: ../smpl-ik
               relative to this file, i.e. the clone made by setup.sh).
  SMPLIK_CKPT  path to the trained checkpoint (.ckpt). REQUIRED.
Requires the AMASS dataset caches to be present in the checkout (the script
reads the first test batch to pick demo samples); run it once to produce
demo_data.json, which demo/server.py then serves.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch a GPU

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("SMPLIK_REPO", DEMO_DIR.parent / "smpl-ik")
).expanduser().resolve()
if not (REPO_ROOT / "smplik").is_dir():
    sys.exit(f"SMPLIK_REPO is not an smpl-ik checkout: {REPO_ROOT} "
             "(run setup.sh, or set SMPLIK_REPO)")
if not os.environ.get("SMPLIK_CKPT"):
    sys.exit("SMPLIK_CKPT is required: path to the trained .ckpt")
CKPT_PATH = Path(os.environ["SMPLIK_CKPT"]).expanduser().resolve()
if not CKPT_PATH.is_file():
    sys.exit(f"SMPLIK_CKPT does not exist: {CKPT_PATH}")
ORIG_CKPT = str(CKPT_PATH)  # recorded in demo_data.json for provenance

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

torch.set_num_threads(8)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from hydra.experimental import compose, initialize_config_dir
from hydra.utils import instantiate
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion

from smplik.data.smpl_module import _batched_collate
from smplik.utils.model_factory import ModelFactory
from smplik.smpl.smpl_info import SMPL_JOINT_NAMES
from smplik.metrics.metric_utils import calc_mpjpe
import smplik.models  # noqa: F401 (registers models)


def resolve_experiment(path):
    with Path(path).open() as stream:
        experiment = yaml.load(stream, Loader=yaml.SafeLoader)
    parameters = experiment["parameters"]
    for key in parameters:
        if not isinstance(parameters[key], list):
            parameters[key] = [parameters[key]]
    grid = list(ParameterGrid(parameters))
    if len(grid) != 1:
        raise RuntimeError(f"{path} must resolve to exactly one experiment")
    overrides = [f"{key}={value}" for key, value in grid[0].items()]
    return compose(experiment["base_config"] + ".yaml", overrides=overrides).model


def r4(x):
    """Round nested float structure to 4 decimals."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return np.round(x.astype(np.float64), 4).tolist()
    if isinstance(x, float):
        return round(x, 4)
    return x


def empty_rot_lookat(batch_size, like):
    return {
        "rotation_data": torch.zeros((batch_size, 0, 6)).type_as(like),
        "rotation_weight": torch.zeros((batch_size, 0)).type_as(like),
        "rotation_tolerance": torch.zeros((batch_size, 0)).type_as(like),
        "rotation_id": torch.zeros((batch_size, 0), dtype=torch.int64),
        "lookat_data": torch.zeros((batch_size, 0, 6)).type_as(like),
        "lookat_weight": torch.zeros((batch_size, 0)).type_as(like),
        "lookat_tolerance": torch.zeros((batch_size, 0)).type_as(like),
        "lookat_id": torch.zeros((batch_size, 0), dtype=torch.int64),
    }


def build_position_input(model, betas, gender, position_data, position_id):
    """Manual input_data exactly as get_data_from_batch builds it (no noise, weight=1, tolerance=0)."""
    b, n, _ = position_data.shape
    input_data = {
        "betas": betas,
        "gender": gender,
        "position_data": position_data,
        "position_weight": torch.ones((b, n)).type_as(position_data),
        "position_tolerance": torch.zeros((b, n)).type_as(position_data),
        "position_id": position_id,
    }
    input_data.update(empty_rot_lookat(b, position_data))
    return input_data


def fk_refine(model, input_data, predicted):
    """FK-refined joint positions exactly as in SmplModel.shared_step."""
    nbj = model.nb_joints
    rot6d = predicted["joint_rotations"]
    rot_mat = rotation_6d_to_matrix(rot6d.view(-1, 6)).view(-1, nbj, 3, 3)
    rot_quat = matrix_to_quaternion(rot_mat)
    fk_pos, _ = model.apply_smpl_quat(betas=input_data["betas"],
                                      joint_rotations_quat=rot_quat,
                                      root_position=predicted["root_joint_position"],
                                      gender=input_data["gender"])
    return fk_pos


def mpjpe_mm(pred, gt, align=True):
    """Per-sample MPJPE in mm. align=True matches repo metric (pelvis-aligned, calc_mpjpe x1000)."""
    if align:
        return calc_mpjpe(pred, gt, align_inds=[0], sample_wise=True)  # already mm
    return (pred - gt).norm(dim=2).mean(dim=1) * 1000.0


def main():
    summary_lines = []

    def log(msg):
        print(msg, flush=True)
        summary_lines.append(msg)

    initialize_config_dir(config_dir=str(REPO_ROOT / "configs"))
    cfg = resolve_experiment(str(REPO_ROOT / "configs/experiments/smplik_amass.yaml"))

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dm = instantiate(cfg.dataset)
    dm.prepare_data()
    model = ModelFactory.instantiate(cfg, data_components=dm.get_data_specific_components())

    # ---- load checkpoint (CPU) ----
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    state_dict = ckpt["state_dict"]
    # strip a leading prefix if present (none expected: model IS the LightningModule)
    model_keys = set(model.state_dict().keys())
    if not (set(state_dict.keys()) & model_keys):
        prefix = next(iter(state_dict)).split(".")[0] + "."
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
        log("checkpoint loaded strict=True: OK")
    except RuntimeError as e:
        # only torchmetrics state buffers may legitimately differ
        missing = [k for k in model_keys - set(state_dict.keys())]
        unexpected = [k for k in set(state_dict.keys()) - model_keys]
        bad = [k for k in missing + unexpected if "metric" not in k]
        if bad:
            raise RuntimeError(f"Non-metric state_dict mismatch: {bad[:10]}") from e
        model.load_state_dict(state_dict, strict=False)
        log(f"checkpoint loaded strict=False (metric-state-only diffs: "
            f"{len(missing)} missing / {len(unexpected)} unexpected)")
    model.eval()
    log(f"ckpt epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')}")

    # ---- data ----
    dm.setup()
    loader = DataLoader(dm.test_dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=_batched_collate)
    batch = next(iter(loader))  # first 2048-row block of the test BatchedDataset
    log(f"test batch shapes: {{k: list(v.shape) for k, v in batch.items()}} = "
        f"{ {k: list(v.shape) for k, v in batch.items()} }")

    gt_all = batch["joint_positions"]  # [2048, 24, 3]
    n_rows = gt_all.shape[0]

    # data-scale sanity: body span head->mean(ankles) in these units
    span = (gt_all[:, 15] - 0.5 * (gt_all[:, 7] + gt_all[:, 8])).norm(dim=1)
    log(f"data scale check: head-to-ankle span mean={span.mean():.3f} min={span.min():.3f} "
        f"max={span.max():.3f} (≈1.2-1.6 => units are meters)")
    log(f"coordinate ranges per axis: min={gt_all.view(-1,3).min(0).values.tolist()} "
        f"max={gt_all.view(-1,3).max(0).values.tolist()}")

    # ---- select 8 well-separated rows: per 256-row chunk, max pose variance (root-centered) ----
    centered = gt_all - gt_all[:, 0:1, :]
    variance = centered.view(n_rows, -1).var(dim=1)
    chunk = n_rows // 8
    sel_rows = []
    for c in range(8):
        lo, hi = c * chunk, (c + 1) * chunk
        sel_rows.append(int(lo + variance[lo:hi].argmax()))
    log(f"selected rows (within test block 0): {sel_rows}")

    sub_batch = {k: v[sel_rows] for k, v in batch.items()}

    with torch.no_grad():
        # =========================================================
        # (4) fixed effector scheme on the 8 samples
        # =========================================================
        torch.manual_seed(123)  # get_data_from_batch permutes effector order via multinomial
        input_data, target_data = model.get_data_from_batch(sub_batch, fixed_effector_setup=True)
        assert float(input_data["position_tolerance"].abs().max()) == 0.0
        predicted = model.forward(input_data)
        pred_fk = fk_refine(model, input_data, predicted)

        gt8 = target_data["joint_positions"]
        mp_aligned = mpjpe_mm(pred_fk, gt8, align=True)
        mp_global = mpjpe_mm(pred_fk, gt8, align=False)

        samples = []
        for i, row in enumerate(sel_rows):
            samples.append({
                "id": row,
                "gender": int(sub_batch["gender"][i, 0]),
                "gt": r4(gt8[i]),
                "effector_ids": input_data["position_id"][i].tolist(),
                "effector_targets": r4(input_data["position_data"][i]),
                "pred": r4(pred_fk[i]),
                "mpjpe_mm": r4(float(mp_aligned[i])),
                "mpjpe_global_mm": r4(float(mp_global[i])),
            })
        log("fixed-scheme per-sample MPJPE (pelvis-aligned, mm): "
            + ", ".join(f"{row}:{float(m):.2f}" for row, m in zip(sel_rows, mp_aligned)))
        log("fixed-scheme per-sample MPJPE (global, mm): "
            + ", ".join(f"{row}:{float(m):.2f}" for row, m in zip(sel_rows, mp_global)))
        log(f"fixed-scheme mean MPJPE aligned={float(mp_aligned.mean()):.2f} mm "
            f"global={float(mp_global.mean()):.2f} mm (val_random reference: 58.39 mm)")

        # =========================================================
        # (5) self-transfer: ALL 24 joints as position effectors, sample #0
        # =========================================================
        st_idx = 0
        st_row = sel_rows[st_idx]
        betas1 = sub_batch["betas"][st_idx:st_idx + 1]
        gender1 = sub_batch["gender"][st_idx:st_idx + 1]
        gt1 = gt8[st_idx:st_idx + 1]
        all_ids = torch.arange(24, dtype=torch.int64).unsqueeze(0)
        st_input = build_position_input(model, betas1, gender1, gt1.clone(), all_ids)
        st_pred = model.forward(st_input)
        st_fk = fk_refine(model, st_input, st_pred)
        st_mp_aligned = float(mpjpe_mm(st_fk, gt1, align=True)[0])
        st_mp_global = float(mpjpe_mm(st_fk, gt1, align=False)[0])
        log(f"self-transfer (24 position effectors, sample {st_row}): "
            f"MPJPE aligned={st_mp_aligned:.2f} mm global={st_mp_global:.2f} mm")

        self_transfer = {
            "sample_id": st_row,
            "effector_ids": all_ids[0].tolist(),
            "pred": r4(st_fk[0]),
            "mpjpe_mm": r4(st_mp_aligned),
            "mpjpe_global_mm": r4(st_mp_global),
        }

        # =========================================================
        # (6) effector sweeps on sample #0
        # manual effector set: pelvis(0), head(15), left_foot(10), right_foot(11),
        #                      left_wrist(20), right_wrist(21)
        # =========================================================
        sweep_effector_ids = [0, 15, 10, 11, 20, 21]
        base_targets = gt1[0, sweep_effector_ids, :].clone()  # [6,3]

        # coronal-plane axes from the sample's own body frame:
        # u = left-right (left_hip - right_hip), v = up (head - pelvis), orthogonalized
        u = gt1[0, 1] - gt1[0, 2]
        u = u / u.norm()
        v = gt1[0, 15] - gt1[0, 0]
        v = v - (v @ u) * u
        v = v / v.norm()
        log(f"sweep axes: left-right u={r4(u)} up v={r4(v)}")

        def run_sweep(name, swept_joint, grid_n, spacing):
            half = grid_n // 2
            swept_slot = sweep_effector_ids.index(swept_joint)
            center = base_targets[swept_slot]
            offsets, grid_points = [], []
            for j in range(grid_n):        # rows: up-down (top row = +up)
                for i in range(grid_n):    # cols: left-right
                    off = (i - half) * spacing * u + (half - j) * spacing * v
                    offsets.append(off)
                    grid_points.append(center + off)
            n_pts = len(grid_points)
            targets = torch.stack(grid_points)  # [n,3]

            pos_data = base_targets.unsqueeze(0).repeat(n_pts, 1, 1)  # [n,6,3]
            pos_data[:, swept_slot, :] = targets
            pos_ids = torch.tensor(sweep_effector_ids, dtype=torch.int64).unsqueeze(0).repeat(n_pts, 1)
            sw_input = build_position_input(model,
                                            betas1.repeat(n_pts, 1),
                                            gender1.repeat(n_pts, 1),
                                            pos_data, pos_ids)
            sw_pred = model.forward(sw_input)
            sw_fk = fk_refine(model, sw_input, sw_pred)  # [n,24,3]

            track = (sw_fk[:, swept_joint, :] - targets).norm(dim=1) * 1000.0  # mm
            log(f"sweep '{name}' (joint {swept_joint}, {grid_n}x{grid_n}, {spacing*100:.0f} cm): "
                f"target-vs-predicted-{name} distance mean={float(track.mean()):.2f} mm "
                f"median={float(track.median()):.2f} mm max={float(track.max()):.2f} mm")
            # spread sanity: predicted swept joint must move with the target
            pred_span = (sw_fk[:, swept_joint, :].max(0).values - sw_fk[:, swept_joint, :].min(0).values).norm()
            targ_span = (targets.max(0).values - targets.min(0).values).norm()
            log(f"sweep '{name}': predicted {name} span={float(pred_span):.3f} m "
                f"vs target span={float(targ_span):.3f} m")

            return {
                "name": name,
                "sample_id": st_row,
                "swept_joint": swept_joint,
                "effector_ids": sweep_effector_ids,
                "base_targets": r4(base_targets),
                "grid_shape": [grid_n, grid_n],
                "grid_spacing_m": spacing,
                "axes": {"left_right": r4(u), "up": r4(v)},
                "grid_points": r4(targets),
                "preds": r4(sw_fk),
                "track_error_mm": r4(track),
            }

        sweeps = [
            run_sweep("right_wrist", 21, 7, 0.08),
            run_sweep("head", 15, 5, 0.06),
        ]

    # ---- (7) write JSON ----
    parents = model.smpl_neutral.parents[:24].tolist()  # from SMPL model kintree_table, parents[0] = -1
    out = {
        "units": "meters (mpjpe values in mm; mpjpe_mm is pelvis-aligned as in repo metric, "
                 "mpjpe_global_mm is unaligned)",
        "joint_names": SMPL_JOINT_NAMES[:24],
        "parents": parents,
        "checkpoint": {
            "path": ORIG_CKPT,
            "epoch": int(ckpt.get("epoch", -1)),
            "val_random_mpjpe": 58.39,
        },
        "fixed_scheme_effectors": {
            "names": list(model.hparams.validation_effectors),
            "indices": model.get_joint_indices(list(model.hparams.validation_effectors)),
        },
        "samples": samples,
        "self_transfer": self_transfer,
        "sweeps": sweeps,
    }
    out_path = DEMO_DIR / "demo_data.json"
    with out_path.open("w") as f:
        json.dump(out, f)
    log(f"parents: {parents}")
    log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    (DEMO_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
