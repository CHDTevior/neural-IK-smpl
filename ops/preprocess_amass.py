#!/usr/bin/env python3
"""Rebuild the unavailable SMPL-IK AMASS gender-augmented dataset."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch3d.transforms import axis_angle_to_quaternion
from smplx import SMPL

from scripts.make_dataset_AMASS_SMPL_from_SMPLH_fast import (
    SMPL_JOINT_NAMES,
    SMPL_POSES_NAMES,
    SMPL_SHAPE_NAMES,
    make_dataset_settings_json,
)


TRAIN_DATASETS = [
    "ACCAD",
    "BMLhandball",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "KIT",
    "MPI_Limits",
    "TCD_handMocap",
    "TotalCapture",
]
VALIDATION_DATASETS = ["HumanEva", "MPI_HDM05", "MPI_mosh", "SFU"]
TEST_DATASETS = ["SSM_synced", "Transitions_mocap"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def headers() -> Tuple[List[str], List[str], List[str], List[str]]:
    positions = [
        f"BonePositions_{joint}_{axis}"
        for joint in SMPL_JOINT_NAMES
        for axis in ("X", "Y", "Z")
    ]
    rotations = [
        f"BoneRotations_{joint}_{axis}"
        for joint in SMPL_POSES_NAMES
        for axis in ("W", "X", "Y", "Z")
    ]
    betas = [f"{name}_V" for name in SMPL_SHAPE_NAMES]
    all_columns = positions + rotations + betas + ["Gender_V"] + [
        "RootPosition_X",
        "RootPosition_Y",
        "RootPosition_Z",
    ]
    return positions, rotations, betas, all_columns


class GenderedSmpl:
    def __init__(self, model_root: Path, device: str):
        filenames = {
            "m": ("male", "basicModel_m_lbs_10_207_0_v1.0.0.pkl"),
            "f": ("female", "basicModel_f_lbs_10_207_0_v1.0.0.pkl"),
            "n": ("neutral", "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
        }
        self.device = torch.device(device)
        self.models = {}
        for code, (gender, filename) in filenames.items():
            model = SMPL(
                model_path=str(model_root / filename),
                gender=gender,
                create_transl=False,
            )
            self.models[code] = model.to(self.device).eval()

    def joints(
        self,
        betas: np.ndarray,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        gender: str,
        batch_size: int,
    ) -> np.ndarray:
        pieces = []
        model = self.models[gender]
        with torch.no_grad():
            for start in range(0, len(body_pose), batch_size):
                stop = start + batch_size
                output = model(
                    betas=torch.from_numpy(betas[start:stop]).to(self.device),
                    global_orient=torch.from_numpy(global_orient[start:stop]).to(
                        self.device
                    ),
                    body_pose=torch.from_numpy(body_pose[start:stop]).to(self.device),
                )
                pieces.append(
                    output.joints[:, : len(SMPL_JOINT_NAMES)]
                    .reshape(-1, len(SMPL_JOINT_NAMES) * 3)
                    .cpu()
                    .numpy()
                )
        return np.concatenate(pieces, axis=0)


def parse_dataset_spec(spec: str) -> Tuple[str, str]:
    if ":" in spec:
        source_name, output_name = spec.split(":", 1)
    else:
        source_name = output_name = spec
    if not source_name or not output_name:
        raise ValueError(f"Invalid dataset spec: {spec}")
    return source_name, output_name


def motion_files(dataset_root: Path) -> List[Path]:
    return sorted(dataset_root.glob("*/*.npz"))


def load_motion(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as annotation:
        if "poses" not in annotation.files:
            return tuple()
        required = {"poses", "betas", "gender", "trans"}
        missing = required.difference(annotation.files)
        if missing:
            raise KeyError(f"{path} is missing required AMASS fields: {sorted(missing)}")

        poses = np.asarray(annotation["poses"])
        if poses.ndim != 2 or poses.shape[1] < 66:
            raise ValueError(f"{path} has unexpected poses shape {poses.shape}")
        frame_count = poses.shape[0]
        global_orient = poses[:, :3].astype(np.float32, copy=False)
        body_pose = np.concatenate(
            [
                poses[:, 3:66].astype(np.float32, copy=False),
                np.zeros((frame_count, 6), dtype=np.float32),
            ],
            axis=1,
        )
        shape = np.asarray(annotation["betas"])[: len(SMPL_SHAPE_NAMES)]
        if shape.shape != (len(SMPL_SHAPE_NAMES),):
            raise ValueError(f"{path} has unexpected betas shape {shape.shape}")
        betas = np.repeat(shape[None, :], frame_count, axis=0).astype(
            np.float32, copy=False
        )
        root_position = np.asarray(annotation["trans"], dtype=np.float32)
        if root_position.shape != (frame_count, 3):
            raise ValueError(
                f"{path} has unexpected trans shape {root_position.shape}"
            )
    return global_orient, body_pose, betas, root_position


def concatenate(chunks: Sequence[Tuple[np.ndarray, ...]]) -> Tuple[np.ndarray, ...]:
    return tuple(np.concatenate(parts, axis=0) for parts in zip(*chunks))


def gender_augmented_frame(
    smpl: GenderedSmpl,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    betas: np.ndarray,
    root_position: np.ndarray,
    smpl_batch_size: int,
) -> pd.DataFrame:
    position_headers, _, _, all_headers = headers()
    rotation_input = np.concatenate([global_orient, body_pose], axis=1)
    quaternions = (
        axis_angle_to_quaternion(
            torch.from_numpy(rotation_input).reshape(-1, 3)
        )
        .reshape(len(rotation_input), -1)
        .numpy()
    )

    frames = []
    for gender_code, numeric_gender in (("m", 0), ("f", 1), ("n", 2)):
        joint_positions = smpl.joints(
            betas,
            global_orient,
            body_pose,
            gender=gender_code,
            batch_size=smpl_batch_size,
        )
        values = np.concatenate(
            [
                joint_positions,
                quaternions,
                betas,
                np.full((len(betas), 1), numeric_gender, dtype=np.float32),
                root_position,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        frames.append(pd.DataFrame(values, columns=all_headers))

    dataframe = pd.concat(frames, ignore_index=True)
    if len(position_headers) != len(SMPL_JOINT_NAMES) * 3:
        raise RuntimeError("Position schema does not match upstream SMPL schema")
    return dataframe


def convert_dataset(
    source_root: Path,
    output_root: Path,
    report_root: Path,
    model_root: Path,
    source_name: str,
    output_name: str,
    device: str,
    chunk_frames: int,
    smpl_batch_size: int,
) -> Dict[str, object]:
    source_dir = source_root / source_name
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    output_dir = output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    final_csv = output_dir / f"{output_name}.csv"
    partial_csv = output_dir / f"{output_name}.csv.partial"
    stats_path = report_root / f"{output_name}.json"

    if final_csv.is_file() and stats_path.is_file():
        print(f"{output_name}: complete output already exists; skipping")
        return json.loads(stats_path.read_text())
    if final_csv.exists() or partial_csv.exists():
        raise RuntimeError(
            f"{output_name}: incomplete output exists; inspect before restarting: "
            f"{final_csv} {partial_csv}"
        )

    started = time.time()
    smpl = GenderedSmpl(model_root, device)
    chunks: List[Tuple[np.ndarray, ...]] = []
    buffered_frames = 0
    source_frames = 0
    rows_written = 0
    auxiliary_files_skipped = 0
    files_seen = 0

    def flush() -> None:
        nonlocal chunks, buffered_frames, rows_written
        if not chunks:
            return
        global_orient, body_pose, betas, root_position = concatenate(chunks)
        dataframe = gender_augmented_frame(
            smpl,
            global_orient,
            body_pose,
            betas,
            root_position,
            smpl_batch_size,
        )
        dataframe.to_csv(
            partial_csv,
            mode="a",
            header=not partial_csv.exists(),
            index=False,
        )
        rows_written += len(dataframe)
        print(
            f"{output_name}: wrote {rows_written} rows "
            f"from {source_frames} source frames",
            flush=True,
        )
        chunks = []
        buffered_frames = 0

    files = motion_files(source_dir)
    for path in files:
        files_seen += 1
        motion = load_motion(path)
        if not motion:
            auxiliary_files_skipped += 1
            continue
        chunks.append(motion)
        frame_count = len(motion[0])
        buffered_frames += frame_count
        source_frames += frame_count
        if buffered_frames > chunk_frames:
            flush()
    flush()

    if rows_written != source_frames * 3:
        raise RuntimeError(
            f"{output_name}: expected {source_frames * 3} rows, wrote {rows_written}"
        )
    os.replace(partial_csv, final_csv)
    elapsed = time.time() - started
    stats = {
        "source_dataset": source_name,
        "output_dataset": output_name,
        "source_path": str(source_dir.resolve()),
        "output_csv": str(final_csv.resolve()),
        "files_seen": files_seen,
        "auxiliary_files_skipped": auxiliary_files_skipped,
        "source_frames": source_frames,
        "gender_augmented_rows": rows_written,
        "csv_bytes": final_csv.stat().st_size,
        "elapsed_seconds": elapsed,
        "device": device,
        "chunk_frames": chunk_frames,
        "smpl_batch_size": smpl_batch_size,
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def split_paths(names: Iterable[str]) -> List[str]:
    return [f"{name}/{name}.csv" for name in names]


def finalize(
    source_root: Path,
    output_root: Path,
    report_root: Path,
    repository_root: Path,
) -> None:
    expected = TRAIN_DATASETS + VALIDATION_DATASETS + TEST_DATASETS
    for name in expected:
        csv_path = output_root / name / f"{name}.csv"
        stats_path = report_root / f"{name}.json"
        if not csv_path.is_file() or not stats_path.is_file():
            raise FileNotFoundError(f"Missing completed dataset output for {name}")

    make_dataset_settings_json(str(output_root))
    split = {
        "training_files": split_paths(TRAIN_DATASETS),
        "validation_files": split_paths(VALIDATION_DATASETS),
        "test_files": split_paths(TEST_DATASETS),
    }
    (output_root / "split.json").write_text(json.dumps(split, indent=2) + "\n")

    stats = [
        json.loads((report_root / f"{name}.json").read_text()) for name in expected
    ]
    upstream_script = (
        repository_root / "scripts/make_dataset_AMASS_SMPL_from_SMPLH_fast.py"
    )
    manifest = {
        "dataset_name": output_root.name,
        "raw_amass_root": str(source_root.resolve()),
        "upstream_preprocessor": str(upstream_script),
        "upstream_preprocessor_sha256": sha256(upstream_script),
        "conversion": {
            "body_pose": "AMASS global orient + 21 SMPL-H body joints + 2 zero SMPL joints",
            "shape": "first 10 AMASS betas",
            "rotations": "PyTorch3D axis-angle to WXYZ quaternion",
            "gender_augmentation": "male=0, female=1, neutral=2 for every source frame",
            "beta_noise": None,
            "root_translation": "stored as RootPosition; not added to BonePositions",
        },
        "split_policy": {
            "basis": "AMASS recommended dataset-held-out train/validation/test split",
            "training": TRAIN_DATASETS,
            "validation": VALIDATION_DATASETS,
            "test": TEST_DATASETS,
            "local_mapping": {
                "KIT_smplh": "KIT",
                "KIT": "excluded local SMPL-X duplicate",
                "newer_unallocated AMASS subsets": "training",
            },
        },
        "datasets": stats,
        "totals": {
            "source_frames": sum(item["source_frames"] for item in stats),
            "gender_augmented_rows": sum(
                item["gender_augmented_rows"] for item in stats
            ),
            "csv_bytes": sum(item["csv_bytes"] for item in stats),
        },
    }
    (output_root / "preprocess_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["totals"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/iridisfs/scratch/ts1v23/workspace/"
            "motion-latent-diffusion-main/datasets/amass/motion_data"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./datasets/amass_gender_augment_cache_v1"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("./run_reports/amass_preprocess"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--model-root", type=Path, default=Path("./tools/smpl/models"))
    convert_parser.add_argument("--device", default="cuda")
    convert_parser.add_argument("--chunk-frames", type=int, default=50 * 1024)
    convert_parser.add_argument("--smpl-batch-size", type=int, default=10 * 1024)
    convert_parser.add_argument("datasets", nargs="+")

    subparsers.add_parser("finalize")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    report_root = args.report_root.resolve()

    if args.command == "convert":
        for spec in args.datasets:
            source_name, output_name = parse_dataset_spec(spec)
            convert_dataset(
                source_root,
                output_root,
                report_root,
                args.model_root.resolve(),
                source_name,
                output_name,
                args.device,
                args.chunk_frames,
                args.smpl_batch_size,
            )
    else:
        finalize(source_root, output_root, report_root, repository_root)


if __name__ == "__main__":
    main()
