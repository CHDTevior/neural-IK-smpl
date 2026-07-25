#!/usr/bin/env python3
"""Live SMPL-IK demo inference server (stdlib HTTP only).

API:
  GET  /       -> serves frontend.html from this directory (404 until it exists)
  GET  /meta   -> model/skeleton/sample metadata (see build_meta)
  POST /solve  -> run learned IK on an arbitrary user effector set

Encodings (documented, verified against the repo):
  gender: integer as used by the AMASS dataset "Gender" column and
          SmplPosingTask.apply_smpl: 0 = male, 1 = female, anything else
          (dataset uses 2) = neutral. We accept exactly {0, 1, 2}.
  rotation effectors ("rot6d"): ortho6d representation of the joint's WORLD
          (global) rotation matrix, i.e. pytorch3d.matrix_to_rotation_6d of the
          FK world rotation = first two ROWS of the 3x3 world rotation matrix,
          flattened [r00,r01,r02,r10,r11,r12]. This is exactly what
          SmplModel.get_data_from_batch feeds the net (it computes
          joint_world_rotations_mat via apply_smpl_quat on GT local quats and
          converts with matrix_to_rotation_6d).
  lookat effectors: lookat_data = [target_point_world(3), local_direction(3)]
          per get_data_from_batch; local_direction is normalized with the
          repo's normalize_vector (eps=1e-5); default [0,0,1].

Model input construction mirrors SmplModel.get_data_from_batch exactly for
arbitrary effector sets, including the exact empty-tensor shapes for unused
categories. make_translation_invariant / pack_data run inside model.forward
and are NOT duplicated here.

FK+mesh path mirrors SmplModel.shared_step: predicted ortho6d local rotations
-> rotation_6d_to_matrix -> matrix_to_quaternion -> apply_smpl_quat with
root_position = predicted root, predict_verts=True (returns the 6890-vertex
skinned SMPL mesh for the gendered model the batch gender selects).

PUBLIC RELEASE EDIT (only change vs. the original cluster script): the
hard-coded absolute cluster paths were replaced by environment variables.
  SMPLIK_REPO  path to the upstream smpl-ik checkout (default: ../smpl-ik
               relative to this file, i.e. the clone made by setup.sh).
  SMPLIK_CKPT  path to the trained checkpoint (.ckpt). REQUIRED.
  SMPLIK_DEMO_DATA  path to demo_data.json produced by run_demo_inference.py
               (default: demo_data.json next to this file).
Optional auth stays env-driven: DEMO_AUTH ("user:pass" for HTTP Basic) and
DEMO_TOKEN (URL token `?k=...`). No secrets are stored in this file.
"""

import gc
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, never touch a GPU

LIVE_DIR = Path(__file__).resolve().parent

if not os.environ.get("SMPLIK_CKPT"):
    sys.exit("SMPLIK_CKPT is required: path to the trained .ckpt "
             "(e.g. smplik_amass_canonical_epoch13.ckpt from the HF release)")
CKPT_PATH = Path(os.environ["SMPLIK_CKPT"]).expanduser().resolve()
if not CKPT_PATH.is_file():
    sys.exit(f"SMPLIK_CKPT does not exist: {CKPT_PATH}")

DEMO_DATA_PATH = Path(
    os.environ.get("SMPLIK_DEMO_DATA", LIVE_DIR / "demo_data.json")
).expanduser().resolve()

FRONTEND_PATH = LIVE_DIR / "frontend.html"
VENDOR_FILES = {
    "/vendor/three.module.js": LIVE_DIR / "vendor" / "three.module.js",
    "/vendor/OrbitControls.js": LIVE_DIR / "vendor" / "OrbitControls.js",
    "/vendor/TransformControls.js": LIVE_DIR / "vendor" / "TransformControls.js",
}
REPO_ROOT = Path(
    os.environ.get("SMPLIK_REPO", LIVE_DIR.parent / "smpl-ik")
).expanduser().resolve()
if not (REPO_ROOT / "smplik").is_dir():
    sys.exit(f"SMPLIK_REPO is not an smpl-ik checkout: {REPO_ROOT} "
             "(run setup.sh, or set SMPLIK_REPO)")
HOST, PORT = "0.0.0.0", 8899

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

torch.set_num_threads(4)  # be gentle: compute node may host trainings

from hydra.experimental import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from sklearn.model_selection import ParameterGrid  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from pytorch3d.transforms import (  # noqa: E402
    rotation_6d_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d)

from smplik.data.smpl_module import _batched_collate  # noqa: E402
from smplik.utils.model_factory import ModelFactory  # noqa: E402
from smplik.smpl.smpl_info import SMPL_JOINT_NAMES  # noqa: E402
from smplik.geometry.vector import normalize_vector  # noqa: E402
import smplik.models  # noqa: F401,E402 (registers models)

GENDER_VALUES = {"male": 0, "female": 1, "neutral": 2}
DEFAULT_WEIGHT = 1.0
DEFAULT_TOLERANCE = 0.0
DEFAULT_LOOKAT_DIR = [0.0, 0.0, 1.0]
MAX_EFFECTORS = 64

MODEL = None
META_JSON = b""
INFER_LOCK = threading.Lock()


class ApiError(Exception):
    """Client error -> HTTP 400."""


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


def load_model_and_meta():
    global MODEL, META_JSON
    t0 = time.time()

    initialize_config_dir(config_dir=str(REPO_ROOT / "configs"))
    cfg = resolve_experiment(str(REPO_ROOT / "configs/experiments/smplik_amass.yaml"))

    seed = int(cfg.seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dm = instantiate(cfg.dataset)
    dm.prepare_data()
    model = ModelFactory.instantiate(cfg, data_components=dm.get_data_specific_components())

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    state_dict = ckpt["state_dict"]
    model_keys = set(model.state_dict().keys())
    if not (set(state_dict.keys()) & model_keys):
        prefix = next(iter(state_dict)).split(".")[0] + "."
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
        log("checkpoint loaded strict=True: OK")
    except RuntimeError as e:
        missing = list(model_keys - set(state_dict.keys()))
        unexpected = list(set(state_dict.keys()) - model_keys)
        bad = [k for k in missing + unexpected if "metric" not in k]
        if bad:
            raise RuntimeError(f"Non-metric state_dict mismatch: {bad[:10]}") from e
        model.load_state_dict(state_dict, strict=False)
        log(f"checkpoint loaded strict=False (metric-state-only diffs: "
            f"{len(missing)} missing / {len(unexpected)} unexpected)")
    model.eval()
    log(f"ckpt epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')}")

    # ---- sample metadata from the test set (same rows as demo_data.json) ----
    demo = json.loads(DEMO_DATA_PATH.read_text())
    sample_rows = [s["id"] for s in demo["samples"]]

    dm.setup()
    loader = DataLoader(dm.test_dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=_batched_collate)
    batch = next(iter(loader))  # first 2048-row test block, deterministic (no augmentation)
    sub = {k: v[sample_rows] for k, v in batch.items()}

    # sanity: GT joints must match what demo_data.json recorded (fail loud otherwise)
    gt_demo = torch.tensor([s["gt"] for s in demo["samples"]], dtype=torch.float32)
    gt_diff = float((sub["joint_positions"] - gt_demo).abs().max())
    if gt_diff > 1e-3:
        raise RuntimeError(f"test-set GT joints deviate from demo_data.json by {gt_diff} m "
                           "- dataset indexing or augmentation mismatch")
    log(f"sample GT matches demo_data.json (max abs diff {gt_diff:.2e} m)")

    # GT world rotations in the exact net encoding (extra meta field 'gt_rot6d')
    with torch.no_grad():
        _, world_rot = model.apply_smpl_quat(betas=sub["betas"],
                                             joint_rotations_quat=sub["joint_rotations"],
                                             gender=sub["gender"])
        gt_rot6d = matrix_to_rotation_6d(world_rot.reshape(-1, 3, 3)).view(
            len(sample_rows), model.nb_joints, 6)

    samples = []
    for i, row in enumerate(sample_rows):
        samples.append({
            "id": int(row),
            "betas": np.round(sub["betas"][i].numpy().astype(np.float64), 6).tolist(),
            "gender": int(sub["gender"][i, 0]),
            "gt_joints": np.round(sub["joint_positions"][i].numpy().astype(np.float64), 6).tolist(),
            # extra (beyond contract, documented): GT WORLD rotation per joint in
            # the exact ortho6d encoding the net consumes for rotation effectors
            "gt_rot6d": np.round(gt_rot6d[i].numpy().astype(np.float64), 6).tolist(),
        })

    faces = model.smpl_neutral.faces
    meta = {
        "joint_names": SMPL_JOINT_NAMES[:24],
        "parents": model.smpl_neutral.parents[:24].tolist(),
        "faces": np.asarray(faces).astype(np.int64).tolist(),
        "n_verts": int(model.smpl_neutral.get_num_verts()),
        "samples": samples,
        "gender_values": GENDER_VALUES,
        "defaults": {"weight": DEFAULT_WEIGHT, "tolerance": DEFAULT_TOLERANCE},
    }
    META_JSON = json.dumps(meta).encode()
    MODEL = model

    # free dataset memory: server only needs the model + cached meta
    del dm, loader, batch, sub
    gc.collect()
    log(f"model + meta ready in {time.time() - t0:.1f}s "
        f"(faces={len(meta['faces'])}, n_verts={meta['n_verts']}, samples={len(samples)})")


# ---------------------------------------------------------------------------
# request validation + input_data construction (mirrors get_data_from_batch)
# ---------------------------------------------------------------------------

def _as_floats(value, n, name):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ApiError(f"{name} must be a list of {n} numbers")
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ApiError(f"{name} must contain only numbers")
    if not all(np.isfinite(out)):
        raise ApiError(f"{name} must be finite")
    return out


def build_input_data(payload):
    if not isinstance(payload, dict):
        raise ApiError("body must be a JSON object")

    betas = _as_floats(payload.get("betas"), 10, "betas")

    gender = payload.get("gender")
    if not isinstance(gender, (int, float)) or isinstance(gender, bool) or \
            float(gender) not in (0.0, 1.0, 2.0):
        raise ApiError("gender must be 0 (male), 1 (female) or 2 (neutral)")
    gender = int(gender)

    effectors = payload.get("effectors")
    if not isinstance(effectors, list) or len(effectors) == 0:
        raise ApiError("effectors must be a non-empty list")
    if len(effectors) > MAX_EFFECTORS:
        raise ApiError(f"too many effectors (max {MAX_EFFECTORS})")

    pos = {"data": [], "w": [], "t": [], "id": []}
    rot = {"data": [], "w": [], "t": [], "id": []}
    la = {"data": [], "w": [], "t": [], "id": []}

    for k, e in enumerate(effectors):
        if not isinstance(e, dict):
            raise ApiError(f"effectors[{k}] must be an object")
        etype = e.get("type")
        if etype not in ("position", "rotation", "lookat"):
            raise ApiError(f"effectors[{k}].type must be position|rotation|lookat")
        joint = e.get("joint")
        if not isinstance(joint, int) or isinstance(joint, bool) or not (0 <= joint <= 23):
            raise ApiError(f"effectors[{k}].joint must be an int in 0..23")
        weight = e.get("weight", DEFAULT_WEIGHT)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or \
                not np.isfinite(weight) or not (0.0 <= float(weight) <= 1.0):
            raise ApiError(f"effectors[{k}].weight must be a number in [0, 1]")
        tolerance = e.get("tolerance", DEFAULT_TOLERANCE)
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or \
                not np.isfinite(tolerance) or float(tolerance) < 0.0:
            raise ApiError(f"effectors[{k}].tolerance must be a number >= 0")

        if etype == "position":
            p = _as_floats(e.get("pos"), 3, f"effectors[{k}].pos")
            pos["data"].append(p)
            pos["w"].append(float(weight))
            pos["t"].append(float(tolerance))
            pos["id"].append(joint)
        elif etype == "rotation":
            r6 = _as_floats(e.get("rot6d"), 6, f"effectors[{k}].rot6d")
            # ortho6d degenerates (NaN) if either 3-vector is ~zero
            if np.linalg.norm(r6[0:3]) < 1e-5 or np.linalg.norm(r6[3:6]) < 1e-5:
                raise ApiError(f"effectors[{k}].rot6d is degenerate (near-zero basis vector)")
            rot["data"].append(r6)
            rot["w"].append(float(weight))
            rot["t"].append(float(tolerance))
            rot["id"].append(joint)
        else:  # lookat
            target = _as_floats(e.get("pos"), 3, f"effectors[{k}].pos")
            direction = _as_floats(e.get("dir", DEFAULT_LOOKAT_DIR), 3, f"effectors[{k}].dir")
            if np.linalg.norm(direction) < 1e-5:
                raise ApiError(f"effectors[{k}].dir must be a non-zero vector")
            la["data"].append(target + direction)
            la["w"].append(float(weight))
            la["t"].append(float(tolerance))
            la["id"].append(joint)

    if len(pos["id"]) == 0 or sum(pos["w"]) <= 0.0:
        raise ApiError("at least one position effector with weight > 0 is required "
                       "(the model is translation-invariant around the weighted "
                       "centroid of position effectors)")

    input_data = {
        "betas": torch.tensor([betas], dtype=torch.float32),
        "gender": torch.tensor([[gender]], dtype=torch.int64),
    }

    # POSITION (exact non-empty/empty shapes of get_data_from_batch)
    if pos["id"]:
        input_data["position_data"] = torch.tensor([pos["data"]], dtype=torch.float32)
        input_data["position_weight"] = torch.tensor([pos["w"]], dtype=torch.float32)
        input_data["position_tolerance"] = torch.tensor([pos["t"]], dtype=torch.float32)
        input_data["position_id"] = torch.tensor([pos["id"]], dtype=torch.int64)
    else:
        input_data["position_data"] = torch.zeros((1, 0, 3), dtype=torch.float32)
        input_data["position_weight"] = torch.zeros((1, 0), dtype=torch.float32)
        input_data["position_tolerance"] = torch.zeros((1, 0), dtype=torch.float32)
        input_data["position_id"] = torch.zeros((1, 0), dtype=torch.int64)

    # ROTATION (world ortho6d, as documented above)
    if rot["id"]:
        input_data["rotation_data"] = torch.tensor([rot["data"]], dtype=torch.float32)
        input_data["rotation_weight"] = torch.tensor([rot["w"]], dtype=torch.float32)
        input_data["rotation_tolerance"] = torch.tensor([rot["t"]], dtype=torch.float32)
        input_data["rotation_id"] = torch.tensor([rot["id"]], dtype=torch.int64)
    else:
        input_data["rotation_data"] = torch.zeros((1, 0, 6), dtype=torch.float32)
        input_data["rotation_weight"] = torch.zeros((1, 0), dtype=torch.float32)
        input_data["rotation_tolerance"] = torch.zeros((1, 0), dtype=torch.float32)
        input_data["rotation_id"] = torch.zeros((1, 0), dtype=torch.int64)

    # LOOK-AT ([target(3), local_dir(3)]; dir normalized like get_data_from_batch)
    if la["id"]:
        la_t = torch.tensor([la["data"]], dtype=torch.float32)  # [1, N, 6]
        n_la = la_t.shape[1]
        dirs = normalize_vector(la_t[0, :, 3:6].reshape(-1, 3), eps=1e-5).view(1, n_la, 3)
        input_data["lookat_data"] = torch.cat([la_t[:, :, 0:3], dirs], dim=2)
        input_data["lookat_weight"] = torch.tensor([la["w"]], dtype=torch.float32)
        input_data["lookat_tolerance"] = torch.tensor([la["t"]], dtype=torch.float32)
        input_data["lookat_id"] = torch.tensor([la["id"]], dtype=torch.int64)
    else:
        input_data["lookat_data"] = torch.zeros((1, 0, 6), dtype=torch.float32)
        input_data["lookat_weight"] = torch.zeros((1, 0), dtype=torch.float32)
        input_data["lookat_tolerance"] = torch.zeros((1, 0), dtype=torch.float32)
        input_data["lookat_id"] = torch.zeros((1, 0), dtype=torch.int64)

    counts = (len(pos["id"]), len(rot["id"]), len(la["id"]))
    return input_data, counts


def run_solve(payload):
    input_data, counts = build_input_data(payload)
    t0 = time.perf_counter()
    with INFER_LOCK:
        with torch.no_grad():
            predicted = MODEL.forward(input_data)
            # FK exactly as shared_step: 6d -> matrix -> quaternion -> apply_smpl_quat
            rot_mat = rotation_6d_to_matrix(
                predicted["joint_rotations"].view(-1, 6)).view(-1, MODEL.nb_joints, 3, 3)
            quat = matrix_to_quaternion(rot_mat)
            fk_pos, fk_world_rot, verts = MODEL.apply_smpl_quat(
                betas=input_data["betas"],
                joint_rotations_quat=quat,
                root_position=predicted["root_joint_position"],
                gender=input_data["gender"],
                predict_verts=True)
    solve_ms = (time.perf_counter() - t0) * 1000.0

    fk_np = fk_pos[0].numpy()
    verts_np = verts[0].numpy()
    rot6d_np = (
        matrix_to_rotation_6d(fk_world_rot.reshape(-1, 3, 3))
        .view(-1, MODEL.nb_joints, 6)[0]
        .numpy()
    )
    if not (np.isfinite(fk_np).all() and np.isfinite(verts_np).all()):
        raise RuntimeError("model produced non-finite output for this input")

    return {
        "joints": np.round(fk_np.astype(np.float64), 6).tolist(),
        "verts": np.round(verts_np.astype(np.float64), 4).tolist(),
        "rot6d": np.round(rot6d_np.astype(np.float64), 6).tolist(),
        "solve_ms": round(solve_ms, 2),
    }, counts


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence default stderr logging; we log ourselves
        pass

    def _send(self, code, body, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if getattr(self, "_set_cookie", False):
            self.send_header(
                "Set-Cookie",
                f"k={os.environ.get('DEMO_TOKEN', '')}; Path=/; HttpOnly; SameSite=Lax",
            )
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code, obj):
        self._send(code, json.dumps(obj).encode())

    def _token_ok(self):
        token = os.environ.get("DEMO_TOKEN", "")
        if not token:
            return False
        from urllib.parse import urlparse, parse_qs
        query = parse_qs(urlparse(self.path).query)
        if query.get("k", [""])[0] == token:
            self._set_cookie = True
            return True
        return f"k={token}" in self.headers.get("Cookie", "")

    def _authorized(self):
        expected = os.environ.get("DEMO_AUTH", "")
        if not expected:
            return True
        if self._token_ok():
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            import base64
            try:
                supplied = base64.b64decode(header[6:]).decode()
            except Exception:
                supplied = ""
            if supplied == expected:
                return True
        body = json.dumps({"error": "authentication required"}).encode()
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="smplik-demo"')
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        log(f"AUTH-DENIED {self.command} {self.path} from {self.client_address[0]}")
        return False

    def do_GET(self):
        if not self._authorized():
            return
        t0 = time.perf_counter()
        path = self.path.split("?")[0]
        if path == "/":
            if FRONTEND_PATH.exists():
                body = FRONTEND_PATH.read_bytes()
                self._send(200, body, "text/html; charset=utf-8")
                status = 200
            else:
                self._send_json(404, {"error": "frontend.html not found"})
                status = 404
        elif path == "/meta":
            self._send(200, META_JSON)
            status = 200
        elif path in VENDOR_FILES and VENDOR_FILES[path].exists():
            self._send(
                200,
                VENDOR_FILES[path].read_bytes(),
                "application/javascript; charset=utf-8",
            )
            status = 200
        else:
            self._send_json(404, {"error": f"unknown path {path}"})
            status = 404
        log(f"GET {path} -> {status} ({(time.perf_counter() - t0) * 1000:.1f} ms) "
            f"from {self.client_address[0]}")

    def do_POST(self):
        if not self._authorized():
            return
        t0 = time.perf_counter()
        path = self.path.split("?")[0]
        if path != "/solve":
            self._send_json(404, {"error": f"unknown path {path}"})
            log(f"POST {path} -> 404 from {self.client_address[0]}")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                raise ApiError("empty request body")
            if length > 10_000_000:
                raise ApiError("request body too large")
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ApiError(f"invalid JSON: {e}")
            result, counts = run_solve(payload)
            self._send_json(200, result)
            log(f"POST /solve -> 200 pos/rot/lookat={counts[0]}/{counts[1]}/{counts[2]} "
                f"solve={result['solve_ms']:.1f}ms total={(time.perf_counter() - t0) * 1000:.1f}ms "
                f"from {self.client_address[0]}")
        except ApiError as e:
            self._send_json(400, {"error": str(e)})
            log(f"POST /solve -> 400 ({e}) from {self.client_address[0]}")
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": f"internal error: {e}"})
            log(f"POST /solve -> 500 ({e}) from {self.client_address[0]}")


def main():
    log(f"loading model (repo={REPO_ROOT}, ckpt={CKPT_PATH}) ...")
    load_model_and_meta()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    log(f"serving on http://{HOST}:{PORT} (frontend={FRONTEND_PATH})")
    server.serve_forever()


if __name__ == "__main__":
    main()
