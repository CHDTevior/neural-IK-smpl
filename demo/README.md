# Interactive SMPL-IK demo — server, frontend, API

A dependency-light live demo of the trained SMPL-IK solver: a Python
**stdlib-HTTP CPU inference server** (`server.py`) serving a **self-contained
three.js frontend** (`frontend.html`, vendored three.js r160 — no CDN, no
build step). You drag position / rotation / look-at constraint gizmos in the
browser; every edit round-trips through `POST /solve` and the full SMPL body
(24-joint skeleton + 6890-vertex skinned mesh) re-poses live.

How the demo maps browser constraints onto the model's native interface, the
no-jump interaction design, and the numerical verification gates are
documented separately in [`../docs/DEMO_ALGORITHMS.md`](../docs/DEMO_ALGORITHMS.md).

## 1. Architecture

```
                browser (frontend.html, three.js r160, self-contained)
                   |
                   |  GET  /            frontend.html
                   |  GET  /vendor/*    vendored three.js modules (same-origin)
                   |  GET  /meta        skeleton, mesh faces, demo samples (once)
                   |  POST /solve       constraints -> full pose (~80 ms cadence
                   v                                   while dragging)
  +------------------------------------------------------------------+
  |  demo/server.py  --  Python stdlib http.server, 0.0.0.0:8899     |
  |    ThreadingHTTPServer; solves serialized by one global lock     |
  |    optional auth: HTTP Basic (DEMO_AUTH) + URL token (DEMO_TOKEN)|
  |    CPU-only torch (CUDA_VISIBLE_DEVICES forced empty), 4 threads |
  |                                                                  |
  |    upstream SmplModel  <- SMPLIK_REPO (clone made by setup.sh)   |
  |    trained checkpoint  <- SMPLIK_CKPT                            |
  |    demo_data.json      <- SMPLIK_DEMO_DATA (run_demo_inference.py)|
  +------------------------------------------------------------------+
                   ^
                   |  optional public exposure (see section 4.4)
       cloudflared tunnel --url http://localhost:8899   (quick tunnel)
       or            ssh -L 8899:<compute-node>:8899 <user>@<cluster>
```

Three files do all the work:

| File | Role |
|---|---|
| `run_demo_inference.py` | One-shot offline pass: loads the checkpoint, reads the first test-set block, picks 8 high-variance samples, runs the fixed 6-effector scheme + an all-24-effector self-transfer + two precomputed drag sweeps, writes `demo_data.json` (the reference data the server and both viewers consume). |
| `server.py` | Long-running inference server. Loads the model once, rebuilds the 8 samples' metadata from the dataset (with a fail-loud GT cross-check against `demo_data.json`), then serves `/meta` and `/solve`. |
| `frontend.html` | The interactive client (UI text is Chinese). Served at `/`; talks only to the same-origin API. `offline_viewer.html` is a separate, server-less viewer (section 7). |

## 2. Prerequisites

Everything from the main [`../README.md`](../README.md) up to (but not
including) training:

1. **Upstream clone + overlay** — `bash setup.sh` from the repo root, which
   clones `boreshkinai/smpl-ik` at the pinned commit into `./smpl-ik`.
   `server.py` refuses to start if `$SMPLIK_REPO/smplik/` does not exist.
2. **The canonical conda environment** (README "Setup"): python 3.8.20,
   torch 1.9.0+cu111, pytorch3d 0.6.2, hydra-core 1.0.4, etc. The server
   imports `hydra`, `pytorch3d.transforms`, `sklearn`, and the upstream
   `smplik` package.
3. **SMPL body model files** under `smpl-ik/tools/smpl/models/` (README
   "Data preparation", step 1) — the server instantiates all three gendered
   SMPL bodies for the FK/mesh path.
4. **AMASS Feather caches** (README "Data preparation", step 2) — needed by
   *both* `run_demo_inference.py` and `server.py`: each reads the first
   2048-row test block to (re)derive the 8 demo samples deterministically.
5. **A trained checkpoint**, e.g. `smplik_amass_canonical_epoch13.ckpt` from
   [huggingface.co/Tevior/neural-ik-smpl](https://huggingface.co/Tevior/neural-ik-smpl):

   ```bash
   hf download Tevior/neural-ik-smpl smplik_amass_canonical_epoch13.ckpt --local-dir ./weights
   ```

No GPU is needed or used: the demo is CPU-only by construction.

## 3. Configuration reference

All configuration is environment-driven; there are no secrets and no
machine-specific paths in the code.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `SMPLIK_REPO` | no | `../smpl-ik` relative to this directory (i.e. the clone `setup.sh` creates at the repo root) | Path to the upstream smpl-ik checkout. Validated at startup: must contain a `smplik/` package directory, else the process exits with an explanatory message. Both `server.py` and `run_demo_inference.py` `sys.path`-insert it and `chdir` into it (Hydra config composition and dataset paths resolve relative to the checkout). |
| `SMPLIK_CKPT` | **yes** (both scripts) | — | Path to the trained `.ckpt` (PyTorch Lightning checkpoint). Loaded `map_location="cpu"`, `strict=True` first; falls back to `strict=False` only if the *only* mismatched keys are torchmetrics state buffers, and aborts otherwise. |
| `SMPLIK_DEMO_DATA` | no | `demo_data.json` next to `server.py` | Where the server reads the reference data written by `run_demo_inference.py`. The server fails loudly at startup if the live test-set GT deviates from this file by more than 1e-3 m (dataset/indexing mismatch guard). |
| `DEMO_AUTH` | no | unset = **open mode**, no authentication | `"user:pass"` — enables auth for *every* route. Requests must present matching HTTP Basic credentials (or a valid token, below); everything else gets `401` + `WWW-Authenticate: Basic realm="smplik-demo"`. |
| `DEMO_TOKEN` | no | unset | URL-token alternative credential, **only active when `DEMO_AUTH` is also set** (with `DEMO_AUTH` unset the server is open and the token is ignored). A request carrying `?k=<token>` is authorized and receives `Set-Cookie: k=<token>; Path=/; HttpOnly; SameSite=Lax`; subsequent requests are authorized by that cookie. This is what makes a shareable link work: opening `http://host:8899/?k=<token>` once logs the browser in for all follow-up `/meta` and `/solve` calls. |
| `CUDA_VISIBLE_DEVICES` | no | — | `server.py` **force-sets it to empty** before importing torch — the server can never touch a GPU regardless of what you export. `run_demo_inference.py` only *defaults* it to empty (`setdefault`), and likewise runs its whole pipeline on CPU. |

Hard-coded constants (edit `server.py` to change): listen address
`0.0.0.0:8899`; `torch.set_num_threads(4)` (deliberately gentle — the server
was designed to coexist with trainings on a shared node); at most **64
effectors** per request; request bodies capped at 10 MB; effector defaults
`weight=1.0`, `tolerance=0.0`.

## 4. Running

### 4.1 One-time: generate `demo_data.json`

```bash
cd <repo-root>
export SMPLIK_REPO=$PWD/smpl-ik
export SMPLIK_CKPT=$PWD/weights/smplik_amass_canonical_epoch13.ckpt

./envs/smplik-py38/bin/python demo/run_demo_inference.py
```

Writes `demo/demo_data.json` (~70 kB; not committed — it is derived data) and
`demo/summary.txt` with the per-sample MPJPE numbers. Re-run this whenever
the checkpoint changes, so the reference predictions match the model the
server is about to load.

### 4.2 Local, open mode

```bash
./envs/smplik-py38/bin/python demo/server.py
```

Wait for the `model + meta ready in ...s` log line, then open
<http://localhost:8899>. No authentication: fine on a trusted machine or
behind an SSH port-forward, wrong for anything reachable by others.

### 4.3 Authed mode

```bash
DEMO_AUTH="demo:$(openssl rand -hex 9)" \
DEMO_TOKEN="$(openssl rand -hex 9)" \
  ./envs/smplik-py38/bin/python demo/server.py
```

(Echo the values once for yourself; they live only in that process's
environment — nothing is written to disk or committed.) Browsers get the
standard Basic-auth prompt, or you hand out the token link
`http://<host>:8899/?k=<token>`. Failed attempts are logged as
`AUTH-DENIED`.

Caveats worth knowing: Basic credentials and the `?k=` token are cleartext
over plain HTTP — pair auth with TLS (the cloudflared tunnel below
terminates TLS for you) or keep it inside SSH. The token also lands in
browser history and in the address bar of anyone you screen-share with.

### 4.4 Public exposure (optional)

**Cloudflared quick tunnel** — no account, one command:

```bash
cloudflared tunnel --url http://localhost:8899
```

`cloudflared` prints a randomly-generated public HTTPS URL and proxies it to
the local port. Honest caveats:

- **The URL is ephemeral.** Every restart of `cloudflared` mints a different
  random hostname; there is no uptime guarantee on quick tunnels. Fine for a
  supervised showing, wrong for a permanent link.
- **A token is strongly recommended** (`DEMO_AUTH` + `DEMO_TOKEN`, section
  4.3). A quick tunnel has no access control of its own — without auth the
  solver is open to anyone who guesses or is leaked the URL. Share the
  `/?k=<token>` form of the link.
- **Check your acceptable-use policy first.** If the server runs on a shared
  HPC cluster, exposing an inbound public service from a compute or login
  node may violate the site's AUP; some sites also block or frown on
  long-lived outbound tunnels. Keep it short-lived, authed, and sanctioned.

**SSH port-forward** — the no-new-attack-surface alternative when everyone
who needs the demo has cluster access:

```bash
ssh -L 8899:<compute-node>:8899 <user>@<cluster-login>
# then open http://localhost:8899 locally
```

## 5. HTTP API reference

All responses are JSON unless noted, with `Access-Control-Allow-Origin: *`
(you may call the API from scripts or other origins). Protocol is HTTP/1.1
with explicit `Content-Length`. When `DEMO_AUTH` is set, every route below
first passes the auth check described in section 3.

### `GET /`

Serves `frontend.html` (`text/html`) from this directory; `404` JSON if the
file is missing. `GET /vendor/three.module.js`, `/vendor/OrbitControls.js`,
`/vendor/TransformControls.js` serve the vendored modules
(`application/javascript`). Any other path: `404` `{"error": "unknown path ..."}`.

### `GET /meta`

Model/skeleton/sample metadata, built once at startup and cached. Fields:

| Field | Type / shape | Meaning |
|---|---|---|
| `joint_names` | `[24]` strings | SMPL joint names, index = joint id used everywhere else (`0 pelvis ... 15 head ... 20/21 left/right wrist ...`). |
| `parents` | `[24]` ints | Kinematic tree; `parents[0] == -1` for the pelvis root. |
| `faces` | `[13776][3]` ints | SMPL mesh triangle indices (shared by all genders). |
| `n_verts` | int (`6890`) | Vertex count of the skinned mesh returned by `/solve`. |
| `samples` | `[8]` objects | Demo start poses from AMASS test block 0. Each: `id` (row index in the block), `betas` `[10]`, `gender` (int, see `gender_values`), `gt_joints` `[24][3]` ground-truth joint positions (meters), and `gt_rot6d` `[24][6]` — the ground-truth **world** rotation of every joint in the exact ortho6d encoding the network consumes for rotation effectors (documented extra beyond the minimal contract; the frontend uses it to seed rotation constraints, and it is the correct value to send back as a `rotation` effector). |
| `gender_values` | object | `{"male": 0, "female": 1, "neutral": 2}` — the wire encoding of `gender`. |
| `defaults` | object | `{"weight": 1.0, "tolerance": 0.0}` — server-side defaults for omitted effector fields. |

### `POST /solve`

Request body (JSON object):

```jsonc
{
  "betas": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   // 10 finite floats (SMPL shape)
  "gender": 2,                                // 0 = male, 1 = female, 2 = neutral (exactly these)
  "effectors": [ /* 1..64 effector objects */ ]
}
```

Common effector fields: `type` (`"position" | "rotation" | "lookat"`),
`joint` (int `0..23`), `weight` (float in `[0, 1]`, default `1.0`),
`tolerance` (float `>= 0`, meters, default `0.0`). Type-specific payloads —
these encodings are exactly what the network was trained on:

- **`position`** — `"pos": [x, y, z]` in meters, world frame (same frame as
  `gt_joints`; Z-up).
- **`rotation`** — `"rot6d": [r00, r01, r02, r10, r11, r12]`: the **first two
  ROWS** of the joint's 3x3 **world** (global, not parent-relative) rotation
  matrix, flattened — i.e. pytorch3d `matrix_to_rotation_6d` of the FK world
  rotation. Identity is `[1, 0, 0, 0, 1, 0]`. Requests where either 3-vector
  has near-zero norm (< 1e-5) are rejected as degenerate. The `rot6d` field
  of a previous `/solve` reply (or `gt_rot6d` from `/meta`) is directly
  valid here. Note the ROWS convention: sending the first two *columns*
  yields the transposed (inverse) rotation — see
  [`../docs/DEMO_ALGORITHMS.md`](../docs/DEMO_ALGORITHMS.md).
- **`lookat`** — `"pos": [x, y, z]` world-space target point, plus optional
  `"dir": [x, y, z]` local direction (default `[0, 0, 1]`), the joint-local
  axis that should point at the target. `dir` must be non-zero; the server
  normalizes it with the repo's `normalize_vector` (eps 1e-5), matching
  training.

**Constraint: at least one `position` effector with `weight > 0` is
required.** The model is translation-invariant — inputs are re-referenced
around the weighted centroid of position effectors — so a request with none
has no defined world frame. The server does not guess; it fails loudly:

```json
HTTP 400
{"error": "at least one position effector with weight > 0 is required (the model is translation-invariant around the weighted centroid of position effectors)"}
```

All other validation errors (wrong shapes, non-finite numbers, bad gender,
out-of-range joint/weight/tolerance, > 64 effectors, > 10 MB body, invalid
JSON) also return `400` with a specific `{"error": ...}` message. Unexpected
server-side failures return `500`; a non-finite model output is one of them
(the server checks and refuses to ship NaNs).

Success reply:

```jsonc
{
  "joints": [[x, y, z], ...],   // [24][3] FK joint positions, meters (6 dp)
  "verts":  [[x, y, z], ...],   // [6890][3] skinned SMPL mesh vertices, meters (4 dp)
  "rot6d":  [[...6...], ...],   // [24][6] world rotation of every joint of the
                                //   SOLVED pose, same ortho6d encoding as the
                                //   rotation-effector input (6 dp)
  "solve_ms": 35.1              // model forward + FK time on the server, ms
}
```

`rot6d` closing the loop with the `rotation` input encoding is what lets a
client seed a new rotation constraint from the current pose with zero jump.

### curl examples

Smoke the metadata:

```bash
curl -s http://localhost:8899/meta | python3 -c \
  'import json,sys; m=json.load(sys.stdin); \
   print(len(m["joint_names"]), "joints,", len(m["faces"]), "faces,", \
         m["n_verts"], "verts, samples:", [s["id"] for s in m["samples"]])'
```

Minimal literal solve (neutral body, pelvis pinned at the origin, head 0.6 m
above it):

```bash
curl -s -X POST http://localhost:8899/solve \
  -H 'Content-Type: application/json' \
  -d '{"betas":[0,0,0,0,0,0,0,0,0,0],"gender":2,"effectors":[
        {"type":"position","joint":0,"pos":[0,0,0]},
        {"type":"position","joint":15,"pos":[0,0,0.6]}]}'
```

Realistic solve driven by `/meta` (first demo sample, its six starter joints
pinned at ground truth):

```bash
curl -s http://localhost:8899/meta | python3 - <<'PY' > /tmp/solve_req.json
import json, sys
m = json.load(sys.stdin); s = m["samples"][0]
req = {"betas": s["betas"], "gender": s["gender"],
       "effectors": [{"type": "position", "joint": j, "pos": s["gt_joints"][j]}
                     for j in (0, 15, 20, 21, 10, 11)]}
print(json.dumps(req))
PY
curl -s -X POST http://localhost:8899/solve -H 'Content-Type: application/json' \
  --data @/tmp/solve_req.json | python3 -c \
  'import json,sys; r=json.load(sys.stdin); \
   print(len(r["joints"]), "joints,", len(r["verts"]), "verts, solve_ms", r["solve_ms"])'
```

The fail-loud path (no position effector):

```bash
curl -s -X POST http://localhost:8899/solve -H 'Content-Type: application/json' \
  -d '{"betas":[0,0,0,0,0,0,0,0,0,0],"gender":2,"effectors":[
        {"type":"rotation","joint":0,"rot6d":[1,0,0,0,1,0]}]}'
# -> HTTP 400 {"error": "at least one position effector with weight > 0 is required ..."}
```

With auth enabled:

```bash
curl -s -u "$DEMO_AUTH" http://localhost:8899/meta        # HTTP Basic
curl -s "http://localhost:8899/meta?k=$DEMO_TOKEN"        # URL token (+ cookie in browsers)
```

## 6. Operations notes

- **Startup is minutes, not seconds.** Hydra config composition, datamodule
  setup, three gendered SMPL bodies, checkpoint load, and the first
  test-block read all happen before the socket opens; readiness is the
  `model + meta ready in ...s` log line. Our recorded runs took between
  ~80 s (warm filesystem cache) and 181 s (cold, shared cluster filesystem —
  see the timing in the startup log). The dataset objects are freed after
  metadata extraction; steady-state memory is the model plus the cached
  `/meta` JSON.
- **Solve latency.** With `torch.set_num_threads(4)`: ~10 ms per solve on an
  otherwise idle CPU, 35-38 ms steady-state in our recorded smoke run on a
  busy shared compute node, and a slower first solve (~120 ms) while
  allocator pools warm up. The reply's `solve_ms` is pure model+FK time; the
  frontend also displays the HTTP round-trip.
- **Threading model.** `ThreadingHTTPServer` (daemon threads) accepts
  concurrent clients, but every solve runs under one global
  `threading.Lock` — inference is serialized, so N simultaneous users share
  one solver and queue. This is a deliberate simplicity/footprint choice for
  a demo, not a scalability design.
- **Log format.** One line per request to stdout, e.g.
  `[2026-01-01 12:00:00] POST /solve -> 200 pos/rot/lookat=6/0/0 solve=35.1ms total=45.2ms from 203.0.113.7`,
  plus `GET` lines, `400` lines with the validation message, and
  `AUTH-DENIED` lines when auth is on. Default `BaseHTTPRequestHandler`
  stderr logging is silenced in favor of these.
- **Restarting.** There is no reload endpoint; env or checkpoint changes
  require a restart. The practical pattern:

  ```bash
  pkill -f '[s]erver.py'    # the [s] keeps pkill from matching its own command line
  SMPLIK_CKPT=... nohup ./envs/smplik-py38/bin/python demo/server.py > server.log 2>&1 &
  ```

- **Keep `demo_data.json` in sync with the checkpoint.** The startup GT
  cross-check catches dataset drift, but the *predictions* stored in
  `demo_data.json` (used as the reference by the offline viewer and the
  verification tests) are checkpoint-specific — regenerate after swapping
  checkpoints (section 4.1).
- **Frontend behavior under failure is fail-loud:** server errors surface
  verbatim in a red banner (with a retry button), `/meta` fetch retries with
  backoff, and the client never masks a `400` by silently "fixing" the
  request.

## 7. Offline viewer (`offline_viewer.html`)

A second, entirely static viewer that needs **no server and no network**: a
single canvas-2D HTML file showing GT-vs-predicted skeletons for the 8 demo
samples, the all-24-effector self-transfer, and two *precomputed* drag sweeps
(right wrist on a 7x7 grid at 8 cm spacing, head on a 5x5 grid at 6 cm — every
grid point an independent solve baked in by `run_demo_inference.py`).

**Out of the box:** this repo ships `demo/demo_data.json` (8 single-frame
AMASS test poses with GT joints, effector sets, epoch-13 model predictions,
the self-transfer case, and both precomputed sweeps) and the pre-injected
`demo/offline_viewer_prebuilt.html` — open the latter directly in any browser
(`file://` is fine), nothing else needed.

Provenance/license note: `demo_data.json` is a de-minimis excerpt derived
from the AMASS test split (8 static single-frame poses, joint coordinates and
betas only — no motion sequences, no mesh data). It is provided solely to
make this viewer self-contained; the AMASS license terms
(https://amass.is.tue.mpg.de) apply to the underlying data. To regenerate it
against your own checkpoint, run `run_demo_inference.py`.

To rebuild the viewer from a fresh `demo_data.json`, the committed
`offline_viewer.html` is the template (data slot `const DATA = /*__DATA__*/null;`):

```python
from pathlib import Path
html = Path("demo/offline_viewer.html").read_text()
Path("demo/offline_viewer_prebuilt.html").write_text(
    html.replace("/*__DATA__*/null", Path("demo/demo_data.json").read_text()))
``` Use it when you want to show results without standing up the
environment — it trades interactivity (no free dragging, no mesh, no live
solver) for zero moving parts.

## 8. Files in this directory

| File | Committed | Notes |
|---|---|---|
| `server.py` | yes | Inference server (this document, sections 3-6). |
| `frontend.html` | yes | Live client, served at `/`. |
| `run_demo_inference.py` | yes | Generates `demo_data.json` + `summary.txt`. |
| `offline_viewer.html` | yes | Static viewer template (section 7). |
| `vendor/three.module.js`, `vendor/OrbitControls.js`, `vendor/TransformControls.js` | yes | three.js r160, MIT (see `../NOTICE.md`). |
| `demo_data.json`, `summary.txt` | no (generated) | Derived from your checkpoint + dataset. |
