# neural-IK-smpl — reproducing SMPL-IK, with multi-GPU engineering

Reproduction and multi-GPU engineering study of **SMPL-IK: Learned
Morphology-Aware Inverse Kinematics** ([arXiv:2208.08274](https://arxiv.org/abs/2208.08274),
upstream code [boreshkinai/smpl-ik](https://github.com/boreshkinai/smpl-ik)),
which extends ProtoRes ([arXiv:2106.01981](https://arxiv.org/abs/2106.01981),
ICLR 2022) to the SMPL body model: a learned IK solver that maps an arbitrary
sparse set of positional / rotational / look-at effectors plus body shape and
gender to a full-body SMPL pose.

**This repository is an overlay.** It contains only *our* work: training/ops
infrastructure (`ops/`), an interactive browser demo (`demo/`), engineering
docs (`docs/`), and a setup script that clones the upstream code at the pinned
commit `ea74d47f8adafed7af95f9c8eac4de6a715bf0bc`. It contains **zero**
upstream source files, **zero** SMPL body model files, **zero** dataset files,
and **zero** checkpoints — all of those have their own licenses and are
obtained by the steps below.

What is in here beyond a plain rerun:

- Exact-environment reproduction of the 2021-era stack (torch 1.9.0+cu111 on
  A100) and a verified dataset rebuild after the upstream download links died.
- A root-caused, bit-exact fix for a **torch >= 1.10 autograd semantics change**
  that makes the upstream geodesic loss NaN-flood 91% of all gradients on
  modern PyTorch (`docs/HOPPER_NUMERICS.md`).
- Linear-scaled multi-GPU training (Goyal et al. 2017) across **separate Slurm
  allocations** — same-node and cross-node — over NCCL/IB, with LR warmup,
  crash-tolerant resume, and honest instability reporting (`docs/MULTI_GPU.md`).
- An interactive three.js demo (drag position/rotation/look-at constraints,
  live CPU inference) and a released pretrained checkpoint.

## Results

Validation, AMASS random-effector protocol (the paper's headline setting);
mm, lower is better.

| Source | MPJPE | PA-MPJPE |
|---|---:|---:|
| Paper (SMPL-IK, AMASS, random effectors) | 59.3 | 52.5 |
| **Ours** (canonical run, val @ epoch 13; campaign concluded at epoch 21 with val random MPJPE holding 58.4-59.1) | **58.39** | **53.53** |

Protocol note: our numbers are the repository's own validation metrics
(`calc_mpjpe`; MPJPE pelvis-aligned, PA-MPJPE Procrustes-aligned) computed on
our rebuilt AMASS validation split. The paper reports numbers on its original
preprocessed dataset, whose download is no longer available (HTTP 403), so we
rebuilt the gender-augmented dataset from raw licensed AMASS with the upstream
conversion code — same semantics, not byte-identical data lineage. Final
epoch-120 and test-set numbers will be added when training completes. Full
campaign status: `docs/REPRODUCTION.md`.

## Setup

```bash
git clone https://github.com/CHDTevior/neural-IK-smpl.git
cd neural-IK-smpl
bash setup.sh          # clones upstream smpl-ik @ ea74d47 into ./smpl-ik and overlays ./ops into it
```

### Canonical environment (single-GPU A100, exact historical stack)

Python 3.8.20, torch 1.9.0+cu111, pytorch-lightning 1.4.7, hydra-core 1.0.4,
pyarrow 4.0.0, pytorch3d 0.6.2, torchmetrics 0.4.0 — with the conda build
strings pinned so the solver cannot silently substitute CPU builds:

```bash
conda create -y -p ./envs/smplik-py38 \
  --override-channels -c pytorch3d -c pytorch -c defaults -c conda-forge \
  python=3.8.20 pip=21.3.1 setuptools=59.5.0 wheel=0.37.1 numpy=1.21.6 \
  'pytorch=1.9.0=py3.8_cuda11.1_cudnn8.0.5_0' \
  'torchvision=0.10.0=py38_cu111' \
  cudatoolkit=11.1.1 \
  'pytorch3d=0.6.2=py38_cu111_pyt190' \
  fvcore=0.1.5.post20221221 iopath=0.1.10

./envs/smplik-py38/bin/python -m pip install \
  --constraint smpl-ik/ops/constraints.compat.txt \
  pytorch-lightning==1.4.7 pandas==1.1.4 pyarrow==4.0.0 \
  scikit-learn==0.23.2 hydra-core==1.0.4 GitPython==3.1.11 plotly==5.6.0 \
  onnx==1.11.0 chumpy==0.70 'smplx[all]==0.1.28' \
  vector-quantize-pytorch==0.2.2 h5py==3.7.0 torchmetrics==0.4.0 \
  roma==1.2.5 markupsafe==2.0.1 tensorboard==2.11.2

./envs/smplik-py38/bin/python -m pip check
```

Equivalent declarative specs ship in the overlay:
`ops/conda-environment.compat.yml`, `ops/constraints.compat.txt`, and
`ops/bootstrap.sh` (set `SMPLIK_ENV_PREFIX` to your target prefix). The full
frozen environment we trained with is the authoritative reference — key pins:
`numpy 1.21.6`, `omegaconf 2.0.6`, `protobuf 3.20.3`, `MarkupSafe 2.0.1`,
`setuptools 59.5.0`. Note `torchmetrics==0.4.0` is *yanked* on PyPI (a DDP
issue that does not affect the single-GPU canonical run) but remains
installable by exact version; keeping it is required for exact reproduction.

### Hopper environment (H100/H200 multi-GPU only)

torch 1.9's CUDA 11.1 binaries cannot execute reliably on `sm_90`, so the
multi-GPU branch uses a separate, isolated environment: Python 3.8.20,
**torch 2.4.1+cu124**, torchvision 0.19.1+cu124, pytorch-lightning 1.4.7,
**torchmetrics 0.4.1** (`ops/constraints.hopper.txt`; 0.4.1 fixes the DDP
metric-state issue 0.4.0 was yanked for), and **pytorch3d v0.7.9** installed
from the release tag with `PYTORCH3D_NO_EXTENSION=1` (SMPL-IK only uses the
pure-Python `pytorch3d.transforms`; fixed-input comparisons confirmed
identical outputs for all five transforms the project imports):

```bash
# tag v0.7.9 == commit 33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba (pin the hash)
PYTORCH3D_NO_EXTENSION=1 pip install \
  'git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba'
```

Read `docs/HOPPER_NUMERICS.md` **before** training on torch >= 1.10 — the
upstream loss is numerically unsafe there without `ops/hopper_numerics_patch.py`
(which the Hopper entry point `ops/run_hopper_ddp.py` applies automatically).

## Data preparation

### 1. SMPL body models (registration required)

Register at <https://smpl.is.tue.mpg.de>, download SMPL v1.0.0 (10 shape
coefficients), and place exactly these five files under
`smpl-ik/tools/smpl/models/` (upstream's `scripts/get_smpl_models.sh`
documents the same layout). Expected SHA-256:

| File | SHA-256 |
|---|---|
| `basicModel_f_lbs_10_207_0_v1.0.0.pkl` | `a583c1b98e4afc19042641f1bae5cd8a1f712a6724886291a7627ec07acd408d` |
| `basicModel_m_lbs_10_207_0_v1.0.0.pkl` | `0e8c0bbbbc635dcb166ed29c303fb4bef16ea5f623e5a89263495a9e403575bd` |
| `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` | `98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42` |
| `J_regressor_extra.npy` | `cc968ea4f9855571e82f90203280836b01f13ee42a8e1b89d8d580b801242a89` |
| `J_regressor_h36m.npy` | `c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0` |

### 2. AMASS (registration required)

The upstream preprocessed dataset download is dead (HTTP 403), so the dataset
is rebuilt from raw AMASS. Register at <https://amass.is.tue.mpg.de> and
download the **SMPL+H** body-data archives for these 18 subsets
(`ops/preprocess_amass.py` encodes the split):

- **Train (12):** ACCAD, BMLhandball, BMLmovi, BioMotionLab_NTroje, CMU,
  DFaust_67, EKUT, Eyes_Japan_Dataset, KIT, MPI_Limits, TCD_handMocap,
  TotalCapture
- **Validation (4):** HumanEva, MPI_HDM05, MPI_mosh, SFU
- **Test (2):** SSM_synced, Transitions_mocap

Extract them so each subset is a directory of `<subset>/<subject>/*.npz` under
one source root, then convert (gender-augmented: every frame is rendered with
the male, female, and neutral SMPL model, matching upstream semantics):

```bash
cd smpl-ik
# GPU recommended; runs subset-by-subset, resumable, refuses ambiguous partial output
../envs/smplik-py38/bin/python ops/preprocess_amass.py \
  --source-root /path/to/raw_amass \
  --output-root ./datasets/amass_gender_augment_cache_v1 \
  convert ACCAD BMLhandball BMLmovi BioMotionLab_NTroje CMU DFaust_67 EKUT \
          Eyes_Japan_Dataset KIT MPI_Limits TCD_handMocap TotalCapture \
          HumanEva MPI_HDM05 MPI_mosh SFU SSM_synced Transitions_mocap

# writes dataset_settings.json, split.json, and a provenance manifest with SHA-256s
../envs/smplik-py38/bin/python ops/preprocess_amass.py \
  --source-root /path/to/raw_amass \
  --output-root ./datasets/amass_gender_augment_cache_v1 \
  finalize
```

(If your KIT copy is the SMPL-X variant, use the SMPL+H one — spec syntax
`KIT_smplh:KIT` maps a differently-named source directory onto the expected
output name.)

Then build the Feather caches through the unmodified repository loader:

```bash
../envs/smplik-py38/bin/python ops/build_dataset_cache.py
```

> **WARNING — memory.** The official cache build loads and deduplicates the
> full 182 GiB CSV set in RAM. Our run peaked at **389.79 GiB RSS**. Run this
> on a node with **>= 800 GiB RAM** (a 200 GiB cgroup is not enough; our first
> attempt had to be stopped). `ops/cache_highmem_job.sh` is the Slurm wrapper
> we used (adapt its cluster-specific headers/paths).

Expected result: 12/4/2 train/val/test files; 50,480,676 / 3,960,168 /
353,013 rows after official deduplication. `ops/dataset_audit.py` and
`ops/validate_batch.py` verify the caches and one real batch;
`ops/smoke_test.py` runs one real 2048-sample train + val step before you
commit to the full run.

## Training — canonical (the reproduction of record)

```bash
cd smpl-ik
../envs/smplik-py38/bin/python run.py --config=configs/experiments/smplik_amass.yaml
```

- Hardware used: 1x NVIDIA A100-SXM4-80GB (~8.4 GiB VRAM used, ~89% util).
- Throughput: **~2h04m per epoch**, 120 epochs total, 24,648 optimizer steps
  per epoch at batch 2048.
- Outputs land in `logs/protores_amass/seed-0_num_blocks_stage1-3/`:
  `checkpoints/epoch=N-step=S.ckpt` (+ `last.ckpt`), TensorBoard events,
  `hparams.yaml`, periodic ONNX exports, and a final `model.onnx` + test pass
  at the end.
- **Resume is repo-native and automatic:** re-running the exact same command
  finds the newest checkpoint for this experiment (after verifying config
  equality) and continues. `ops/formal_resume_job.sh` is our Slurm wrapper
  that chains this across allocation lifetimes (adapt its site-specific
  headers); `ops/monitor_formal_training.sh` is the matching watchdog.

No upstream file is modified for this run — `ops/` is pure addition, and the
launcher-side checksum gate refuses to start if any protected upstream
config/model file differs from the pinned commit.

## Training — multi-GPU (linear-scaled variants)

> These are **linear-scaled approximations** (global batch and LR scaled by
> world size k, per Goyal et al. 2017), not exact reproductions. Full design,
> spike history, and pitfalls: `docs/MULTI_GPU.md`.

```bash
cd smpl-ik
# smoke first, always (validates NCCL rendezvous + gradient finiteness):
SMPLIK_WARMUP_STEPS=7044  SMPLIK_WARMUP_START_LR=0.0002 \
  bash ops/launch_hopper_variant.sh h100x7 smoke
SMPLIK_WARMUP_STEPS=7044  SMPLIK_WARMUP_START_LR=0.0002 \
  bash ops/launch_hopper_variant.sh h100x7 formal

SMPLIK_WARMUP_STEPS=12324 SMPLIK_WARMUP_START_LR=0.0002 \
  bash ops/launch_hopper_variant.sh h200x4 smoke
SMPLIK_WARMUP_STEPS=12324 SMPLIK_WARMUP_START_LR=0.0002 \
  bash ops/launch_hopper_variant.sh h200x4 formal
```

- Variants: `h100x7` = 7 ranks, global batch 14336, lr 1.4e-3;
  `h200x4` = 4 ranks, global batch 8192, lr 8e-4. Per-rank batch stays 2048,
  FP32, no AMP/TF32.
- **Required env:** `SMPLIK_WARMUP_STEPS` / `SMPLIK_WARMUP_START_LR` as above
  (linear LR ramp over the first 2 epochs; resume-exact). Running the scaled
  LRs without warmup is documented to explode — see the spike table in
  `docs/MULTI_GPU.md`.
- **Optional env:** `SMPLIK_GRAD_CLIP_NORM=<float>` enables global-norm
  gradient clipping (default off).
- The launcher aggregates GPUs from several running Slurm allocations via
  `srun --overlap`, one cgroup-isolated rank per GPU, static rendezvous over
  the IB interface, and a per-rank **`NCCL_HOSTID`** override — without it,
  NCCL's NVML topology discovery fails across same-host allocation/cgroup
  boundaries (`nvmlDeviceGetHandleByPciBusId() failed: Not Found`).
- `ops/launch_hopper_variant.sh` and `ops/monitor_hopper_variant.sh` hard-code
  *our* cluster's allocation job IDs, node names, and repo path near the top —
  edit those for your site before use.

Honest status: the unclipped 7x LR (h100x7) remained spike-prone even with
warmup and that line was terminated as an instability study; h200x4 is the
surviving scaled line. On torch >= 1.10 the runs additionally require the
numerics patch, applied automatically — background in `docs/HOPPER_NUMERICS.md`.

## Pretrained weights

Canonical-run checkpoint (epoch 13, the one behind the results table and the
demo) is published at
[huggingface.co/Tevior/neural-ik-smpl](https://huggingface.co/Tevior/neural-ik-smpl):

```bash
pip install -U huggingface_hub
hf download Tevior/neural-ik-smpl smplik_amass_canonical_epoch13.ckpt \
  --local-dir ./weights
```

It is a standard PyTorch Lightning checkpoint for the upstream `SmplModel`
(`state_dict` + optimizer state + hyperparameters); the demo below shows the
exact load path (instantiate model from the composed config, then
`load_state_dict`). License: non-commercial research only (see below).

## Interactive demo

A dependency-light live demo: stdlib-HTTP CPU inference server + a vendored
three.js (r160, MIT) frontend. Drag effector gizmos, the solver re-poses the
full SMPL body in real time.

```bash
# needs: the conda env, the upstream clone (setup.sh), SMPL models, AMASS caches
# (demo_data.json generation reads the test set once), and a checkpoint.
export SMPLIK_REPO=$PWD/smpl-ik
export SMPLIK_CKPT=$PWD/weights/smplik_amass_canonical_epoch13.ckpt

./envs/smplik-py38/bin/python demo/run_demo_inference.py   # writes demo/demo_data.json once
./envs/smplik-py38/bin/python demo/server.py               # serves on 0.0.0.0:8899
```

Open <http://localhost:8899>. Features:

- **Multi-constraint joints:** any of the 24 SMPL joints can carry position,
  rotation (ortho6d world), and look-at effectors simultaneously, each with
  per-effector weight and tolerance — mirroring the model's native input
  exactly (`demo/server.py` documents the encodings).
- **three.js gizmos:** translate/rotate `TransformControls` per effector,
  orbit camera, GT-vs-prediction skeleton overlay, and the full 6890-vertex
  skinned SMPL mesh returned by the gendered FK path.
- Test-set sample poses, per-solve latency readout, and a fail-loud GT
  consistency check at startup.
- Optional auth for exposing it beyond localhost, **env-driven only** (no
  secrets in code or repo): `DEMO_AUTH="user:pass"` enables HTTP Basic;
  `DEMO_TOKEN=<random>` additionally allows `?k=<token>` cookie login.

## Troubleshooting

**NaN gradients / non-finite loss on torch >= 1.10 (the big one).** torch 1.10
changed elementwise `torch.min`/`torch.max` backward at ties from "route the
whole gradient to the other argument" (torch 1.9: the model side gets an exact
0) to "split 0.5/0.5 between the tied inputs". The upstream geodesic loss
clamps `cos` with an eps-less min/max before `acos`; fp32 rounding produces
exact `cos == +/-1` ties every step, where `acos'` is `-inf`. Under the new
rule this leaks `-inf/2` (rotation branch) or `0 * -inf = NaN` (weighted
branch — even zero-weight joints) into ~91% of all gradient values on every
rank, while the forward loss stays finite. The fix is
`ops/hopper_numerics_patch.py` — a custom autograd Function that is forward
bit-identical and backward bit-equal to torch 1.9's semantics (0-mismatch over
600,780 CPU + 500,000 CUDA probed gradient values) — applied automatically by
`ops/run_hopper_ddp.py`. Full evidence chain and verification tables:
`docs/HOPPER_NUMERICS.md`.

Other quick hits:

- `nvmlDeviceGetHandleByPciBusId() failed: Not Found` in NCCL -> you are
  crossing Slurm cgroup/allocation boundaries on one host; set a unique
  `NCCL_HOSTID` per rank (done by `ops/hopper_rank.sh`).
- Cache build OOM-killed -> you ignored the 389.79 GiB peak-RSS warning; use a
  >= 800 GiB node.
- `torchmetrics==0.4.0` "yanked" pip warning -> expected; exact version pin
  still installs and is required for the canonical environment.
- Demo exits with `SMPLIK_CKPT is required` / `not an smpl-ik checkout` ->
  export `SMPLIK_CKPT` and `SMPLIK_REPO` as in the demo section.

## License

- **Our code** (everything tracked in this repo: `ops/`, `demo/server.py`,
  `demo/run_demo_inference.py`, `demo/frontend.html`, `docs/`, `setup.sh`):
  MIT, © 2026 CHDTevior — see `LICENSE`. The MIT grant applies **only to our
  files**, not to anything the setup steps download.
- **Upstream smpl-ik:** non-commercial academic research/reference use only
  (its `LICENSE.md`). **Not included here** — `setup.sh` clones it from the
  upstream repository; the combined working tree inherits the non-commercial
  restriction.
- **SMPL body models** (MPI) and **AMASS**: separate individual registration
  and licenses required; never committed here.
- **three.js** (`demo/vendor/`): MIT, header preserved.
- **Pretrained weights:** trained with the non-commercial upstream code on
  AMASS — **non-commercial research use only**.

Full attributions: `NOTICE.md`.

---

## 中文快速开始

本仓库是 SMPL-IK ([arXiv:2208.08274](https://arxiv.org/abs/2208.08274)) 的
复现 + 多卡工程化 overlay：只包含我们自己写的训练基础设施 (`ops/`)、交互式
演示 (`demo/`) 和文档 (`docs/`)，不含任何上游源码 / SMPL 模型 / 数据集 /
权重（各有独立许可，需按下述步骤自行获取）。

```bash
# 1. 克隆并搭好工作树（自动 clone 上游 smpl-ik @ ea74d47 并叠加 ops/）
git clone https://github.com/CHDTevior/neural-IK-smpl.git
cd neural-IK-smpl && bash setup.sh

# 2. 建环境（精确历史版本：python 3.8.20 + torch 1.9.0+cu111 等，见上文 Setup）
# 3. 数据：在 smpl.is.tue.mpg.de 和 amass.is.tue.mpg.de 注册下载（见 Data preparation）
#    注意：官方缓存构建峰值内存 389.79 GiB，请用 >= 800 GiB 内存节点
# 4. 单卡正式训练（约 2h04m/epoch，共 120 epochs，可自动断点续训）
cd smpl-ik && ../envs/smplik-py38/bin/python run.py --config=configs/experiments/smplik_amass.yaml

# 5. 或直接下载我们的权重跑交互 demo（浏览器里拖拽约束、实时求解）
cd .. && hf download Tevior/neural-ik-smpl smplik_amass_canonical_epoch13.ckpt --local-dir ./weights
export SMPLIK_REPO=$PWD/smpl-ik SMPLIK_CKPT=$PWD/weights/smplik_amass_canonical_epoch13.ckpt
./envs/smplik-py38/bin/python demo/run_demo_inference.py
./envs/smplik-py38/bin/python demo/server.py   # 打开 http://localhost:8899
```

结果：论文 AMASS 随机 effector 验证 MPJPE 59.3 / PA-MPJPE 52.5；我们的
canonical 运行在第 13/120 epoch 已达 **58.39 / 53.53**（训练进行中）。
多卡（7xH100 / 4xH200 跨 Slurm allocation 的 NCCL/IB DDP、线性缩放 + LR
warmup）与 torch>=1.10 min/max 平局梯度语义变更导致 91% NaN 梯度的根因分析，
分别见 `docs/MULTI_GPU.md` 与 `docs/HOPPER_NUMERICS.md`。许可：本仓库代码
MIT；上游代码与训练权重仅限非商业学术研究使用。
