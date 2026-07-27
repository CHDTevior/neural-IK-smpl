# Reproduction campaign summary

Reproduction of SMPL-IK (arXiv:2208.08274, upstream
`boreshkinai/smpl-ik@ea74d47`) on an academic Slurm cluster, July 2026.
Cluster-specific paths and job IDs appear below as historical record; adapt
them to your site.

## Timeline

**2026-07-24**
- Official Docker path unavailable (`docker: command not found` on all nodes);
  built the pinned conda compatibility environment instead (Python 3.8.20,
  torch 1.9.0+cu111 — exact historical builds, see README Setup).
- Upstream GCS bucket returns HTTP 403 for both the preprocessed AMASS ZIP and
  the SMPL helper objects. Rebuilt the gender-augmented dataset from licensed
  local AMASS with the upstream conversion semantics
  (`ops/preprocess_amass.py`; provenance manifest + per-subset SHA-256
  recorded). 18 subsets, 50.5M/3.96M/353K train/val/test rows after official
  deduplication, 182 GiB of CSVs.
- Feather cache build through the unmodified repository loader needed a
  high-memory node: Slurm job with an 800 GiB cgroup, **peak RSS 389.79 GiB**,
  zero OOM events (an earlier 200 GiB attempt was stopped pre-emptively).
- Batch validation, smoke test (finite loss 12.270475, one real
  train/val step) passed; canonical training launched 10:45:28 +01:00 on one
  A100-SXM4-80GB: `python run.py --config=configs/experiments/smplik_amass.yaml`,
  ~2h04m/epoch, 120 epochs, with an `afterany` continuation chain and the
  repo-native auto-resume.
- A same-node 4xA100 `ddp_spawn` conversion was audited and **rejected for the
  canonical line** (it would change global batch 2048 -> 8192); kept as a
  config artifact (`ops/smplik_amass_ddp4_linear*.yaml`).
- First 7xH100 cross-allocation smoke: NaN model/Adam state after the first
  update despite a finite first loss. Evidence preserved; per-rank
  pre-all-reduce gradient probes captured (~91% NaN on all 7 ranks before
  reduction).

**2026-07-25**
- Root cause isolated: torch >= 1.10 elementwise min/max tie-gradient
  semantics change x the eps-less acos clamp in the upstream geodesic loss.
  Bit-exact compatibility patch written, adversarially reviewed, verified with
  0-mismatch bit tables in both environments (full narrative:
  `docs/HOPPER_NUMERICS.md`).
- v1 no-warmup formal attempts on both Hopper variants hit large loss
  excursions (spike table in `docs/MULTI_GPU.md`); v2 added the
  Goyal-style linear LR warmup (`ops/hopper_lr_warmup.py`), archived v1.
- h200x4 survived an allocation expiry mid-training: allocation migration +
  crash-tolerant resume from `last.ckpt` (fingerprint-checked, finite-checked)
  with zero non-finite values at the resumed step.
- External cross-model closure review (GPT-5.6 class, max effort): **PASS**
  on the numerics patch, the warmup wire-in, and the migration/resume; the
  canonical path verified isolated (protected-file checksums clean, no `ops`
  import reachable from `run.py`).
- Interactive demo built (CPU inference server + three.js frontend) against
  the canonical epoch-13 checkpoint.

## The three training lines

| Line | Hardware | Status | Role |
|---|---|---|---|
| canonical | 1x A100-SXM4-80GB, torch 1.9.0+cu111 | **concluded at epoch 21/120** (2026-07-27, by owner decision; val random MPJPE reached the paper's 59.3 level at epoch 13 and plateaued at 58.4-59.1 thereafter) | The reproduction of record. Unmodified command, config, and code. |
| h100x7 | 7x H100 across 4 same-node Slurm allocations, torch 2.4.1+cu124 | **terminated; kept as instability study** | Exposed the torch tie-gradient regression; its unclipped 7x lr (1.4e-3) remained spike-prone even with warmup. Evidence archived. |
| h200x4 | 4x H200 across 2 nodes / 2 allocations, torch 2.4.1+cu124 | **concluded at epoch 75/120** (2026-07-27; val random MPJPE ~82.6mm and improving, but epoch-for-epoch behind canonical) | Large-batch study: naive linear scaling proved spike-prone (see docs/MULTI_GPU.md); a modernized sqrt-scaling + cosine-decay + grad-clip recipe (v4) was designed and fully reviewed but the campaign ended before deployment. |

## Current metrics

Validation, random-effector protocol (the paper's headline AMASS setting);
mm, lower is better. PA = Procrustes-aligned.

| Source | MPJPE | PA-MPJPE |
|---|---:|---:|
| Paper (SMPL-IK, AMASS random effectors) | 59.3 | 52.5 |
| Ours, canonical val @ epoch 13/120 (final campaign metrics plateaued at this level) | **58.39** | **53.53** |
| Ours, canonical val @ epoch 0 (first validation) | 70.24 | 60.88 |

Protocol note: our numbers are the repo's own validation metrics
(`calc_mpjpe`, pelvis-index alignment for MPJPE, Procrustes for PA) on the
rebuilt AMASS validation split; the paper reports test-set numbers of its
original (now unavailable) preprocessed dataset, so this is a
close-but-not-identical data lineage. Final test-set numbers follow at
epoch 120.

## What "done" means

The campaign is complete when, for the canonical line:

1. **120 epochs** finish (native auto-resume chain);
2. the official **test pass** runs and test-set MPJPE / PA-MPJPE are recorded;
3. the final **ONNX export** (`model.onnx`) is produced by the unmodified
   export path and load-verified;
4. **visual QA** passes: rendered skeleton/mesh demos (not just metrics) are
   inspected for the failure modes numbers can hide — frozen poses, jitter,
   collapse, interpenetration.

The h200x4 line is additionally judged on end-to-end stability evidence
(no unrecovered non-finite events across its 120 epochs) and a final
side-by-side metric comparison against the canonical line at equal data
traversal counts.
