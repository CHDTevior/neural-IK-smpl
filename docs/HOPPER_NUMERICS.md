# The torch >= 1.10 min/max tie-gradient regression, and a bit-exact fix

This document is the full root-cause narrative behind
`ops/hopper_numerics_patch.py`: why every multi-GPU run of SMPL-IK under
PyTorch 2.4 flooded ~91% of all gradient values with NaN on every rank at the
very first optimizer step, while the identical model, data, and hyperparameters
trained flawlessly under PyTorch 1.9 — and how the fix was verified to be
bit-exact against torch 1.9's semantics.

## Symptom

The first complete 7xH100 DDP smoke attempt (`20260724T2310Z-h100-smoke3`)
produced a **finite first loss** (`15.302417...`) but non-finite model weights
and Adam state after the first update. A per-rank pre-all-reduce gradient
probe (a DDP comm hook that records unreduced bucket statistics before
delegating to the normal FP32 all-reduce) showed that **all seven ranks
independently** already carried 36.26M–36.50M NaN gradient values — about
91.24% of all gradient elements — *before* any NCCL communication. This
immediately excluded the network stack (NCCL, IB transport, cross-allocation
cgroups) and the DDP reduction itself.

## Root cause

Two ingredients interact:

### 1. The upstream geodesic loss clamps with eps-less min/max

`smplik/geometry/rotations.py` (upstream, unmodified) computes the geodesic
distance between rotation matrices as:

```python
def compute_geodesic_distance_from_two_matrices(m1, m2):
    m = torch.bmm(m1, m2.transpose(1, 2))            # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.ones_like(cos))       # clamp to [-1, 1] ...
    cos = torch.max(cos, -torch.ones_like(cos))      # ... with NO epsilon
    return torch.acos(cos)
```

With fp32 rounding, near-identical rotations regularly produce **exactly**
`cos == 1.0` (and occasionally `-1.0`). At a per-rank batch of 2048 samples x
24 joint-rotation pairs, exact ties occur every single step. At `cos = ±1`,
`d/dx acos(x) = -1/sqrt(1-x²) = -inf`.

### 2. torch >= 1.10 changed elementwise min/max tie gradients

- **torch 1.9**: at a tie `min(a, b)` with `a == b` routes the *entire*
  gradient to the *other* argument. Here the "other" argument is the constant
  `ones_like` tensor, so the model side receives an **exact 0** — the `-inf`
  from `acos'` is multiplied by that exact hard 0 along a branch that autograd
  prunes; the model never sees it.
- **torch >= 1.10** (still true in 2.4.1): the tie gradient is **split
  0.5/0.5** between the tied inputs. The model side now receives
  `0.5 * grad`, and the chain through `acos` contributes `-inf`:
  - in the **rotation-loss branch**, `-inf/2 = -inf` leaks directly;
  - in the **weighted global branch** (`smplik/losses/weighted_geodesic.py`),
    the per-joint weight multiplies it: `0 * -inf = NaN` — so even
    **zero-weight joints** poison the graph.

One NaN/Inf in the loss graph back-propagates through the shared trunk and
floods the gradients of essentially every parameter — hence ~91% NaN on every
rank, with a perfectly finite forward loss.

The canonical single-GPU A100 run uses torch **1.9.0** and is unaffected *by
construction*. Only the Hopper (H100/H200) environments require torch 2.4.1
(the CUDA 11.1 binaries of torch 1.9 cannot execute reliably on `sm_90`), so
only the multi-GPU branch hit the regression.

## Evidence chain

1. **Per-rank pre-all-reduce probe** — all 7 ranks NaN *before* reduction,
   finite first loss (`run_reports/h100x7_smoke_attempts/20260724T2253Z-h100-gradprobe/`
   in the working tree; see `ops/run_hopper_ddp.py`, `local_gradient_probe_hook`).
2. **CPU A/B probe in both pinned environments**
   (`nan_rootcause_cpu_probe_20260725.py`): the identical eps-less
   min/max->acos composite yields model-side gradient exactly 0 at ties under
   torch 1.9 and `-inf/2` / `0*-inf = NaN` under torch 2.4 — on CPU, with no
   NCCL, no DDP, no GPU.
3. **Single-rank replay diagnostic**
   (`ops/diagnose_single_rank_gradient.py`): replayed rank 0's exact first
   `DistributedSampler` batch on one GPU and reproduced the live failure
   **bit-for-bit** — loss `15.302417429474527`, `36,350,216` NaN gradient
   values, identical to the DDP evidence — and isolated the `rotation` and
   `lookat` loss branches as the sources. With `--numerics-patch torch19` the
   same replay yields **zero** NaN and zero Inf gradient values with the
   bit-identical loss.

## The patch (`ops/hopper_numerics_patch.py`)

Design goals: (a) forward bit-identical to the official composite for every
non-NaN input, (b) backward bitwise equal to what torch 1.9's original
composite computed at every probed point, (c) applied **only** in the Hopper
entry points — the canonical path never imports it.

```python
class _AcosOfClampedCos(torch.autograd.Function):
    """acos(clamp(cos, -1, 1)) with torch-1.9 tie-routing backward."""

    @staticmethod
    def forward(ctx, cos):
        ctx.save_for_backward(cos)
        return torch.acos(torch.clamp(cos, -1.0, 1.0))

    @staticmethod
    def backward(ctx, grad_output):
        (cos,) = ctx.saved_tensors
        inside = (cos > -1.0) & (cos < 1.0)
        # Mirror autograd's acos derivative op-for-op (grad * -rsqrt(-x*x+1),
        # see derivatives.yaml) so the inside-gradient is bitwise equal to
        # what both torch 1.9 and torch 2.4 autograd compute.
        safe = torch.where(inside, cos, torch.zeros_like(cos))
        grad_cos = grad_output * -torch.rsqrt(-safe * safe + 1)
        return torch.where(inside, grad_cos, torch.zeros_like(grad_output))
```

Key decisions:

- **Forward** uses `clamp(cos, -1, 1)`: for every ordered non-NaN float this
  selects the same value as the official nested `min`/`max` before the
  identical `acos` call, so the forward is bit-identical (a NaN input stays
  NaN in both, differing only in NaN payload bits).
- **Backward at `|cos| >= 1`** returns an exact positive zero — precisely what
  torch 1.9 computed there, *including* for NaN/Inf upstream gradients
  (torch 1.9's hard-zero branch multiplied them away; the patch reproduces
  that contract explicitly).
- **Backward strictly inside `(-1, 1)`** mirrors autograd's own op sequence
  `grad * -rsqrt(-cos*cos + 1)`. An earlier draft used division
  (`grad / -sqrt(...)`); adversarial review caught a 1–3 ULP deviation vs.
  autograd's `rsqrt` sequence, and the patch was changed to the op-for-op
  mirror and re-verified to bitwise equality.
- `apply_hopper_numerics_patch()` rebinds
  `compute_geodesic_distance_from_two_matrices` in both consuming namespaces
  (`smplik.geometry.rotations` and `smplik.losses.weighted_geodesic`, whose
  `from ... import` created a second binding). The forward-only metric binding
  in `smplik.metrics.rotation_matrix_error` is intentionally not rebound — it
  never runs backward, and its forward is bit-identical anyway.
- `ops/run_hopper_ddp.py` applies the patch at entry in every Hopper rank;
  `ops/diagnose_single_rank_gradient.py` applies it only under
  `--numerics-patch torch19`. `run.py` (the canonical entry point) has no
  import path to it.

## Verification (0-mismatch tables)

The actual patch and the official `min(max(cos, -1), 1) -> acos` composite
were probed in both pinned environments (torch 1.9.0 CPU and
torch 2.4.1+cu124), plus CUDA probes on H100:

| Check | Inputs | torch 1.9 | torch 2.4.1 |
|---|---:|---:|---:|
| Non-NaN scalar forward, bit comparison | 996,065 float32 values (incl. exhaustive boundary sweep) | 0 mismatches | 0 mismatches |
| Full matrix->geodesic forward, bit comparison | 100,000 matrix pairs | 0 mismatches | 0 mismatches |
| Strict-interior backward, bit comparison | 600,780 cases, incl. non-finite upstream gradients | 0 mismatches | 0 mismatches |
| Finite `abs(cos) >= 1` with finite/NaN/Inf upstream gradients | targeted Cartesian probe | exact torch-1.9 positive-zero bits | patch returns +0; native autograd produces NaN/Inf at ties |
| CUDA backward vs. torch 2.4 autograd (interior) | 500,000 values on H100 | — | 0 mismatched bits |

The forward equality also holds analytically: for every ordered non-NaN float,
`clamp(x, -1, 1)` and the nested min/max select the same float before the same
`acos` call. The patched run's per-branch losses are bit-equal to the
unpatched run; only gradients at exact ties change (from NaN/-inf to the
torch-1.9 exact zero).

Known non-blocking edge: a NaN **cosine input** stays NaN in forward, but the
patch returns 0 in backward where the torch-1.9 composite would propagate NaN.
This is outside the stated contract (non-NaN forward inputs, finite
`|cos| >= 1`), and the Hopper runner independently rejects a non-finite loss
before backward, so it cannot mask a real failure.

Review: three independent adversarial review lenses (math: PASS; integration
and regression both independently found the division-vs-rsqrt ULP deviation,
which was fixed and re-verified), then an external cross-model closure review
(GPT-5.6 class, max reasoning effort) issued a **PASS** with no blockers and no
required fixes on 2026-07-25, independently re-running the equivalence probes
in both environments.

## Guardrails that caught this (and stay in the repo)

- **Non-finite checkpoint rejection** (`ops/hopper_checkpoints.py`): resume
  selection walks every floating/complex tensor in model *and* optimizer state
  and refuses non-finite checkpoints, so a poisoned checkpoint can never be
  silently resumed.
- **Synchronized bad-loss abort** (`ops/run_hopper_ddp.py`): all ranks
  all-reduce a scalar bad-loss flag before backward, so one rank's non-finite
  loss can never leave its peers blocked in a collective.
- **Smoke-phase gradient probe**: smoke runs record per-rank unreduced DDP
  bucket statistics before the all-reduce — the instrument that localized this
  failure to "before communication" in the first place.
