#!/usr/bin/env python3
"""Hopper-branch-only numerics compatibility patch (torch-1.9 tie semantics).

Root cause being compensated: torch >= 1.10 changed the backward of
elementwise torch.min/torch.max at ties from "route the whole gradient to
the other argument" (torch 1.9 behaviour, model side receives an exact 0)
to "split the gradient 0.5/0.5 between the tied inputs". The official
compute_geodesic_distance_from_two_matrices (smplik/geometry/rotations.py)
clamps cos with eps-less min/max before acos; fp32 rounding produces exact
cos == +/-1.0 ties every ~2048*24-pair step, where acos'(cos) = -inf.
Under torch 2.4 the split rule leaks -inf/2 (or 0 * -inf = NaN via the
weighted branch) into the model gradients, which floods ~91% of all
gradient values with NaN before the DDP all-reduce. Evidence:
run_reports/h100x7_smoke_attempts/20260724T2253Z-h100-gradprobe/ and
run_reports/nan_rootcause_cpu_probe_20260725.py (A/B in both envs).

This patch keeps the forward bit-identical for every non-NaN cos (acos of
the [-1, 1]-clamped cos, the same values the official min/max composition
produces; an already-NaN cos stays NaN in both, differing only in NaN
payload bits) and only replaces the backward at |cos| >= 1 with an exact
0 -- precisely what torch 1.9 computed there, including for NaN/inf
upstream gradients. Strictly inside (-1, 1) the backward mirrors
autograd's own op sequence (grad * -rsqrt(-cos*cos+1)) and is bitwise
equal to what torch 1.9 and torch 2.4 autograd compute.

The canonical A100 run never imports this module. Only the Hopper
entrypoints that execute backward (ops/run_hopper_ddp.py and
ops/diagnose_single_rank_gradient.py with --numerics-patch torch19)
apply it.
"""

import torch

import smplik.geometry.rotations as _rotations
import smplik.losses.weighted_geodesic as _weighted_geodesic

PATCH_NAME = "geodesic-acos-clamp-torch19-tie-backward"


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


def _compute_geodesic_distance_from_two_matrices_torch19(m1, m2):
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    return _AcosOfClampedCos.apply(cos)


def apply_hopper_numerics_patch():
    """Rebind every import-site of the unguarded geodesic distance."""
    patched = _compute_geodesic_distance_from_two_matrices_torch19
    _rotations.compute_geodesic_distance_from_two_matrices = patched
    _weighted_geodesic.compute_geodesic_distance_from_two_matrices = patched
    return PATCH_NAME
