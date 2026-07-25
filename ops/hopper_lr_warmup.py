#!/usr/bin/env python3
"""Linear learning-rate warmup for the Hopper linear-scaled branches (v2).

Rationale: the k-scaled Adam learning rates (7x=0.0014, 4x=0.0008) without
warmup are outside the stable region for this model: rare ill-conditioned
batches produce single-step loss explosions (observed at 7x: epochs 1 and 5,
aggregates 1.91e5 / 3.21e5, running peak 5.65e7; at 4x: epoch 2, single step
-1.94 -> 888), each costing ~3 epochs of recovery. This is the exact failure
mode the Goyal et al. 2017 gradual-warmup prescription addresses. The
canonical 1x run (lr 0.0002) shows zero spikes in >200k updates and is not
touched by this file.

Schedule: lr(step) = start + (target - start) * min(1, step / warmup_steps),
i.e. linear from the canonical base lr to the scaled target over the first
two epochs of optimizer updates, constant afterwards. The schedule is a pure
function of trainer.global_step, so checkpoint resume restores it exactly.

Activation is env-driven and default-off: SMPLIK_WARMUP_STEPS (int, 0 or
unset disables) and SMPLIK_WARMUP_START_LR (float). The target lr is the
config's optimizer.lr. User approved this recipe change (v2) on 2026-07-25;
prior no-warmup runs are archived, not resumed.
"""

from pytorch_lightning.callbacks import Callback


class LinearLrWarmup(Callback):
    def __init__(self, start_lr, target_lr, warmup_steps):
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if start_lr <= 0 or target_lr <= 0:
            raise ValueError("learning rates must be positive")
        self.start_lr = float(start_lr)
        self.target_lr = float(target_lr)
        self.warmup_steps = int(warmup_steps)

    def current_lr(self, global_step):
        fraction = min(1.0, float(global_step) / float(self.warmup_steps))
        return self.start_lr + (self.target_lr - self.start_lr) * fraction

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        lr = self.current_lr(trainer.global_step)
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
