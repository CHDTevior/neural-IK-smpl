# Multi-GPU engineering: linear scaling, warmup, cross-allocation NCCL, crash-tolerant resume

This documents the multi-GPU (Hopper) branch of the reproduction: how the
single-GPU recipe was linearly scaled, what went wrong, what fixed it, and the
Slurm/NCCL plumbing that makes N ranks spread across *separate Slurm
allocations* (same node and cross-node) behave as one synchronous DDP world.
Everything here is explicitly classified as a **linear-scaled approximation**,
not an exact reproduction of the canonical single-GPU run.

## 1. Linear scaling (Goyal et al. 2017)

The canonical recipe is per-GPU batch 2048, Adam lr 2e-4, 120 epochs. The
scaled variants keep the **per-rank batch fixed at 2048** and scale the global
batch and learning rate together by the world size k:

| Variant | World size | Per-rank batch | Global batch | Adam lr | Steps/epoch | Config |
|---|---:|---:|---:|---:|---:|---|
| canonical (A100) | 1 | 2048 | 2048 | 0.0002 | 24,648 | `configs/experiments/smplik_amass.yaml` (upstream) |
| a100 ddp4 (audited, not the campaign path) | 4 | 2048 | 8192 | 0.0008 | 6,162 | `ops/smplik_amass_ddp4_linear.yaml` |
| h100x7 | 7 | 2048 | 14336 | 0.0014 | 3,522 | `ops/smplik_amass_h100x7_linear.yaml` |
| h200x4 | 4 | 2048 | 8192 | 0.0008 | 6,162 | `ops/smplik_amass_h200x4_linear.yaml` |

All variants keep FP32 precision, `accumulate_grad_batches: 1`,
`replace_sampler_ddp: True`, seed 0, and 120 epochs. TF32, AMP, compilation,
and fused optimizers stay **disabled** so the only intentional deviation from
the canonical run is the batch/lr pair (plus warmup, below).

## 2. Warmup: why, and the spike history

The k-scaled Adam learning rates are outside this model's always-stable
region: rare ill-conditioned batches produce single-step loss explosions,
each costing roughly 3 epochs of recovery. The canonical 1x run (lr 2e-4)
shows **zero** spikes in >200k updates.

| Run | Warmup | Observed instability |
|---|---|---|
| h100x7 v1 (no warmup) | none | epoch-aggregate spikes `1.91e5` (ep 1) and `3.21e5` (ep 5); running peak `5.65e7` |
| h200x4 v1 (no warmup) | none | epoch-2 escalation from losses near `-1.94` through displayed values `19.5, 686, 782, 861, 875` to `888` |
| h100x7 v2 (warmup) | 0.0002 -> 0.0014 over 7,044 steps (2 epochs) | mid-warmup aggregate spikes ~`545 / 654 / 2933`, then long stable stretches |
| h200x4 v2 (warmup) | 0.0002 -> 0.0008 over 12,324 steps (2 epochs) | mid-warmup aggregate spikes ~`1248 / 1729`, then long stable stretches; one late `~3.1e6` aggregate event at epoch 46 (checkpoint backed up, run recovered and continued finite) |

Two honest conclusions from that table:

1. Warmup is a **mitigation, not a cure** — it removed the catastrophic
   early-training explosions but did not eliminate every excursion (the ep46
   event happened long after warmup ended). Stability must be judged from run
   evidence, not inferred from the callback.
2. The unclipped 7x learning rate remained the most fragile configuration;
   the h100x7 line was ultimately kept as an instability study (see
   `docs/REPRODUCTION.md`).

### Warmup implementation (`ops/hopper_lr_warmup.py`)

```
lr(step) = start_lr + (target_lr - start_lr) * min(1, step / warmup_steps)
```

- Pure function of `trainer.global_step` -> **resume-exact**: restoring a
  checkpoint restores the schedule with no hidden callback state.
- Env-driven, default-off. When `SMPLIK_WARMUP_STEPS` is unset or 0 no
  callback is even imported and the execution path is byte-equal to the
  no-warmup runner.
- The shape follows the gradual-warmup prescription of Goyal et al. 2017,
  adapted to this Adam configuration (it is not a literal reproduction of
  that paper's 5-epoch SGD recipe).

Values used by the formal runs (exported before `launch_hopper_variant.sh`):

| Variant | `SMPLIK_WARMUP_STEPS` | `SMPLIK_WARMUP_START_LR` | ramp |
|---|---:|---:|---|
| h100x7 | 7044 | 0.0002 | 2e-4 -> 1.4e-3 over 2 epochs |
| h200x4 | 12324 | 0.0002 | 2e-4 -> 8e-4 over 2 epochs |

### Optional gradient clipping

`SMPLIK_GRAD_CLIP_NORM` (default off/empty) enables PL's global-norm clipping
(`clip_grad_norm_` over all optimizer params, applied after `on_after_backward`
so the evidence hooks still record RAW pre-clip gradients). The runner refuses
to silently override a non-zero config value. This is the documented follow-up
lever for excursions that survive warmup.

## 3. Cross-allocation DDP over NCCL/IB

The scaled runs do not own one clean `sbatch` allocation. They aggregate GPUs
from **several independent Slurm allocations** — h100x7: 7 ranks from four
allocations on one H100 node; h200x4: 4 ranks from two allocations on **two
different H200 nodes**. The wiring that makes this a single DDP world:

### Static rendezvous, external TorchElastic identity

`ops/hopper_rank.sh` gives every rank a fully manual TorchElastic environment
(no `torchrun` host election, which fails when hostname != IB hostname):

```bash
export RANK=$GLOBAL_RANK  WORLD_SIZE=$W  LOCAL_RANK=0  LOCAL_WORLD_SIZE=1
export GROUP_RANK=$GLOBAL_RANK  NODE_RANK=$GLOBAL_RANK
export MASTER_ADDR=<master-node>-ib0  MASTER_PORT=<fixed port>
```

Each Slurm task is cgroup-scoped to exactly one GPU, so every rank presents
itself as a one-GPU logical node (`LOCAL_WORLD_SIZE=1`) and Lightning 1.4
selects `TorchElasticEnvironment` instead of its Slurm plugin.

### The `NCCL_HOSTID` cross-cgroup trick

Slurm applies a one-GPU device cgroup to each task. NCCL 2.20.5's local-host
topology discovery then fails **across task/allocation boundaries on the same
physical host** — a rank cannot inspect a peer's GPU through NVML and dies
with `nvmlDeviceGetHandleByPciBusId() failed: Not Found`, even with P2P and
SHM disabled. The fix: give every rank a unique host identity so NCCL treats
each rank as its own host and uses the verified IB transport for everything:

```bash
export NCCL_HOSTID="smplik-${VARIANT}-rank-${RANK}"
export NCCL_SOCKET_IFNAME=ib   NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

This is a communication-topology compatibility setting only — it does not
alter model, data, precision, or update arithmetic. A seven-rank all-reduce
smoke (expected sum 28 on every rank) validated it before any training.

### `srun --overlap` wiring

`ops/launch_hopper_variant.sh` drives one `srun` step per allocation into the
already-running interactive allocations:

```bash
srun --jobid="$job_id" --overlap --kill-on-bad-exit=1 \
     --ntasks="$n" --gpus-per-task=1 --cpus-per-task="$c" --cpu-bind=cores \
     bash ops/hopper_rank.sh "$rank_base" ...
```

plus: per-variant `flock` single-instance lock, per-attempt evidence
directories, named steps (`smplik-<variant>-<phase>-<attempt>`) so cleanup can
`scancel --signal` exactly its own steps and nothing else, protected-file
SHA-256 + `git diff` gate before every launch, and a config audit
(`ops/audit_hopper_configs.py`) asserting the scaled YAMLs differ from
canonical only in the intended keys.

### Lightning 1.4 Slurm-handler disarm

PL 1.4 installs a Slurm SIGTERM handler that *ignores termination* and a
SIGUSR1 handler that requeues the enclosing allocation — both wrong for
externally orchestrated ranks spanning several allocations. Ranks present the
step as interactive (`SLURM_JOB_NAME=bash`) to disable only that legacy
integration.

## 4. Crash-tolerant checkpoint resume (`ops/hopper_checkpoints.py`)

Allocations expire mid-training (one migration is documented in
`docs/REPRODUCTION.md`), so resume must be safe against both config drift and
NaN-poisoned checkpoints:

- **Config fingerprint check**: the resolved experiment config (with the
  operational `resume_from_checkpoint` field normalized to null) is hashed;
  a checkpoint is only eligible for resume if its recorded fingerprint matches
  the current launch. No silent resumes across recipe changes.
- **Non-finite rejection**: every floating/complex tensor in `state_dict`
  *and* `optimizer_states` is checked with `torch.isfinite`; any non-finite
  value disqualifies the checkpoint (fault-injection tested).
- Selection prefers the newest eligible checkpoint (`last.ckpt` chain), and
  the chosen fingerprint/epoch/step is logged by rank 0 for audit.
- Rank 0 additionally maintains a durable `run_state.json` (epoch, global
  step, last loss, world size, non-finite event flags) as the
  monitoring/verification source of truth.

## 5. Known operational pitfalls

- **`pam_slurm_adopt` kills SSH-launched orchestrators.** On clusters with
  pam_slurm_adopt, an SSH session to a compute node is adopted into one of
  that node's job cgroups. When *that* job ends — even if it is not yours —
  everything the session spawned (launchers, monitors) is killed with it.
  Long-lived orchestrators must be started with `setsid nohup ... &` so they
  are re-parented to init, and must live on an allocation whose lifetime
  covers the run; verify survival with `ps -o ppid= -p <pid>` -> `1`.
- **`pkill`/`pgrep` self-match.** A monitor or cleanup that greps for the
  training command pattern can match *itself* (its own command line contains
  the pattern) or a peer allocation's ranks on the same node, then kill the
  wrong thing or abort a healthy launch. Use exact named Slurm steps
  (`squeue --steps` + step-name match, as `launch_hopper_variant.sh` does) or
  PID files — never bare `pkill -f` on shared nodes.
- **Buffered pipeline logging looks like a hang.** `srun ... | sed`-style
  console pipelines block-buffer; judge liveness from GPU utilization and the
  rank-0 log/`run_state.json`, not from the orchestrator's console file.
- **One launcher per experiment directory.** The `flock` lock plus
  attempt-directory `mkdir` guard prevent two launchers from driving the same
  log/checkpoint directory — double launches were the first failure mode we
  designed out.

## 6. Honest classification

These runs change global batch and learning rate (and add warmup). They are
throughput/effect studies of the linear scaling rule on this model, not
canonical results. The canonical numbers in the README come exclusively from
the unmodified single-GPU `run.py` path, whose config, protected files, and
environment are isolated from everything described here (verified by
checksum gate + external review; see `docs/REPRODUCTION.md`).
