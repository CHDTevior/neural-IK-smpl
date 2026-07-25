#!/usr/bin/env python3
"""External-launch DDP entry point for the isolated Hopper compatibility runs."""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import yaml
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from smplik.utils.model_factory import ModelFactory
from smplik.utils.tensorboard import TensorBoardLoggerWithMetrics
from smplik.utils.versioning import get_git_diff
from ops.hopper_checkpoints import set_latest_hopper_checkpoint

# Register the structured Hydra model schema.
import smplik.models  # noqa: F401,E402


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def scalar_metrics(metrics):
    result = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[name] = float(value.detach().cpu())
        elif isinstance(value, (float, int)):
            result[name] = float(value)
    return result


class LocalGradientProbeState:
    """Accumulate rank-local bucket statistics before DDP reduction."""

    def __init__(self, evidence_root, rank):
        self.evidence_root = Path(evidence_root)
        self.rank = rank
        self.backward_index = 0
        self.bucket_count = 0
        self.numel = 0
        self.nan_count = 0
        self.inf_count = 0
        self.finite_abs_max = 0.0
        self.first_bad_bucket = None

    def record(self, bucket):
        buffer = bucket.buffer().detach()
        nan_count = int(torch.isnan(buffer).sum().cpu())
        inf_count = int(torch.isinf(buffer).sum().cpu())
        finite = torch.isfinite(buffer)
        if bool(finite.any()):
            finite_abs_max = float(buffer[finite].abs().max().cpu())
        else:
            finite_abs_max = None

        if nan_count + inf_count > 0 and self.first_bad_bucket is None:
            self.first_bad_bucket = int(bucket.index())
        self.bucket_count += 1
        self.numel += int(buffer.numel())
        self.nan_count += nan_count
        self.inf_count += inf_count
        if finite_abs_max is not None:
            self.finite_abs_max = max(self.finite_abs_max, finite_abs_max)

        if bucket.is_last():
            payload = {
                "time_utc": utc_now(),
                "rank": self.rank,
                "backward_index": self.backward_index,
                "bucket_count": self.bucket_count,
                "numel": self.numel,
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
                "finite_abs_max": self.finite_abs_max,
                "first_bad_bucket": self.first_bad_bucket,
            }
            atomic_json(
                self.evidence_root
                / "local_gradient_probe"
                / f"rank-{self.rank}-backward-{self.backward_index}.json",
                payload,
            )
            print(f"LOCAL_GRADIENT_PROBE {json.dumps(payload, sort_keys=True)}", flush=True)
            self.backward_index += 1
            self.bucket_count = 0
            self.numel = 0
            self.nan_count = 0
            self.inf_count = 0
            self.finite_abs_max = 0.0
            self.first_bad_bucket = None


def local_gradient_probe_hook(state, bucket):
    state.record(bucket)
    return default_hooks.allreduce_hook(dist.group.WORLD, bucket)


class RunEvidence(Callback):
    """Write bounded operational evidence without changing training decisions."""

    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.evidence_root = Path(
            os.environ.get("HOPPER_EVIDENCE_ROOT", self.log_dir / "rank_evidence")
        )
        self.first_batch_written = False
        self.first_local_loss_written = False
        self.gradient_checks_remaining = (
            2 if os.environ.get("HOPPER_PHASE") == "smoke" else 1
        )

    def on_before_backward(self, trainer, pl_module, loss):
        loss_value = float(loss.detach().cpu())
        loss_finite = bool(torch.isfinite(loss.detach()).cpu())
        rank = int(os.environ["RANK"])
        bad_loss = torch.tensor(
            0 if loss_finite else 1,
            dtype=torch.int32,
            device=loss.device,
        )
        dist.all_reduce(bad_loss, op=dist.ReduceOp.MAX)
        any_rank_bad = bool(bad_loss.cpu())

        if not self.first_local_loss_written:
            atomic_json(
                self.evidence_root / f"rank-{rank}-first_local_loss.json",
                {
                    "time_utc": utc_now(),
                    "rank": rank,
                    "epoch": int(trainer.current_epoch),
                    "global_step": int(trainer.global_step),
                    "loss": loss_value if loss_finite else None,
                    "loss_repr": repr(loss_value),
                    "loss_finite": loss_finite,
                },
            )
            self.first_local_loss_written = True
        if not loss_finite:
            atomic_json(
                self.evidence_root
                / (
                    f"rank-{rank}-nonfinite_loss-epoch-{trainer.current_epoch}"
                    f"-step-{trainer.global_step}.json"
                ),
                {
                    "time_utc": utc_now(),
                    "rank": rank,
                    "epoch": int(trainer.current_epoch),
                    "global_step": int(trainer.global_step),
                    "loss": None,
                    "loss_repr": repr(loss_value),
                    "loss_finite": False,
                },
            )
        if any_rank_bad:
            raise RuntimeError(
                f"At least one rank produced a non-finite training loss at "
                f"epoch={trainer.current_epoch} global_step={trainer.global_step}"
            )

    def on_after_backward(self, trainer, pl_module):
        if self.gradient_checks_remaining <= 0:
            return
        self.gradient_checks_remaining -= 1
        rank = int(os.environ["RANK"])
        gradients = [
            parameter.grad
            for parameter in pl_module.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError(f"Rank {rank} produced no gradients")
        nonfinite = sum(
            int((~torch.isfinite(gradient)).sum().cpu())
            for gradient in gradients
        )
        atomic_json(
            self.evidence_root
            / f"rank-{rank}-reduced_gradient-step-{trainer.global_step}.json",
            {
                "time_utc": utc_now(),
                "rank": rank,
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "gradient_tensors": len(gradients),
                "nonfinite_values": nonfinite,
            },
        )
        if nonfinite:
            raise RuntimeError(
                f"Rank {rank} has {nonfinite} non-finite reduced gradient "
                f"values at epoch={trainer.current_epoch} "
                f"global_step={trainer.global_step}"
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not trainer.is_global_zero or self.first_batch_written:
            return
        loss = outputs
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        loss_value = None
        loss_finite = None
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            loss_value = float(loss.detach().cpu())
            loss_finite = bool(torch.isfinite(loss.detach()).cpu())
        atomic_json(
            self.log_dir / "first_train_batch.json",
            {
                "time_utc": utc_now(),
                "global_step": int(trainer.global_step),
                "epoch": int(trainer.current_epoch),
                "loss": loss_value,
                "loss_finite": loss_finite,
                "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
                "world_size": int(os.environ["WORLD_SIZE"]),
            },
        )
        self.first_batch_written = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        # TorchMetrics may synchronize while callback_metrics is materialized.
        # Every rank must therefore execute this before rank-zero-only I/O.
        metrics = scalar_metrics(trainer.callback_metrics)
        if not trainer.is_global_zero:
            return
        atomic_json(
            self.log_dir / "run_state.json",
            {
                "time_utc": utc_now(),
                "status": "RUNNING",
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "metrics": metrics,
                "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
                "world_size": int(os.environ["WORLD_SIZE"]),
                "attempt_id": os.environ.get("HOPPER_ATTEMPT_ID", "manual"),
            },
        )


def resolve_experiment(filepath):
    with open(filepath) as stream:
        experiment = yaml.load(stream, Loader=yaml.SafeLoader)

    parameters = experiment["parameters"]
    for key in parameters:
        if not isinstance(parameters[key], list):
            parameters[key] = [parameters[key]]

    grid = list(ParameterGrid(parameters))
    if len(grid) != 1:
        raise RuntimeError(
            f"{filepath} resolves to {len(grid)} experiments; exactly one is required"
        )

    overrides = [f"{key}={value}" for key, value in grid[0].items()]
    cfg = compose(experiment["base_config"] + ".yaml", overrides=overrides)
    return cfg


def validate_external_launch(cfg):
    required = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"Missing external DDP variables: {missing}")

    expected_world_size = int(cfg.trainer.gpus) * int(
        cfg.trainer.num_nodes
    )
    actual_world_size = int(os.environ["WORLD_SIZE"])
    if actual_world_size != expected_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={actual_world_size}, config expects {expected_world_size}"
        )
    if int(os.environ["LOCAL_RANK"]) != 0 or int(os.environ["LOCAL_WORLD_SIZE"]) != 1:
        raise RuntimeError("Each external rank must see exactly one local GPU")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError(
            f"Expected one visible CUDA GPU, found {torch.cuda.device_count()}"
        )
    if cfg.trainer.accelerator != "ddp":
        raise RuntimeError("Hopper compatibility runs require trainer.accelerator=ddp")
    if int(cfg.trainer.precision) != 32:
        raise RuntimeError("Hopper compatibility runs must remain FP32")
    if int(cfg.trainer.accumulate_grad_batches) != 1:
        raise RuntimeError("Gradient accumulation is not allowed")


def run(cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, True)
    validate_external_launch(cfg)

    rank = int(os.environ["RANK"])
    if rank == 0:
        print(OmegaConf.to_yaml(cfg), flush=True)

    # The canonical stack predates TF32. Keep Hopper arithmetic in FP32 mode.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if rank == 0:
        print(f"Random Seed: {seed}", flush=True)

    dm = instantiate(cfg.dataset)
    dm.prepare_data()
    model = ModelFactory.instantiate(
        cfg, data_components=dm.get_data_specific_components()
    )

    metrics = model.get_metrics()
    tb_logger = TensorBoardLoggerWithMetrics(
        save_dir=cfg.logging.path,
        name=cfg.logging.name,
        version=cfg.logging.version,
        metrics=metrics,
    )
    log_dir = Path(os.path.normpath(tb_logger.log_dir)).resolve()
    if rank == 0:
        print(f"Logging saved to: {log_dir}", flush=True)

    checkpoint = ModelCheckpoint(
        dirpath=str(log_dir / "checkpoints"),
        save_top_k=1,
        save_last=True,
        monitor=None,
        mode="min",
        every_n_epochs=1,
    )
    callbacks = [checkpoint, RunEvidence(log_dir)]

    warmup_steps = int(os.environ.get("SMPLIK_WARMUP_STEPS", "0"))
    warmup_start_lr = None
    if warmup_steps > 0:
        from ops.hopper_lr_warmup import LinearLrWarmup

        warmup_start_lr = float(os.environ["SMPLIK_WARMUP_START_LR"])
        callbacks.append(
            LinearLrWarmup(warmup_start_lr, float(cfg.optimizer.lr), warmup_steps)
        )
        print(
            f"lr_warmup=linear {warmup_start_lr}->{float(cfg.optimizer.lr)} "
            f"over {warmup_steps} steps",
            flush=True,
        )

    # Env-gated global-norm gradient clipping (default off, byte-equal when
    # unset/empty/0). The composed cfg.trainer already carries
    # gradient_clip_val=0.0 from the TrainerOptions schema and is expanded via
    # **cfg.trainer, so a duplicate Trainer kwarg would raise; override the
    # cfg value instead. PL 1.4.7's Trainer default
    # gradient_clip_algorithm="norm" applies (cfg has no such key): a single
    # torch.nn.utils.clip_grad_norm_ over all optimizer params, executed after
    # the on_after_backward hook and before optimizer.step, so
    # RunEvidence.on_after_backward sees the RAW (pre-clip) reduced gradients.
    grad_clip_env = os.environ.get("SMPLIK_GRAD_CLIP_NORM", "").strip()
    grad_clip_norm = float(grad_clip_env) if grad_clip_env else 0.0
    if grad_clip_norm < 0:
        raise RuntimeError(
            f"SMPLIK_GRAD_CLIP_NORM must be >= 0, got {grad_clip_norm}"
        )
    if grad_clip_norm > 0:
        cfg_clip_val = cfg.trainer.gradient_clip_val
        if cfg_clip_val not in (None, 0, 0.0):
            raise RuntimeError(
                f"cfg.trainer.gradient_clip_val={cfg_clip_val} conflicts with "
                f"SMPLIK_GRAD_CLIP_NORM={grad_clip_norm}; refusing to "
                "silently override a non-zero config value"
            )
        cfg.trainer.gradient_clip_val = grad_clip_norm
        print(f"grad_clip=global_norm {grad_clip_norm}", flush=True)

    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        git_diff = get_git_diff()
        if git_diff:
            (log_dir / "git_diff.patch").write_text(git_diff)
        atomic_json(
            log_dir / "launch_state.json",
            {
                "time_utc": utc_now(),
                "status": "STARTING",
                "rank": rank,
                "world_size": int(os.environ["WORLD_SIZE"]),
                "master_addr": os.environ["MASTER_ADDR"],
                "master_port": int(os.environ["MASTER_PORT"]),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "lightning": pl.__version__,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "resume_from_checkpoint": cfg.trainer.resume_from_checkpoint,
                "attempt_id": os.environ.get("HOPPER_ATTEMPT_ID", "manual"),
                "lr_warmup_steps": warmup_steps,
                "lr_warmup_start_lr": warmup_start_lr,
                "grad_clip_norm": grad_clip_norm if grad_clip_norm > 0 else None,
            },
        )
    logging.info("Logging saved to: %s", log_dir)

    plugins = None
    if os.environ.get("HOPPER_LOCAL_GRAD_PROBE") == "1":
        probe_state = LocalGradientProbeState(
            os.environ.get("HOPPER_EVIDENCE_ROOT", log_dir / "rank_evidence"),
            rank,
        )
        plugins = [
            DDPPlugin(
                ddp_comm_state=probe_state,
                ddp_comm_hook=local_gradient_probe_hook,
            )
        ]

    trainer = pl.Trainer(
        logger=[tb_logger],
        callbacks=callbacks,
        plugins=plugins,
        **cfg.trainer,
    )
    trainer.fit(model, datamodule=dm)

    # This property can trigger a distributed reduction when its cache is not
    # populated. Materialize it on every rank before rank-zero-only I/O.
    fit_metrics = scalar_metrics(trainer.callback_metrics)
    if trainer.is_global_zero:
        atomic_json(
            log_dir / "fit_completed.json",
            {
                "time_utc": utc_now(),
                "status": "FIT_COMPLETED",
                "epoch": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
                "metrics": fit_metrics,
                "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
                "world_size": int(os.environ["WORLD_SIZE"]),
            },
        )


def main(filepath):
    os.chdir(REPO_ROOT)
    from ops.hopper_numerics_patch import apply_hopper_numerics_patch

    print(f"numerics_patch={apply_hopper_numerics_patch()}", flush=True)
    initialize(config_path="../configs")
    cfg = resolve_experiment(filepath)
    set_latest_hopper_checkpoint(cfg)
    run(cfg.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Externally launched SMPL-IK DDP")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
