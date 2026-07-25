#!/usr/bin/env python3
"""Run one real official-config optimization step and one validation batch."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from pytorch_lightning import Callback, Trainer
from sklearn.model_selection import ParameterGrid

import smplik.models  # noqa: F401 - registers the model schemas
from smplik.utils.model_factory import ModelFactory


EXPERIMENT_CONFIG = Path("configs/experiments/smplik_amass.yaml")


def compose_experiment():
    experiment = yaml.safe_load(EXPERIMENT_CONFIG.read_text())
    parameters = experiment["parameters"]
    for key in parameters:
        if not isinstance(parameters[key], list):
            parameters[key] = [parameters[key]]
    parameter_sets = list(ParameterGrid(parameters))
    if len(parameter_sets) != 1:
        raise RuntimeError("The official AMASS experiment no longer resolves to one run")
    overrides = [f"{key}={value}" for key, value in parameter_sets[0].items()]
    initialize(config_path="../configs")
    return compose(experiment["base_config"] + ".yaml", overrides=overrides).model


class SmokeEvidence(Callback):
    def __init__(self):
        self.train_batch_sizes = []
        self.validation_batch_sizes = {}
        self.training_loss = None
        self.backward_completed = False
        self.parameters_with_gradients = 0

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        self.train_batch_sizes.append(int(batch["joint_positions"].shape[0]))

    def on_before_backward(self, trainer, pl_module, loss):
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite smoke training loss: {loss}")
        self.training_loss = float(loss.detach().cpu())

    def on_after_backward(self, trainer, pl_module):
        gradients = [
            parameter.grad
            for parameter in pl_module.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("Smoke backward pass produced no gradients")
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("Smoke backward pass produced non-finite gradients")
        self.parameters_with_gradients = len(gradients)
        self.backward_completed = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        self.validation_batch_sizes.setdefault(str(dataloader_idx), []).append(
            int(batch["joint_positions"].shape[0])
        )


def main() -> None:
    cfg = compose_experiment()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    dm = instantiate(cfg.dataset)
    dm.prepare_data()

    model = ModelFactory.instantiate(
        cfg, data_components=dm.get_data_specific_components()
    )
    evidence = SmokeEvidence()
    torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(
        callbacks=[evidence],
        logger=False,
        checkpoint_callback=False,
        default_root_dir="./run_reports/smoke_test_runtime",
        gpus=cfg.trainer.gpus,
        precision=cfg.trainer.precision,
        max_epochs=1,
        min_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        benchmark=cfg.trainer.benchmark,
        deterministic=cfg.trainer.deterministic,
        progress_bar_refresh_rate=cfg.trainer.progress_bar_refresh_rate,
        weights_summary=cfg.trainer.weights_summary,
    )
    trainer.fit(model, datamodule=dm)

    if evidence.train_batch_sizes != [2048]:
        raise RuntimeError(
            f"Smoke train batch was not exactly 2048: {evidence.train_batch_sizes}"
        )
    if evidence.validation_batch_sizes != {"0": [2048], "1": [2048]}:
        raise RuntimeError(
            "Smoke validation did not cover one full batch from both official "
            f"validation loaders: {evidence.validation_batch_sizes}"
        )
    if not evidence.backward_completed or trainer.global_step != 1:
        raise RuntimeError(
            "Smoke test did not complete exactly one backward/optimizer step"
        )

    callback_metrics = {
        key: float(value.detach().cpu())
        for key, value in trainer.callback_metrics.items()
        if torch.is_tensor(value) and value.numel() == 1
    }
    if not all(np.isfinite(value) for value in callback_metrics.values()):
        raise RuntimeError(f"Non-finite smoke callback metrics: {callback_metrics}")

    report = {
        "experiment_config": str(EXPERIMENT_CONFIG),
        "configured_batch_size": int(cfg.dataset.batch_size),
        "observed_train_batch_sizes": evidence.train_batch_sizes,
        "observed_validation_batch_sizes": evidence.validation_batch_sizes,
        "training_loss": evidence.training_loss,
        "callback_metrics": callback_metrics,
        "global_step": trainer.global_step,
        "parameters_with_gradients": evidence.parameters_with_gradients,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "optimizer": cfg.optimizer._target_,
        "learning_rate": float(cfg.optimizer.lr),
    }
    report_path = Path("./run_reports/smoke_test.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Smoke test passed: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
