#!/usr/bin/env python3
"""Single-GPU test and ONNX export after a Hopper DDP fit finishes."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from smplik.utils.model_factory import ModelFactory
from smplik.utils.tensorboard import TensorBoardLoggerWithMetrics
from ops.hopper_checkpoints import (
    assert_saved_hparams_match,
    select_valid_checkpoint,
)

import smplik.models  # noqa: F401,E402


def resolve_experiment(filepath):
    with open(filepath) as stream:
        experiment = yaml.load(stream, Loader=yaml.SafeLoader)
    parameters = experiment["parameters"]
    for key in parameters:
        if not isinstance(parameters[key], list):
            parameters[key] = [parameters[key]]
    grid = list(ParameterGrid(parameters))
    if len(grid) != 1:
        raise RuntimeError(f"{filepath} must resolve to exactly one experiment")
    overrides = [f"{key}={value}" for key, value in grid[0].items()]
    return compose(experiment["base_config"] + ".yaml", overrides=overrides)


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu()) if value.numel() == 1 else None
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    return str(value)


def main(filepath):
    os.chdir(REPO_ROOT)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Postprocessing requires exactly one visible CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    initialize(config_path="../configs")
    cfg = resolve_experiment(filepath)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    OmegaConf.set_struct(cfg, True)

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dm = instantiate(cfg.dataset)
    dm.prepare_data()
    model = ModelFactory.instantiate(
        cfg, data_components=dm.get_data_specific_components()
    )
    tb_logger = TensorBoardLoggerWithMetrics(
        save_dir=cfg.logging.path,
        name=cfg.logging.name,
        version=cfg.logging.version,
        metrics=model.get_metrics(),
    )
    log_dir = Path(os.path.normpath(tb_logger.log_dir)).resolve()
    if not assert_saved_hparams_match(cfg, log_dir):
        raise RuntimeError(f"No saved hparams found under {log_dir}")
    checkpoint, checkpoint_metadata, rejected = select_valid_checkpoint(
        log_dir / "checkpoints", cfg
    )
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint_payload["state_dict"], strict=True)
    del checkpoint_payload

    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        logger=[tb_logger],
        checkpoint_callback=False,
        weights_summary=None,
        progress_bar_refresh_rate=1,
        limit_test_batches=cfg.trainer.limit_test_batches,
        benchmark=False,
        deterministic=False,
    )
    results = trainer.test(model=model, datamodule=dm, ckpt_path=None)
    onnx_path = log_dir / "model.onnx"
    model.export(str(onnx_path))
    if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
        raise RuntimeError(f"ONNX export is empty: {onnx_path}")

    attempt_id = os.environ.get("HOPPER_ATTEMPT_ID", "manual")
    payload = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "status": "POSTPROCESS_COMPLETED",
        "attempt_id": attempt_id,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_size": Path(checkpoint).stat().st_size,
        "checkpoint_metadata": checkpoint_metadata,
        "rejected_checkpoints": rejected,
        "onnx": str(onnx_path),
        "onnx_size": onnx_path.stat().st_size,
        "test_results": json_safe(results),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "lightning": pl.__version__,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    result_path = log_dir / "postprocess_results" / f"{attempt_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    for destination in (result_path, log_dir / "postprocess_completed.json"):
        temporary = destination.with_name(
            destination.name + f".tmp.{os.getpid()}"
        )
        temporary.write_text(encoded)
        os.replace(str(temporary), str(destination))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test/export a Hopper DDP result")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
