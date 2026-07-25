#!/usr/bin/env python3
"""Identify which official loss branch first creates non-finite gradients."""

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import yaml
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from sklearn.model_selection import ParameterGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from smplik.data.smpl_module import _batched_collate
from smplik.utils.model_factory import ModelFactory

import smplik.models  # noqa: F401,E402


def resolve_experiment(path):
    with Path(path).open() as stream:
        experiment = yaml.load(stream, Loader=yaml.SafeLoader)
    parameters = experiment["parameters"]
    for key in parameters:
        if not isinstance(parameters[key], list):
            parameters[key] = [parameters[key]]
    grid = list(ParameterGrid(parameters))
    if len(grid) != 1:
        raise RuntimeError(f"{path} must resolve to exactly one experiment")
    overrides = [f"{key}={value}" for key, value in grid[0].items()]
    return compose(experiment["base_config"] + ".yaml", overrides=overrides).model


def gradient_summary(model):
    result = {
        "gradient_tensors": 0,
        "gradient_values": 0,
        "nan_count": 0,
        "inf_count": 0,
        "finite_abs_max": 0.0,
        "first_bad_parameter": None,
    }
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        nan_count = int(torch.isnan(gradient).sum().cpu())
        inf_count = int(torch.isinf(gradient).sum().cpu())
        finite = torch.isfinite(gradient)
        result["gradient_tensors"] += 1
        result["gradient_values"] += gradient.numel()
        result["nan_count"] += nan_count
        result["inf_count"] += inf_count
        if nan_count + inf_count and result["first_bad_parameter"] is None:
            result["first_bad_parameter"] = name
        if bool(finite.any()):
            result["finite_abs_max"] = max(
                result["finite_abs_max"],
                float(gradient[finite].abs().max().cpu()),
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--numerics-patch", choices=["off", "torch19"], default="off")
    args = parser.parse_args()

    applied_patch = None
    if args.numerics_patch == "torch19":
        from ops.hopper_numerics_patch import apply_hopper_numerics_patch

        applied_patch = apply_hopper_numerics_patch()
        print(f"numerics_patch={applied_patch}", flush=True)

    os.chdir(REPO_ROOT)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Diagnostic requires exactly one visible CUDA GPU")
    if not 0 <= args.rank < args.world_size:
        raise RuntimeError("rank must be in [0, world_size)")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initialize(config_path="../configs")
    cfg = resolve_experiment(args.config)

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dm = instantiate(cfg.dataset)
    dm.prepare_data()
    model = ModelFactory.instantiate(
        cfg, data_components=dm.get_data_specific_components()
    ).cuda()
    model.train()

    dm.setup()
    sampler = DistributedSampler(
        dm.training_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    sampler.set_epoch(0)
    loader = DataLoader(
        dm.training_dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=int(cfg.dataset.num_workers),
        collate_fn=_batched_collate,
    )
    batch = {
        key: value.cuda(non_blocking=False)
        for key, value in next(iter(loader)).items()
    }
    batch_finite = {
        key: bool(torch.isfinite(value).all().cpu())
        for key, value in batch.items()
        if value.is_floating_point()
    }

    losses = model.shared_step(batch, step="gradient_diagnostic")
    component_names = ("rotation", "fk", "lookat", "true_lookat", "total")
    results = {}
    for index, name in enumerate(component_names):
        model.zero_grad(set_to_none=True)
        loss = losses[name]
        if loss is None:
            results[name] = {"loss": None, "skipped": True}
            continue
        loss.backward(retain_graph=index < len(component_names) - 1)
        results[name] = {
            "loss": float(loss.detach().cpu()),
            "loss_finite": bool(torch.isfinite(loss.detach()).cpu()),
            **gradient_summary(model),
        }

    payload = {
        "numerics_patch": applied_patch,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "config": str(Path(args.config).resolve()),
        "world_size": args.world_size,
        "rank": args.rank,
        "sampler_first_index": int(next(iter(sampler))),
        "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "batch_finite": batch_finite,
        "loss_components": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(output))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
