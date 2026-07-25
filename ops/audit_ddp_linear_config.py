#!/usr/bin/env python3
"""Resolve and compare the canonical and linear-scaled DDP experiment configs."""

import argparse
import json
from pathlib import Path
import sys

import yaml
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid

# Register the structured Hydra schemas, matching run.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import smplik.models  # noqa: F401


def resolve_experiment(path):
    with open(path) as stream:
        experiment = yaml.load(stream, Loader=yaml.SafeLoader)

    parameters = experiment["parameters"]
    for key in parameters:
        if not isinstance(parameters[key], list):
            parameters[key] = [parameters[key]]

    grid = list(ParameterGrid(parameters))
    if len(grid) != 1:
        raise RuntimeError(f"{path} resolves to {len(grid)} experiments, expected exactly one")

    overrides = [f"{key}={value}" for key, value in grid[0].items()]
    config = compose(experiment["base_config"] + ".yaml", overrides=overrides)
    return OmegaConf.to_container(config.model, resolve=True)


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(flatten(child, child_prefix))
    else:
        result[prefix] = value
    return result


def compare(left, right):
    left_flat = flatten(left)
    right_flat = flatten(right)
    keys = sorted(set(left_flat) | set(right_flat))
    return {
        key: {"canonical": left_flat.get(key), "candidate": right_flat.get(key)}
        for key in keys
        if (
            left_flat.get(key) != right_flat.get(key)
            or type(left_flat.get(key)) is not type(right_flat.get(key))
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--formal", required=True)
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-rows", type=int, required=True)
    args = parser.parse_args()

    initialize(config_path="../configs")
    canonical = resolve_experiment(args.canonical)
    formal = resolve_experiment(args.formal)
    smoke = resolve_experiment(args.smoke)

    formal_differences = compare(canonical, formal)
    allowed_formal_differences = {
        "logging.export_period",
        "logging.name",
        "logging.path",
        "optimizer.lr",
        "trainer.accelerator",
        "trainer.gpus",
    }
    unexpected = sorted(set(formal_differences) - allowed_formal_differences)
    missing = sorted(allowed_formal_differences - set(formal_differences))

    required_canonical = {
        "dataset.batch_size": 2048,
        "optimizer.lr": 0.0002,
        "trainer.accumulate_grad_batches": 1,
        "trainer.gpus": 1,
        "trainer.max_epochs": 120,
        "trainer.precision": 32,
    }
    required_formal = {
        "dataset.batch_size": 2048,
        "optimizer.lr": 0.0008,
        "trainer.accelerator": "ddp_spawn",
        "trainer.accumulate_grad_batches": 1,
        "trainer.gpus": 4,
        "trainer.max_epochs": 120,
        "trainer.precision": 32,
        "trainer.replace_sampler_ddp": True,
    }

    canonical_flat = flatten(canonical)
    formal_flat = flatten(formal)
    for key, expected in required_canonical.items():
        actual = canonical_flat.get(key)
        if actual != expected:
            raise RuntimeError(f"canonical {key}: expected {expected!r}, got {actual!r}")
    for key, expected in required_formal.items():
        actual = formal_flat.get(key)
        if actual != expected:
            raise RuntimeError(f"formal {key}: expected {expected!r}, got {actual!r}")
    if unexpected or missing:
        raise RuntimeError(
            f"formal config delta mismatch; unexpected={unexpected}, missing={missing}"
        )

    local_batch = formal["dataset"]["batch_size"]
    world_size = formal["trainer"]["gpus"] * formal["trainer"]["num_nodes"]
    canonical_batches = args.training_rows // canonical["dataset"]["batch_size"]
    ddp_updates = (canonical_batches + world_size - 1) // world_size
    report = {
        "status": "PASS",
        "classification": "linear-scaled approximation, not canonical exact reproduction",
        "canonical_config": str(Path(args.canonical).resolve()),
        "formal_config": str(Path(args.formal).resolve()),
        "smoke_config": str(Path(args.smoke).resolve()),
        "formal_differences": formal_differences,
        "scaling": {
            "world_size": world_size,
            "per_rank_batch": local_batch,
            "global_batch": local_batch * world_size,
            "canonical_lr": canonical["optimizer"]["lr"],
            "scaled_lr": formal["optimizer"]["lr"],
            "max_epochs": formal["trainer"]["max_epochs"],
            "canonical_optimizer_updates_per_epoch": canonical_batches,
            "ddp_optimizer_updates_per_epoch": ddp_updates,
            "canonical_total_optimizer_updates": (
                canonical_batches * canonical["trainer"]["max_epochs"]
            ),
            "ddp_total_optimizer_updates": ddp_updates * formal["trainer"]["max_epochs"],
        },
        "smoke_differences_from_formal": compare(formal, smoke),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
