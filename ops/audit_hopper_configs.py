#!/usr/bin/env python3
"""Resolve and audit both linear-scaled Hopper experiment families."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import yaml
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import smplik.models  # noqa: F401,E402

CANONICAL = REPO_ROOT / "configs/experiments/smplik_amass.yaml"
EXPERIMENTS = {
    "h100x7": {
        "world_size": 7,
        "lr": 0.0014,
        "formal": REPO_ROOT / "ops/smplik_amass_h100x7_linear.yaml",
        "smoke": REPO_ROOT / "ops/smplik_amass_h100x7_linear_smoke.yaml",
    },
    "h200x4": {
        "world_size": 4,
        "lr": 0.0008,
        "formal": REPO_ROOT / "ops/smplik_amass_h200x4_linear.yaml",
        "smoke": REPO_ROOT / "ops/smplik_amass_h200x4_linear_smoke.yaml",
    },
}
TRAINING_ROWS = 50_480_676


def resolve_experiment(path):
    with open(path) as stream:
        experiment = yaml.load(stream, Loader=yaml.SafeLoader)
    parameters = experiment["parameters"]
    for key in parameters:
        if not isinstance(parameters[key], list):
            parameters[key] = [parameters[key]]
    grid = list(ParameterGrid(parameters))
    if len(grid) != 1:
        raise RuntimeError(f"{path} resolves to {len(grid)} experiments")
    overrides = [f"{key}={value}" for key, value in grid[0].items()]
    cfg = compose(experiment["base_config"] + ".yaml", overrides=overrides)
    return OmegaConf.to_container(cfg.model, resolve=True)


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


def require(flattened, expected, label):
    for key, value in expected.items():
        actual = flattened.get(key)
        if actual != value or type(actual) is not type(value):
            raise RuntimeError(
                f"{label} {key}: expected {value!r}, found {actual!r}"
            )


def fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main(output_path):
    os.chdir(REPO_ROOT)
    initialize(config_path="../configs")
    canonical = resolve_experiment(CANONICAL)
    canonical_flat = flatten(canonical)
    require(
        canonical_flat,
        {
            "seed": 0,
            "min_effectors_count": 3,
            "max_effectors_count": 16,
            "dataset.path": "./datasets",
            "dataset.name": "amass_gender_augment_cache_v1",
            "dataset.batch_size": 2048,
            "dataset.num_workers": 4,
            "dataset.rotate": True,
            "dataset.translate": False,
            "dataset.augment_training": True,
            "dataset.augment_validation": False,
            "backbone.num_blocks_stage1": 3,
            "backbone.num_blocks_stage2": 3,
            "backbone.layer_width_enc": 1024,
            "backbone.layer_width_stage1": 1024,
            "backbone.layer_width_stage2": 1024,
            "optimizer.lr": 0.0002,
            "trainer.gpus": 1,
            "trainer.num_nodes": 1,
            "trainer.max_epochs": 120,
            "trainer.precision": 32,
            "trainer.accumulate_grad_batches": 1,
            "use_pos_loss": False,
        },
        "canonical",
    )

    allowed_formal_deltas = {
        "logging.export_period",
        "logging.name",
        "logging.path",
        "optimizer.lr",
        "trainer.accelerator",
        "trainer.num_nodes",
    }
    allowed_smoke_deltas = {
        "logging.path",
        "trainer.limit_test_batches",
        "trainer.limit_train_batches",
        "trainer.limit_val_batches",
        "trainer.max_epochs",
        "trainer.num_sanity_val_steps",
    }
    report = {
        "status": "PASS",
        "classification": "linear-scaled approximations, not canonical exact reproductions",
        "canonical": str(CANONICAL),
        "canonical_fingerprint": fingerprint(canonical),
        "training_rows": TRAINING_ROWS,
        "variants": {},
    }

    canonical_prebatches = TRAINING_ROWS // canonical["dataset"]["batch_size"]
    for name, spec in EXPERIMENTS.items():
        formal = resolve_experiment(spec["formal"])
        smoke = resolve_experiment(spec["smoke"])
        formal_flat = flatten(formal)
        differences = compare(canonical, formal)
        unexpected = sorted(set(differences) - allowed_formal_deltas)
        missing = sorted(allowed_formal_deltas - set(differences))
        if unexpected or missing:
            raise RuntimeError(
                f"{name} formal deltas: unexpected={unexpected}, missing={missing}"
            )

        require(
            formal_flat,
            {
                "seed": 0,
                "min_effectors_count": 3,
                "max_effectors_count": 16,
                "dataset.path": "./datasets",
                "dataset.name": "amass_gender_augment_cache_v1",
                "dataset.batch_size": 2048,
                "dataset.num_workers": 4,
                "dataset.rotate": True,
                "dataset.translate": False,
                "dataset.augment_training": True,
                "dataset.augment_validation": False,
                "backbone.num_blocks_stage1": 3,
                "backbone.num_blocks_stage2": 3,
                "backbone.layer_width_enc": 1024,
                "backbone.layer_width_stage1": 1024,
                "backbone.layer_width_stage2": 1024,
                "optimizer.lr": spec["lr"],
                "trainer.accelerator": "ddp",
                "trainer.gpus": 1,
                "trainer.num_nodes": spec["world_size"],
                "trainer.max_epochs": 120,
                "trainer.precision": 32,
                "trainer.accumulate_grad_batches": 1,
                "trainer.replace_sampler_ddp": True,
                "use_pos_loss": False,
            },
            name,
        )
        if not math.isclose(
            formal["optimizer"]["lr"],
            canonical["optimizer"]["lr"] * spec["world_size"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{name} does not obey the linear LR scaling rule")

        smoke_differences = compare(formal, smoke)
        smoke_unexpected = sorted(set(smoke_differences) - allowed_smoke_deltas)
        smoke_missing = sorted(allowed_smoke_deltas - set(smoke_differences))
        if smoke_unexpected or smoke_missing:
            raise RuntimeError(
                f"{name} smoke deltas: unexpected={smoke_unexpected}, "
                f"missing={smoke_missing}"
            )

        updates_per_epoch = math.ceil(canonical_prebatches / spec["world_size"])
        report["variants"][name] = {
            "formal": str(spec["formal"]),
            "smoke": str(spec["smoke"]),
            "formal_fingerprint": fingerprint(formal),
            "formal_differences": differences,
            "smoke_differences": smoke_differences,
            "world_size": spec["world_size"],
            "per_rank_batch": formal["dataset"]["batch_size"],
            "global_batch": formal["dataset"]["batch_size"] * spec["world_size"],
            "canonical_lr": canonical["optimizer"]["lr"],
            "scaled_lr": formal["optimizer"]["lr"],
            "epochs": formal["trainer"]["max_epochs"],
            "canonical_prebatches_per_epoch": canonical_prebatches,
            "optimizer_updates_per_epoch": updates_per_epoch,
            "optimizer_updates_total": updates_per_epoch
            * formal["trainer"]["max_epochs"],
            "distributed_sampler_padding_prebatches": updates_per_epoch
            * spec["world_size"]
            - canonical_prebatches,
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.output)
