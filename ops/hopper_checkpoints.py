#!/usr/bin/env python3
"""Crash-tolerant checkpoint selection for the isolated Hopper runs."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
import torch
import yaml


REQUIRED_CHECKPOINT_KEYS = {
    "epoch",
    "global_step",
    "hyper_parameters",
    "optimizer_states",
    "pytorch-lightning_version",
    "state_dict",
}


def _assert_finite_tensors(value, location):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            nonfinite = int((~torch.isfinite(value)).sum())
            raise RuntimeError(
                f"{location} contains {nonfinite} non-finite values"
            )
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tensors(child, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tensors(child, f"{location}[{index}]")


def _plain_config(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True)


def normalized_config(config):
    value = deepcopy(_plain_config(config))
    trainer = value.get("trainer")
    if isinstance(trainer, dict):
        # This field is populated operationally only after the experiment
        # identity has been checked and a valid checkpoint has been selected.
        trainer["resume_from_checkpoint"] = None
    return value


def config_fingerprint(config):
    encoded = json.dumps(
        normalized_config(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def assert_saved_hparams_match(expected_config, log_dir):
    hparams_path = Path(log_dir) / "hparams.yaml"
    if not hparams_path.is_file() or hparams_path.stat().st_size == 0:
        return False

    with hparams_path.open() as stream:
        saved_config = yaml.load(stream, Loader=yaml.SafeLoader)

    expected = normalized_config(expected_config)
    saved = normalized_config(saved_config)
    if expected != saved:
        raise RuntimeError(
            "Saved hparams do not match the requested Hopper experiment: "
            f"expected={config_fingerprint(expected)} "
            f"saved={config_fingerprint(saved)} path={hparams_path}"
        )
    return True


def _validate_checkpoint(path, expected_fingerprint):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("checkpoint file is absent or empty")

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint root is {type(checkpoint).__name__}, not dict")

    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(checkpoint))
    if missing:
        raise RuntimeError(f"missing checkpoint keys: {missing}")
    if not isinstance(checkpoint["state_dict"], dict) or not checkpoint["state_dict"]:
        raise RuntimeError("state_dict is empty")
    if (
        not isinstance(checkpoint["optimizer_states"], list)
        or not checkpoint["optimizer_states"]
    ):
        raise RuntimeError("optimizer_states is empty")
    _assert_finite_tensors(checkpoint["state_dict"], "state_dict")
    _assert_finite_tensors(checkpoint["optimizer_states"], "optimizer_states")

    checkpoint_fingerprint = config_fingerprint(checkpoint["hyper_parameters"])
    if checkpoint_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "checkpoint config fingerprint mismatch: "
            f"expected={expected_fingerprint} found={checkpoint_fingerprint}"
        )

    metadata = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "config_fingerprint": checkpoint_fingerprint,
    }
    del checkpoint
    return metadata


def select_valid_checkpoint(checkpoint_dir, expected_config):
    checkpoint_dir = Path(checkpoint_dir)
    expected_fingerprint = config_fingerprint(expected_config)
    candidates = sorted(
        checkpoint_dir.glob("*.ckpt"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"No checkpoint candidates under {checkpoint_dir}")

    rejected = {}
    for candidate in candidates:
        try:
            metadata = _validate_checkpoint(candidate, expected_fingerprint)
        except Exception as error:
            rejected[str(candidate.resolve())] = (
                f"{type(error).__name__}: {error}"
            )
            continue
        return str(candidate.resolve()), metadata, rejected

    raise RuntimeError(
        "No valid checkpoint remains after crash-tolerant validation: "
        + json.dumps(rejected, sort_keys=True)
    )


def _write_selection_evidence(log_dir, metadata, rejected):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    attempt_id = os.environ.get("HOPPER_ATTEMPT_ID", "manual")
    destination = Path(log_dir) / "checkpoint_selections" / f"{attempt_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "attempt_id": attempt_id,
        "selected": metadata,
        "rejected": rejected,
    }
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(destination))


def set_latest_hopper_checkpoint(cfg):
    if cfg.model.trainer.resume_from_checkpoint is not None:
        raise RuntimeError("Hopper experiment files must not set an explicit resume path")

    log_dir = Path(cfg.model.logging.path) / cfg.model.logging.name
    if not assert_saved_hparams_match(cfg.model, log_dir):
        return None

    selected, metadata, rejected = select_valid_checkpoint(
        log_dir / "checkpoints", cfg.model
    )
    cfg.model.trainer.resume_from_checkpoint = selected
    _write_selection_evidence(log_dir, metadata, rejected)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "Crash-tolerant checkpoint selection: "
            f"{selected} epoch={metadata['epoch']} "
            f"global_step={metadata['global_step']}",
            flush=True,
        )
    return selected
