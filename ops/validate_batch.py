#!/usr/bin/env python3
"""Validate one real AMASS batch using the formal DataModule parameters."""

import json
from pathlib import Path

import torch

from smplik.data.smpl_module import SmplDataModule


def main() -> None:
    dm = SmplDataModule(
        path="./datasets",
        name="amass_gender_augment_cache_v1",
        batch_size=2048,
        num_workers=4,
        rotate=True,
        translate=False,
        augment_training=True,
        augment_validation=False,
    )
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    for key, value in batch.items():
        print(key, tuple(value.shape), value.dtype, flush=True)

    assert batch["joint_positions"].ndim == 3
    assert batch["joint_positions"].shape[1:] == (24, 3)
    assert batch["joint_rotations"].ndim == 3
    assert batch["joint_rotations"].shape[1:] == (24, 4)
    assert batch["betas"].ndim == 2
    assert batch["betas"].shape[1] == 10
    assert batch["gender"].ndim == 2
    assert batch["gender"].shape[1] == 1
    assert batch["joint_positions"].shape[0] == 2048

    finite = {}
    for key, value in batch.items():
        if torch.is_floating_point(value):
            finite[key] = bool(torch.isfinite(value).all())
            assert finite[key], f"Non-finite values in {key}"

    report = {
        "fields": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in batch.items()
        },
        "finite": finite,
    }
    Path("./run_reports/batch_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("Real batch validation passed.", flush=True)


if __name__ == "__main__":
    main()
