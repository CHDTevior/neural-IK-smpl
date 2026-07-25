#!/usr/bin/env python3
"""Record the official loader view of the rebuilt AMASS dataset."""

import json
from pathlib import Path

from smplik.data.datasets import DatasetLoader


DATASET_ROOT = Path("./datasets")
DATASET_NAME = "amass_gender_augment_cache_v1"


def main() -> None:
    loader = DatasetLoader(str(DATASET_ROOT))
    dataset_path = Path(loader.pull(DATASET_NAME))
    split = loader.get_split(DATASET_NAME)
    files = [path for path in dataset_path.rglob("*") if path.is_file()]

    report = {
        "dataset_name": DATASET_NAME,
        "dataset_path": str(dataset_path.resolve()),
        "available": loader.is_available(DATASET_NAME),
        "valid": loader.is_valid(DATASET_NAME),
        "settings_path": loader.settings_file_of(DATASET_NAME),
        "settings": loader.get_settings(DATASET_NAME),
        "dataset_bytes": sum(path.stat().st_size for path in files),
        "file_count": len(files),
        "training_split_files": len(split["Training"]),
        "validation_split_files": len(split["Validation"]),
        "test_split_files": len(split["Test"]) if split["Test"] is not None else 0,
        "split_path": split["SplitFile"],
    }
    if not report["available"] or not report["valid"]:
        raise RuntimeError("DatasetLoader does not accept the rebuilt dataset")
    if (
        report["training_split_files"] == 0
        or report["validation_split_files"] == 0
        or report["test_split_files"] == 0
    ):
        raise RuntimeError("The rebuilt dataset has an empty official split")

    report_path = Path("./run_reports/dataset_audit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Dataset audit passed: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
