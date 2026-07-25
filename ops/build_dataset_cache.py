#!/usr/bin/env python3
"""Build the three SMPL-IK Feather caches through the repository loader."""

import gc
import json
import resource
import time
from pathlib import Path

from smplik.data.dataset.typed_table import TypedColumnDataset
from smplik.data.datasets import DatasetLoader


DATASET_ROOT = Path("./datasets")
DATASET_NAME = "amass_gender_augment_cache_v1"


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main() -> None:
    loader = DatasetLoader(str(DATASET_ROOT))
    dataset_path = Path(loader.pull(DATASET_NAME))
    split = loader.get_split(DATASET_NAME)

    report = {
        "dataset_name": DATASET_NAME,
        "dataset_path": str(dataset_path.resolve()),
        "splits": {},
    }
    for subset in ("Training", "Validation", "Test"):
        started = time.time()
        dataset = TypedColumnDataset(split, subset=subset)
        cache_path = dataset_path / f"{subset}_cache.feather"
        if len(dataset) == 0 or not cache_path.is_file() or cache_path.stat().st_size == 0:
            raise RuntimeError(f"{subset} cache was not built correctly: {cache_path}")
        report["splits"][subset] = {
            "source_files": len(split[subset]),
            "rows_after_official_deduplication": len(dataset),
            "cache_path": str(cache_path.resolve()),
            "cache_bytes": cache_path.stat().st_size,
            "elapsed_seconds": time.time() - started,
            "max_rss_gib": max_rss_gib(),
        }
        print(json.dumps({subset: report["splits"][subset]}, indent=2), flush=True)
        del dataset
        gc.collect()

    report["max_rss_gib"] = max_rss_gib()
    report_path = Path("./run_reports/dataset_cache_build.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Cache build report: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
