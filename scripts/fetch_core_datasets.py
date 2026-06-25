#!/usr/bin/env python
"""Fetch core long-term forecasting datasets without committing data files.

The primary source is the THUML Time-Series-Library Hugging Face dataset. The
script writes CSV files into the layout expected by Time-Series-Library and
mirrors ETT-small files into the legacy HalluGuard real-pipeline directory.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ID = "thuml/Time-Series-Library"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_config: str
    subdir: str
    filename: str
    group: str
    mirror_ett: bool = False


DATASETS = {
    "ETTm1": DatasetSpec("ETTm1", "ETTm1", "ETT-small", "ETTm1.csv", "core", True),
    "ETTh1": DatasetSpec("ETTh1", "ETTh1", "ETT-small", "ETTh1.csv", "core", True),
    "Weather": DatasetSpec("Weather", "weather", "weather", "weather.csv", "core"),
    "Electricity": DatasetSpec("Electricity", "electricity", "electricity", "electricity.csv", "core"),
    "ECL": DatasetSpec("ECL", "electricity", "electricity", "electricity.csv", "core"),
    "ETTm2": DatasetSpec("ETTm2", "ETTm2", "ETT-small", "ETTm2.csv", "optional_extended", True),
    "ETTh2": DatasetSpec("ETTh2", "ETTh2", "ETT-small", "ETTh2.csv", "optional_extended", True),
    "Traffic": DatasetSpec("Traffic", "traffic", "traffic", "traffic.csv", "optional_extended"),
}


def parse_dataset_names(raw: str) -> list[str]:
    if raw == "core":
        return [name for name, spec in DATASETS.items() if spec.group == "core"]
    if raw == "extended":
        return [name for name, spec in DATASETS.items() if spec.group in {"core", "optional_extended"}]
    if raw == "all":
        return list(DATASETS)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Valid: {', '.join(DATASETS)}")
    return names


def load_hf_dataset(config_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install with `pip install -r requirements.txt` first."
        ) from exc
    ds = load_dataset(REPO_ID, name=config_name)
    split = "train" if "train" in ds else next(iter(ds.keys()))
    return ds[split].to_pandas()


def write_csv(df, path: Path, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return "written"


def iter_destinations(repo_root: Path, spec: DatasetSpec, mirror_ett: bool) -> Iterable[Path]:
    yield repo_root / "external" / "Time-Series-Library" / "dataset" / spec.subdir / spec.filename
    if spec.mirror_ett and mirror_ett:
        yield repo_root / "external" / "ETDataset" / "ETT-small" / spec.filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HalluGuard core-table datasets.")
    parser.add_argument(
        "--datasets",
        default="core",
        help="core, extended, all, or a comma-separated list such as ETTm1,ETTh1,Weather",
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-ett-mirror", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    selected = parse_dataset_names(args.datasets)
    print(f"Source: {REPO_ID}")
    print(f"Repo root: {repo_root}")
    print(f"Datasets: {', '.join(selected)}")

    for name in selected:
        spec = DATASETS[name]
        destinations = list(iter_destinations(repo_root, spec, mirror_ett=not args.no_ett_mirror))
        if args.dry_run:
            print(f"[dry-run] {name}: hf_config={spec.hf_config} -> {', '.join(str(p) for p in destinations)}")
            continue
        df = load_hf_dataset(spec.hf_config)
        for dest in destinations:
            status = write_csv(df, dest, overwrite=args.overwrite)
            print(f"{name}: {status} {dest} rows={len(df)} cols={len(df.columns)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
