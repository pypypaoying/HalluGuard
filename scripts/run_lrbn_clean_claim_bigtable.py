#!/usr/bin/env python
"""Run HalluGuard-LRBN clean-claim big table.

This is the server-side one-command orchestrator for the claim-clean method:
`HalluGuard-LRBN unified_revin_rdn_hybrid`.

The runner deliberately records every requested row as completed or blocked.
Current lightweight in-repo backbones are DLinear and PatchTST; larger official
backbones can be added later through the same manifest/table contract without
changing the claim method.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LRBN_SCRIPT = REPO_ROOT / "scripts" / "run_halluguard_lrbn.py"
CORE12_SCRIPT = REPO_ROOT / "scripts" / "run_core12_predictions.py"
FETCH_DATA_SCRIPT = REPO_ROOT / "scripts" / "fetch_core_datasets.py"

DEFAULT_DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Traffic")
SUPPORTED_DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Traffic")
DEFAULT_BACKBONES = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "TimeMixer")
SUPPORTED_BACKBONES = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "TimeMixer")
DEFAULT_HORIZONS = (96, 192, 336, 720)
DEFAULT_SEEDS = (2026, 2027, 2028)
DEFAULT_ADAPTATION_METHODS = (
    "raw_no_correction",
    "HalluGuard-LRBN",
    "RevIN",
    "DishTS",
    "SAN",
    "NST",
    "TAFAS",
)
SMOOTHING_METHODS = ("naive_smoothing", "ema_smoothing", "median_smoothing", "matched_sparse_smoothing")


@dataclass(frozen=True)
class Config:
    dataset: str
    backbone: str
    horizon: int
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.backbone}_{self.horizon}_seed{self.seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HalluGuard-LRBN clean-claim big table.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES))
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--methods", default=",".join(DEFAULT_ADAPTATION_METHODS))
    parser.add_argument("--include-smoothing-controls", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("external/ETDataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_clean_claim_bigtable_v1"))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--tail-len", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--san-period-len", type=int, default=24)
    parser.add_argument("--san-station-lr", type=float, default=1e-4)
    parser.add_argument("--san-pretrain-epochs", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=8192)
    parser.add_argument("--max-eval-windows", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fetch-data", action="store_true")
    parser.add_argument("--fetch-datasets", default="", help="Dataset list for fetch_core_datasets.py; defaults to requested datasets.")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke subset while preserving output schema.")
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    backbones = parse_list(args.backbones)
    horizons = [int(x) for x in parse_list(args.horizons)]
    seeds = [int(x) for x in parse_list(args.seeds)]
    methods = parse_list(args.methods)
    if args.include_smoothing_controls:
        methods = list(dict.fromkeys([*methods, *SMOOTHING_METHODS]))
    if args.smoke:
        datasets = ["ETTm1"]
        backbones = ["DLinear", "PatchTST"]
        horizons = [96]
        seeds = [seeds[0]]
        args.epochs = min(args.epochs, 1)
        args.max_train_windows = min(args.max_train_windows, 128)
        args.max_eval_windows = min(args.max_eval_windows, 32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)

    if args.fetch_data:
        fetch_datasets = args.fetch_datasets.strip() or ",".join(datasets)
        print_progress(f"fetch-data datasets={fetch_datasets} log={args.output_dir / 'logs' / 'fetch_data.log'}")
        run_command([sys.executable, str(FETCH_DATA_SCRIPT), "--datasets", fetch_datasets], args.output_dir / "logs" / "fetch_data.log")

    configs = [Config(d, b, h, s) for d in datasets for b in backbones for h in horizons for s in seeds]
    rows: List[Dict[str, object]] = []
    for idx, cfg in enumerate(configs, start=1):
        print_progress(f"config {idx}/{len(configs)} {cfg.tag}")
        if cfg.dataset not in SUPPORTED_DATASETS:
            print_progress(f"blocked {cfg.tag}: unsupported dataset")
            rows.extend(blocked_config_rows(cfg, methods, "dataset not supported by current lightweight in-repo exporter; provide official prediction adapter or extend data loader"))
            continue
        if cfg.backbone not in SUPPORTED_BACKBONES:
            print_progress(f"blocked {cfg.tag}: unsupported backbone")
            rows.extend(blocked_config_rows(cfg, methods, "backbone not supported by current lightweight in-repo exporter; integrate official TSLib/adapter for this backbone"))
            continue
        try:
            rows.extend(run_supported_config(cfg, methods, args))
        except Exception as exc:
            rows.extend(blocked_config_rows(cfg, methods, f"{type(exc).__name__}: {exc}"))
            if not args.continue_on_error:
                raise
        write_outputs(rows, args.output_dir)

    write_outputs(rows, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows), "completed": sum(r["status"] == "completed" for r in rows)}))


def run_supported_config(cfg: Config, methods: Sequence[str], args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    lrbn_methods = [m for m in methods if m in ("raw_no_correction", "HalluGuard-LRBN")]
    adapter_methods = [m for m in methods if m in ("RevIN", "DishTS", "SAN", "NST", "TAFAS")]
    smoothing_methods = [m for m in methods if m in SMOOTHING_METHODS]

    if lrbn_methods:
        lrbn_dir = args.output_dir / "predictions" / "halluguard_lrbn" / cfg.tag
        raw_dir = args.output_dir / "predictions" / "raw" / cfg.tag
        out_dir = args.output_dir / "runs" / "halluguard_lrbn" / cfg.tag
        if not (args.skip_existing and (out_dir / "lrbn_metrics.csv").exists()):
            print_progress(f"run LRBN/raw {cfg.tag} log={args.output_dir / 'logs' / f'{cfg.tag}_lrbn.log'}")
            cmd = [
                sys.executable,
                str(LRBN_SCRIPT),
                "--datasets",
                cfg.dataset,
                "--models",
                cfg.backbone,
                "--horizons",
                str(cfg.horizon),
                "--variants",
                "unified_revin_rdn_hybrid",
                "--data-root",
                str(args.data_root),
                "--prediction-dir",
                str(lrbn_dir),
                "--raw-prediction-dir",
                str(raw_dir),
                "--output-dir",
                str(out_dir),
                "--seq-len",
                str(args.seq_len),
                "--tail-len",
                str(args.tail_len),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--max-train-windows",
                str(args.max_train_windows),
                "--max-eval-windows",
                str(args.max_eval_windows),
                "--seed",
                str(cfg.seed),
                "--device",
                args.device,
                "--continue-on-error",
            ]
            run_command(cmd, args.output_dir / "logs" / f"{cfg.tag}_lrbn.log")
        else:
            print_progress(f"skip existing LRBN/raw {cfg.tag}")
        rows.extend(lrbn_rows_from_metrics(cfg, out_dir, methods))

    if adapter_methods:
        adapter_dir = args.output_dir / "predictions" / "adaptation_baselines" / cfg.tag
        if not (args.skip_existing and (adapter_dir / "manifest.csv").exists()):
            print_progress(f"run adapters {cfg.tag} methods={','.join(adapter_methods)} log={args.output_dir / 'logs' / f'{cfg.tag}_adaptation.log'}")
            cmd = [
                sys.executable,
                str(CORE12_SCRIPT),
                "--datasets",
                cfg.dataset,
                "--models",
                cfg.backbone,
                "--horizons",
                str(cfg.horizon),
                "--methods",
                ",".join(adapter_methods),
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(adapter_dir),
                "--seq-len",
                str(args.seq_len),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--san-period-len",
                str(args.san_period_len),
                "--san-station-lr",
                str(args.san_station_lr),
                "--san-pretrain-epochs",
                str(args.san_pretrain_epochs),
                "--max-train-windows",
                str(args.max_train_windows),
                "--max-eval-windows",
                str(args.max_eval_windows),
                "--seed",
                str(cfg.seed),
                "--device",
                args.device,
                "--continue-on-error",
            ]
            run_command(cmd, args.output_dir / "logs" / f"{cfg.tag}_adaptation.log")
        else:
            print_progress(f"skip existing adapters {cfg.tag}")
        rows.extend(adapter_rows_from_predictions(cfg, adapter_dir, adapter_methods))

    if smoothing_methods:
        raw_path = raw_prediction_path_for_cfg(args, cfg)
        smoothing_dir = args.output_dir / "predictions" / "smoothing_controls" / cfg.tag
        smoothing_dir.mkdir(parents=True, exist_ok=True)
        for method in smoothing_methods:
            try:
                out_path = smoothing_dir / f"{cfg.dataset}_{cfg.backbone}_{cfg.horizon}_{method}.jsonl"
                if not (args.skip_existing and out_path.exists()):
                    print_progress(f"write smoothing {cfg.tag} method={method}")
                    write_smoothing_predictions(raw_path, out_path, method)
                mse, mae = prediction_metrics(out_path)
                rows.append(metric_row(cfg, method, mse, mae, "", "", str(out_path), str(smoothing_dir), "completed", ""))
            except Exception as exc:
                rows.append(blocked_row(cfg, method, f"{type(exc).__name__}: {exc}"))
    return rows


def raw_prediction_path_for_cfg(args: argparse.Namespace, cfg: Config) -> Path:
    raw_dir = args.output_dir / "predictions" / "raw" / cfg.tag
    return raw_dir / f"{cfg.dataset}_{cfg.backbone}_{cfg.horizon}_raw_no_correction.jsonl"


def write_smoothing_predictions(raw_path: Path, out_path: Path, method: str) -> None:
    samples = read_jsonl(raw_path)
    if not samples:
        raise ValueError(f"empty raw prediction file: {raw_path}")
    rough_threshold = calibration_roughness_threshold(samples) if method == "matched_sparse_smoothing" else None
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            updated = dict(sample)
            pred = np.asarray(sample["prediction"], dtype=float)
            if method == "naive_smoothing":
                corrected = moving_average(pred, window=5)
            elif method == "ema_smoothing":
                corrected = ema_smooth(pred, alpha=0.35)
            elif method == "median_smoothing":
                corrected = median_smooth(pred, window=5)
            elif method == "matched_sparse_smoothing":
                if roughness_score(sample["context"], pred) >= float(rough_threshold):
                    corrected = moving_average(pred, window=5)
                else:
                    corrected = pred.copy()
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
            updated["prediction"] = round_list(corrected)
            updated["method"] = method
            updated["variant"] = method
            updated["model"] = f"{sample.get('model', 'raw')}+{method}"
            handle.write(json.dumps(updated) + "\n")


def calibration_roughness_threshold(samples: Sequence[dict]) -> float:
    val_scores = [
        roughness_score(sample["context"], np.asarray(sample["prediction"], dtype=float))
        for sample in samples
        if sample.get("split") == "val"
    ]
    if not val_scores:
        raise ValueError("matched_sparse_smoothing requires validation rows for threshold calibration")
    return float(np.percentile(np.asarray(val_scores, dtype=float), 75.0))


def roughness_score(context: Sequence[float], prediction: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    ctx = np.asarray(context, dtype=float)
    if pred.size < 3:
        return 0.0
    pred_rough = float(np.mean(np.abs(np.diff(pred, n=2))))
    ctx_rough = float(np.mean(np.abs(np.diff(ctx, n=2)))) if ctx.size >= 3 else 0.0
    scale = float(np.std(ctx)) + 1e-6
    return max(0.0, pred_rough - ctx_rough) / scale


def moving_average(prediction: np.ndarray, window: int = 5) -> np.ndarray:
    pred = np.asarray(prediction, dtype=float)
    if window <= 1 or pred.size < 3:
        return pred.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(pred, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def ema_smooth(prediction: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    pred = np.asarray(prediction, dtype=float)
    if pred.size < 2:
        return pred.copy()
    out = np.empty_like(pred)
    out[0] = pred[0]
    for idx in range(1, pred.size):
        out[idx] = alpha * pred[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def median_smooth(prediction: np.ndarray, window: int = 5) -> np.ndarray:
    pred = np.asarray(prediction, dtype=float)
    if window <= 1 or pred.size < 3:
        return pred.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(pred, (pad, pad), mode="edge")
    return np.asarray([np.median(padded[i : i + window]) for i in range(pred.size)], dtype=float)


def round_list(values: np.ndarray) -> List[float]:
    return [round(float(v), 6) for v in np.asarray(values, dtype=float).tolist()]


def lrbn_rows_from_metrics(cfg: Config, out_dir: Path, requested_methods: Sequence[str]) -> List[Dict[str, object]]:
    path = out_dir / "lrbn_metrics.csv"
    if not path.exists():
        return [blocked_row(cfg, m, f"missing LRBN metrics: {path}") for m in requested_methods if m in ("raw_no_correction", "HalluGuard-LRBN")]
    raw_rows = read_csv(path)
    out = []
    for method in ("raw_no_correction", "HalluGuard-LRBN"):
        if method not in requested_methods:
            continue
        variant = "raw_no_correction" if method == "raw_no_correction" else "unified_revin_rdn_hybrid"
        match = [r for r in raw_rows if r.get("variant") == variant and r.get("status") == "completed"]
        if not match:
            out.append(blocked_row(cfg, method, f"missing completed variant {variant} in {path}"))
            continue
        r = match[0]
        out.append(metric_row(cfg, method, r.get("mse"), r.get("mae"), r.get("mse_delta_pct_vs_raw", ""), r.get("mae_delta_pct_vs_raw", ""), r.get("prediction_path", ""), str(out_dir), "completed", ""))
    return out


def adapter_rows_from_predictions(cfg: Config, adapter_dir: Path, methods: Sequence[str]) -> List[Dict[str, object]]:
    out = []
    manifest_path = adapter_dir / "manifest.csv"
    manifest = read_csv(manifest_path) if manifest_path.exists() else []
    by_method = {r.get("method"): r for r in manifest}
    for method in methods:
        record = by_method.get(method)
        if not record:
            out.append(blocked_row(cfg, method, f"missing adapter manifest row for {method}"))
            continue
        if record.get("status") != "completed":
            out.append(blocked_row(cfg, method, record.get("blocker_reason", "adapter blocked")))
            continue
        pred_path = Path(record.get("output", ""))
        try:
            mse, mae = prediction_metrics(pred_path)
            out.append(metric_row(cfg, method, mse, mae, "", "", str(pred_path), str(adapter_dir), "completed", ""))
        except Exception as exc:
            out.append(blocked_row(cfg, method, f"{type(exc).__name__}: {exc}"))
    return out


def prediction_metrics(path: Path) -> tuple[float, float]:
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test = [s for s in samples if s.get("split") == "test"]
    if not test:
        raise ValueError(f"{path} has no test rows")
    preds = np.asarray([s["prediction"] for s in test], dtype=float)
    targets = np.asarray([s["target"] for s in test], dtype=float)
    return float(np.mean((preds - targets) ** 2)), float(np.mean(np.abs(preds - targets)))


def metric_row(cfg: Config, method: str, mse, mae, mse_delta, mae_delta, prediction_path: str, output_dir: str, status: str, blocker: str) -> Dict[str, object]:
    return {
        "dataset": cfg.dataset,
        "backbone": cfg.backbone,
        "horizon": cfg.horizon,
        "seed": cfg.seed,
        "method": method,
        "status": status,
        "mse": float(mse) if mse not in ("", None) else "",
        "mae": float(mae) if mae not in ("", None) else "",
        "mse_delta_pct_vs_raw": mse_delta,
        "mae_delta_pct_vs_raw": mae_delta,
        "prediction_path": prediction_path,
        "output_dir": output_dir,
        "test_threshold_leakage": False,
        "blocker_reason": blocker,
    }


def blocked_row(cfg: Config, method: str, reason: str) -> Dict[str, object]:
    return metric_row(cfg, method, "", "", "", "", "", "", "blocked", reason)


def blocked_config_rows(cfg: Config, methods: Sequence[str], reason: str) -> List[Dict[str, object]]:
    return [blocked_row(cfg, method, reason) for method in methods]


def write_outputs(rows: List[Dict[str, object]], output_dir: Path) -> None:
    write_csv(rows, output_dir / "combined_metrics.csv")
    summary = summarize(rows)
    write_csv(summary, output_dir / "summary_by_method.csv")
    by_backbone = summarize_by(rows, "backbone")
    write_csv(by_backbone, output_dir / "summary_by_backbone.csv")
    by_dataset = summarize_by(rows, "dataset")
    write_csv(by_dataset, output_dir / "summary_by_dataset.csv")
    payload = {"rows": rows, "summary_by_method": summary, "summary_by_backbone": by_backbone, "summary_by_dataset": by_dataset}
    (output_dir / "combined_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(summary_md(rows, summary), encoding="utf-8")


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    raw_lookup = completed_raw_lookup(rows)
    for method in sorted({str(r["method"]) for r in rows}):
        selected = [r for r in rows if str(r["method"]) == method]
        completed = [r for r in selected if r["status"] == "completed"]
        out.append(summary_row({"method": method}, selected, completed, raw_lookup))
    return out


def summarize_by(rows: List[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    out = []
    raw_lookup = completed_raw_lookup(rows)
    for value in sorted({str(r[key]) for r in rows}):
        for method in sorted({str(r["method"]) for r in rows}):
            selected = [r for r in rows if str(r[key]) == value and str(r["method"]) == method]
            if not selected:
                continue
            completed = [r for r in selected if r["status"] == "completed"]
            out.append(summary_row({key: value, "method": method}, selected, completed, raw_lookup))
    return out


def completed_raw_lookup(rows: List[Dict[str, object]]) -> Dict[tuple, Dict[str, object]]:
    return {
        (r["dataset"], r["backbone"], r["horizon"], r["seed"]): r
        for r in rows
        if r["method"] == "raw_no_correction" and r["status"] == "completed"
    }


def summary_row(prefix: Dict[str, object], selected: List[Dict[str, object]], completed: List[Dict[str, object]], raw_lookup: Dict[tuple, Dict[str, object]]) -> Dict[str, object]:
    deltas = []
    for r in completed:
        key = (r["dataset"], r["backbone"], r["horizon"], r["seed"])
        raw = raw_lookup.get(key)
        if raw and r["method"] != "raw_no_correction" and raw.get("mse") not in ("", 0):
            deltas.append(100.0 * (float(r["mse"]) - float(raw["mse"])) / float(raw["mse"]))
    return {
        **prefix,
        "completed_rows": len(completed),
        "total_rows": len(selected),
        "mean_mse": mean(float(r["mse"]) for r in completed) if completed else "",
        "mean_mae": mean(float(r["mae"]) for r in completed) if completed else "",
        "mean_mse_delta_pct_vs_raw_local": mean(deltas) if deltas else "",
        "blocked_rows": len(selected) - len(completed),
    }


def summary_md(rows: List[Dict[str, object]], summary: List[Dict[str, object]]) -> str:
    completed = sum(r["status"] == "completed" for r in rows)
    lines = [
        "# HalluGuard-LRBN Clean Claim BigTable v1",
        "",
        f"- Completed rows: {completed} / {len(rows)}",
        "- Claim method: `HalluGuard-LRBN unified_revin_rdn_hybrid`",
        "- Test threshold leakage: False",
        "",
        "## Method Summary",
        "",
    ]
    for row in summary:
        lines.append(f"- `{row['method']}`: completed {row['completed_rows']} / {row['total_rows']}, mean MSE {row['mean_mse']}, blocked {row['blocked_rows']}")
    blocked = [r for r in rows if r["status"] != "completed"]
    if blocked:
        lines.extend(["", "## Blocked Rows", ""])
        for row in blocked[:200]:
            lines.append(f"- {row['dataset']} {row['backbone']} h{row['horizon']} seed{row['seed']} `{row['method']}`: {row['blocker_reason']}")
    return "\n".join(lines) + "\n"


def run_command(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\nexit_code={proc.returncode} elapsed_sec={time.time() - start:.2f}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}; see {log_path}")


def print_progress(message: str) -> None:
    print(f"[halluguard-bigtable] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def parse_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
