#!/usr/bin/env python
"""Stage 3 validation for HalluGuard-LRBN + optional Boundary Projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from halluguard_lrbn_bp import (
    EPS,
    ForecastBatch,
    boundary_projection_batched,
    boundary_quantile_masks,
    choose_lrbn_bp_always_alpha,
    choose_lrbn_bp_params,
    choose_raw_bp_alpha,
    load_forecast_batch_from_metrics,
    lrbn_optional_bp,
    mae_per_sample,
    mse_per_sample,
    normalized_boundary_gap,
    paired_bootstrap_delta,
    summarize_against,
)


ALPHA_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
TAU_QUANTILES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
SMOOTHING_METHODS = [
    "matched_sparse_smoothing",
    "naive_smoothing",
    "ema_smoothing",
    "median_smoothing",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_stage3"))
    parser.add_argument("--tail", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch = load_forecast_batch_from_metrics(args.metrics_csv, include_methods=SMOOTHING_METHODS)
    split = batch.meta["split"].to_numpy()
    val = batch.subset(split == "val")
    test = batch.subset(split == "test")

    gated_params, gated_grid = choose_lrbn_bp_params(
        val,
        alpha_grid=ALPHA_GRID,
        tau_quantiles=TAU_QUANTILES,
        tail=args.tail,
        decay="linear",
    )
    always_params, always_grid = choose_lrbn_bp_always_alpha(
        val,
        alpha_grid=ALPHA_GRID,
        tail=args.tail,
        decay="linear",
    )
    raw_bp_params, raw_bp_grid = choose_raw_bp_alpha(val, alpha_grid=ALPHA_GRID, decay="linear")
    (args.output_dir / "selected_lrbn_bp_params.json").write_text(
        json.dumps(
            {
                "HalluGuard-LRBN-BP-gated": gated_params,
                "HalluGuard-LRBN-BP-always": always_params,
                "HalluGuard-BP-global": raw_bp_params,
                "test_threshold_leakage": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.concat([gated_grid, always_grid, raw_bp_grid], ignore_index=True, sort=False).to_csv(
        args.output_dir / "lrbn_bp_calibration_grid.csv", index=False
    )

    predictions = {
        "val": make_predictions(val, gated_params, always_params, raw_bp_params),
        "test": make_predictions(test, gated_params, always_params, raw_bp_params),
    }
    overall = []
    slices = []
    per_config = []
    sample_tables = []
    for split_name, pred_map in predictions.items():
        active = val if split_name == "val" else test
        overall.extend(overall_rows(active, pred_map, split_name))
        slices.extend(boundary_slice_rows(active, pred_map, split_name, args.tail))
        per_config.extend(per_config_rows(active, pred_map, split_name, args.tail))
        sample_tables.append(sample_level_table(active, pred_map, split_name, args.tail))
    overall_df = pd.DataFrame(overall)
    slice_df = pd.DataFrame(slices)
    per_config_df = pd.DataFrame(per_config)
    sample_df = pd.concat(sample_tables, ignore_index=True)

    overall_df.to_csv(args.output_dir / "lrbn_bp_overall.csv", index=False)
    slice_df.to_csv(args.output_dir / "lrbn_bp_boundary_slices.csv", index=False)
    per_config_df.to_csv(args.output_dir / "lrbn_bp_per_config.csv", index=False)
    write_failure_cases(sample_df, args.output_dir / "lrbn_bp_failure_cases.csv")

    ci = bootstrap_report(test, predictions["test"], args.n_bootstrap, args.seed)
    (args.output_dir / "lrbn_bp_bootstrap_ci.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")

    verdict = direction_verdict(overall_df, slice_df, per_config_df)
    (args.output_dir / "lrbn_bp_direction_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    write_summary(args.output_dir, batch, overall_df, slice_df, per_config_df, verdict, ci)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(sample_df), "status": verdict["status"]}))


def make_predictions(batch: ForecastBatch, gated_params: dict, always_params: dict, raw_bp_params: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    gated_pred, gated_selected, _ = lrbn_optional_bp(
        batch.context,
        batch.lrbn_pred,
        alpha=float(gated_params["alpha"]),
        tau=float(gated_params["tau"]),
        tail=int(gated_params.get("tail", 24)),
        decay=str(gated_params.get("decay", "linear")),
        horizons=batch.meta["horizon"].to_numpy(int),
    )
    always_alpha = float(always_params["alpha"])
    always_pred = boundary_projection_batched(
        batch.context,
        batch.lrbn_pred,
        alpha=always_alpha,
        horizons=batch.meta["horizon"].to_numpy(int),
        decay=str(always_params.get("decay", "linear")),
    )
    always_selected = np.repeat(always_alpha > 0.0, len(batch.meta))
    raw_alpha = float(raw_bp_params["alpha"])
    raw_bp_pred = boundary_projection_batched(
        batch.context,
        batch.raw_pred,
        alpha=raw_alpha,
        horizons=batch.meta["horizon"].to_numpy(int),
        decay=str(raw_bp_params.get("decay", "linear")),
    )
    raw_bp_selected = np.repeat(raw_alpha > 0.0, len(batch.meta))
    out = {
        "raw_no_correction": (batch.raw_pred, np.zeros(len(batch.meta), dtype=bool)),
        "HalluGuard-LRBN": (batch.lrbn_pred, np.zeros(len(batch.meta), dtype=bool)),
        "HalluGuard-LRBN-BP-gated": (gated_pred, gated_selected),
        "HalluGuard-LRBN-BP-always": (always_pred, always_selected),
        "HalluGuard-BP-global": (raw_bp_pred, raw_bp_selected),
    }
    if batch.extra_preds:
        for name, pred in batch.extra_preds.items():
            if not np.all(np.isnan(pred)):
                out[name] = (pred, np.zeros(len(batch.meta), dtype=bool))
    return out


def overall_rows(batch: ForecastBatch, pred_map: Dict[str, Tuple[np.ndarray, np.ndarray]], split_name: str) -> List[dict]:
    rows = []
    lrbn_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    for method, (pred, selected) in pred_map.items():
        baseline = batch.raw_pred if method == "raw_no_correction" else batch.lrbn_pred
        method_mse = mse_per_sample(pred, batch.y_true)
        row = summarize_against(method, pred, batch.y_true, baseline_pred=baseline, raw_pred=batch.raw_pred, selected=selected)
        row.update(
            {
                "split": split_name,
                "baseline": "raw_no_correction" if method == "raw_no_correction" else "HalluGuard-LRBN",
                "lrbn_mean_mse": float(np.mean(lrbn_mse)),
                "delta_vs_lrbn": float(np.mean(method_mse - lrbn_mse)),
                "delta_pct_vs_lrbn": float((np.mean(method_mse) - np.mean(lrbn_mse)) / (np.mean(lrbn_mse) + EPS) * 100.0),
                "win_rate_vs_lrbn": float(np.mean(method_mse < lrbn_mse)),
                "harm_rate_vs_lrbn": float(np.mean(method_mse > lrbn_mse + 1e-12)),
                "test_threshold_leakage": False,
            }
        )
        rows.append(row)
    return rows


def boundary_slice_rows(batch: ForecastBatch, pred_map: Dict[str, Tuple[np.ndarray, np.ndarray]], split_name: str, tail: int) -> List[dict]:
    rows = []
    gap = normalized_boundary_gap(batch.context, batch.lrbn_pred, tail=tail)
    masks = boundary_quantile_masks(gap)
    lrbn_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    raw_mse = mse_per_sample(batch.raw_pred, batch.y_true)
    for method, (pred, selected) in pred_map.items():
        method_mse = mse_per_sample(pred, batch.y_true)
        for bin_name, mask in masks.items():
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "boundary_bin": bin_name,
                    "n": int(np.sum(mask)),
                    "mean_boundary_gap": float(np.mean(gap[mask])),
                    "coverage_in_bin": float(np.mean(selected[mask])),
                    "raw_mse": float(np.mean(raw_mse[mask])),
                    "lrbn_mse": float(np.mean(lrbn_mse[mask])),
                    "method_mse": float(np.mean(method_mse[mask])),
                    "delta_vs_lrbn": float(np.mean(method_mse[mask] - lrbn_mse[mask])),
                    "delta_pct_vs_lrbn": float(
                        (np.mean(method_mse[mask]) - np.mean(lrbn_mse[mask])) / (np.mean(lrbn_mse[mask]) + EPS) * 100.0
                    ),
                    "win_rate_vs_lrbn": float(np.mean(method_mse[mask] < lrbn_mse[mask])),
                    "harm_rate_vs_lrbn": float(np.mean(method_mse[mask] > lrbn_mse[mask] + 1e-12)),
                }
            )
    return rows


def per_config_rows(batch: ForecastBatch, pred_map: Dict[str, Tuple[np.ndarray, np.ndarray]], split_name: str, tail: int) -> List[dict]:
    rows = []
    gap = normalized_boundary_gap(batch.context, batch.lrbn_pred, tail=tail)
    lrbn_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    raw_mse = mse_per_sample(batch.raw_pred, batch.y_true)
    for method, (pred, selected) in pred_map.items():
        method_mse = mse_per_sample(pred, batch.y_true)
        method_mae = mae_per_sample(pred, batch.y_true)
        df = batch.meta.copy()
        df["gap"] = gap
        df["raw_mse"] = raw_mse
        df["lrbn_mse"] = lrbn_mse
        df["method_mse"] = method_mse
        df["method_mae"] = method_mae
        df["selected"] = selected
        for keys, g in df.groupby(["dataset", "backbone", "horizon", "seed"]):
            dataset, backbone, horizon, seed = keys
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "dataset": dataset,
                    "backbone": backbone,
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "n": len(g),
                    "coverage": float(g["selected"].mean()),
                    "mean_boundary_gap": float(g["gap"].mean()),
                    "raw_mse": float(g["raw_mse"].mean()),
                    "lrbn_mse": float(g["lrbn_mse"].mean()),
                    "method_mse": float(g["method_mse"].mean()),
                    "method_mae": float(g["method_mae"].mean()),
                    "delta_vs_lrbn": float((g["method_mse"] - g["lrbn_mse"]).mean()),
                    "delta_pct_vs_lrbn": float(
                        (g["method_mse"].mean() - g["lrbn_mse"].mean()) / (g["lrbn_mse"].mean() + EPS) * 100.0
                    ),
                    "win_rate_vs_lrbn": float((g["method_mse"] < g["lrbn_mse"]).mean()),
                    "harm_rate_vs_lrbn": float((g["method_mse"] > g["lrbn_mse"] + 1e-12).mean()),
                }
            )
    return rows


def sample_level_table(batch: ForecastBatch, pred_map: Dict[str, Tuple[np.ndarray, np.ndarray]], split_name: str, tail: int) -> pd.DataFrame:
    meta = batch.meta.copy()
    gap = normalized_boundary_gap(batch.context, batch.lrbn_pred, tail=tail)
    lrbn_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    raw_mse = mse_per_sample(batch.raw_pred, batch.y_true)
    rows = []
    for method, (pred, selected) in pred_map.items():
        method_mse = mse_per_sample(pred, batch.y_true)
        for i, r in meta.iterrows():
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "config_id": r["config_id"],
                    "sample_key": r["sample_key"],
                    "dataset": r["dataset"],
                    "backbone": r["backbone"],
                    "horizon": int(r["horizon"]),
                    "seed": int(r["seed"]),
                    "boundary_gap": float(gap[i]),
                    "selected": bool(selected[i]),
                    "raw_mse": float(raw_mse[i]),
                    "lrbn_mse": float(lrbn_mse[i]),
                    "method_mse": float(method_mse[i]),
                    "delta_vs_lrbn": float(method_mse[i] - lrbn_mse[i]),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_report(batch: ForecastBatch, pred_map: Dict[str, Tuple[np.ndarray, np.ndarray]], n_boot: int, seed: int) -> Dict[str, dict]:
    lrbn_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    report = {}
    for method in ("HalluGuard-LRBN-BP-gated", "HalluGuard-LRBN-BP-always", "HalluGuard-BP-global"):
        method_mse = mse_per_sample(pred_map[method][0], batch.y_true)
        report[method] = paired_bootstrap_delta(method_mse, lrbn_mse, n_boot=n_boot, seed=seed)
    return report


def write_failure_cases(sample_df: pd.DataFrame, path: Path) -> None:
    target = sample_df[(sample_df["split"].eq("test")) & (sample_df["method"].eq("HalluGuard-LRBN-BP-gated"))].copy()
    target["case_type"] = ""
    worst = target.nlargest(50, "delta_vs_lrbn").copy()
    worst["case_type"] = "bp_worse_than_lrbn_top50"
    best = target.nsmallest(50, "delta_vs_lrbn").copy()
    best["case_type"] = "bp_better_than_lrbn_top50"
    q75 = target["boundary_gap"].quantile(0.75)
    q50 = target["boundary_gap"].quantile(0.50)
    high = target[target["boundary_gap"] > q75]
    high_success = high.nsmallest(30, "delta_vs_lrbn").copy()
    high_success["case_type"] = "q4_bp_success_top30"
    high_fail = high.nlargest(30, "delta_vs_lrbn").copy()
    high_fail["case_type"] = "q4_bp_fail_top30"
    low = target[target["boundary_gap"] <= q50]
    low_harm = low.nlargest(30, "delta_vs_lrbn").copy()
    low_harm["case_type"] = "q1q2_low_gap_harm_top30"
    pd.concat([worst, best, high_success, high_fail, low_harm], ignore_index=True).to_csv(path, index=False)


def direction_verdict(overall: pd.DataFrame, slices: pd.DataFrame, per_config: pd.DataFrame) -> dict:
    test_overall = overall[(overall["split"].eq("test")) & (overall["method"].eq("HalluGuard-LRBN-BP-gated"))].iloc[0]
    q = slices[
        (slices["split"].eq("test"))
        & (slices["method"].eq("HalluGuard-LRBN-BP-gated"))
        & (slices["boundary_bin"].isin(["q1_low", "q2", "q4_high"]))
    ]
    q4 = q[q["boundary_bin"].eq("q4_high")].iloc[0]
    low = q[q["boundary_bin"].isin(["q1_low", "q2"])]
    low_worse_pct = float(low["delta_pct_vs_lrbn"].max())
    cfg = per_config[(per_config["split"].eq("test")) & (per_config["method"].eq("HalluGuard-LRBN-BP-gated"))]
    configs_improved_ratio = float(np.mean(cfg["delta_vs_lrbn"] <= 0.0))
    overall_delta_pct = float(test_overall["delta_pct_vs_lrbn"])
    q4_improve_pct = float(-q4["delta_pct_vs_lrbn"])
    harm = float(test_overall["harm_rate_vs_baseline"])
    strong = (
        overall_delta_pct <= -0.5
        and q4_improve_pct >= 2.0
        and harm <= 0.02
        and low_worse_pct <= 0.5
        and configs_improved_ratio >= 0.60
    )
    weak = (
        -0.2 <= overall_delta_pct <= 0.2
        and q4_improve_pct >= 2.0
        and harm <= 0.02
        and low_worse_pct <= 0.5
    )
    if strong:
        status = "strong_pass"
        decision = "enter_full_table"
    elif weak:
        status = "weak_pass"
        decision = "appendix_only"
    else:
        status = "fail"
        decision = "bp_ablation_only"
    return {
        "status": status,
        "decision": decision,
        "overall_delta_pct_vs_lrbn": overall_delta_pct,
        "overall_delta_vs_lrbn": float(test_overall["delta_vs_lrbn"]),
        "q4_delta_pct_vs_lrbn": float(q4["delta_pct_vs_lrbn"]),
        "q4_improvement_pct_vs_lrbn": q4_improve_pct,
        "harm_extra_pp_vs_lrbn": harm,
        "low_slice_worse_pct": low_worse_pct,
        "configs_improved_ratio": configs_improved_ratio,
        "test_threshold_leakage": False,
    }


def write_summary(out: Path, batch: ForecastBatch, overall: pd.DataFrame, slices: pd.DataFrame, per_config: pd.DataFrame, verdict: dict, ci: dict) -> None:
    test = overall[overall["split"].eq("test")].copy()
    lines = [
        "# Stage 3 LRBN + Optional Boundary Projection Summary",
        "",
        f"- Samples: {len(batch.meta)}",
        f"- Test samples: {int(batch.meta['split'].eq('test').sum())}",
        f"- Configs: {batch.meta['config_id'].nunique()}",
        "- Calibration: validation-only alpha/tau selection; test-only evaluation.",
        f"- Verdict: `{verdict['status']}` / `{verdict['decision']}`",
        f"- Test threshold leakage: `{verdict['test_threshold_leakage']}`",
        "",
        "## Test Overall",
        "",
        "| Method | Mean MSE | Mean MAE | Delta % vs LRBN | Coverage | Harm vs LRBN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in test.sort_values("mean_mse").iterrows():
        lines.append(
            f"| {r['method']} | {r['mean_mse']:.6f} | {r['mean_mae']:.6f} | "
            f"{r['delta_pct_vs_lrbn']:.6f} | {r['coverage']:.6f} | {r['harm_rate_vs_lrbn']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary Slice: LRBN-BP-gated",
            "",
            "| Bin | Method MSE | LRBN MSE | Delta % vs LRBN | Coverage | Harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    target_slices = slices[(slices["split"].eq("test")) & (slices["method"].eq("HalluGuard-LRBN-BP-gated"))]
    for _, r in target_slices.iterrows():
        lines.append(
            f"| {r['boundary_bin']} | {r['method_mse']:.6f} | {r['lrbn_mse']:.6f} | "
            f"{r['delta_pct_vs_lrbn']:.6f} | {r['coverage_in_bin']:.6f} | {r['harm_rate_vs_lrbn']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Bootstrap vs LRBN",
            "",
            "```json",
            json.dumps(ci, indent=2),
            "```",
            "",
            "## Decision",
            "",
            f"`HalluGuard-LRBN-BP-gated` status: `{verdict['status']}`. Decision: `{verdict['decision']}`.",
        ]
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
