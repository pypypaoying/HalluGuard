#!/usr/bin/env python
"""Run Stage 4 BP harm attribution and safe-controller validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from halluguard_lrbn_bp import ForecastBatch, mse_per_sample, paired_bootstrap_delta
from halluguard_stage4_bp_harm_control import (
    add_gate_metrics,
    apply_candidate,
    boundary_features,
    boundary_slice_rows,
    candidate_grid,
    choose_best_per_method,
    make_boundary_delta,
    per_config_rows,
    segment_delta_rows,
    load_forecast_batch_from_metrics,
    evaluate_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv"),
    )
    parser.add_argument("--stage3-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_stage3"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_stage4"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch = load_forecast_batch_from_metrics(args.metrics_csv)
    split = batch.meta["split"].to_numpy()
    val = batch.subset(split == "val")
    test = batch.subset(split == "test")

    calibration = run_calibration(val)
    calibration.to_csv(args.output_dir / "stage4b_calibration_grid.csv", index=False)
    selected = choose_best_per_method(calibration)
    selected_params = selected_params_from_grid(selected)
    stage3_params = load_stage3_params(args.stage3_dir)

    test_results = evaluate_selected(val, test, selected_params, stage3_params, args.n_bootstrap, args.seed)
    write_selected_outputs(args.output_dir, test_results)

    always_test = test_results["results"]["LRBN-BP-always"]["test"]
    run_stage4a(args.output_dir, test, always_test)

    verdict = direction_verdict(test_results)
    (args.output_dir / "stage4c_direction_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    write_stage4a_summary(args.output_dir)
    write_stage4c_summary(args.output_dir, test_results, verdict)
    print(json.dumps({"output_dir": str(args.output_dir), "status": verdict["status"], "decision": verdict["decision"]}))


def run_calibration(val: ForecastBatch) -> pd.DataFrame:
    rows = []
    for params in candidate_grid(val):
        result = apply_candidate(val, params)
        row = evaluate_candidate(val, result, split="val")
        slices = pd.DataFrame(boundary_slice_rows(val, result, "val"))
        configs = pd.DataFrame(per_config_rows(val, result, "val"))
        row = add_gate_metrics(row, slices, configs)
        rows.append(row)
    return pd.DataFrame(rows)


def selected_params_from_grid(selected: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for _, row in selected.iterrows():
        method = str(row["method"])
        params: Dict[str, object] = {"method": method}
        for k, v in row.items():
            if not k.startswith("param_"):
                continue
            key = k.replace("param_", "", 1)
            if pd.isna(v):
                continue
            params[key] = to_scalar(v)
        out[method] = params
    return out


def to_scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def load_stage3_params(stage3_dir: Path) -> Dict[str, object]:
    p = stage3_dir / "selected_lrbn_bp_params.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    g = data.get("HalluGuard-LRBN-BP-gated", {})
    return {
        "method": "LRBN-BP-stage3-gated",
        "alpha": float(g.get("alpha", 0.5)),
        "tau": float(g.get("tau", float("inf"))),
        "anchor_mode": "last",
        "bridge_mode": "linear",
    }


def apply_stage3_gated(batch: ForecastBatch, params: Dict[str, object]):
    from halluguard_stage4_bp_harm_control import CandidateResult
    from halluguard_lrbn_bp import lrbn_optional_bp

    pred, selected, gap = lrbn_optional_bp(
        batch.context,
        batch.lrbn_pred,
        alpha=float(params.get("alpha", 0.5)),
        tau=float(params.get("tau", float("inf"))),
        tail=int(params.get("tail", 24)),
        decay="linear",
        horizons=batch.meta["horizon"].to_numpy(int),
    )
    delta, info = make_boundary_delta(batch, batch.lrbn_pred, alpha=float(params.get("alpha", 0.5)))
    feats = boundary_features(batch, pred - batch.lrbn_pred, info["anchor"], info["scale"])
    strength = selected.astype(float)
    feats["post_lrbn_gap_stage3_scale"] = gap
    info.update(feats)
    info["effective_strength"] = strength
    return CandidateResult("LRBN-BP-stage3-gated", pred, strength, info, params)


def evaluate_selected(
    val: ForecastBatch,
    test: ForecastBatch,
    selected_params: Dict[str, Dict[str, object]],
    stage3_params: Dict[str, object],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    eval_methods = [
        "LRBN",
        "LRBN-BP-always",
        "LRBN-BP-gap-strength",
        "LRBN-BP-bounded",
        "LRBN-BP-robust-anchor",
        "LRBN-BP-short-bridge",
        "LRBN-BP-conflict-filter",
        "LRBN-BP-repair-gate",
        "LRBN-BP-safe-controller",
    ]
    if stage3_params:
        eval_methods.append("LRBN-BP-stage3-gated")

    overall_rows = []
    slice_rows = []
    config_rows = []
    segment_rows = []
    bootstrap = {}
    results = {}
    for method in eval_methods:
        params = stage3_params if method == "LRBN-BP-stage3-gated" else selected_params.get(method, {"method": method})
        results[method] = {}
        for split_name, batch in [("val", val), ("test", test)]:
            result = apply_stage3_gated(batch, params) if method == "LRBN-BP-stage3-gated" else apply_candidate(batch, params)
            results[method][split_name] = result
            overall_rows.append(evaluate_candidate(batch, result, split=split_name))
            slice_rows.extend(boundary_slice_rows(batch, result, split_name))
            config_rows.extend(per_config_rows(batch, result, split_name))
            segment_rows.extend(segment_delta_rows(batch, result, split_name))
        method_mse = mse_per_sample(results[method]["test"].pred, test.y_true)
        parent_mse = mse_per_sample(test.lrbn_pred, test.y_true)
        bootstrap[method] = paired_bootstrap_delta(method_mse, parent_mse, n_boot=n_bootstrap, seed=seed)

    overall = pd.DataFrame(overall_rows)
    slices = pd.DataFrame(slice_rows)
    configs = pd.DataFrame(config_rows)
    enriched = []
    for _, row in overall.iterrows():
        enriched.append(add_gate_metrics(row.to_dict(), slices, configs))
    overall = pd.DataFrame(enriched)
    return {
        "overall": overall,
        "slices": slices,
        "configs": configs,
        "segments": pd.DataFrame(segment_rows),
        "bootstrap": bootstrap,
        "results": results,
    }


def write_selected_outputs(out: Path, data: Dict[str, object]) -> None:
    data["overall"].to_csv(out / "stage4c_overall.csv", index=False)
    data["slices"].to_csv(out / "stage4c_boundary_slices.csv", index=False)
    data["configs"].to_csv(out / "stage4c_per_config.csv", index=False)
    segment_summary = (
        data["segments"]
        .groupby(["split", "method", "segment"], as_index=False)
        .agg(
            rows=("delta_mse_vs_lrbn", "size"),
            mean_delta_mse_vs_lrbn=("delta_mse_vs_lrbn", "mean"),
            lrbn_mse=("lrbn_mse", "mean"),
            method_mse=("method_mse", "mean"),
        )
    )
    segment_summary["delta_pct_vs_lrbn"] = (
        (segment_summary["method_mse"] - segment_summary["lrbn_mse"]) / (segment_summary["lrbn_mse"] + 1e-8) * 100.0
    )
    segment_summary.to_csv(out / "stage4c_horizon_segments.csv", index=False)
    (out / "stage4c_bootstrap_ci.json").write_text(json.dumps(data["bootstrap"], indent=2), encoding="utf-8")


def run_stage4a(out: Path, test: ForecastBatch, always_result) -> None:
    delta = always_result.pred - test.lrbn_pred
    anchor = always_result.info.get("anchor")
    scale = always_result.info.get("scale")
    if anchor is None or scale is None:
        _, info = make_boundary_delta(test, test.lrbn_pred, alpha=0.0)
        anchor = info["anchor"]
        scale = info["scale"]
    feats = boundary_features(test, delta, anchor, scale)
    attrs = pd.DataFrame(feats)
    method_mse = mse_per_sample(always_result.pred, test.y_true)
    parent_mse = mse_per_sample(test.lrbn_pred, test.y_true)
    attrs["mse_lrbn"] = parent_mse
    attrs["mse_bp"] = method_mse
    attrs["delta_mse_vs_lrbn"] = method_mse - parent_mse
    attrs["harm_vs_lrbn"] = (attrs["delta_mse_vs_lrbn"] > 1e-12).astype(int)
    attrs["effective_strength"] = always_result.strength
    attrs = pd.concat([test.meta.reset_index(drop=True), attrs], axis=1)
    seg_rows = segment_delta_rows(test, always_result, "test")
    seg_sample = pd.DataFrame(seg_rows).groupby(["segment"], as_index=False).agg(
        rows=("delta_mse_vs_lrbn", "size"),
        mean_delta_mse_vs_lrbn=("delta_mse_vs_lrbn", "mean"),
        harm_rate=("delta_mse_vs_lrbn", lambda x: float(np.mean(np.asarray(x) > 1e-12))),
        lrbn_mse=("lrbn_mse", "mean"),
        method_mse=("method_mse", "mean"),
    )
    seg_sample["delta_pct_vs_lrbn"] = (seg_sample["method_mse"] - seg_sample["lrbn_mse"]) / (seg_sample["lrbn_mse"] + 1e-8) * 100.0

    attrs.to_csv(out / "stage4a_failure_attribution.csv", index=False)
    bin_summary(attrs, "post_lrbn_gap", mode="quantile").to_csv(out / "stage4a_boundary_gap_bins.csv", index=False)
    bin_summary(attrs, "repair_ratio", bins=[-np.inf, 0.0, 0.3, 0.7, np.inf]).to_csv(out / "stage4a_repair_ratio_bins.csv", index=False)
    bin_summary(attrs, "conflict_cosine", bins=[-np.inf, -0.2, 0.2, np.inf]).to_csv(out / "stage4a_conflict_cosine_bins.csv", index=False)
    bin_summary(attrs, "norm_ratio", bins=[-np.inf, 0.25, 0.5, 1.0, np.inf]).to_csv(out / "stage4a_norm_ratio_bins.csv", index=False)
    pd.concat(
        [
            bin_summary(attrs, "tail_volatility", mode="quantile"),
            bin_summary(attrs, "anchor_disagreement", mode="quantile"),
        ],
        ignore_index=True,
    ).to_csv(out / "stage4a_anchor_reliability_bins.csv", index=False)
    seg_sample.to_csv(out / "stage4a_horizon_segment_mse.csv", index=False)
    topk = pd.concat(
        [
            attrs.nlargest(50, "delta_mse_vs_lrbn").assign(case_type="top_harm"),
            attrs.nsmallest(50, "delta_mse_vs_lrbn").assign(case_type="top_win"),
        ],
        ignore_index=True,
    )
    topk.to_csv(out / "stage4a_failure_cases_topk.csv", index=False)


def bin_summary(df: pd.DataFrame, col: str, mode: str = "explicit", bins=None) -> pd.DataFrame:
    d = df.copy()
    if mode == "quantile":
        d["bin"] = pd.qcut(d[col], q=4, duplicates="drop")
    else:
        d["bin"] = pd.cut(d[col], bins=bins, include_lowest=True)
    rows = []
    for label, g in d.groupby("bin", observed=True):
        wins = g[g["delta_mse_vs_lrbn"] < 0]["delta_mse_vs_lrbn"]
        losses = g[g["delta_mse_vs_lrbn"] > 0]["delta_mse_vs_lrbn"]
        rows.append(
            {
                "feature": col,
                "bin": str(label),
                "n": len(g),
                "mean_feature": float(g[col].mean()),
                "mean_delta_mse_vs_lrbn": float(g["delta_mse_vs_lrbn"].mean()),
                "harm_rate_vs_lrbn": float(g["harm_vs_lrbn"].mean()),
                "mean_win_size": float((-wins).mean()) if len(wins) else 0.0,
                "mean_loss_size": float(losses.mean()) if len(losses) else 0.0,
                "win_loss_ratio": float((-wins).mean() / (losses.mean() + 1e-8)) if len(wins) and len(losses) else 0.0,
                "true_boundary_jump": float(g["true_boundary_jump"].mean()),
            }
        )
    return pd.DataFrame(rows)


def direction_verdict(data: Dict[str, object]) -> dict:
    overall = data["overall"]
    test = overall[overall["split"].eq("test")].copy()
    rows = {r.method: r for r in test.itertuples()}
    safe = rows["LRBN-BP-safe-controller"]
    always = rows["LRBN-BP-always"]
    stage3 = rows.get("LRBN-BP-stage3-gated")
    harm_reduction_vs_always = 1.0 - (safe.harm_rate_vs_lrbn / max(always.harm_rate_vs_lrbn, 1e-12))
    ci = data["bootstrap"]["LRBN-BP-safe-controller"]
    safe_pass = (
        safe.delta_pct_vs_lrbn <= -1.0
        and (safe.harm_rate_vs_lrbn <= 0.02 or harm_reduction_vs_always >= 0.30)
        and safe.q4_improvement_pct_vs_lrbn >= 2.0
        and safe.low_delta_pct_vs_lrbn <= 0.5
        and safe.config_improved_ratio >= 0.60
        and ci["ci95_low"] <= 0.0
    )
    perf_pass = always.delta_pct_vs_lrbn <= -3.0 and always.win_loss_ratio > 1.0
    if safe_pass:
        status = "safe_pass"
        decision = "enter_expanded_validation_as_safe_candidate"
    elif perf_pass:
        status = "perf_only"
        decision = "keep_lrbn_main_report_bp_perf_ablation"
    else:
        status = "fail"
        decision = "keep_stage3_gated_or_lrbn_main"
    return {
        "status": status,
        "decision": decision,
        "safe_delta_pct_vs_lrbn": float(safe.delta_pct_vs_lrbn),
        "safe_delta_mse_vs_lrbn": float(safe.delta_mse_vs_lrbn),
        "safe_harm_rate_vs_lrbn": float(safe.harm_rate_vs_lrbn),
        "safe_q4_improvement_pct_vs_lrbn": float(safe.q4_improvement_pct_vs_lrbn),
        "safe_low_delta_pct_vs_lrbn": float(safe.low_delta_pct_vs_lrbn),
        "safe_config_improved_ratio": float(safe.config_improved_ratio),
        "safe_ci95_low": float(ci["ci95_low"]),
        "safe_ci95_high": float(ci["ci95_high"]),
        "harm_reduction_vs_bp_always": float(harm_reduction_vs_always),
        "bp_always_delta_pct_vs_lrbn": float(always.delta_pct_vs_lrbn),
        "bp_always_harm_rate_vs_lrbn": float(always.harm_rate_vs_lrbn),
        "stage3_gated_delta_pct_vs_lrbn": float(stage3.delta_pct_vs_lrbn) if stage3 is not None else None,
        "stage3_gated_harm_rate_vs_lrbn": float(stage3.harm_rate_vs_lrbn) if stage3 is not None else None,
        "performance_variant_pass": bool(perf_pass),
        "test_threshold_leakage": False,
    }


def write_stage4a_summary(out: Path) -> None:
    attrs = pd.read_csv(out / "stage4a_failure_attribution.csv")
    gap = pd.read_csv(out / "stage4a_boundary_gap_bins.csv")
    conflict = pd.read_csv(out / "stage4a_conflict_cosine_bins.csv")
    norm = pd.read_csv(out / "stage4a_norm_ratio_bins.csv")
    seg = pd.read_csv(out / "stage4a_horizon_segment_mse.csv")
    lines = [
        "# Stage 4A BP-Always Harm Attribution",
        "",
        f"- Test samples: {len(attrs)}",
        f"- BP-always mean delta vs LRBN: {attrs['delta_mse_vs_lrbn'].mean():.6f}",
        f"- BP-always harm rate vs LRBN: {attrs['harm_vs_lrbn'].mean():.6f}",
        "",
        "## Boundary Gap Bins",
        "",
        df_to_markdown(gap),
        "",
        "## Conflict Cosine Bins",
        "",
        df_to_markdown(conflict),
        "",
        "## Norm Ratio Bins",
        "",
        df_to_markdown(norm),
        "",
        "## Horizon Segments",
        "",
        df_to_markdown(seg),
        "",
    ]
    (out / "stage4a_summary.md").write_text("\n".join(lines), encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_stage4c_summary(out: Path, data: Dict[str, object], verdict: dict) -> None:
    test = data["overall"][data["overall"]["split"].eq("test")].sort_values("mean_mse")
    lines = [
        "# Stage 4C BP Safe Controller Summary",
        "",
        f"- Verdict: `{verdict['status']}` / `{verdict['decision']}`",
        f"- Test threshold leakage: `{verdict['test_threshold_leakage']}`",
        "",
        "## Test Overall",
        "",
        "| Method | MSE | MAE | Delta % vs LRBN | Harm | Coverage | q4 improvement | low-gap delta | config improved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in test.iterrows():
        lines.append(
            f"| {r['method']} | {r['mean_mse']:.6f} | {r['mean_mae']:.6f} | {r['delta_pct_vs_lrbn']:.6f} | "
            f"{r['harm_rate_vs_lrbn']:.6f} | {r['coverage']:.6f} | {r['q4_improvement_pct_vs_lrbn']:.6f} | "
            f"{r['low_delta_pct_vs_lrbn']:.6f} | {r['config_improved_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Bootstrap CI",
            "",
            "```json",
            json.dumps(data["bootstrap"], indent=2),
            "```",
            "",
            "## Verdict",
            "",
            "```json",
            json.dumps(verdict, indent=2),
            "```",
        ]
    )
    (out / "stage4c_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
