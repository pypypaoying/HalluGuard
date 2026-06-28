#!/usr/bin/env python
"""Stage 4.5 BP-always failure attribution validation.

This runner reuses existing compact real-forecast assets and produces
sample-level mechanism evidence for dense Boundary Projection after LRBN.
All split-derived thresholds are calibrated on validation samples only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from halluguard_lrbn_bp import (
    EPS,
    ForecastBatch,
    load_forecast_batch_from_metrics,
    lrbn_optional_bp,
    mae_per_sample,
    mse_per_sample,
)
from halluguard_stage4_bp_harm_control import (
    CandidateResult,
    apply_candidate,
    boundary_features,
    make_boundary_delta,
    robust_anchor,
    tail_scale,
)


DEFAULT_METRICS = Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv")


def horizons(batch: ForecastBatch) -> np.ndarray:
    return batch.meta["horizon"].to_numpy(int)


def finite_mean(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def safe_pct(method_mean: float, baseline_mean: float) -> float:
    return float((method_mean - baseline_mean) / (baseline_mean + EPS) * 100.0)


def flat_valid(x: np.ndarray, i: int, h: int) -> np.ndarray:
    return np.asarray(x[i, :h, :], dtype=float).reshape(-1)


def alignment_scores(delta: np.ndarray, residual: np.ndarray, hs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a_vals = np.zeros(len(hs), dtype=float)
    cos_vals = np.zeros(len(hs), dtype=float)
    for i, h in enumerate(hs):
        d = flat_valid(delta, i, int(h))
        e = flat_valid(residual, i, int(h))
        d_norm2 = float(np.dot(d, d))
        e_norm = float(np.linalg.norm(e))
        d_norm = math.sqrt(max(d_norm2, 0.0))
        dot = float(np.dot(d, e))
        a_vals[i] = 2.0 * dot / (d_norm2 + EPS)
        cos_vals[i] = dot / (d_norm * e_norm + EPS)
    return a_vals, cos_vals


def segment_bounds(h: int) -> Dict[str, Tuple[int, int]]:
    q1 = max(1, h // 4)
    q3 = max(q1 + 1, 3 * h // 4)
    return {
        "early": (0, q1),
        "mid": (q1, q3),
        "late": (q3, h),
    }


def segment_alignment_records(batch: ForecastBatch, pred: np.ndarray, method: str) -> pd.DataFrame:
    hs = horizons(batch)
    rows: List[dict] = []
    delta = pred - batch.lrbn_pred
    residual = batch.y_true - batch.lrbn_pred
    for i, h in enumerate(hs):
        meta = batch.meta.iloc[i].to_dict()
        for segment, (a, b) in segment_bounds(int(h)).items():
            d = delta[i, a:b, :].reshape(-1)
            e = residual[i, a:b, :].reshape(-1)
            base = batch.lrbn_pred[i, a:b, :]
            y = batch.y_true[i, a:b, :]
            p = pred[i, a:b, :]
            d_norm2 = float(np.dot(d, d))
            dot = float(np.dot(d, e))
            cos = dot / (math.sqrt(max(d_norm2, 0.0)) * float(np.linalg.norm(e)) + EPS)
            parent_mse = float(np.mean((base - y) ** 2))
            method_mse = float(np.mean((p - y) ** 2))
            rows.append(
                {
                    **meta,
                    "method": method,
                    "segment": segment,
                    "lrbn_mse": parent_mse,
                    "method_mse": method_mse,
                    "delta_vs_lrbn": method_mse - parent_mse,
                    "A": 2.0 * dot / (d_norm2 + EPS),
                    "cos": cos,
                    "harm": bool(method_mse > parent_mse + 1e-12),
                }
            )
    return pd.DataFrame(rows)


def load_stage3_params(stage3_dir: Path) -> Dict[str, object]:
    path = stage3_dir / "selected_lrbn_bp_params.json"
    if not path.exists():
        return {"alpha": 0.5, "tau": float("inf"), "tail": 24, "decay": "linear"}
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data.get("HalluGuard-LRBN-BP-gated", {})
    return {
        "alpha": float(row.get("alpha", 0.5)),
        "tau": float(row.get("tau", float("inf"))),
        "tail": int(row.get("tail", 24)),
        "decay": str(row.get("decay", "linear")),
    }


def load_repair_gate_params(stage4_dir: Path) -> Dict[str, object]:
    path = stage4_dir / "stage4c_overall.csv"
    fallback = {
        "method": "LRBN-BP-repair-gate",
        "alpha": 0.2,
        "anchor_mode": "last",
        "bridge_mode": "linear",
        "high_repair": 0.3,
    }
    if not path.exists():
        return fallback
    df = pd.read_csv(path)
    rows = df[(df["method"] == "LRBN-BP-repair-gate") & (df["split"] == "test")]
    if rows.empty:
        rows = df[df["method"] == "LRBN-BP-repair-gate"]
    if rows.empty:
        return fallback
    r = rows.iloc[0]
    out = {"method": "LRBN-BP-repair-gate"}
    for col in df.columns:
        if not col.startswith("param_") or pd.isna(r[col]):
            continue
        out[col.replace("param_", "", 1)] = scalar(r[col])
    out.setdefault("alpha", 0.2)
    out.setdefault("anchor_mode", "last")
    out.setdefault("bridge_mode", "linear")
    out.setdefault("high_repair", 0.3)
    return out


def scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def apply_bp_always(batch: ForecastBatch, alpha: float = 0.5, anchor_mode: str = "last") -> CandidateResult:
    delta, info = make_boundary_delta(
        batch=batch,
        parent=batch.lrbn_pred,
        alpha=alpha,
        anchor_mode=anchor_mode,
        bridge_mode="linear",
        tail_len=16,
    )
    pred = batch.lrbn_pred + delta
    feats = boundary_features(batch, delta, info["anchor"], info["scale"])
    info.update(feats)
    strength = np.ones(len(batch.meta), dtype=float)
    return CandidateResult(f"LRBN-BP-always-{anchor_mode}", pred, strength, info, {"alpha": alpha, "anchor_mode": anchor_mode})


def apply_stage3(batch: ForecastBatch, params: Dict[str, object]) -> CandidateResult:
    pred, selected, gap = lrbn_optional_bp(
        batch.context,
        batch.lrbn_pred,
        alpha=float(params.get("alpha", 0.5)),
        tau=float(params.get("tau", float("inf"))),
        tail=int(params.get("tail", 24)),
        decay=str(params.get("decay", "linear")),
        horizons=horizons(batch),
    )
    delta, info = make_boundary_delta(batch, batch.lrbn_pred, alpha=float(params.get("alpha", 0.5)))
    feats = boundary_features(batch, pred - batch.lrbn_pred, info["anchor"], info["scale"])
    info.update(feats)
    info["stage3_gap"] = gap
    info["effective_strength"] = selected.astype(float)
    return CandidateResult("LRBN-BP-stage3-gated", pred, selected.astype(float), info, params)


def calibrate_thresholds(val_table: pd.DataFrame) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for col in ["g_L", "g_y", "norm_ratio", "anchor_disagreement", "raw_gap"]:
        qs = val_table[col].quantile([0.25, 0.50, 0.75]).to_dict()
        thresholds[f"{col}_q25"] = float(qs[0.25])
        thresholds[f"{col}_q50"] = float(qs[0.50])
        thresholds[f"{col}_q75"] = float(qs[0.75])
    thresholds["repair_low"] = 0.3
    thresholds["repair_high"] = 0.7
    return thresholds


def quantile_label(v: float, q25: float, q50: float, q75: float) -> str:
    if v <= q25:
        return "q1_low"
    if v <= q50:
        return "q2"
    if v <= q75:
        return "q3"
    return "q4_high"


def repair_label(v: float) -> str:
    if v < 0.0:
        return "<0"
    if v <= 0.3:
        return "0-0.3"
    if v <= 0.7:
        return "0.3-0.7"
    return ">0.7"


def conflict_label(v: float) -> str:
    if v < -0.2:
        return "<-0.2"
    if v <= 0.2:
        return "[-0.2,0.2]"
    return ">0.2"


def make_sample_table(
    batch: ForecastBatch,
    split_name: str,
    bp: CandidateResult,
    stage3: CandidateResult,
    repair: CandidateResult,
    thresholds: Dict[str, float] | None = None,
) -> pd.DataFrame:
    hs = horizons(batch)
    residual = batch.y_true - batch.lrbn_pred
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)

    method_specs = {
        "bp": bp.pred,
        "stage3": stage3.pred,
        "repair_gate": repair.pred,
    }
    align = {}
    for key, pred in method_specs.items():
        a, c = alignment_scores(pred - batch.lrbn_pred, residual, hs)
        align[f"A_{key}"] = a
        align[f"cos_{key}"] = c

    feature_keys = [
        "raw_gap",
        "post_lrbn_gap",
        "repair_ratio",
        "conflict_cosine",
        "norm_ratio",
        "tail_volatility",
        "anchor_disagreement",
        "true_boundary_jump",
    ]
    feats = {k: np.asarray(bp.info[k], dtype=float) for k in feature_keys}
    table = batch.meta.reset_index(drop=True).copy()
    table["split"] = split_name
    table["horizon"] = hs
    table["mse_lrbn"] = base_mse
    table["mae_lrbn"] = base_mae
    for key, pred in method_specs.items():
        mm = mse_per_sample(pred, batch.y_true)
        ma = mae_per_sample(pred, batch.y_true)
        table[f"mse_{key}"] = mm
        table[f"mae_{key}"] = ma
        table[f"delta_{key}_vs_lrbn"] = mm - base_mse
        table[f"gain_{key}_vs_lrbn"] = base_mse - mm
        table[f"harm_{key}_vs_lrbn"] = mm > base_mse + 1e-12
        table[f"A_{key}"] = align[f"A_{key}"]
        table[f"cos_{key}"] = align[f"cos_{key}"]

    table["stage3_selected"] = stage3.strength > 1e-8
    table["repair_strength"] = repair.strength
    table["repair_selected"] = repair.strength > 1e-8
    table["g_L"] = feats["post_lrbn_gap"]
    table["g_y"] = feats["true_boundary_jump"]
    table["raw_gap"] = feats["raw_gap"]
    table["repair_ratio"] = feats["repair_ratio"]
    table["conflict_cosine"] = feats["conflict_cosine"]
    table["norm_ratio"] = feats["norm_ratio"]
    table["tail_volatility"] = feats["tail_volatility"]
    table["anchor_disagreement"] = feats["anchor_disagreement"]
    if thresholds:
        table = add_calibrated_labels(table, thresholds)
    return table


def add_calibrated_labels(table: pd.DataFrame, t: Dict[str, float]) -> pd.DataFrame:
    out = table.copy()
    out["g_L_bin"] = out["g_L"].apply(lambda v: quantile_label(v, t["g_L_q25"], t["g_L_q50"], t["g_L_q75"]))
    out["g_y_bin"] = out["g_y"].apply(lambda v: quantile_label(v, t["g_y_q25"], t["g_y_q50"], t["g_y_q75"]))
    out["norm_ratio_bin"] = out["norm_ratio"].apply(
        lambda v: quantile_label(v, t["norm_ratio_q25"], t["norm_ratio_q50"], t["norm_ratio_q75"])
    )
    out["anchor_disagreement_bin"] = out["anchor_disagreement"].apply(
        lambda v: quantile_label(
            v,
            t["anchor_disagreement_q25"],
            t["anchor_disagreement_q50"],
            t["anchor_disagreement_q75"],
        )
    )
    out["repair_ratio_bin"] = out["repair_ratio"].apply(repair_label)
    out["conflict_cosine_bin"] = out["conflict_cosine"].apply(conflict_label)
    out["pred_gap_flag"] = out["g_L"] >= t["g_L_q75"]
    out["true_jump_flag"] = out["g_y"] >= t["g_y_q75"]
    out["low_gap_flag"] = out["g_L"] <= t["g_L_q25"]
    out["high_repair_flag"] = out["repair_ratio"] > t["repair_high"]
    out["low_repair_flag"] = out["repair_ratio"] <= t["repair_low"]

    def truth_bin(r: pd.Series) -> str:
        if r["low_gap_flag"] and not r["true_jump_flag"]:
            return "low_gL_low_gY"
        if r["pred_gap_flag"] and not r["true_jump_flag"]:
            return "high_gL_low_gY"
        if r["pred_gap_flag"] and r["true_jump_flag"]:
            return "high_gL_high_gY"
        if r["low_gap_flag"] and r["true_jump_flag"]:
            return "low_gL_high_gY"
        return "mid_or_mixed"

    out["oracle_boundary_bin"] = out.apply(truth_bin, axis=1)
    return out


def summarize_samples(df: pd.DataFrame, mask: np.ndarray | pd.Series, label: str, kind: str) -> dict:
    g = df.loc[np.asarray(mask, dtype=bool)]
    if g.empty:
        return {
            "slice_type": kind,
            "slice_name": label,
            "count": 0,
            "mean_A": float("nan"),
            "A_gt_1_rate": float("nan"),
            "mean_cos": float("nan"),
            "mean_delta_vs_lrbn": float("nan"),
            "harm_rate": float("nan"),
        }
    return {
        "slice_type": kind,
        "slice_name": label,
        "count": int(len(g)),
        "mean_A": float(g["A_bp"].mean()),
        "A_gt_1_rate": float((g["A_bp"] > 1.0).mean()),
        "mean_cos": float(g["cos_bp"].mean()),
        "mean_delta_vs_lrbn": float(g["delta_bp_vs_lrbn"].mean()),
        "harm_rate": float(g["harm_bp_vs_lrbn"].mean()),
    }


def oracle_boundary_truth(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["low_gL_low_gY", "high_gL_low_gY", "high_gL_high_gY", "low_gL_high_gY", "mid_or_mixed"]:
        g = test[test["oracle_boundary_bin"] == label]
        rows.append(
            {
                "bin": label,
                "count": int(len(g)),
                "mean_delta_vs_lrbn": float(g["delta_bp_vs_lrbn"].mean()) if len(g) else float("nan"),
                "harm_rate": float(g["harm_bp_vs_lrbn"].mean()) if len(g) else float("nan"),
                "mean_g_L": float(g["g_L"].mean()) if len(g) else float("nan"),
                "mean_g_y": float(g["g_y"].mean()) if len(g) else float("nan"),
                "true_jump_rate": float(g["true_jump_flag"].mean()) if len(g) else float("nan"),
                "bp_needed_rate": float((g["A_bp"] > 1.0).mean()) if len(g) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def residual_alignment_by_slice(test: pd.DataFrame, segment_records: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for col, kind in [
        ("g_L_bin", "post_lrbn_gap_quantile"),
        ("repair_ratio_bin", "repair_ratio_bin"),
        ("conflict_cosine_bin", "conflict_cosine_bin"),
        ("norm_ratio_bin", "norm_ratio_quantile"),
        ("anchor_disagreement_bin", "anchor_disagreement_quantile"),
        ("dataset", "dataset"),
        ("backbone", "backbone"),
        ("horizon", "horizon"),
        ("seed", "seed"),
    ]:
        for label, _ in test.groupby(col, observed=True):
            rows.append(summarize_samples(test, test[col] == label, str(label), kind))
    for label, g in segment_records.groupby("segment", observed=True):
        rows.append(
            {
                "slice_type": "horizon_segment",
                "slice_name": str(label),
                "count": int(len(g)),
                "mean_A": float(g["A"].mean()),
                "A_gt_1_rate": float((g["A"] > 1.0).mean()),
                "mean_cos": float(g["cos"].mean()),
                "mean_delta_vs_lrbn": float(g["delta_vs_lrbn"].mean()),
                "harm_rate": float(g["harm"].mean()),
            }
        )
    return pd.DataFrame(rows)


def gap_repair_interaction(test: pd.DataFrame) -> pd.DataFrame:
    def gap_group(r: pd.Series) -> str:
        if r["low_gap_flag"]:
            return "low"
        if r["pred_gap_flag"]:
            return "high"
        return "mid"

    def repair_group(r: pd.Series) -> str:
        if r["low_repair_flag"]:
            return "low"
        if r["high_repair_flag"]:
            return "high"
        return "mid"

    d = test.copy()
    d["gap_group"] = d.apply(gap_group, axis=1)
    d["repair_group"] = d.apply(repair_group, axis=1)
    rows = []
    for gap in ["low", "mid", "high"]:
        for repair in ["low", "mid", "high"]:
            g = d[(d["gap_group"] == gap) & (d["repair_group"] == repair)]
            wins = -g.loc[g["delta_bp_vs_lrbn"] < 0, "delta_bp_vs_lrbn"]
            losses = g.loc[g["delta_bp_vs_lrbn"] > 0, "delta_bp_vs_lrbn"]
            rows.append(
                {
                    "gap_group": gap,
                    "repair_group": repair,
                    "count": int(len(g)),
                    "mean_delta_vs_lrbn": float(g["delta_bp_vs_lrbn"].mean()) if len(g) else float("nan"),
                    "harm_rate": float(g["harm_bp_vs_lrbn"].mean()) if len(g) else float("nan"),
                    "mean_A": float(g["A_bp"].mean()) if len(g) else float("nan"),
                    "A_gt_1_rate": float((g["A_bp"] > 1.0).mean()) if len(g) else float("nan"),
                    "win_loss_ratio": float(wins.mean() / (losses.mean() + EPS)) if len(wins) and len(losses) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def gini_positive(x: np.ndarray) -> float:
    values = np.sort(np.asarray(x, dtype=float))
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return 0.0
    n = len(values)
    return float((2.0 * np.arange(1, n + 1) @ values) / (n * values.sum() + EPS) - (n + 1.0) / n)


def win_loss_stats(df: pd.DataFrame, scope: str) -> dict:
    gain = df["gain_bp_vs_lrbn"].to_numpy(float)
    wins = gain[gain > 0]
    losses = -gain[gain < 0]
    top_n = max(1, int(math.ceil(0.05 * max(len(wins), 1))))
    top1_n = max(1, int(math.ceil(0.01 * max(len(wins), 1))))
    sorted_wins = np.sort(wins)[::-1]
    total_gain = float(np.sum(gain))
    positive_gain = float(np.sum(wins))
    return {
        "scope": scope,
        "count": int(len(df)),
        "win_rate": float(np.mean(gain > 0)) if len(gain) else float("nan"),
        "harm_rate": float(np.mean(gain < 0)) if len(gain) else float("nan"),
        "mean_win": float(np.mean(wins)) if len(wins) else 0.0,
        "median_win": float(np.median(wins)) if len(wins) else 0.0,
        "mean_loss": float(np.mean(losses)) if len(losses) else 0.0,
        "median_loss": float(np.median(losses)) if len(losses) else 0.0,
        "p95_loss": float(np.quantile(losses, 0.95)) if len(losses) else 0.0,
        "top1_gain_share": float(np.sum(sorted_wins[:top1_n]) / (positive_gain + EPS)) if positive_gain > 0 else 0.0,
        "top5_gain_share": float(np.sum(sorted_wins[:top_n]) / (positive_gain + EPS)) if positive_gain > 0 else 0.0,
        "gini_positive_gain": gini_positive(wins),
        "total_gain": total_gain,
    }


def win_loss_distribution(test: pd.DataFrame) -> pd.DataFrame:
    scopes = [
        ("overall", np.ones(len(test), dtype=bool)),
        ("low_gap", test["low_gap_flag"]),
        ("high_gap", test["pred_gap_flag"]),
        ("high_repair", test["high_repair_flag"]),
        ("low_gap_high_repair", test["low_gap_flag"] & test["high_repair_flag"]),
        ("high_gap_low_repair", test["pred_gap_flag"] & test["low_repair_flag"]),
    ]
    return pd.DataFrame([win_loss_stats(test.loc[np.asarray(mask, dtype=bool)], name) for name, mask in scopes])


def horizon_segment_attribution(segment_records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment in ["early", "mid", "late"]:
        g = segment_records[segment_records["segment"] == segment]
        wins = -g.loc[g["delta_vs_lrbn"] < 0, "delta_vs_lrbn"]
        losses = g.loc[g["delta_vs_lrbn"] > 0, "delta_vs_lrbn"]
        rows.append(
            {
                "segment": segment,
                "count": int(len(g)),
                "mse_delta_pct_vs_lrbn": safe_pct(float(g["method_mse"].mean()), float(g["lrbn_mse"].mean())) if len(g) else float("nan"),
                "harm_rate": float(g["harm"].mean()) if len(g) else float("nan"),
                "mean_A": float(g["A"].mean()) if len(g) else float("nan"),
                "A_gt_1_rate": float((g["A"] > 1.0).mean()) if len(g) else float("nan"),
                "win_loss_ratio": float(wins.mean() / (losses.mean() + EPS)) if len(wins) and len(losses) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def anchor_tables(test_batch: ForecastBatch, thresholds: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    anchor_map = {
        "last": "last",
        "trend": "last_plus_slope",
        "robust": "hybrid",
    }
    parent_mse = mse_per_sample(test_batch.lrbn_pred, test_batch.y_true)
    hs = horizons(test_batch)
    residual = test_batch.y_true - test_batch.lrbn_pred
    for label, mode in anchor_map.items():
        result = apply_bp_always(test_batch, alpha=0.5, anchor_mode=mode)
        method_mse = mse_per_sample(result.pred, test_batch.y_true)
        a_vals, _ = alignment_scores(result.pred - test_batch.lrbn_pred, residual, hs)
        oracle_anchor_error = np.nanmean(np.abs(test_batch.y_true[:, :1, :] - result.info["anchor"]) / result.info["scale"], axis=(1, 2))
        pred_anchor_gap = np.nanmean(np.abs(test_batch.lrbn_pred[:, :1, :] - result.info["anchor"]) / result.info["scale"], axis=(1, 2))
        rows.append(
            {
                "anchor": label,
                "mean_oracle_anchor_error": float(np.mean(oracle_anchor_error)),
                "mean_pred_anchor_gap": float(np.mean(pred_anchor_gap)),
                "mean_delta_vs_lrbn": float(np.mean(method_mse - parent_mse)),
                "delta_pct_vs_lrbn": safe_pct(float(method_mse.mean()), float(parent_mse.mean())),
                "harm_rate": float(np.mean(method_mse > parent_mse + 1e-12)),
                "mean_A": float(np.mean(a_vals)),
                "A_gt_1_rate": float(np.mean(a_vals > 1.0)),
            }
        )

    main = apply_bp_always(test_batch, alpha=0.5, anchor_mode="last")
    stage3 = apply_stage3(test_batch, {"alpha": 0.0, "tau": float("inf")})
    repair = apply_candidate(test_batch, {"method": "LRBN-BP-repair-gate", "alpha": 0.0, "high_repair": 0.3})
    sample = make_sample_table(test_batch, "test", main, stage3, repair, thresholds)
    rows2 = []
    for label in ["q1_low", "q2", "q3", "q4_high"]:
        g = sample[sample["anchor_disagreement_bin"] == label]
        wins = -g.loc[g["delta_bp_vs_lrbn"] < 0, "delta_bp_vs_lrbn"]
        losses = g.loc[g["delta_bp_vs_lrbn"] > 0, "delta_bp_vs_lrbn"]
        rows2.append(
            {
                "disagreement_bin": label,
                "count": int(len(g)),
                "mean_delta_vs_lrbn": float(g["delta_bp_vs_lrbn"].mean()) if len(g) else float("nan"),
                "harm_rate": float(g["harm_bp_vs_lrbn"].mean()) if len(g) else float("nan"),
                "win_loss_ratio": float(wins.mean() / (losses.mean() + EPS)) if len(wins) and len(losses) else 0.0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(rows2)


def method_sample_metrics(test: pd.DataFrame, method_key: str) -> pd.DataFrame:
    d = test.copy()
    d["method"] = method_key
    d["delta"] = d[f"delta_{method_key}_vs_lrbn"]
    d["harm"] = d[f"harm_{method_key}_vs_lrbn"]
    d["A"] = d[f"A_{method_key}"]
    return d


def per_config_attribution(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["bp", "stage3", "repair_gate"]:
        d = method_sample_metrics(test, method)
        for keys, g in d.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
            dataset, backbone, horizon, seed = keys
            q4 = g[g["g_L_bin"] == "q4_high"]
            low_gap = g[g["low_gap_flag"]]
            high_repair = g[g["high_repair_flag"]]
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "method": method,
                    "count": int(len(g)),
                    "mse_delta_pct_vs_lrbn": safe_pct(float(g[f"mse_{method}"].mean()), float(g["mse_lrbn"].mean())),
                    "harm_rate": float(g["harm"].mean()),
                    "q4_gain": safe_pct(float(q4[f"mse_{method}"].mean()), float(q4["mse_lrbn"].mean())) * -1.0 if len(q4) else float("nan"),
                    "low_gap_harm": float(low_gap["harm"].mean()) if len(low_gap) else float("nan"),
                    "high_repair_harm": float(high_repair["harm"].mean()) if len(high_repair) else float("nan"),
                    "mean_A": float(g["A"].mean()),
                    "A_gt_1_rate": float((g["A"] > 1.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def failure_cases_topk(test: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    group_cols = ["dataset", "backbone", "horizon", "seed"]
    for _, g in test.groupby(group_cols, observed=True):
        candidates = [
            ("top_harm", g.nlargest(20, "delta_bp_vs_lrbn")),
            ("top_win", g.nsmallest(20, "delta_bp_vs_lrbn")),
            ("low_gap_high_repair_harm", g[g["low_gap_flag"] & g["high_repair_flag"]].nlargest(20, "delta_bp_vs_lrbn")),
            ("high_gap_low_repair_win", g[g["pred_gap_flag"] & g["low_repair_flag"]].nsmallest(20, "delta_bp_vs_lrbn")),
        ]
        for case_type, part in candidates:
            if part.empty:
                continue
            keep = part[
                [
                    "sample_id",
                    "dataset",
                    "backbone",
                    "horizon",
                    "seed",
                    "split",
                    "delta_bp_vs_lrbn",
                    "g_L",
                    "g_y",
                    "repair_ratio",
                    "A_bp",
                    "conflict_cosine",
                ]
            ].copy()
            keep["case_type"] = case_type
            keep["segment_harm_pattern"] = ""
            keep["save_path"] = ""
            rows.append(keep)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bootstrap_ci(test: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    methods = ["bp", "stage3", "repair_gate"]
    out: Dict[str, object] = {"n_bootstrap": int(n_boot), "seed": int(seed)}
    n = len(test)
    for method in methods:
        delta = test[f"delta_{method}_vs_lrbn"].to_numpy(float)
        harm = test[f"harm_{method}_vs_lrbn"].to_numpy(float)
        boot_delta = np.empty(n_boot, dtype=float)
        boot_harm = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_delta[b] = np.mean(delta[idx])
            boot_harm[b] = np.mean(harm[idx])
        out[method] = {
            "mean_delta": float(np.mean(delta)),
            "delta_ci95_low": float(np.quantile(boot_delta, 0.025)),
            "delta_ci95_high": float(np.quantile(boot_delta, 0.975)),
            "p_improve": float(np.mean(boot_delta < 0.0)),
            "harm_rate": float(np.mean(harm)),
            "harm_ci95_low": float(np.quantile(boot_harm, 0.025)),
            "harm_ci95_high": float(np.quantile(boot_harm, 0.975)),
        }
    for label, mask in [
        ("low_gap_high_repair", test["low_gap_flag"] & test["high_repair_flag"]),
        ("high_gap_low_repair", test["pred_gap_flag"] & test["low_repair_flag"]),
    ]:
        g = test.loc[np.asarray(mask, dtype=bool)]
        if g.empty:
            continue
        delta = g["delta_bp_vs_lrbn"].to_numpy(float)
        vals = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, len(delta), size=len(delta))
            vals[b] = np.mean(delta[idx])
        out[f"slice_{label}"] = {
            "count": int(len(g)),
            "mean_delta": float(delta.mean()),
            "delta_ci95_low": float(np.quantile(vals, 0.025)),
            "delta_ci95_high": float(np.quantile(vals, 0.975)),
        }
    return out


def write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def try_write_parquet(df: pd.DataFrame, path: Path) -> str:
    try:
        df.to_parquet(path, index=False)
        return str(path)
    except Exception as exc:  # pragma: no cover - optional dependency path
        return f"not_written: {type(exc).__name__}: {exc}"


def build_verdict(
    test: pd.DataFrame,
    oracle: pd.DataFrame,
    interaction: pd.DataFrame,
    wins: pd.DataFrame,
    segments: pd.DataFrame,
    anchors: pd.DataFrame,
) -> Dict[str, object]:
    overall_harm = float(test["harm_bp_vs_lrbn"].mean())
    low_gap = test[test["low_gap_flag"]]
    high_repair = test[test["high_repair_flag"]]
    wins_df = test[test["delta_bp_vs_lrbn"] < 0]
    harms_df = test[test["delta_bp_vs_lrbn"] > 0]

    inter_nonempty = interaction[interaction["count"] > 0]
    lghr = interaction[(interaction["gap_group"] == "low") & (interaction["repair_group"] == "high")]
    hglr = interaction[(interaction["gap_group"] == "high") & (interaction["repair_group"] == "low")]
    lghr_harm = float(lghr["harm_rate"].iloc[0]) if len(lghr) else float("nan")
    hglr_delta = float(hglr["mean_delta_vs_lrbn"].iloc[0]) if len(hglr) else float("nan")
    max_inter_harm = float(inter_nonempty["harm_rate"].max()) if len(inter_nonempty) else float("nan")

    early = segments[segments["segment"] == "early"].iloc[0]
    mid = segments[segments["segment"] == "mid"].iloc[0]
    late = segments[segments["segment"] == "late"].iloc[0]
    wl_overall = wins[wins["scope"] == "overall"].iloc[0]
    non_last = anchors[anchors["anchor"] != "last"]
    best_alt_harm = float(non_last["harm_rate"].min()) if len(non_last) else float("nan")

    defect_conditions = {
        "low_gap_harm_above_overall": bool(len(low_gap) and float(low_gap["harm_bp_vs_lrbn"].mean()) > overall_harm),
        "high_repair_harm_above_overall": bool(len(high_repair) and float(high_repair["harm_bp_vs_lrbn"].mean()) > overall_harm),
        "low_gap_high_repair_highest_harm": bool(np.isfinite(lghr_harm) and lghr_harm >= max_inter_harm - 1e-12),
        "harmful_slices_lower_A_gt_1": bool(
            len(harms_df) and len(wins_df) and float((harms_df["A_bp"] > 1.0).mean()) < float((wins_df["A_bp"] > 1.0).mean())
        ),
        "early_gain_stronger_than_mid_late": bool(
            float(early["mse_delta_pct_vs_lrbn"]) < float(mid["mse_delta_pct_vs_lrbn"])
            and float(early["mse_delta_pct_vs_lrbn"]) < float(late["mse_delta_pct_vs_lrbn"])
        ),
        "top_gains_concentrated": bool(float(wl_overall["top5_gain_share"]) >= 0.50 or float(wl_overall["gini_positive_gain"]) >= 0.50),
        "anchor_swap_does_not_solve_harm": bool(np.isfinite(best_alt_harm) and best_alt_harm > 0.25),
    }
    defect_score = sum(defect_conditions.values())

    selected_stage3 = test[test["stage3_selected"]]
    selected_repair = test[test["repair_selected"]]
    q4_stage3 = per_config_attribution(test)
    stage3_rows = q4_stage3[q4_stage3["method"] == "stage3"]
    sparse_conditions = {
        "high_gap_low_repair_gain": bool(np.isfinite(hglr_delta) and hglr_delta < 0.0),
        "low_gap_high_repair_high_harm": bool(np.isfinite(lghr_harm) and lghr_harm > overall_harm),
        "selected_samples_higher_A": bool(
            (len(selected_stage3) and float(selected_stage3["A_bp"].mean()) > float(test["A_bp"].mean()))
            or (len(selected_repair) and float(selected_repair["A_bp"].mean()) > float(test["A_bp"].mean()))
        ),
        "stage3_low_harm_q4_gain_supported": bool(
            float(test["harm_stage3_vs_lrbn"].mean()) <= 0.05
            and len(stage3_rows)
            and float(stage3_rows["q4_gain"].mean()) > 0.0
        ),
    }
    sparse_score = sum(sparse_conditions.values())
    return {
        "test_threshold_leakage": False,
        "bp_always_mechanism_defect_supported": "yes" if defect_score >= 5 else ("partial" if defect_score >= 3 else "no"),
        "bp_always_defect_score": int(defect_score),
        "bp_always_defect_conditions": defect_conditions,
        "sparse_repair_aware_bp_supported": "yes" if sparse_score >= 3 else ("partial" if sparse_score >= 2 else "no"),
        "sparse_repair_aware_score": int(sparse_score),
        "sparse_repair_aware_conditions": sparse_conditions,
        "overall_bp_delta_vs_lrbn": float(test["delta_bp_vs_lrbn"].mean()),
        "overall_bp_delta_pct_vs_lrbn": safe_pct(float(test["mse_bp"].mean()), float(test["mse_lrbn"].mean())),
        "overall_bp_harm_rate": overall_harm,
        "stage3_delta_pct_vs_lrbn": safe_pct(float(test["mse_stage3"].mean()), float(test["mse_lrbn"].mean())),
        "stage3_harm_rate": float(test["harm_stage3_vs_lrbn"].mean()),
        "repair_gate_delta_pct_vs_lrbn": safe_pct(float(test["mse_repair_gate"].mean()), float(test["mse_lrbn"].mean())),
        "repair_gate_harm_rate": float(test["harm_repair_gate_vs_lrbn"].mean()),
        "recommendation": (
            "Use LRBN + sparse repair-aware boundary expert as the next BP line; keep BP-always as performance attribution only."
        ),
    }


def df_to_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    show = df.head(max_rows)
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in show.iterrows():
        vals = []
        for col in cols:
            v = r[col]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(out: Path, config: Dict[str, object], verdict: Dict[str, object]) -> None:
    oracle = pd.read_csv(out / "oracle_boundary_truth.csv")
    interaction = pd.read_csv(out / "gap_repair_interaction.csv")
    wins = pd.read_csv(out / "win_loss_distribution.csv")
    segments = pd.read_csv(out / "horizon_segment_attribution.csv")
    anchors = pd.read_csv(out / "anchor_reliability.csv")
    per_config = pd.read_csv(out / "per_config_attribution.csv")
    lines = [
        "# Stage 4.5 BP-Always Failure Attribution",
        "",
        "## Setup",
        "",
        f"- Input metrics: `{config['metrics_csv']}`",
        f"- Output directory: `{config['output_dir']}`",
        f"- Compact configs: `{config['n_test_configs']}` test configs",
        f"- Validation samples: `{config['n_val_samples']}`",
        f"- Test samples: `{config['n_test_samples']}`",
        f"- Test threshold leakage: `{verdict['test_threshold_leakage']}`",
        "",
        "## Headline",
        "",
        f"- BP-always MSE delta vs LRBN: `{verdict['overall_bp_delta_pct_vs_lrbn']:.6f}%`; harm rate `{verdict['overall_bp_harm_rate']:.6f}`.",
        f"- Stage3 gated MSE delta vs LRBN: `{verdict['stage3_delta_pct_vs_lrbn']:.6f}%`; harm rate `{verdict['stage3_harm_rate']:.6f}`.",
        f"- Repair-gate MSE delta vs LRBN: `{verdict['repair_gate_delta_pct_vs_lrbn']:.6f}%`; harm rate `{verdict['repair_gate_harm_rate']:.6f}`.",
        "",
        "## Verdict",
        "",
        f"- BP-always mechanism defect: `{verdict['bp_always_mechanism_defect_supported']}` ({verdict['bp_always_defect_score']}/7 conditions).",
        f"- Sparse repair-aware BP support: `{verdict['sparse_repair_aware_bp_supported']}` ({verdict['sparse_repair_aware_score']}/4 conditions).",
        f"- Recommendation: {verdict['recommendation']}",
        "",
        "## Oracle Boundary Truth",
        "",
        df_to_md(oracle),
        "",
        "## Gap x Repair Interaction",
        "",
        df_to_md(interaction),
        "",
        "## Win/Loss Distribution",
        "",
        df_to_md(wins),
        "",
        "## Horizon Segment Attribution",
        "",
        df_to_md(segments),
        "",
        "## Anchor Reliability",
        "",
        df_to_md(anchors),
        "",
        "## Per-Config Attribution (first rows)",
        "",
        df_to_md(per_config.head(12)),
        "",
        "## Output Files",
        "",
        "- `attribution_config.json`",
        "- `attribution_sample_table.csv` and optional `attribution_sample_table.parquet`",
        "- `oracle_boundary_truth.csv`",
        "- `residual_alignment_by_slice.csv`",
        "- `gap_repair_interaction.csv`",
        "- `win_loss_distribution.csv`",
        "- `horizon_segment_attribution.csv`",
        "- `anchor_reliability.csv`",
        "- `anchor_disagreement_slices.csv`",
        "- `per_config_attribution.csv`",
        "- `bootstrap_ci.json`",
        "- `failure_cases_topk.csv`",
        "- `verdict.json`",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--stage3-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_stage3"))
    parser.add_argument("--stage4-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_stage4"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_attribution_stage45"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures" / "failure_cases").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures" / "slice_plots").mkdir(parents=True, exist_ok=True)

    batch = load_forecast_batch_from_metrics(args.metrics_csv)
    split = batch.meta["split"].to_numpy()
    val = batch.subset(split == "val")
    test = batch.subset(split == "test")
    stage3_params = load_stage3_params(args.stage3_dir)
    repair_params = load_repair_gate_params(args.stage4_dir)

    val_bp = apply_bp_always(val, alpha=0.5, anchor_mode="last")
    val_stage3 = apply_stage3(val, stage3_params)
    val_repair = apply_candidate(val, repair_params)
    val_table_uncalibrated = make_sample_table(val, "val", val_bp, val_stage3, val_repair, thresholds=None)
    thresholds = calibrate_thresholds(val_table_uncalibrated)
    val_table = add_calibrated_labels(val_table_uncalibrated, thresholds)

    test_bp = apply_bp_always(test, alpha=0.5, anchor_mode="last")
    test_stage3 = apply_stage3(test, stage3_params)
    test_repair = apply_candidate(test, repair_params)
    test_table = make_sample_table(test, "test", test_bp, test_stage3, test_repair, thresholds=thresholds)

    sample_table = pd.concat([val_table, test_table], ignore_index=True)
    sample_csv = args.output_dir / "attribution_sample_table.csv"
    sample_table.to_csv(sample_csv, index=False)
    parquet_status = try_write_parquet(sample_table, args.output_dir / "attribution_sample_table.parquet")

    seg_records = segment_alignment_records(test, test_bp.pred, "bp")
    oracle = oracle_boundary_truth(test_table)
    residual = residual_alignment_by_slice(test_table, seg_records)
    interaction = gap_repair_interaction(test_table)
    wins = win_loss_distribution(test_table)
    segments = horizon_segment_attribution(seg_records)
    anchors, anchor_slices = anchor_tables(test, thresholds)
    per_config = per_config_attribution(test_table)
    failures = failure_cases_topk(test_table)
    ci = bootstrap_ci(test_table, args.n_bootstrap, args.seed)
    verdict = build_verdict(test_table, oracle, interaction, wins, segments, anchors)

    write_table(oracle, args.output_dir / "oracle_boundary_truth.csv")
    write_table(residual, args.output_dir / "residual_alignment_by_slice.csv")
    write_table(interaction, args.output_dir / "gap_repair_interaction.csv")
    write_table(wins, args.output_dir / "win_loss_distribution.csv")
    write_table(segments, args.output_dir / "horizon_segment_attribution.csv")
    write_table(anchors, args.output_dir / "anchor_reliability.csv")
    write_table(anchor_slices, args.output_dir / "anchor_disagreement_slices.csv")
    write_table(per_config, args.output_dir / "per_config_attribution.csv")
    write_table(failures, args.output_dir / "failure_cases_topk.csv")
    (args.output_dir / "bootstrap_ci.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")
    (args.output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    config = {
        "metrics_csv": str(args.metrics_csv),
        "stage3_dir": str(args.stage3_dir),
        "stage4_dir": str(args.stage4_dir),
        "output_dir": str(args.output_dir),
        "scope": "compact_attribution",
        "datasets": sorted(test.meta["dataset"].unique().tolist()),
        "backbones": sorted(test.meta["backbone"].unique().tolist()),
        "horizons": sorted([int(x) for x in test.meta["horizon"].unique().tolist()]),
        "seeds": sorted([int(x) for x in test.meta["seed"].unique().tolist()]),
        "n_val_samples": int(len(val.meta)),
        "n_test_samples": int(len(test.meta)),
        "n_test_configs": int(test.meta.groupby(["dataset", "backbone", "horizon", "seed"]).ngroups),
        "thresholds_validation_only": thresholds,
        "stage3_params": stage3_params,
        "repair_gate_params": repair_params,
        "bp_always_params": {"alpha": 0.5, "anchor_mode": "last", "bridge_mode": "linear"},
        "parquet_status": parquet_status,
        "test_threshold_leakage": False,
    }
    (args.output_dir / "attribution_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_summary(args.output_dir, config, verdict)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "bp_defect": verdict["bp_always_mechanism_defect_supported"],
                "sparse_bp": verdict["sparse_repair_aware_bp_supported"],
                "bp_delta_pct": verdict["overall_bp_delta_pct_vs_lrbn"],
                "bp_harm": verdict["overall_bp_harm_rate"],
                "test_threshold_leakage": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
