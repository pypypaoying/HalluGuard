#!/usr/bin/env python
"""Run Stage 5 validation for Sparse Repair-Aware Boundary Projection."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from halluguard_sra_bp import (
    SRABPParams,
    apply_sra_bp,
    apply_sra_bp_from_features,
    compute_sra_features,
    residual_alignment,
)
from halluguard_stage4_bp_harm_control import CandidateResult, apply_candidate


DEFAULT_METRICS = Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv")


def horizons(batch: ForecastBatch) -> np.ndarray:
    return batch.meta["horizon"].to_numpy(int)


def split_batch(batch: ForecastBatch) -> Tuple[ForecastBatch, ForecastBatch]:
    split = batch.meta["split"].to_numpy(str)
    return batch.subset(split == "val"), batch.subset(split == "test")


def finite_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    return f


def safe_pct(method_mean: float, baseline_mean: float) -> float:
    return float((method_mean - baseline_mean) / (baseline_mean + EPS) * 100.0)


def load_stage3_params(stage3_dir: Path) -> Dict[str, Any]:
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


def load_stage4_params(stage4_dir: Path, method: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    path = stage4_dir / "stage4c_overall.csv"
    if not path.exists():
        return fallback
    df = pd.read_csv(path)
    rows = df[(df["method"] == method) & (df["split"] == "test")]
    if rows.empty:
        rows = df[df["method"] == method]
    if rows.empty:
        return fallback
    r = rows.iloc[0]
    out: Dict[str, Any] = {"method": method}
    for col in df.columns:
        if col.startswith("param_") and not pd.isna(r[col]):
            val = r[col]
            if isinstance(val, np.generic):
                val = val.item()
            out[col.replace("param_", "", 1)] = val
    for k, v in fallback.items():
        out.setdefault(k, v)
    return out


def apply_stage3(batch: ForecastBatch, params: Dict[str, Any]) -> CandidateResult:
    pred, selected, gap = lrbn_optional_bp(
        batch.context,
        batch.lrbn_pred,
        alpha=float(params.get("alpha", 0.5)),
        tau=float(params.get("tau", float("inf"))),
        tail=int(params.get("tail", 24)),
        decay=str(params.get("decay", "linear")),
        horizons=horizons(batch),
    )
    info = {
        "stage3_gap": gap,
        "effective_strength": selected.astype(float),
    }
    return CandidateResult("LRBN-BP-stage3-gated", pred, selected.astype(float), info, params)


def compute_thresholds(val: ForecastBatch) -> Dict[str, float]:
    feats = compute_sra_features(val.context, val.raw_pred, val.lrbn_pred, val.y_true)
    thresholds: Dict[str, float] = {}
    for key in ["g_l", "g_raw", "g_y", "repair_ratio", "jump_support"]:
        qs = pd.Series(feats[key]).quantile([0.25, 0.50, 0.75, 0.80, 0.90]).to_dict()
        for q, value in qs.items():
            thresholds[f"{key}_q{int(q * 100)}"] = float(value)
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


def gap_group(v: float, thresholds: Dict[str, float]) -> str:
    if v <= thresholds["g_l_q25"]:
        return "low"
    if v >= thresholds["g_l_q75"]:
        return "high"
    return "mid"


def repair_group(v: float, thresholds: Dict[str, float]) -> str:
    if v <= thresholds["repair_low"]:
        return "low"
    if v > thresholds["repair_high"]:
        return "high"
    return "mid"


def add_feature_labels(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["g_l_bin"] = out["g_l"].apply(
        lambda v: quantile_label(v, thresholds["g_l_q25"], thresholds["g_l_q50"], thresholds["g_l_q75"])
    )
    out["g_y_bin"] = out["g_y"].apply(
        lambda v: quantile_label(v, thresholds["g_y_q25"], thresholds["g_y_q50"], thresholds["g_y_q75"])
    )
    out["gap_group"] = out["g_l"].apply(lambda v: gap_group(v, thresholds))
    out["repair_group"] = out["repair_ratio"].apply(lambda v: repair_group(v, thresholds))
    out["pred_gap_flag"] = out["gap_group"].eq("high")
    out["low_gap_flag"] = out["gap_group"].eq("low")
    out["high_repair_flag"] = out["repair_group"].eq("high")
    out["low_repair_flag"] = out["repair_group"].eq("low")
    out["true_jump_flag"] = out["g_y"] >= thresholds["g_y_q75"]

    def oracle_bin(row: pd.Series) -> str:
        if row["low_gap_flag"] and not row["true_jump_flag"]:
            return "low_gL_low_gY"
        if row["pred_gap_flag"] and not row["true_jump_flag"]:
            return "high_gL_low_gY"
        if row["pred_gap_flag"] and row["true_jump_flag"]:
            return "high_gL_high_gY"
        if row["low_gap_flag"] and row["true_jump_flag"]:
            return "low_gL_high_gY"
        return "mid_or_mixed"

    out["oracle_boundary_bin"] = out.apply(oracle_bin, axis=1)
    return out


def base_feature_frame(batch: ForecastBatch, thresholds: Dict[str, float]) -> pd.DataFrame:
    feats = compute_sra_features(batch.context, batch.raw_pred, batch.lrbn_pred, batch.y_true)
    df = batch.meta.reset_index(drop=True).copy()
    df["g_raw"] = feats["g_raw"]
    df["g_l"] = feats["g_l"]
    df["g_y"] = feats["g_y"]
    df["repair_ratio"] = feats["repair_ratio"]
    df["jump_support"] = feats["jump_support"]
    df["trend_support"] = feats["trend_support"]
    df["vol_support"] = feats["vol_support"]
    df["smooth_support"] = feats["smooth_support"]
    return add_feature_labels(df, thresholds)


def slice_delta_pct(method_mse: np.ndarray, base_mse: np.ndarray, mask: np.ndarray, default: float = 0.0) -> float:
    idx = np.asarray(mask, dtype=bool)
    if not idx.any():
        return default
    return safe_pct(float(np.mean(method_mse[idx])), float(np.mean(base_mse[idx])))


def win_loss_ratio(delta: np.ndarray, mask: np.ndarray) -> float:
    idx = np.asarray(mask, dtype=bool)
    d = np.asarray(delta, dtype=float)[idx]
    wins = -d[d < 0]
    losses = d[d > 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(np.mean(wins) / (np.mean(losses) + EPS))


def config_improved_ratio(meta: pd.DataFrame, delta: np.ndarray) -> float:
    df = meta[["dataset", "backbone", "horizon", "seed"]].copy()
    df["delta"] = delta
    vals = df.groupby(["dataset", "backbone", "horizon", "seed"], observed=True)["delta"].mean()
    return float((vals <= 0.0).mean()) if len(vals) else float("nan")


def evaluate_arrays_with_features(
    batch: ForecastBatch,
    pred: np.ndarray,
    method: str,
    strength: Optional[np.ndarray],
    thresholds: Dict[str, float],
    split_name: str,
    feat: pd.DataFrame,
) -> Dict[str, Any]:
    base = mse_per_sample(batch.lrbn_pred, batch.y_true)
    mm = mse_per_sample(pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    ma = mae_per_sample(pred, batch.y_true)
    delta = mm - base
    q4 = feat["g_l_bin"].eq("q4_high").to_numpy()
    low_high = (feat["low_gap_flag"] & feat["high_repair_flag"]).to_numpy()
    high_low = (feat["pred_gap_flag"] & feat["low_repair_flag"]).to_numpy()
    selected = np.zeros(len(batch.meta), dtype=bool) if strength is None else np.asarray(strength, dtype=float) > 1e-8
    return {
        "split": split_name,
        "method": method,
        "n": int(len(batch.meta)),
        "mse": float(np.mean(mm)),
        "mae": float(np.mean(ma)),
        "mse_lrbn": float(np.mean(base)),
        "mae_lrbn": float(np.mean(base_mae)),
        "mean_delta": float(np.mean(delta)),
        "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(mm)), float(np.mean(base))),
        "mae_delta_pct_vs_lrbn": safe_pct(float(np.mean(ma)), float(np.mean(base_mae))),
        "harm_rate": float(np.mean(delta > 1e-12)),
        "win_rate": float(np.mean(delta < 0.0)),
        "coverage": float(np.mean(selected)),
        "selected_count": int(np.sum(selected)),
        "mean_strength": float(np.mean(strength)) if strength is not None else 0.0,
        "q4_improvement_pct": -slice_delta_pct(mm, base, q4, default=float("nan")),
        "low_gap_high_repair_delta_pct": slice_delta_pct(mm, base, low_high, default=0.0),
        "high_gap_low_repair_delta_pct": slice_delta_pct(mm, base, high_low, default=float("nan")),
        "low_gap_high_repair_coverage": float(np.mean(selected[low_high])) if low_high.any() else 0.0,
        "high_gap_low_repair_coverage": float(np.mean(selected[high_low])) if high_low.any() else 0.0,
        "config_improved_ratio": config_improved_ratio(batch.meta, delta),
        "test_threshold_leakage": False,
    }


def evaluate_arrays(
    batch: ForecastBatch,
    pred: np.ndarray,
    method: str,
    strength: Optional[np.ndarray],
    thresholds: Dict[str, float],
    split_name: str,
) -> Dict[str, Any]:
    feat = base_feature_frame(batch, thresholds)
    return evaluate_arrays_with_features(batch, pred, method, strength, thresholds, split_name, feat)


def param_grid(val: ForecastBatch, thresholds: Dict[str, float], val_g_l: Optional[np.ndarray] = None) -> List[Tuple[str, float, SRABPParams]]:
    if val_g_l is None:
        val_g_l = compute_sra_features(val.context, val.raw_pred, val.lrbn_pred)["g_l"]
    tau_g_items = [
        (0.50, thresholds["g_l_q50"]),
        (0.60, float(np.quantile(val_g_l, 0.60))),
        (0.70, float(np.quantile(val_g_l, 0.70))),
        (0.80, thresholds["g_l_q80"]),
        (0.90, thresholds["g_l_q90"]),
    ]
    tau_r_values = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
    tau_j_values = [0.3, 0.5, 0.7]
    alpha_values = [0.10, 0.25, 0.50, 0.75]
    short_k = [4, 8, 16, 24, "H_div_4"]
    rows: List[Tuple[str, float, SRABPParams]] = []
    for q, tau_g in tau_g_items:
        for tau_r in tau_r_values:
            for alpha in alpha_values:
                rows.append(
                    (
                        "basic",
                        q,
                        SRABPParams(method_family="basic", tau_g=tau_g, tau_r=tau_r, tau_j=None, alpha=alpha, K="H"),
                    )
                )
                for k in short_k:
                    rows.append(
                        (
                            "short",
                            q,
                            SRABPParams(method_family="short", tau_g=tau_g, tau_r=tau_r, tau_j=None, alpha=alpha, K=k),
                        )
                    )
                    for tau_j in tau_j_values:
                        rows.append(
                            (
                                "support",
                                q,
                                SRABPParams(
                                    method_family="support",
                                    tau_g=tau_g,
                                    tau_r=tau_r,
                                    tau_j=tau_j,
                                    alpha=alpha,
                                    K=k,
                                ),
                            )
                        )
                    for tau_j in [0.3, 0.5, 0.7, None]:
                        rows.append(
                            (
                                "continuous",
                                q,
                                SRABPParams(
                                    method_family="continuous",
                                    tau_g=tau_g,
                                    tau_r=tau_r,
                                    tau_j=tau_j,
                                    alpha=alpha,
                                    K=k,
                                    continuous=True,
                                ),
                            )
                        )
    return rows


def safe_feasible(row: pd.Series) -> bool:
    return bool(
        row["mse_delta_pct_vs_lrbn"] <= -0.5
        and row["harm_rate"] <= 0.05
        and row["q4_improvement_pct"] >= 2.0
        and row["low_gap_high_repair_delta_pct"] <= 0.5
        and row["config_improved_ratio"] >= 0.75
    )


def balanced_feasible(row: pd.Series) -> bool:
    return bool(
        row["mse_delta_pct_vs_lrbn"] <= -1.5
        and row["harm_rate"] <= 0.15
        and row["q4_improvement_pct"] >= 4.0
        and row["config_improved_ratio"] >= 0.75
    )


def calibrate_grid(val: ForecastBatch, thresholds: Dict[str, float]) -> pd.DataFrame:
    rows: List[dict] = []
    feature_cache = compute_sra_features(val.context, val.raw_pred, val.lrbn_pred)
    items = param_grid(val, thresholds, feature_cache["g_l"])
    feat = base_feature_frame(val, thresholds)
    for i, (family, q_g, params) in enumerate(items):
        pred, aux = apply_sra_bp_from_features(val.lrbn_pred, horizons(val), feature_cache, params)
        row = evaluate_arrays_with_features(
            val,
            pred,
            f"LRBN-SRA-BP-{family}",
            aux["strength"],
            thresholds,
            "val",
            feat,
        )
        row.update(params.to_dict())
        row["q_g"] = float(q_g)
        row["candidate_index"] = int(i)
        row["safe_feasible"] = safe_feasible(pd.Series(row))
        row["balanced_feasible"] = balanced_feasible(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def select_params(grid: pd.DataFrame, objective: str) -> Optional[Dict[str, Any]]:
    col = "safe_feasible" if objective == "safe" else "balanced_feasible"
    feasible = grid[grid[col].astype(bool)].copy()
    if feasible.empty:
        return None
    deployable = feasible[~feasible["method_family"].eq("basic")].copy()
    if not deployable.empty:
        feasible = deployable
    feasible = feasible.sort_values(
        ["mean_delta", "harm_rate", "low_gap_high_repair_delta_pct", "candidate_index"],
        ascending=[True, True, True, True],
    )
    row = feasible.iloc[0].to_dict()
    return row_to_params(row)


def select_family_params(grid: pd.DataFrame, family: str) -> Optional[Dict[str, Any]]:
    g = grid[grid["method_family"].eq(family)].copy()
    if g.empty:
        return None
    safer = g[g["harm_rate"] <= 0.15].copy()
    if safer.empty:
        safer = g.copy()
    safer = safer.sort_values(["mean_delta", "harm_rate", "candidate_index"], ascending=[True, True, True])
    return row_to_params(safer.iloc[0].to_dict())


def row_to_params(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["method_family", "anchor_mode", "tail_len", "tau_g", "tau_r", "tau_j", "alpha", "K", "continuous", "kg", "kr", "kj"]
    out = {}
    for key in keys:
        value = row.get(key)
        if key == "tau_j":
            out[key] = finite_float(value)
        elif key in {"tail_len"}:
            out[key] = int(value)
        elif key in {"tau_g", "tau_r", "alpha", "kg", "kr", "kj"}:
            out[key] = float(value)
        elif key == "continuous":
            out[key] = bool(value)
        else:
            out[key] = value
    return out


def method_sample_table(
    batch: ForecastBatch,
    method: str,
    pred: np.ndarray,
    strength: Optional[np.ndarray],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    feat = base_feature_frame(batch, thresholds)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    pred_mse = mse_per_sample(pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    pred_mae = mae_per_sample(pred, batch.y_true)
    delta = pred - batch.lrbn_pred
    align = residual_alignment(delta, batch.lrbn_pred, batch.y_true, horizons(batch))
    feat["method"] = method
    feat["mse_lrbn"] = base_mse
    feat["mse_method"] = pred_mse
    feat["mae_lrbn"] = base_mae
    feat["mae_method"] = pred_mae
    feat["delta_mse_vs_lrbn"] = pred_mse - base_mse
    feat["harm"] = feat["delta_mse_vs_lrbn"] > 1e-12
    feat["win"] = feat["delta_mse_vs_lrbn"] < 0.0
    if strength is None:
        feat["strength"] = 0.0
        feat["selected"] = False
    else:
        s = np.asarray(strength, dtype=float)
        feat["strength"] = s
        feat["selected"] = s > 1e-8
    feat["A"] = align["A"]
    feat["A_gt_1"] = align["A_gt_1"]
    feat["alignment_cosine"] = align["cosine"]
    return feat


def summarize_slice(df: pd.DataFrame, mask: np.ndarray, slice_type: str, slice_name: str) -> Dict[str, Any]:
    g = df.loc[np.asarray(mask, dtype=bool)]
    if g.empty:
        return {
            "method": str(df["method"].iloc[0]) if len(df) else "",
            "slice_type": slice_type,
            "slice_name": slice_name,
            "count": 0,
            "coverage": 0.0,
            "mse_delta_pct_vs_lrbn": float("nan"),
            "mean_delta": float("nan"),
            "harm_rate": float("nan"),
            "win_rate": float("nan"),
            "win_loss_ratio": 0.0,
        }
    delta = g["delta_mse_vs_lrbn"].to_numpy(float)
    return {
        "method": str(g["method"].iloc[0]),
        "slice_type": slice_type,
        "slice_name": slice_name,
        "count": int(len(g)),
        "coverage": float(g["selected"].mean()),
        "mse_delta_pct_vs_lrbn": safe_pct(float(g["mse_method"].mean()), float(g["mse_lrbn"].mean())),
        "mean_delta": float(delta.mean()),
        "harm_rate": float(g["harm"].mean()),
        "win_rate": float(g["win"].mean()),
        "win_loss_ratio": win_loss_ratio(delta, np.ones(len(delta), dtype=bool)),
        "A_gt_1_rate": float(g["A_gt_1"].mean()),
        "mean_A": float(g["A"].mean()),
    }


def boundary_gap_slices(tables: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in tables:
        for label in ["q1_low", "q2", "q3", "q4_high"]:
            rows.append(summarize_slice(df, df["g_l_bin"].eq(label).to_numpy(), "g_l_bin", label))
    return pd.DataFrame(rows)


def gap_repair_slices(tables: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in tables:
        for gap in ["low", "mid", "high"]:
            for repair in ["low", "mid", "high"]:
                mask = df["gap_group"].eq(gap) & df["repair_group"].eq(repair)
                rows.append(summarize_slice(df, mask.to_numpy(), "gap_repair", f"{gap}_gap__{repair}_repair"))
    return pd.DataFrame(rows)


def oracle_truth_slices(tables: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    labels = ["low_gL_low_gY", "high_gL_low_gY", "high_gL_high_gY", "low_gL_high_gY", "mid_or_mixed"]
    for df in tables:
        for label in labels:
            rows.append(summarize_slice(df, df["oracle_boundary_bin"].eq(label).to_numpy(), "oracle_boundary", label))
    return pd.DataFrame(rows)


def segment_rows(batch: ForecastBatch, method: str, pred: np.ndarray, strength: Optional[np.ndarray]) -> pd.DataFrame:
    rows = []
    selected = np.zeros(len(batch.meta), dtype=bool) if strength is None else np.asarray(strength, dtype=float) > 1e-8
    for i, h in enumerate(horizons(batch)):
        cuts = {
            "early": (0, max(1, int(h) // 4)),
            "mid": (max(1, int(h) // 4), max(max(1, int(h) // 4) + 1, 3 * int(h) // 4)),
            "late": (max(max(1, int(h) // 4) + 1, 3 * int(h) // 4), int(h)),
        }
        meta = batch.meta.iloc[i].to_dict()
        for seg, (a, b) in cuts.items():
            base_mse = float(np.mean((batch.lrbn_pred[i, a:b, :] - batch.y_true[i, a:b, :]) ** 2))
            pred_mse = float(np.mean((pred[i, a:b, :] - batch.y_true[i, a:b, :]) ** 2))
            rows.append(
                {
                    **meta,
                    "method": method,
                    "segment": seg,
                    "selected": bool(selected[i]),
                    "mse_lrbn": base_mse,
                    "mse_method": pred_mse,
                    "delta_mse_vs_lrbn": pred_mse - base_mse,
                    "harm": pred_mse > base_mse + 1e-12,
                }
            )
    return pd.DataFrame(rows)


def horizon_segments(batch: ForecastBatch, method_preds: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]) -> pd.DataFrame:
    raw = []
    for method, (pred, strength) in method_preds.items():
        raw.append(segment_rows(batch, method, pred, strength))
    d = pd.concat(raw, ignore_index=True)
    rows = []
    for (method, segment), g in d.groupby(["method", "segment"], observed=True):
        delta = g["delta_mse_vs_lrbn"].to_numpy(float)
        rows.append(
            {
                "method": method,
                "segment": segment,
                "count": int(len(g)),
                "coverage": float(g["selected"].mean()),
                "mse_delta_pct_vs_lrbn": safe_pct(float(g["mse_method"].mean()), float(g["mse_lrbn"].mean())),
                "mean_delta": float(delta.mean()),
                "harm_rate": float(g["harm"].mean()),
                "win_loss_ratio": win_loss_ratio(delta, np.ones(len(delta), dtype=bool)),
            }
        )
    return pd.DataFrame(rows)


def selected_alignment(tables: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in tables:
        for label, mask in [
            ("selected", df["selected"].to_numpy()),
            ("unselected", ~df["selected"].to_numpy()),
            ("overall", np.ones(len(df), dtype=bool)),
        ]:
            g = df.loc[np.asarray(mask, dtype=bool)]
            if g.empty:
                continue
            wins = -g.loc[g["delta_mse_vs_lrbn"] < 0, "delta_mse_vs_lrbn"]
            losses = g.loc[g["delta_mse_vs_lrbn"] > 0, "delta_mse_vs_lrbn"]
            rows.append(
                {
                    "method": str(g["method"].iloc[0]),
                    "selection_slice": label,
                    "count": int(len(g)),
                    "A_gt_1_rate": float(g["A_gt_1"].mean()),
                    "mean_A": float(g["A"].mean()),
                    "mean_alignment_cosine": float(g["alignment_cosine"].mean()),
                    "harm_rate": float(g["harm"].mean()),
                    "mean_win_size": float(wins.mean()) if len(wins) else 0.0,
                    "mean_loss_size": float(losses.mean()) if len(losses) else 0.0,
                    "true_jump_rate": float(g["true_jump_flag"].mean()),
                    "high_gap_low_repair_rate": float((g["pred_gap_flag"] & g["low_repair_flag"]).mean()),
                    "low_gap_high_repair_rate": float((g["low_gap_flag"] & g["high_repair_flag"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def per_config_table(tables: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in tables:
        for keys, g in df.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
            dataset, backbone, horizon, seed = keys
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "method": str(g["method"].iloc[0]),
                    "count": int(len(g)),
                    "coverage": float(g["selected"].mean()),
                    "mse_delta_pct_vs_lrbn": safe_pct(float(g["mse_method"].mean()), float(g["mse_lrbn"].mean())),
                    "harm_rate": float(g["harm"].mean()),
                    "q4_improvement_pct": -slice_delta_pct(
                        g["mse_method"].to_numpy(float),
                        g["mse_lrbn"].to_numpy(float),
                        g["g_l_bin"].eq("q4_high").to_numpy(),
                        default=float("nan"),
                    ),
                    "low_gap_high_repair_delta_pct": slice_delta_pct(
                        g["mse_method"].to_numpy(float),
                        g["mse_lrbn"].to_numpy(float),
                        (g["low_gap_flag"] & g["high_repair_flag"]).to_numpy(),
                        default=0.0,
                    ),
                    "A_gt_1_rate": float(g["A_gt_1"].mean()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(tables: List[pd.DataFrame], n_boot: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {"n_bootstrap": int(n_boot), "seed": int(seed)}
    for df in tables:
        method = str(df["method"].iloc[0])
        out[method] = {}
        for slice_name, mask in [
            ("overall", np.ones(len(df), dtype=bool)),
            ("high_gap_low_repair", (df["pred_gap_flag"] & df["low_repair_flag"]).to_numpy()),
            ("low_gap_high_repair", (df["low_gap_flag"] & df["high_repair_flag"]).to_numpy()),
        ]:
            g = df.loc[np.asarray(mask, dtype=bool)]
            if g.empty:
                continue
            delta = g["delta_mse_vs_lrbn"].to_numpy(float)
            vals = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                idx = rng.integers(0, len(delta), size=len(delta))
                vals[i] = float(delta[idx].mean())
            out[method][slice_name] = {
                "count": int(len(g)),
                "mean_delta": float(delta.mean()),
                "delta_ci95_low": float(np.quantile(vals, 0.025)),
                "delta_ci95_high": float(np.quantile(vals, 0.975)),
                "p_improve": float(np.mean(vals < 0.0)),
            }
    return out


def failure_cases(tables: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in tables:
        for keys, g in df.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
            for case_type, part in [
                ("top_harm", g.nlargest(20, "delta_mse_vs_lrbn")),
                ("top_win", g.nsmallest(20, "delta_mse_vs_lrbn")),
                (
                    "low_gap_high_repair_harm",
                    g[g["low_gap_flag"] & g["high_repair_flag"]].nlargest(20, "delta_mse_vs_lrbn"),
                ),
                (
                    "high_gap_low_repair_win",
                    g[g["pred_gap_flag"] & g["low_repair_flag"]].nsmallest(20, "delta_mse_vs_lrbn"),
                ),
            ]:
                if part.empty:
                    continue
                keep = part[
                    [
                        "method",
                        "sample_id",
                        "dataset",
                        "backbone",
                        "horizon",
                        "seed",
                        "split",
                        "delta_mse_vs_lrbn",
                        "selected",
                        "strength",
                        "g_l",
                        "g_y",
                        "repair_ratio",
                        "jump_support",
                        "A",
                    ]
                ].copy()
                keep["case_type"] = case_type
                rows.append(keep)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def apply_sra_method(batch: ForecastBatch, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pred, aux = apply_sra_bp(batch.context, batch.raw_pred, batch.lrbn_pred, horizons(batch), params)
    return pred, aux["strength"]


def build_methods(
    val: ForecastBatch,
    test: ForecastBatch,
    thresholds: Dict[str, float],
    grid: pd.DataFrame,
    stage3_params: Dict[str, Any],
    stage4_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]], Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]]:
    selected: Dict[str, Dict[str, Any]] = {
        "safe": select_params(grid, "safe"),
        "balanced": select_params(grid, "balanced"),
    }
    for family in ["basic", "short", "support", "continuous"]:
        selected[f"family_{family}"] = select_family_params(grid, family)

    val_methods: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {
        "LRBN": (val.lrbn_pred, np.zeros(len(val.meta))),
        "raw_no_correction": (val.raw_pred, np.zeros(len(val.meta))),
    }
    test_methods: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {
        "LRBN": (test.lrbn_pred, np.zeros(len(test.meta))),
        "raw_no_correction": (test.raw_pred, np.zeros(len(test.meta))),
    }
    if val.extra_preds:
        for name, arr in val.extra_preds.items():
            val_methods[name] = (arr, np.zeros(len(val.meta)))
    if test.extra_preds:
        for name, arr in test.extra_preds.items():
            test_methods[name] = (arr, np.zeros(len(test.meta)))

    always_params = {"method": "LRBN-BP-always", "alpha": 0.5, "anchor_mode": "last", "bridge_mode": "linear"}
    repair_params = load_stage4_params(
        stage4_dir,
        "LRBN-BP-repair-gate",
        {"method": "LRBN-BP-repair-gate", "alpha": 0.2, "anchor_mode": "last", "bridge_mode": "linear", "high_repair": 0.3},
    )
    short_params = load_stage4_params(
        stage4_dir,
        "LRBN-BP-short-bridge",
        {"method": "LRBN-BP-short-bridge", "alpha": 0.5, "anchor_mode": "last", "bridge_mode": "short_linear", "k_value": 4},
    )
    for split_name, batch, methods in [("val", val, val_methods), ("test", test, test_methods)]:
        bp = apply_candidate(batch, always_params)
        st = apply_stage3(batch, stage3_params)
        rep = apply_candidate(batch, repair_params)
        sh = apply_candidate(batch, short_params)
        methods["LRBN-BP-always"] = (bp.pred, bp.strength)
        methods["LRBN-BP-stage3-gated"] = (st.pred, st.strength)
        methods["LRBN-BP-repair-gate"] = (rep.pred, rep.strength)
        methods["LRBN-BP-short-bridge"] = (sh.pred, sh.strength)

        for key, params in selected.items():
            if params is None:
                continue
            if key == "safe":
                name = "LRBN-SRA-BP-safe"
            elif key == "balanced":
                name = "LRBN-SRA-BP-balanced"
            else:
                name = "LRBN-SRA-BP-" + str(params["method_family"])
            pred, strength = apply_sra_method(batch, params)
            methods[name] = (pred, strength)
    return selected, val_methods, test_methods


def overall_tables(
    val: ForecastBatch,
    test: ForecastBatch,
    val_methods: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    test_methods: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for split_name, batch, methods in [("val", val, val_methods), ("test", test, test_methods)]:
        for method, (pred, strength) in methods.items():
            rows.append(evaluate_arrays(batch, pred, method, strength, thresholds, split_name))
    return pd.DataFrame(rows)


def test_tables(
    test: ForecastBatch,
    methods: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    thresholds: Dict[str, float],
) -> List[pd.DataFrame]:
    keep = [
        "LRBN-BP-always",
        "LRBN-BP-stage3-gated",
        "LRBN-BP-repair-gate",
        "LRBN-BP-short-bridge",
        "LRBN-SRA-BP-safe",
        "LRBN-SRA-BP-balanced",
        "LRBN-SRA-BP-basic",
        "LRBN-SRA-BP-short",
        "LRBN-SRA-BP-support",
        "LRBN-SRA-BP-continuous",
    ]
    tables = []
    for method in keep:
        if method not in methods:
            continue
        pred, strength = methods[method]
        tables.append(method_sample_table(test, method, pred, strength, thresholds))
    return tables


def verdict_from_outputs(
    overall: pd.DataFrame,
    ci: Dict[str, Any],
    per_config: pd.DataFrame,
    selected: Dict[str, Any],
) -> Dict[str, Any]:
    test_rows = overall[overall["split"].eq("test")].set_index("method")

    def get(method: str, key: str, default: float = float("nan")) -> float:
        if method not in test_rows.index:
            return default
        return float(test_rows.loc[method, key])

    def safe_pass(method: str) -> bool:
        if method not in test_rows.index:
            return False
        upper = ci.get(method, {}).get("overall", {}).get("delta_ci95_high", float("inf"))
        return bool(
            get(method, "mse_delta_pct_vs_lrbn") <= -0.5
            and get(method, "harm_rate") <= 0.05
            and get(method, "q4_improvement_pct") >= 2.0
            and get(method, "low_gap_high_repair_delta_pct") <= 0.5
            and get(method, "config_improved_ratio") >= 0.75
            and float(upper) < 0.0
        )

    def balanced_pass(method: str) -> bool:
        if method not in test_rows.index:
            return False
        upper = ci.get(method, {}).get("high_gap_low_repair", {}).get("delta_ci95_high", float("inf"))
        return bool(
            get(method, "mse_delta_pct_vs_lrbn") <= -1.5
            and get(method, "harm_rate") <= 0.15
            and get(method, "q4_improvement_pct") >= 4.0
            and float(upper) < 0.0
            and get(method, "config_improved_ratio") >= 0.75
        )

    safe = "LRBN-SRA-BP-safe"
    balanced = "LRBN-SRA-BP-balanced"
    safe_ok = safe_pass(safe)
    balanced_ok = balanced_pass(balanced)
    if safe_ok and balanced_ok:
        status = "safe_and_balanced_pass"
        decision = "promote_sra_bp_to_mini_extension"
    elif safe_ok:
        status = "safe_pass"
        decision = "promote_safe_sra_to_mini_extension"
    elif balanced_ok:
        status = "balanced_pass"
        decision = "keep_as_risk_performance_variant_and_run_mini_extension"
    else:
        status = "fail"
        decision = "do_not_promote_current_sra_bp"
    return {
        "status": status,
        "decision": decision,
        "safe_pass": safe_ok,
        "balanced_pass": balanced_ok,
        "safe_params_found": selected.get("safe") is not None,
        "balanced_params_found": selected.get("balanced") is not None,
        "safe_metrics": test_rows.loc[safe].to_dict() if safe in test_rows.index else None,
        "balanced_metrics": test_rows.loc[balanced].to_dict() if balanced in test_rows.index else None,
        "bp_always_harm_rate": get("LRBN-BP-always", "harm_rate"),
        "stage3_harm_rate": get("LRBN-BP-stage3-gated", "harm_rate"),
        "repair_gate_harm_rate": get("LRBN-BP-repair-gate", "harm_rate"),
        "test_threshold_leakage": False,
    }


def df_to_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    show = df.head(max_rows)
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(out: Path, config: Dict[str, Any], verdict: Dict[str, Any]) -> None:
    overall = pd.read_csv(out / "stage5_overall.csv")
    overall_test = overall[overall["split"].eq("test")].sort_values("mse")
    gap = pd.read_csv(out / "stage5_gap_repair_interaction.csv")
    selected = pd.read_csv(out / "stage5_selected_alignment.csv")
    segments = pd.read_csv(out / "stage5_horizon_segments.csv")
    lines = [
        "# Stage 5 SRA-BP Validation Summary",
        "",
        "## Setup",
        "",
        f"- Input metrics: `{config['metrics_csv']}`",
        f"- Output directory: `{config['output_dir']}`",
        f"- Validation samples: `{config['n_val_samples']}`",
        f"- Test samples: `{config['n_test_samples']}`",
        f"- Test configs: `{config['n_test_configs']}`",
        f"- Test threshold leakage: `{verdict['test_threshold_leakage']}`",
        "",
        "## Verdict",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Decision: `{verdict['decision']}`",
        f"- Safe-SRA pass: `{verdict['safe_pass']}`",
        f"- Balanced-SRA pass: `{verdict['balanced_pass']}`",
        "",
        "## Test Overall",
        "",
        df_to_md(
            overall_test[
                [
                    "method",
                    "mse",
                    "mae",
                    "mse_delta_pct_vs_lrbn",
                    "harm_rate",
                    "coverage",
                    "q4_improvement_pct",
                    "low_gap_high_repair_delta_pct",
                    "high_gap_low_repair_delta_pct",
                    "config_improved_ratio",
                ]
            ],
            max_rows=30,
        ),
        "",
        "## Gap x Repair Slices",
        "",
        df_to_md(gap[gap["method"].isin(["LRBN-SRA-BP-safe", "LRBN-SRA-BP-balanced"])], max_rows=30),
        "",
        "## Selected Alignment",
        "",
        df_to_md(selected[selected["method"].isin(["LRBN-SRA-BP-safe", "LRBN-SRA-BP-balanced"])], max_rows=20),
        "",
        "## Horizon Segments",
        "",
        df_to_md(segments[segments["method"].isin(["LRBN-SRA-BP-safe", "LRBN-SRA-BP-balanced"])], max_rows=20),
        "",
        "## Interpretation",
        "",
    ]
    if verdict["safe_pass"] or verdict["balanced_pass"]:
        lines.append("Current SRA-BP has enough compact evidence to enter mini-extension, but not full TableA.")
    else:
        lines.append("Current SRA-BP did not pass compact gates; do not promote this specific proxy to the next table.")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `stage5_config.json`",
            "- `stage5_selected_safe_params.json`",
            "- `stage5_selected_balanced_params.json`",
            "- `stage5_calibration_grid.csv`",
            "- `stage5_overall.csv`",
            "- `stage5_boundary_gap_slices.csv`",
            "- `stage5_gap_repair_interaction.csv`",
            "- `stage5_oracle_boundary_truth_slices.csv`",
            "- `stage5_horizon_segments.csv`",
            "- `stage5_selected_alignment.csv`",
            "- `stage5_per_config.csv`",
            "- `stage5_bootstrap_ci.json`",
            "- `stage5_failure_cases_topk.csv`",
            "- `stage5_verdict.json`",
        ]
    )
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--stage3-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_stage3"))
    parser.add_argument("--stage4-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_stage4"))
    parser.add_argument("--stage45-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_attribution_stage45"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_sra_bp_stage5"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch = load_forecast_batch_from_metrics(args.metrics_csv)
    val, test = split_batch(batch)
    thresholds = compute_thresholds(val)
    stage3_params = load_stage3_params(args.stage3_dir)
    grid = calibrate_grid(val, thresholds)
    selected, val_methods, test_methods = build_methods(val, test, thresholds, grid, stage3_params, args.stage4_dir)

    overall = overall_tables(val, test, val_methods, test_methods, thresholds)
    overall.to_csv(args.output_dir / "stage5_overall.csv", index=False)
    grid.to_csv(args.output_dir / "stage5_calibration_grid.csv", index=False)
    write_json(args.output_dir / "stage5_selected_safe_params.json", selected.get("safe"))
    write_json(args.output_dir / "stage5_selected_balanced_params.json", selected.get("balanced"))

    tables = test_tables(test, test_methods, thresholds)
    pd.concat(tables, ignore_index=True).to_csv(args.output_dir / "stage5_sample_table.csv", index=False)
    boundary_gap_slices(tables).to_csv(args.output_dir / "stage5_boundary_gap_slices.csv", index=False)
    gap_repair_slices(tables).to_csv(args.output_dir / "stage5_gap_repair_interaction.csv", index=False)
    oracle_truth_slices(tables).to_csv(args.output_dir / "stage5_oracle_boundary_truth_slices.csv", index=False)
    horizon_segments(test, {k: v for k, v in test_methods.items() if k in {t["method"].iloc[0] for t in tables}}).to_csv(
        args.output_dir / "stage5_horizon_segments.csv",
        index=False,
    )
    selected_alignment(tables).to_csv(args.output_dir / "stage5_selected_alignment.csv", index=False)
    per_config = per_config_table(tables)
    per_config.to_csv(args.output_dir / "stage5_per_config.csv", index=False)
    ci = bootstrap_ci(tables, args.n_bootstrap, args.seed)
    write_json(args.output_dir / "stage5_bootstrap_ci.json", ci)
    failure_cases(tables).to_csv(args.output_dir / "stage5_failure_cases_topk.csv", index=False)
    verdict = verdict_from_outputs(overall, ci, per_config, selected)
    write_json(args.output_dir / "stage5_verdict.json", verdict)

    config = {
        "metrics_csv": str(args.metrics_csv),
        "stage3_dir": str(args.stage3_dir),
        "stage4_dir": str(args.stage4_dir),
        "stage45_dir": str(args.stage45_dir),
        "output_dir": str(args.output_dir),
        "scope": "compact_stage5_sra_bp",
        "datasets": sorted(test.meta["dataset"].unique().tolist()),
        "backbones": sorted(test.meta["backbone"].unique().tolist()),
        "horizons": sorted([int(x) for x in test.meta["horizon"].unique().tolist()]),
        "seeds": sorted([int(x) for x in test.meta["seed"].unique().tolist()]),
        "n_val_samples": int(len(val.meta)),
        "n_test_samples": int(len(test.meta)),
        "n_test_configs": int(test.meta.groupby(["dataset", "backbone", "horizon", "seed"]).ngroups),
        "thresholds_validation_only": thresholds,
        "stage3_params": stage3_params,
        "selected_params": selected,
        "test_threshold_leakage": False,
    }
    write_json(args.output_dir / "stage5_config.json", config)
    write_summary(args.output_dir, config, verdict)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "status": verdict["status"],
                "decision": verdict["decision"],
                "safe_pass": verdict["safe_pass"],
                "balanced_pass": verdict["balanced_pass"],
                "test_threshold_leakage": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
