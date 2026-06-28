#!/usr/bin/env python
"""Stage 4 BP harm attribution and mechanism-level control utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from halluguard_lrbn_bp import (
    EPS,
    ForecastBatch,
    load_forecast_batch_from_metrics,
    mae_per_sample,
    mse_per_sample,
    paired_bootstrap_delta,
)


ALPHA_GRID = [0.05, 0.10, 0.20, 0.30, 0.50]


@dataclass
class CandidateResult:
    method: str
    pred: np.ndarray
    strength: np.ndarray
    info: Dict[str, np.ndarray]
    params: Dict[str, object]


def horizons(batch: ForecastBatch) -> np.ndarray:
    return batch.meta["horizon"].to_numpy(int)


def valid_values(x: np.ndarray, h: int) -> np.ndarray:
    return x[:h]


def tail_scale(context: np.ndarray, tail_len: int = 16, eps: float = 1e-6) -> np.ndarray:
    x_tail = context[:, -min(tail_len, context.shape[1]) :, :]
    dx = np.diff(x_tail, axis=1)
    if dx.shape[1] == 0:
        return np.ones((context.shape[0], 1, context.shape[2]), dtype=float) * eps
    scale = np.nanstd(dx, axis=1, keepdims=True)
    return np.maximum(scale, eps)


def robust_anchor(context: np.ndarray, tail_len: int = 16, mode: str = "hybrid") -> np.ndarray:
    x_tail = context[:, -min(tail_len, context.shape[1]) :, :]
    x_last = context[:, -1:, :]
    dx = np.diff(x_tail, axis=1)
    dx_mean = np.nanmean(dx, axis=1, keepdims=True) if dx.shape[1] else np.zeros_like(x_last)
    x_median = np.nanmedian(x_tail, axis=1, keepdims=True)
    if mode == "last":
        return x_last
    if mode == "last_plus_slope":
        return x_last + dx_mean
    if mode == "robust_median_slope":
        return x_median + dx_mean
    if mode == "hybrid":
        return 0.5 * x_last + 0.3 * (x_last + dx_mean) + 0.2 * (x_median + dx_mean)
    raise ValueError(f"Unknown anchor mode: {mode}")


def bridge_matrix(horizons_arr: np.ndarray, max_h: int, mode: str = "linear", k_value: object = 16, tau: float = 16.0) -> np.ndarray:
    bridge = np.zeros((len(horizons_arr), max_h, 1), dtype=float)
    for i, h in enumerate(horizons_arr):
        t = np.arange(h, dtype=float)
        if mode == "linear":
            w = np.clip(1.0 - t / max(h - 1, 1), 0.0, 1.0)
        elif mode == "exp":
            w = np.exp(-t / max(float(tau), 1e-6))
        elif mode == "short_linear":
            k = max(1, h // 4) if k_value == "H_div_4" else int(k_value)
            w = np.clip(1.0 - t / float(max(k, 1)), 0.0, 1.0)
        elif mode == "constant":
            w = np.ones(h, dtype=float)
        else:
            raise ValueError(f"Unknown bridge mode: {mode}")
        bridge[i, :h, 0] = w
    return bridge


def make_boundary_delta(
    batch: ForecastBatch,
    parent: np.ndarray,
    alpha: float,
    anchor_mode: str = "last",
    bridge_mode: str = "linear",
    tail_len: int = 16,
    k_value: object = 16,
    tau: float = 16.0,
    boundary_clip: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    hs = horizons(batch)
    max_h = parent.shape[1]
    anchor = robust_anchor(batch.context, tail_len=tail_len, mode=anchor_mode)
    scale = tail_scale(batch.context, tail_len=tail_len)
    pull = anchor - parent[:, :1, :]
    if boundary_clip is not None:
        pull = np.clip(pull / scale, -boundary_clip, boundary_clip) * scale
    bridge = bridge_matrix(hs, max_h, mode=bridge_mode, k_value=k_value, tau=tau)
    delta = alpha * pull * bridge
    return delta, {"anchor": anchor, "scale": scale, "pull": pull, "bridge": bridge}


def flat_cosine_valid(a: np.ndarray, b: np.ndarray, hs: np.ndarray) -> np.ndarray:
    out = np.zeros(len(hs), dtype=float)
    for i, h in enumerate(hs):
        av = a[i, :h, :].reshape(-1)
        bv = b[i, :h, :].reshape(-1)
        denom = np.linalg.norm(av) * np.linalg.norm(bv) + EPS
        out[i] = float(np.dot(av, bv) / denom)
    return out


def norm_valid(x: np.ndarray, hs: np.ndarray) -> np.ndarray:
    out = np.zeros(len(hs), dtype=float)
    for i, h in enumerate(hs):
        out[i] = float(np.linalg.norm(x[i, :h, :]))
    return out


def boundary_features(batch: ForecastBatch, delta_bp: np.ndarray, anchor: np.ndarray, scale: np.ndarray) -> Dict[str, np.ndarray]:
    raw_abs = np.abs(batch.raw_pred[:, :1, :] - anchor)
    lrbn_abs = np.abs(batch.lrbn_pred[:, :1, :] - anchor)
    raw_gap = np.nanmean(raw_abs / scale, axis=(1, 2))
    post_gap = np.nanmean(lrbn_abs / scale, axis=(1, 2))
    repair_ratio = 1.0 - np.nanmean(lrbn_abs / (raw_abs + EPS), axis=(1, 2))
    delta_l = batch.lrbn_pred - batch.raw_pred
    hs = horizons(batch)
    conflict = flat_cosine_valid(delta_l, delta_bp, hs)
    norm_ratio = norm_valid(delta_bp, hs) / (norm_valid(delta_l, hs) + EPS)
    x_tail = batch.context[:, -min(16, batch.context.shape[1]) :, :]
    dx = np.diff(x_tail, axis=1)
    tail_vol = np.nanstd(dx, axis=1).mean(axis=1) if dx.shape[1] else np.zeros(len(batch.meta))
    last_anchor = robust_anchor(batch.context, mode="last")
    hybrid_anchor = robust_anchor(batch.context, mode="hybrid")
    anchor_disagreement = np.nanmean(np.abs(last_anchor - hybrid_anchor) / scale, axis=(1, 2))
    true_boundary_jump = np.nanmean(np.abs(batch.y_true[:, :1, :] - anchor) / scale, axis=(1, 2))
    return {
        "raw_gap": raw_gap,
        "post_lrbn_gap": post_gap,
        "repair_ratio": repair_ratio,
        "conflict_cosine": conflict,
        "norm_ratio": norm_ratio,
        "tail_volatility": tail_vol,
        "anchor_disagreement": anchor_disagreement,
        "true_boundary_jump": true_boundary_jump,
    }


def strength_gap(post_gap: np.ndarray, tau: float, temp: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(post_gap - tau) / max(temp, 1e-6)))


def strength_repair(repair_ratio: np.ndarray, high_repair: float) -> np.ndarray:
    return np.clip((high_repair - repair_ratio) / max(high_repair, EPS), 0.0, 1.0)


def strength_conflict(conflict_cosine: np.ndarray, eta: float, temp: float = 0.1) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(conflict_cosine - eta) / max(temp, 1e-6)))


def strength_norm(delta_bp: np.ndarray, scale: np.ndarray, hs: np.ndarray, norm_clip: float) -> np.ndarray:
    delta_norm = norm_valid(delta_bp, hs)
    max_norm = norm_clip * scale.mean(axis=(1, 2)) * np.sqrt(hs)
    return np.minimum(1.0, max_norm / (delta_norm + EPS))


def apply_candidate(batch: ForecastBatch, params: Dict[str, object]) -> CandidateResult:
    method = str(params["method"])
    if method == "LRBN":
        pred = batch.lrbn_pred.copy()
        strength = np.zeros(len(batch.meta), dtype=float)
        return CandidateResult(method, pred, strength, {}, params)
    delta, info = make_boundary_delta(
        batch=batch,
        parent=batch.lrbn_pred,
        alpha=float(params.get("alpha", 0.3)),
        anchor_mode=str(params.get("anchor_mode", "last")),
        bridge_mode=str(params.get("bridge_mode", "linear")),
        tail_len=int(params.get("tail_len", 16)),
        k_value=params.get("k_value", 16),
        tau=float(params.get("tau", 16.0)),
        boundary_clip=params.get("boundary_clip"),
    )
    feats = boundary_features(batch, delta, info["anchor"], info["scale"])
    hs = horizons(batch)
    strength = np.ones(len(batch.meta), dtype=float)
    if method == "LRBN-BP-gap-strength":
        strength *= strength_gap(feats["post_lrbn_gap"], float(params["gap_tau"]), float(params.get("gap_temp", 0.5)))
    elif method == "LRBN-BP-bounded":
        strength *= strength_norm(delta, info["scale"], hs, float(params.get("norm_clip", 0.25)))
    elif method == "LRBN-BP-robust-anchor":
        pass
    elif method == "LRBN-BP-short-bridge":
        pass
    elif method == "LRBN-BP-conflict-filter":
        strength *= strength_conflict(feats["conflict_cosine"], float(params.get("conflict_eta", -0.2)))
    elif method == "LRBN-BP-repair-gate":
        strength *= strength_repair(feats["repair_ratio"], float(params.get("high_repair", 0.7)))
    elif method == "LRBN-BP-safe-controller":
        strength *= strength_gap(feats["post_lrbn_gap"], float(params["gap_tau"]), float(params.get("gap_temp", 0.5)))
        strength *= strength_repair(feats["repair_ratio"], float(params.get("high_repair", 0.7)))
        strength *= strength_conflict(feats["conflict_cosine"], float(params.get("conflict_eta", -0.2)))
        strength *= strength_norm(delta, info["scale"], hs, float(params.get("norm_clip", 0.25)))
    elif method == "LRBN-BP-always":
        pass
    else:
        raise ValueError(f"Unknown Stage4 method: {method}")
    pred = batch.lrbn_pred + strength.reshape(-1, 1, 1) * delta
    info.update(feats)
    info["effective_strength"] = strength
    return CandidateResult(method, pred, strength, info, params)


def evaluate_candidate(batch: ForecastBatch, result: CandidateResult, split: str) -> Dict[str, object]:
    parent_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mse = mse_per_sample(result.pred, batch.y_true)
    parent_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    method_mae = mae_per_sample(result.pred, batch.y_true)
    delta = method_mse - parent_mse
    wins = delta < 0
    harms = delta > 1e-12
    mean_win = float((-delta[wins]).mean()) if wins.any() else 0.0
    mean_loss = float(delta[harms].mean()) if harms.any() else 0.0
    out = {
        "split": split,
        "method": result.method,
        "mean_mse": float(np.mean(method_mse)),
        "mean_mae": float(np.mean(method_mae)),
        "lrbn_mse": float(np.mean(parent_mse)),
        "lrbn_mae": float(np.mean(parent_mae)),
        "delta_mse_vs_lrbn": float(np.mean(delta)),
        "delta_pct_vs_lrbn": float((np.mean(method_mse) - np.mean(parent_mse)) / (np.mean(parent_mse) + EPS) * 100.0),
        "delta_mae_vs_lrbn": float(np.mean(method_mae - parent_mae)),
        "delta_mae_pct_vs_lrbn": float((np.mean(method_mae) - np.mean(parent_mae)) / (np.mean(parent_mae) + EPS) * 100.0),
        "harm_rate_vs_lrbn": float(np.mean(harms)),
        "win_rate_vs_lrbn": float(np.mean(wins)),
        "mean_win_size": mean_win,
        "mean_loss_size": mean_loss,
        "win_loss_ratio": float(mean_win / (mean_loss + EPS)),
        "coverage": float(np.mean(np.abs(result.strength) > 1e-8)),
        "mean_strength": float(np.mean(result.strength)),
        "test_threshold_leakage": False,
    }
    out.update({f"param_{k}": v for k, v in result.params.items() if k != "method"})
    return out


def segment_delta_rows(batch: ForecastBatch, result: CandidateResult, split: str) -> List[dict]:
    rows = []
    for i, h in enumerate(horizons(batch)):
        cuts = {
            "early": (0, max(1, h // 4)),
            "mid": (max(1, h // 4), max(max(1, h // 4) + 1, 3 * h // 4)),
            "late": (max(max(1, h // 4) + 1, 3 * h // 4), h),
        }
        for name, (a, b) in cuts.items():
            pm = np.mean((batch.lrbn_pred[i, a:b, :] - batch.y_true[i, a:b, :]) ** 2)
            cm = np.mean((result.pred[i, a:b, :] - batch.y_true[i, a:b, :]) ** 2)
            r = batch.meta.iloc[i]
            rows.append(
                {
                    "split": split,
                    "method": result.method,
                    "segment": name,
                    "config_id": r["config_id"],
                    "dataset": r["dataset"],
                    "backbone": r["backbone"],
                    "horizon": int(r["horizon"]),
                    "seed": int(r["seed"]),
                    "delta_mse_vs_lrbn": float(cm - pm),
                    "lrbn_mse": float(pm),
                    "method_mse": float(cm),
                }
            )
    return rows


def boundary_slice_rows(batch: ForecastBatch, result: CandidateResult, split: str) -> List[dict]:
    post_gap = result.info.get("post_lrbn_gap")
    if post_gap is None:
        delta, info = make_boundary_delta(batch, batch.lrbn_pred, alpha=0.0)
        post_gap = boundary_features(batch, delta, info["anchor"], info["scale"])["post_lrbn_gap"]
    labels = pd.qcut(post_gap, q=4, labels=["q1_low", "q2", "q3", "q4_high"], duplicates="drop")
    parent_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mse = mse_per_sample(result.pred, batch.y_true)
    rows = []
    for label in sorted(pd.Series(labels).dropna().unique()):
        mask = np.asarray(labels == label)
        rows.append(
            {
                "split": split,
                "method": result.method,
                "boundary_bin": str(label),
                "n": int(mask.sum()),
                "coverage": float(np.mean(result.strength[mask] > 1e-8)),
                "mean_strength": float(np.mean(result.strength[mask])),
                "mean_post_gap": float(np.mean(post_gap[mask])),
                "lrbn_mse": float(np.mean(parent_mse[mask])),
                "method_mse": float(np.mean(method_mse[mask])),
                "delta_mse_vs_lrbn": float(np.mean(method_mse[mask] - parent_mse[mask])),
                "delta_pct_vs_lrbn": float(
                    (np.mean(method_mse[mask]) - np.mean(parent_mse[mask])) / (np.mean(parent_mse[mask]) + EPS) * 100.0
                ),
                "harm_rate_vs_lrbn": float(np.mean(method_mse[mask] > parent_mse[mask] + 1e-12)),
                "win_rate_vs_lrbn": float(np.mean(method_mse[mask] < parent_mse[mask])),
            }
        )
    return rows


def per_config_rows(batch: ForecastBatch, result: CandidateResult, split: str) -> List[dict]:
    parent_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mse = mse_per_sample(result.pred, batch.y_true)
    method_mae = mae_per_sample(result.pred, batch.y_true)
    df = batch.meta.copy()
    df["parent_mse"] = parent_mse
    df["method_mse"] = method_mse
    df["method_mae"] = method_mae
    df["strength"] = result.strength
    rows = []
    for keys, g in df.groupby(["dataset", "backbone", "horizon", "seed"]):
        dataset, backbone, horizon, seed = keys
        rows.append(
            {
                "split": split,
                "method": result.method,
                "dataset": dataset,
                "backbone": backbone,
                "horizon": int(horizon),
                "seed": int(seed),
                "n": len(g),
                "coverage": float(np.mean(g["strength"] > 1e-8)),
                "mean_strength": float(g["strength"].mean()),
                "lrbn_mse": float(g["parent_mse"].mean()),
                "method_mse": float(g["method_mse"].mean()),
                "method_mae": float(g["method_mae"].mean()),
                "delta_mse_vs_lrbn": float((g["method_mse"] - g["parent_mse"]).mean()),
                "delta_pct_vs_lrbn": float(
                    (g["method_mse"].mean() - g["parent_mse"].mean()) / (g["parent_mse"].mean() + EPS) * 100.0
                ),
                "harm_rate_vs_lrbn": float((g["method_mse"] > g["parent_mse"] + 1e-12).mean()),
                "win_rate_vs_lrbn": float((g["method_mse"] < g["parent_mse"]).mean()),
            }
        )
    return rows


def candidate_grid(val: ForecastBatch) -> List[Dict[str, object]]:
    delta0, info0 = make_boundary_delta(val, val.lrbn_pred, alpha=0.0)
    post_gap = boundary_features(val, delta0, info0["anchor"], info0["scale"])["post_lrbn_gap"]
    gap_quantiles = [0.50, 0.60, 0.70, 0.80]
    gap_taus = [(q, float(np.quantile(post_gap, q))) for q in gap_quantiles]
    out: List[Dict[str, object]] = [{"method": "LRBN"}]
    for alpha in ALPHA_GRID:
        out.append({"method": "LRBN-BP-always", "alpha": alpha, "anchor_mode": "last", "bridge_mode": "linear"})
        for q, tau in gap_taus:
            for temp in [0.25, 0.50, 1.00]:
                out.append(
                    {
                        "method": "LRBN-BP-gap-strength",
                        "alpha": alpha,
                        "anchor_mode": "last",
                        "bridge_mode": "linear",
                        "gap_tau_quantile": q,
                        "gap_tau": tau,
                        "gap_temp": temp,
                    }
                )
        for boundary_clip in [0.5, 1.0, 2.0, 3.0]:
            for norm_clip in [0.10, 0.20, 0.30, 0.50]:
                out.append(
                    {
                        "method": "LRBN-BP-bounded",
                        "alpha": alpha,
                        "anchor_mode": "last",
                        "bridge_mode": "linear",
                        "boundary_clip": boundary_clip,
                        "norm_clip": norm_clip,
                    }
                )
        for anchor_mode in ["last_plus_slope", "robust_median_slope", "hybrid"]:
            out.append({"method": "LRBN-BP-robust-anchor", "alpha": alpha, "anchor_mode": anchor_mode, "bridge_mode": "linear"})
        for k in [4, 8, 16, 24, "H_div_4"]:
            out.append({"method": "LRBN-BP-short-bridge", "alpha": alpha, "anchor_mode": "last", "bridge_mode": "short_linear", "k_value": k})
        for eta in [-0.50, -0.20, 0.00, 0.20]:
            out.append(
                {
                    "method": "LRBN-BP-conflict-filter",
                    "alpha": alpha,
                    "anchor_mode": "last",
                    "bridge_mode": "linear",
                    "conflict_eta": eta,
                }
            )
        for high_repair in [0.30, 0.50, 0.70, 0.90]:
            out.append(
                {
                    "method": "LRBN-BP-repair-gate",
                    "alpha": alpha,
                    "anchor_mode": "last",
                    "bridge_mode": "linear",
                    "high_repair": high_repair,
                }
            )
    for alpha in [0.20, 0.30, 0.50]:
        for anchor_mode in ["hybrid", "robust_median_slope"]:
            for k in [8, 16, "H_div_4"]:
                for boundary_clip in [1.0, 2.0]:
                    for norm_clip in [0.20, 0.30]:
                        for q, tau in gap_taus:
                            for temp in [0.50, 1.00]:
                                for eta in [-0.20, 0.00]:
                                    for high_repair in [0.50, 0.70]:
                                        out.append(
                                            {
                                                "method": "LRBN-BP-safe-controller",
                                                "alpha": alpha,
                                                "anchor_mode": anchor_mode,
                                                "bridge_mode": "short_linear",
                                                "k_value": k,
                                                "boundary_clip": boundary_clip,
                                                "norm_clip": norm_clip,
                                                "gap_tau_quantile": q,
                                                "gap_tau": tau,
                                                "gap_temp": temp,
                                                "conflict_eta": eta,
                                                "high_repair": high_repair,
                                            }
                                        )
    return out


def safe_selection_score(row: pd.Series) -> float:
    penalty = 0.0
    penalty += 100.0 * max(0.0, float(row["harm_rate_vs_lrbn"]) - 0.02)
    penalty += 5.0 * max(0.0, float(row.get("low_delta_pct_vs_lrbn", 0.0)) - 0.5)
    penalty += 2.0 * max(0.0, 2.0 - float(row.get("q4_improvement_pct_vs_lrbn", 0.0)))
    return float(row["delta_pct_vs_lrbn"]) + penalty


def choose_best_per_method(grid_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in grid_df.groupby("method"):
        if method == "LRBN":
            rows.append(g.iloc[0])
            continue
        if method == "LRBN-BP-always":
            rows.append(g.sort_values(["delta_pct_vs_lrbn", "harm_rate_vs_lrbn"]).iloc[0])
            continue
        tmp = g.copy()
        tmp["selection_score"] = tmp.apply(safe_selection_score, axis=1)
        rows.append(tmp.sort_values(["selection_score", "delta_pct_vs_lrbn", "harm_rate_vs_lrbn"]).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def add_gate_metrics(row: dict, slices: pd.DataFrame, configs: pd.DataFrame) -> dict:
    method = row["method"]
    ss = slices[(slices["method"].eq(method)) & (slices["split"].eq(row["split"]))]
    q4 = ss[ss["boundary_bin"].eq("q4_high")]
    low = ss[ss["boundary_bin"].isin(["q1_low", "q2"])]
    cc = configs[(configs["method"].eq(method)) & (configs["split"].eq(row["split"]))]
    row["q4_improvement_pct_vs_lrbn"] = float(-q4["delta_pct_vs_lrbn"].iloc[0]) if len(q4) else np.nan
    row["low_delta_pct_vs_lrbn"] = float(low["delta_pct_vs_lrbn"].max()) if len(low) else np.nan
    row["config_improved_ratio"] = float(np.mean(cc["delta_mse_vs_lrbn"] <= 0.0)) if len(cc) else np.nan
    return row

