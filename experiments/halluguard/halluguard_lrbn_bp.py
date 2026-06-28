#!/usr/bin/env python
"""Reusable HalluGuard-LRBN + optional Boundary Projection utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

Array = np.ndarray
EPS = 1e-8


@dataclass
class ForecastBatch:
    """Matched forecast trajectories for validation/evaluation.

    Shapes:
        context:   [N, L, C]
        y_true:    [N, H, C]
        raw_pred:  [N, H, C]
        lrbn_pred: [N, H, C]
    """

    context: Array
    y_true: Array
    raw_pred: Array
    lrbn_pred: Array
    meta: pd.DataFrame
    extra_preds: Optional[Dict[str, Array]] = None

    def subset(self, mask: Array) -> "ForecastBatch":
        mask = np.asarray(mask, dtype=bool)
        extra = None
        if self.extra_preds is not None:
            extra = {k: v[mask] for k, v in self.extra_preds.items()}
        return ForecastBatch(
            context=self.context[mask],
            y_true=self.y_true[mask],
            raw_pred=self.raw_pred[mask],
            lrbn_pred=self.lrbn_pred[mask],
            meta=self.meta.loc[mask].reset_index(drop=True),
            extra_preds=extra,
        )


def ensure_3d(a: Array) -> Array:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, arr.shape[0], 1)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected [T], [N,T], or [N,T,C], got shape={arr.shape}")


def one_traj(x) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected one trajectory [T] or [T,C], got shape={arr.shape}")


def pad_stack(trajs: List[Array], fill: float = np.nan) -> Array:
    if not trajs:
        raise ValueError("Cannot stack an empty trajectory list")
    max_t = max(t.shape[0] for t in trajs)
    max_c = max(t.shape[1] for t in trajs)
    out = np.full((len(trajs), max_t, max_c), fill, dtype=float)
    for i, t in enumerate(trajs):
        out[i, : t.shape[0], : t.shape[1]] = t
    return out


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_key(sample: dict) -> str:
    return f"{sample.get('split')}::{sample.get('sample_id')}"


def method_path(metrics: pd.DataFrame, method: str) -> Optional[Path]:
    rows = metrics[metrics["method"].eq(method)]
    if rows.empty:
        return None
    p = Path(str(rows.iloc[0]["prediction_path"]))
    return p


def load_forecast_batch_from_metrics(
    metrics_csv: Path,
    include_methods: Iterable[str] = (
        "matched_sparse_smoothing",
        "naive_smoothing",
        "ema_smoothing",
        "median_smoothing",
    ),
) -> ForecastBatch:
    """Load matched raw/LRBN/action trajectories from combined_metrics.csv."""

    metrics = pd.read_csv(metrics_csv)
    completed = metrics[metrics["status"].eq("completed")].copy()
    required = {"raw_no_correction", "HalluGuard-LRBN"}
    contexts: List[Array] = []
    targets: List[Array] = []
    raws: List[Array] = []
    lrbns: List[Array] = []
    meta_rows: List[dict] = []
    extra_lists: Dict[str, List[Array]] = {m: [] for m in include_methods}

    group_cols = ["dataset", "backbone", "horizon", "seed"]
    for keys, group in completed.groupby(group_cols):
        paths = {row["method"]: Path(str(row["prediction_path"])) for _, row in group.iterrows()}
        if not required.issubset(paths):
            continue
        raw_samples = {sample_key(s): s for s in read_jsonl(paths["raw_no_correction"])}
        lrbn_samples = {sample_key(s): s for s in read_jsonl(paths["HalluGuard-LRBN"])}
        extra_samples = {
            method: {sample_key(s): s for s in read_jsonl(paths[method])}
            for method in include_methods
            if method in paths and paths[method].exists()
        }
        dataset, backbone, horizon, seed = keys
        for skey in sorted(raw_samples):
            if skey not in lrbn_samples:
                continue
            raw = raw_samples[skey]
            lrbn = lrbn_samples[skey]
            contexts.append(one_traj(raw["context"]))
            targets.append(one_traj(raw["target"]))
            raws.append(one_traj(raw["prediction"]))
            lrbns.append(one_traj(lrbn["prediction"]))
            meta_rows.append(
                {
                    "config_id": f"{dataset}_{backbone}_{int(horizon)}_seed{int(seed)}",
                    "sample_key": skey,
                    "sample_id": raw.get("sample_id"),
                    "split": raw.get("split"),
                    "dataset": dataset,
                    "backbone": backbone,
                    "horizon": int(horizon),
                    "seed": int(seed),
                }
            )
            for method in include_methods:
                samples = extra_samples.get(method)
                if samples is not None and skey in samples:
                    extra_lists[method].append(one_traj(samples[skey]["prediction"]))
                else:
                    extra_lists[method].append(np.full_like(one_traj(raw["prediction"]), np.nan))

    if not meta_rows:
        raise RuntimeError(f"No matched raw/LRBN samples found in {metrics_csv}")
    extra = {k: pad_stack(v) for k, v in extra_lists.items() if v}
    return ForecastBatch(
        context=pad_stack(contexts),
        y_true=pad_stack(targets),
        raw_pred=pad_stack(raws),
        lrbn_pred=pad_stack(lrbns),
        meta=pd.DataFrame(meta_rows),
        extra_preds=extra,
    )


def _tail_diff_scale(context: Array, tail: int = 24, eps: float = 1e-6) -> Array:
    context = ensure_3d(context)
    x_tail = context[:, -min(tail + 1, context.shape[1]) :, :]
    diffs = np.diff(x_tail, axis=1)
    if diffs.shape[1] == 0:
        return np.ones((context.shape[0], 1, context.shape[2])) * eps
    med = np.median(diffs, axis=1, keepdims=True)
    mad = np.median(np.abs(diffs - med), axis=1, keepdims=True)
    std = np.std(diffs, axis=1, keepdims=True)
    scale = 1.4826 * mad
    scale = np.where(scale > eps, scale, std)
    return np.maximum(scale, eps)


def normalized_boundary_gap(context: Array, pred: Array, tail: int = 24) -> Array:
    """Sample-level normalized boundary gap computed without target leakage."""

    context = ensure_3d(context)
    pred = ensure_3d(pred)
    scale = _tail_diff_scale(context, tail=tail)[:, 0, :]
    gap = np.abs(pred[:, 0, :] - context[:, -1, :]) / scale
    return np.mean(gap, axis=-1)


def decay_vector(horizon: int, mode: str = "linear") -> Array:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if mode == "linear":
        return np.linspace(1.0, 0.0, horizon, endpoint=True)
    if mode == "exp":
        t = np.arange(horizon)
        return np.exp(-3.0 * t / max(horizon - 1, 1))
    if mode == "cosine":
        t = np.linspace(0.0, np.pi, horizon)
        return 0.5 * (1.0 + np.cos(t))
    raise ValueError(f"Unknown decay mode: {mode}")


def boundary_projection(context: Array, pred: Array, alpha: float, decay: str = "linear") -> Array:
    context = ensure_3d(context)
    pred = ensure_3d(pred)
    if alpha == 0.0:
        return pred.copy()
    d = decay_vector(pred.shape[1], mode=decay).reshape(1, pred.shape[1], 1)
    boundary_error = context[:, -1:, :] - pred[:, 0:1, :]
    return pred + alpha * boundary_error * d


def boundary_projection_batched(
    context: Array,
    pred: Array,
    alpha: float,
    horizons: Optional[Array] = None,
    decay: str = "linear",
) -> Array:
    context = ensure_3d(context)
    pred = ensure_3d(pred)
    corrected = pred.copy()
    if alpha == 0.0:
        return corrected
    if horizons is None:
        return boundary_projection(context, pred, alpha=alpha, decay=decay)
    horizons = np.asarray(horizons, dtype=int)
    for h in sorted(set(horizons.tolist())):
        idx = horizons == h
        corrected[idx, :h, :] = boundary_projection(context[idx], pred[idx, :h, :], alpha=alpha, decay=decay)
    return corrected


def lrbn_optional_bp(
    context: Array,
    lrbn_pred: Array,
    alpha: float,
    tau: float,
    tail: int = 24,
    decay: str = "linear",
    horizons: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    gap = normalized_boundary_gap(context, lrbn_pred, tail=tail)
    selected = gap > tau
    corrected = ensure_3d(lrbn_pred).copy()
    if np.any(selected):
        selected_horizons = None if horizons is None else np.asarray(horizons, dtype=int)[selected]
        corrected[selected] = boundary_projection_batched(
            ensure_3d(context)[selected],
            corrected[selected],
            alpha=alpha,
            horizons=selected_horizons,
            decay=decay,
        )
    return corrected, selected, gap


def mse_per_sample(pred: Array, y_true: Array) -> Array:
    return np.nanmean((ensure_3d(pred) - ensure_3d(y_true)) ** 2, axis=(1, 2))


def mae_per_sample(pred: Array, y_true: Array) -> Array:
    return np.nanmean(np.abs(ensure_3d(pred) - ensure_3d(y_true)), axis=(1, 2))


def boundary_quantile_masks(gap: Array) -> Dict[str, Array]:
    q25, q50, q75 = np.quantile(gap, [0.25, 0.50, 0.75])
    return {
        "q1_low": gap <= q25,
        "q2": (gap > q25) & (gap <= q50),
        "q3": (gap > q50) & (gap <= q75),
        "q4_high": gap > q75,
    }


def summarize_against(
    name: str,
    pred: Array,
    y_true: Array,
    baseline_pred: Array,
    raw_pred: Optional[Array] = None,
    selected: Optional[Array] = None,
) -> Dict[str, float]:
    method_mse = mse_per_sample(pred, y_true)
    baseline_mse = mse_per_sample(baseline_pred, y_true)
    out = {
        "method": name,
        "mean_mse": float(np.mean(method_mse)),
        "mean_mae": float(np.mean(mae_per_sample(pred, y_true))),
        "baseline_mean_mse": float(np.mean(baseline_mse)),
        "delta_vs_baseline": float(np.mean(method_mse - baseline_mse)),
        "delta_pct_vs_baseline": float(
            (np.mean(method_mse) - np.mean(baseline_mse)) / (np.mean(baseline_mse) + EPS) * 100.0
        ),
        "win_rate_vs_baseline": float(np.mean(method_mse < baseline_mse)),
        "harm_rate_vs_baseline": float(np.mean(method_mse > baseline_mse + 1e-12)),
    }
    if raw_pred is not None:
        raw_mse = mse_per_sample(raw_pred, y_true)
        out["raw_mean_mse"] = float(np.mean(raw_mse))
        out["delta_vs_raw"] = float(np.mean(method_mse - raw_mse))
        out["delta_pct_vs_raw"] = float(
            (np.mean(method_mse) - np.mean(raw_mse)) / (np.mean(raw_mse) + EPS) * 100.0
        )
    if selected is not None:
        out["coverage"] = float(np.mean(selected))
    return out


def choose_lrbn_bp_params(
    val: ForecastBatch,
    alpha_grid: Iterable[float],
    tau_quantiles: Iterable[float],
    tail: int = 24,
    decay: str = "linear",
    max_overall_worse_pct: float = 0.2,
    max_harm_extra_pp: float = 0.02,
    max_low_slice_worse_pct: float = 0.5,
    min_coverage: float = 0.05,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Validation-only selection of gated LRBN+BP parameters."""

    y = ensure_3d(val.y_true)
    lrbn = ensure_3d(val.lrbn_pred)
    base_mse = mse_per_sample(lrbn, y)
    base_mean = float(np.mean(base_mse))
    gap = normalized_boundary_gap(val.context, lrbn, tail=tail)
    tau_values = [float(np.quantile(gap, q)) for q in tau_quantiles]
    tau_values.append(float("inf"))
    masks = boundary_quantile_masks(gap)
    low = masks["q1_low"] | masks["q2"]
    q4 = masks["q4_high"]

    rows: List[dict] = []
    for alpha in alpha_grid:
        for tau in tau_values:
            pred, selected, _ = lrbn_optional_bp(
                val.context,
                lrbn,
                alpha,
                tau,
                tail=tail,
                decay=decay,
                horizons=val.meta["horizon"].to_numpy(int),
            )
            mse = mse_per_sample(pred, y)
            coverage = float(np.mean(selected))
            delta_pct = float((np.mean(mse) - base_mean) / (base_mean + EPS) * 100.0)
            harm = float(np.mean(mse > base_mse + 1e-12))
            low_delta_pct = float(
                (np.mean(mse[low]) - np.mean(base_mse[low])) / (np.mean(base_mse[low]) + EPS) * 100.0
            )
            q4_impr_pct = float(
                (np.mean(base_mse[q4]) - np.mean(mse[q4])) / (np.mean(base_mse[q4]) + EPS) * 100.0
            )
            feasible = (
                delta_pct <= max_overall_worse_pct
                and harm <= max_harm_extra_pp
                and low_delta_pct <= max_low_slice_worse_pct
                and (coverage >= min_coverage or alpha == 0.0 or np.isinf(tau))
            )
            rows.append(
                {
                    "policy": "gated",
                    "alpha": float(alpha),
                    "tau": float(tau),
                    "coverage": coverage,
                    "delta_pct_vs_lrbn": delta_pct,
                    "harm_rate_vs_lrbn": harm,
                    "low_delta_pct_vs_lrbn": low_delta_pct,
                    "q4_improvement_pct_vs_lrbn": q4_impr_pct,
                    "feasible": bool(feasible),
                }
            )
    grid = pd.DataFrame(rows)
    feasible_grid = grid[grid["feasible"]].copy()
    if feasible_grid.empty:
        best = {
            "alpha": 0.0,
            "tau": float("inf"),
            "tail": tail,
            "decay": decay,
            "selection_reason": "safe_fallback_no_feasible_candidate",
        }
    else:
        feasible_grid = feasible_grid.sort_values(
            by=["q4_improvement_pct_vs_lrbn", "delta_pct_vs_lrbn", "alpha"],
            ascending=[False, True, True],
        )
        best = feasible_grid.iloc[0].to_dict()
        best.update({"tail": tail, "decay": decay, "selection_reason": "validation_feasible_best_q4"})
    return best, grid


def choose_lrbn_bp_always_alpha(
    val: ForecastBatch,
    alpha_grid: Iterable[float],
    tail: int = 24,
    decay: str = "linear",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y = ensure_3d(val.y_true)
    base_mse = mse_per_sample(val.lrbn_pred, y)
    rows = []
    for alpha in alpha_grid:
        pred = boundary_projection_batched(
            val.context,
            val.lrbn_pred,
            alpha=alpha,
            horizons=val.meta["horizon"].to_numpy(int),
            decay=decay,
        )
        mse = mse_per_sample(pred, y)
        rows.append(
            {
                "policy": "always",
                "alpha": float(alpha),
                "tau": float("-inf"),
                "coverage": 1.0 if alpha > 0 else 0.0,
                "delta_pct_vs_lrbn": float(
                    (np.mean(mse) - np.mean(base_mse)) / (np.mean(base_mse) + EPS) * 100.0
                ),
                "harm_rate_vs_lrbn": float(np.mean(mse > base_mse + 1e-12)),
            }
        )
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["delta_pct_vs_lrbn", "harm_rate_vs_lrbn", "alpha"]).iloc[0].to_dict()
    best.update({"tail": tail, "decay": decay, "selection_reason": "validation_min_mse_always"})
    return best, grid


def choose_raw_bp_alpha(
    val: ForecastBatch,
    alpha_grid: Iterable[float],
    decay: str = "linear",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y = ensure_3d(val.y_true)
    raw_mse = mse_per_sample(val.raw_pred, y)
    rows = []
    for alpha in alpha_grid:
        pred = boundary_projection_batched(
            val.context,
            val.raw_pred,
            alpha=alpha,
            horizons=val.meta["horizon"].to_numpy(int),
            decay=decay,
        )
        mse = mse_per_sample(pred, y)
        rows.append(
            {
                "policy": "raw_bp_global",
                "alpha": float(alpha),
                "tau": float("-inf"),
                "coverage": 1.0 if alpha > 0 else 0.0,
                "delta_pct_vs_raw": float((np.mean(mse) - np.mean(raw_mse)) / (np.mean(raw_mse) + EPS) * 100.0),
            }
        )
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["delta_pct_vs_raw", "alpha"]).iloc[0].to_dict()
    best.update({"decay": decay, "selection_reason": "validation_min_mse_raw_bp"})
    return best, grid


def paired_bootstrap_delta(
    method_mse: Array,
    baseline_mse: Array,
    n_boot: int = 2000,
    seed: int = 2026,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(method_mse) - np.asarray(baseline_mse)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(diff), size=len(diff))
        deltas[i] = float(np.mean(diff[idx]))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "mean_delta": float(np.mean(diff)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_improve_bootstrap": float(np.mean(deltas < 0.0)),
    }


def with_extra(batch: ForecastBatch, lrbn_pred: Array) -> ForecastBatch:
    return replace(batch, lrbn_pred=lrbn_pred)
