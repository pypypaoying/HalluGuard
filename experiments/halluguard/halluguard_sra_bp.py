#!/usr/bin/env python
"""Sparse Repair-Aware Boundary Projection utilities for HalluGuard-LRBN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

EPS = 1e-6


@dataclass
class SRABPParams:
    method_family: str = "support"
    anchor_mode: str = "last"
    tail_len: int = 16
    tau_g: float = 1.0
    tau_r: float = 0.4
    tau_j: Optional[float] = None
    alpha: float = 0.5
    K: Any = 16
    continuous: bool = False
    kg: float = 4.0
    kr: float = 4.0
    kj: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def tail_scale(context: np.ndarray, tail_len: int = 16, eps: float = EPS) -> np.ndarray:
    context = np.asarray(context, dtype=float)
    x_tail = context[:, -min(tail_len, context.shape[1]) :, :]
    dx = x_tail[:, 1:, :] - x_tail[:, :-1, :]
    if dx.shape[1] == 0:
        return np.ones((context.shape[0], 1, context.shape[2]), dtype=float) * eps
    scale = np.nanstd(dx, axis=1, keepdims=True)
    return np.maximum(scale, eps)


def make_anchor(context: np.ndarray, mode: str = "last", tail_len: int = 16) -> np.ndarray:
    context = np.asarray(context, dtype=float)
    x_tail = context[:, -min(tail_len, context.shape[1]) :, :]
    x_last = context[:, -1:, :]
    dx = x_tail[:, 1:, :] - x_tail[:, :-1, :]
    slope = np.nanmean(dx, axis=1, keepdims=True) if dx.shape[1] else np.zeros_like(x_last)
    if mode == "last":
        return x_last
    if mode in {"trend", "last_plus_slope"}:
        return x_last + slope
    if mode in {"robust", "hybrid"}:
        med = np.nanmedian(x_tail, axis=1, keepdims=True)
        return 0.5 * x_last + 0.5 * (med + slope)
    raise ValueError(f"Unknown anchor mode: {mode}")


def compute_sra_features(
    context: np.ndarray,
    y_raw: np.ndarray,
    y_lrbn: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    anchor_mode: str = "last",
    tail_len: int = 16,
) -> Dict[str, np.ndarray]:
    context = np.asarray(context, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    y_lrbn = np.asarray(y_lrbn, dtype=float)
    anchor = make_anchor(context, mode=anchor_mode, tail_len=tail_len)
    scale = tail_scale(context, tail_len=tail_len)

    g_raw = np.nanmean(np.abs(y_raw[:, :1, :] - anchor) / scale, axis=(1, 2))
    g_l = np.nanmean(np.abs(y_lrbn[:, :1, :] - anchor) / scale, axis=(1, 2))
    repair_ratio = 1.0 - g_l / (g_raw + EPS)

    x_tail = context[:, -min(tail_len, context.shape[1]) :, :]
    dx = x_tail[:, 1:, :] - x_tail[:, :-1, :]
    slope = np.nanmean(dx, axis=1, keepdims=True) if dx.shape[1] else np.zeros_like(anchor)
    volatility = np.nanstd(dx, axis=1, keepdims=True) if dx.shape[1] else np.zeros_like(anchor)

    boundary_vec = y_lrbn[:, :1, :] - anchor
    trend_agree = (boundary_vec * slope > 0).astype(float)
    trend_support = np.nanmean(
        trend_agree * np.minimum(np.abs(slope) / (np.abs(boundary_vec) + EPS), 1.0),
        axis=(1, 2),
    )
    vol_support = np.nanmean(np.minimum(volatility / (np.abs(boundary_vec) + EPS), 1.0), axis=(1, 2))

    k0 = min(8, y_lrbn.shape[1])
    if k0 > 1:
        early_diff = np.nanmean(np.abs(y_lrbn[:, 1:k0, :] - y_lrbn[:, : k0 - 1, :]), axis=(1, 2))
    else:
        early_diff = np.zeros(y_lrbn.shape[0], dtype=float)
    boundary_mag = np.nanmean(np.abs(boundary_vec), axis=(1, 2))
    smooth_support = np.minimum(early_diff / (boundary_mag + EPS), 1.0)
    jump_support = (trend_support + vol_support + smooth_support) / 3.0

    out: Dict[str, np.ndarray] = {
        "anchor": anchor,
        "scale": scale,
        "g_raw": g_raw,
        "g_l": g_l,
        "repair_ratio": repair_ratio,
        "jump_support": jump_support,
        "trend_support": trend_support,
        "vol_support": vol_support,
        "smooth_support": smooth_support,
    }
    if y_true is not None:
        out["g_y"] = np.nanmean(np.abs(np.asarray(y_true, dtype=float)[:, :1, :] - anchor) / scale, axis=(1, 2))
    return out


def _resolve_k(k_value: Any, horizon: int) -> int:
    if isinstance(k_value, str):
        if k_value == "H_div_4":
            return max(4, int(horizon) // 4)
        if k_value in {"H", "full"}:
            return max(1, int(horizon) - 1)
        return int(float(k_value))
    return int(k_value)


def bridge_matrix(horizons: np.ndarray, max_h: int, k_value: Any) -> np.ndarray:
    bridge = np.zeros((len(horizons), max_h, 1), dtype=float)
    for i, h in enumerate(np.asarray(horizons, dtype=int)):
        k = max(1, _resolve_k(k_value, int(h)))
        t = np.arange(int(h), dtype=float)
        bridge[i, : int(h), 0] = np.maximum(0.0, 1.0 - t / float(k))
    return bridge


def apply_sra_bp(
    context: np.ndarray,
    y_raw: np.ndarray,
    y_lrbn: np.ndarray,
    horizons: np.ndarray,
    params: SRABPParams | Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if isinstance(params, dict):
        params = SRABPParams(**params)

    y_lrbn = np.asarray(y_lrbn, dtype=float)
    feats = compute_sra_features(
        context,
        y_raw,
        y_lrbn,
        anchor_mode=params.anchor_mode,
        tail_len=params.tail_len,
    )
    bridge = bridge_matrix(np.asarray(horizons, dtype=int), y_lrbn.shape[1], params.K)
    unit_delta = (feats["anchor"] - y_lrbn[:, :1, :]) * bridge

    if params.continuous:
        qg = 1.0 / (1.0 + np.exp(-params.kg * (feats["g_l"] - params.tau_g)))
        qr = 1.0 / (1.0 + np.exp(-params.kr * (params.tau_r - feats["repair_ratio"])))
        if params.tau_j is None:
            qj = np.ones_like(qg)
        else:
            qj = 1.0 / (1.0 + np.exp(-params.kj * (params.tau_j - feats["jump_support"])))
        strength = params.alpha * qg * qr * qj
        mask = strength > 1e-6
    else:
        mask = (feats["g_l"] > params.tau_g) & (feats["repair_ratio"] < params.tau_r)
        if params.tau_j is not None:
            mask = mask & (feats["jump_support"] < params.tau_j)
        strength = params.alpha * mask.astype(float)

    delta = strength.reshape(-1, 1, 1) * unit_delta
    y_final = y_lrbn + delta
    feats.update({"mask": mask, "strength": strength, "unit_delta": unit_delta, "delta": delta, "bridge": bridge})
    return y_final, feats


def apply_sra_bp_from_features(
    y_lrbn: np.ndarray,
    horizons: np.ndarray,
    features: Dict[str, np.ndarray],
    params: SRABPParams | Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply SRA-BP while reusing target-free features.

    This is numerically equivalent to `apply_sra_bp` when `features` were
    produced by `compute_sra_features` for the same context/raw/LRBN arrays.
    """
    if isinstance(params, dict):
        params = SRABPParams(**params)
    y_lrbn = np.asarray(y_lrbn, dtype=float)
    bridge = bridge_matrix(np.asarray(horizons, dtype=int), y_lrbn.shape[1], params.K)
    unit_delta = (features["anchor"] - y_lrbn[:, :1, :]) * bridge

    if params.continuous:
        qg = 1.0 / (1.0 + np.exp(-params.kg * (features["g_l"] - params.tau_g)))
        qr = 1.0 / (1.0 + np.exp(-params.kr * (params.tau_r - features["repair_ratio"])))
        if params.tau_j is None:
            qj = np.ones_like(qg)
        else:
            qj = 1.0 / (1.0 + np.exp(-params.kj * (params.tau_j - features["jump_support"])))
        strength = params.alpha * qg * qr * qj
        mask = strength > 1e-6
    else:
        mask = (features["g_l"] > params.tau_g) & (features["repair_ratio"] < params.tau_r)
        if params.tau_j is not None:
            mask = mask & (features["jump_support"] < params.tau_j)
        strength = params.alpha * mask.astype(float)

    delta = strength.reshape(-1, 1, 1) * unit_delta
    y_final = y_lrbn + delta
    aux = dict(features)
    aux.update({"mask": mask, "strength": strength, "unit_delta": unit_delta, "delta": delta, "bridge": bridge})
    return y_final, aux


def residual_alignment(delta: np.ndarray, y_lrbn: np.ndarray, y_true: np.ndarray, horizons: np.ndarray) -> Dict[str, np.ndarray]:
    a_vals = np.zeros(len(horizons), dtype=float)
    cos_vals = np.zeros(len(horizons), dtype=float)
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_lrbn, dtype=float)
    for i, h in enumerate(np.asarray(horizons, dtype=int)):
        d = np.asarray(delta[i, :h, :], dtype=float).reshape(-1)
        e = np.asarray(residual[i, :h, :], dtype=float).reshape(-1)
        norm2 = float(np.dot(d, d))
        dot = float(np.dot(d, e))
        a_vals[i] = 2.0 * dot / (norm2 + EPS)
        cos_vals[i] = dot / (np.linalg.norm(d) * np.linalg.norm(e) + EPS)
    return {"A": a_vals, "A_gt_1": a_vals > 1.0, "cosine": cos_vals}
