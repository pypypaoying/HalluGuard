#!/usr/bin/env python
"""Run learnable HalluGuard boundary-normalization ablations.

This script turns the RDN-level-only observation into testable, trainable
variants:

1. learnable_robust_anchor
   center = alpha * last + (1-alpha) * tail_median

2. learnable_residual_gate
   y = raw + gate(context_features) * (boundary_anchor_forecast - raw)

3. learnable_horizon_gate
   y[t] = raw[t] + gate[t] * (boundary_anchor_forecast[t] - raw[t])

4. unified_revin_rdn_hybrid
   center = beta * boundary_anchor + (1-beta) * instance_mean
   scale  = gamma * robust_tail_scale + (1-gamma) * instance_std

5. combination variants
   robust anchor, fixed level anchor, unified hybrid scale, and output blend
   ablations that test whether the three strongest LRBN components are
   complementary.

All parameters are learned only through the training split. Validation/test
targets are used only for reporting metrics. The exported JSONL schema matches
the HalluGuard external prediction contract.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPO_ROOT / "external" / "halluguard_real_pipeline" / "export_predictions.py"
DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Traffic")
MODELS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "TimeMixer")
HORIZONS = (96, 192, 336, 720)
VARIANTS = (
    "fixed_level_only",
    "learnable_robust_anchor",
    "learnable_residual_gate",
    "learnable_horizon_gate",
    "unified_revin_rdn_hybrid",
    "robust_unified_hybrid",
    "robust_unified_no_scale",
    "fixed_anchor_unified_scale",
    "fixed_hybrid_output_blend",
)


def load_exporter():
    spec = importlib.util.spec_from_file_location("halluguard_real_exporter", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load exporter from {EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = load_exporter()


@dataclass(frozen=True)
class Job:
    dataset: str
    backbone: str
    horizon: int
    variant: str

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.backbone}_{self.horizon}_{self.variant}"

    @property
    def model_label(self) -> str:
        return f"{self.backbone}+HalluGuard-LRBN-{self.variant}"


class RawWindowDataset(Dataset):
    def __init__(self, scaled_series: np.ndarray, starts: Iterable[int], seq_len: int, pred_len: int):
        self.scaled_series = scaled_series.astype(np.float32)
        self.starts = list(starts)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        x = self.scaled_series[start : start + self.seq_len]
        y = self.scaled_series[start + self.seq_len : start + self.seq_len + self.pred_len]
        return torch.from_numpy(x[:, None]), torch.from_numpy(y[:, None])


class FixedLevelOnly(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center = x[:, -1:, :].detach()
        return self.base(x - center) + center


class LearnableRobustAnchor(nn.Module):
    def __init__(self, base: nn.Module, tail_len: int, init_alpha: float = 0.8):
        super().__init__()
        self.base = base
        self.tail_len = int(tail_len)
        self.alpha_logit = nn.Parameter(logit_tensor(init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last, tail_median, _, _ = context_stats(x, self.tail_len)
        alpha = torch.sigmoid(self.alpha_logit)
        center = alpha * last + (1.0 - alpha) * tail_median
        return self.base(x - center) + center


class UnifiedRevINRDNHybrid(nn.Module):
    def __init__(self, base: nn.Module, tail_len: int, init_beta: float = 0.7, init_gamma: float = 0.35, eps: float = 1e-5):
        super().__init__()
        self.base = base
        self.tail_len = int(tail_len)
        self.beta_logit = nn.Parameter(logit_tensor(init_beta))
        self.gamma_logit = nn.Parameter(logit_tensor(init_gamma))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last, tail_median, robust_scale, instance = context_stats(x, self.tail_len, self.eps)
        instance_mean, instance_std = instance
        beta = torch.sigmoid(self.beta_logit)
        gamma = torch.sigmoid(self.gamma_logit)
        boundary_anchor = 0.85 * last + 0.15 * tail_median
        center = beta * boundary_anchor + (1.0 - beta) * instance_mean
        scale = gamma * robust_scale + (1.0 - gamma) * instance_std
        z = (x - center) / scale
        return self.base(z) * scale + center


class RobustUnifiedHybrid(nn.Module):
    """Unified RevIN-RDN hybrid with a learnable robust boundary anchor."""

    def __init__(
        self,
        base: nn.Module,
        tail_len: int,
        init_alpha: float = 0.85,
        init_beta: float = 0.7,
        init_gamma: float = 0.35,
        use_scale: bool = True,
        fixed_last_anchor: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.base = base
        self.tail_len = int(tail_len)
        self.use_scale = bool(use_scale)
        self.fixed_last_anchor = bool(fixed_last_anchor)
        if self.fixed_last_anchor:
            self.register_buffer("alpha_logit", logit_tensor(1.0))
        else:
            self.alpha_logit = nn.Parameter(logit_tensor(init_alpha))
        self.beta_logit = nn.Parameter(logit_tensor(init_beta))
        self.gamma_logit = nn.Parameter(logit_tensor(init_gamma))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last, tail_median, robust_scale, instance = context_stats(x, self.tail_len, self.eps)
        instance_mean, instance_std = instance
        if self.fixed_last_anchor:
            boundary_anchor = last
        else:
            alpha = torch.sigmoid(self.alpha_logit)
            boundary_anchor = alpha * last + (1.0 - alpha) * tail_median
        beta = torch.sigmoid(self.beta_logit)
        center = beta * boundary_anchor + (1.0 - beta) * instance_mean
        if self.use_scale:
            gamma = torch.sigmoid(self.gamma_logit)
            scale = gamma * robust_scale + (1.0 - gamma) * instance_std
        else:
            scale = torch.ones_like(center)
        z = (x - center) / scale
        return self.base(z) * scale + center


class FixedHybridOutputBlend(nn.Module):
    """Diagnostic only: learn whether hard level anchor complements hybrid output."""

    def __init__(self, fixed_base: nn.Module, hybrid_base: nn.Module, tail_len: int, init_blend: float = 0.35, eps: float = 1e-5):
        super().__init__()
        self.fixed = FixedLevelOnly(fixed_base)
        self.hybrid = RobustUnifiedHybrid(hybrid_base, tail_len, eps=eps)
        self.blend_logit = nn.Parameter(logit_tensor(init_blend))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fixed = self.fixed(x)
        hybrid = self.hybrid(x)
        blend = torch.sigmoid(self.blend_logit)
        return blend * fixed + (1.0 - blend) * hybrid


class LearnableHorizonGate(nn.Module):
    def __init__(self, raw_base: nn.Module, anchor_base: nn.Module, pred_len: int, init_gate: float = 0.5):
        super().__init__()
        self.raw_base = raw_base
        self.anchor_base = anchor_base
        self.horizon_logits = nn.Parameter(torch.full((1, int(pred_len), 1), float(logit(init_gate))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_base(x)
        center = x[:, -1:, :].detach()
        anchored = self.anchor_base(x - center) + center
        gate = torch.sigmoid(self.horizon_logits)
        return raw + gate * (anchored - raw)


class LearnableResidualGate(nn.Module):
    def __init__(self, raw_base: nn.Module, anchor_base: nn.Module, tail_len: int, pred_len: int, init_gate: float = 0.5):
        super().__init__()
        self.raw_base = raw_base
        self.anchor_base = anchor_base
        self.tail_len = int(tail_len)
        self.pred_len = int(pred_len)
        self.gate = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1))
        with torch.no_grad():
            self.gate[-1].bias.fill_(float(logit(init_gate)))
            self.gate[-1].weight.zero_()
        self.horizon_logits = nn.Parameter(torch.zeros(1, int(pred_len), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_base(x)
        last, tail_median, robust_scale, instance = context_stats(x, self.tail_len)
        instance_mean, instance_std = instance
        anchored = self.anchor_base(x - last.detach()) + last.detach()
        features = torch.cat(
            [
                torch.abs(last - tail_median) / instance_std,
                robust_scale / instance_std,
                torch.abs(last - instance_mean) / instance_std,
                torch.abs(x[:, -1:, :] - x[:, -2:-1, :]) / instance_std,
            ],
            dim=-1,
        ).squeeze(1)
        sample_gate = torch.sigmoid(self.gate(features)).view(-1, 1, 1)
        horizon_gate = torch.sigmoid(self.horizon_logits)
        gate = sample_gate * horizon_gate
        return raw + gate * (anchored - raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HalluGuard learnable reversible boundary-normalization ablations.")
    parser.add_argument("--datasets", default="ETTm1,ETTh1")
    parser.add_argument("--models", default="DLinear,PatchTST")
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--variants", default="fixed_level_only,learnable_robust_anchor,learnable_residual_gate,learnable_horizon_gate,unified_revin_rdn_hybrid")
    parser.add_argument("--data-root", type=Path, default=Path("external/ETDataset"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("baseline_predictions/halluguard_lrbn"))
    parser.add_argument("--raw-prediction-dir", type=Path, default=Path("baseline_predictions/halluguard_lrbn_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/halluguard_lrbn"))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--tail-len", type=int, default=48)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=4096)
    parser.add_argument("--max-eval-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-raw-baseline", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets, DATASETS, "dataset")
    models = parse_list(args.models, MODELS, "model")
    horizons = [int(v) for v in parse_list(args.horizons, [str(h) for h in HORIZONS], "horizon")]
    variants = parse_list(args.variants, VARIANTS, "variant")

    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    args.raw_prediction_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    raw_metrics = {}
    if not args.skip_raw_baseline:
        for dataset in datasets:
            for backbone in models:
                for horizon in horizons:
                    raw_path = args.raw_prediction_dir / f"{dataset}_{backbone}_{horizon}_raw_no_correction.jsonl"
                    try:
                        raw_manifest = run_raw_job(dataset, backbone, horizon, raw_path, args)
                        raw_metric = prediction_metric_row(raw_path)
                        raw_metrics[(dataset, backbone, horizon)] = raw_metric
                        records.append({**raw_manifest, **raw_metric, "variant": "raw_no_correction", "method": "raw_no_correction", "model_label": f"{backbone}+raw_no_correction", "mse_delta_pct_vs_raw": 0.0, "mae_delta_pct_vs_raw": 0.0, "status": "completed", "blocker_reason": ""})
                        print(json.dumps({"status": "completed", "job": f"{dataset}_{backbone}_{horizon}_raw", "mse": raw_metric["mse"]}), flush=True)
                    except Exception as exc:
                        records.append(blocked_record(dataset, backbone, horizon, "raw_no_correction", raw_path, "raw_no_correction", exc))
                        if not args.continue_on_error:
                            raise

    for job in [Job(d, m, h, v) for d in datasets for m in models for h in horizons for v in variants]:
        out_path = args.prediction_dir / f"{job.tag}.jsonl"
        try:
            manifest = run_job(job, out_path, args)
            metric = prediction_metric_row(out_path)
            deltas = raw_deltas(metric, raw_metrics.get((job.dataset, job.backbone, job.horizon)))
            records.append({**manifest, **metric, **deltas, "status": "completed", "blocker_reason": ""})
            print(json.dumps({"status": "completed", "job": job.tag, "mse": metric["mse"], "delta_vs_raw": deltas["mse_delta_pct_vs_raw"]}), flush=True)
        except Exception as exc:
            records.append(blocked_record(job.dataset, job.backbone, job.horizon, job.variant, out_path, "HalluGuard-LRBN", exc))
            print(json.dumps({"status": "blocked", "job": job.tag, "reason": f"{type(exc).__name__}: {exc}"}), flush=True)
            if not args.continue_on_error:
                raise

    write_csv(records, args.output_dir / "lrbn_metrics.csv")
    summary = summarize(records)
    write_csv(summary, args.output_dir / "lrbn_summary.csv")
    (args.output_dir / "lrbn_metrics.json").write_text(json.dumps({"rows": records, "summary": summary}, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(summary_md(records, summary, args), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "completed": sum(r["status"] == "completed" for r in records), "total": len(records)}))


def run_job(job: Job, out_path: Path, args: argparse.Namespace) -> dict:
    set_seed(args.seed + stable_offset(job.tag))
    device = exporter.choose_device(args.device)
    raw_series, scaler = exporter.load_series(job.dataset, args.data_root)
    scaled = scaler.transform(raw_series).astype(np.float32)
    train_starts = exporter.select_starts(exporter.split_starts(job.dataset, args.seq_len, job.horizon, "train"), args.max_train_windows)
    val_starts = exporter.select_starts(exporter.split_starts(job.dataset, args.seq_len, job.horizon, "val"), args.max_eval_windows)
    test_starts = exporter.select_starts(exporter.split_starts(job.dataset, args.seq_len, job.horizon, "test"), args.max_eval_windows)

    model = build_variant_model(job.variant, job.backbone, args.seq_len, job.horizon, args.tail_len, args.eps).to(device)
    train_model(model, scaled, train_starts, args.seq_len, job.horizon, args, device)
    samples = []
    samples.extend(export_split(model, raw_series, scaled, scaler, val_starts, args, job, "val", device))
    samples.extend(export_split(model, raw_series, scaled, scaler, test_starts, args, job, "test", device))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": job.dataset,
        "backbone": job.backbone,
        "horizon": job.horizon,
        "variant": job.variant,
        "method": "HalluGuard-LRBN",
        "model_label": job.model_label,
        "seq_len": args.seq_len,
        "tail_len": args.tail_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_windows": args.max_train_windows,
        "max_eval_windows": args.max_eval_windows,
        "device": str(device),
        "n_val": len(val_starts),
        "n_test": len(test_starts),
        "prediction_path": str(out_path),
        "adapter_mode": "learnable_reversible_boundary_normalization",
        "learned_params": learned_params(model),
        "test_threshold_leakage": False,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_raw_job(dataset: str, backbone: str, horizon: int, out_path: Path, args: argparse.Namespace) -> dict:
    tag = f"{dataset}_{backbone}_{horizon}_raw_no_correction"
    exporter.set_seed(args.seed + stable_offset(tag))
    device = exporter.choose_device(args.device)
    raw_series, scaler = exporter.load_series(dataset, args.data_root)
    scaled = scaler.transform(raw_series).astype(np.float32)
    train_starts = exporter.select_starts(exporter.split_starts(dataset, args.seq_len, horizon, "train"), args.max_train_windows)
    val_starts = exporter.select_starts(exporter.split_starts(dataset, args.seq_len, horizon, "val"), args.max_eval_windows)
    test_starts = exporter.select_starts(exporter.split_starts(dataset, args.seq_len, horizon, "test"), args.max_eval_windows)
    model = exporter.build_model(backbone, args.seq_len, horizon).to(device)
    exporter.train_model(model, scaled, train_starts, args.seq_len, horizon, args.epochs, args.batch_size, args.learning_rate, device)
    raw_args = SimpleNamespace(seq_len=args.seq_len, horizon=horizon, batch_size=args.batch_size, dataset=dataset, model=f"{backbone}+raw_no_correction")
    samples = []
    samples.extend(exporter.export_split(model, raw_series, scaled, scaler, val_starts, raw_args, "val", device))
    samples.extend(exporter.export_split(model, raw_series, scaled, scaler, test_starts, raw_args, "test", device))
    for sample in samples:
        sample["backbone"] = backbone
        sample["method"] = "raw_no_correction"
        sample["variant"] = "raw_no_correction"
        sample["adapter_mode"] = "raw_backbone"
        sample["test_threshold_leakage"] = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    manifest = {"dataset": dataset, "backbone": backbone, "horizon": horizon, "seq_len": args.seq_len, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "max_train_windows": args.max_train_windows, "max_eval_windows": args.max_eval_windows, "device": str(device), "n_val": len(val_starts), "n_test": len(test_starts), "prediction_path": str(out_path), "adapter_mode": "raw_backbone", "test_threshold_leakage": False}
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_variant_model(variant: str, backbone: str, seq_len: int, pred_len: int, tail_len: int, eps: float) -> nn.Module:
    if variant == "fixed_level_only":
        return FixedLevelOnly(exporter.build_model(backbone, seq_len, pred_len))
    if variant == "learnable_robust_anchor":
        return LearnableRobustAnchor(exporter.build_model(backbone, seq_len, pred_len), tail_len)
    if variant == "learnable_residual_gate":
        return LearnableResidualGate(exporter.build_model(backbone, seq_len, pred_len), exporter.build_model(backbone, seq_len, pred_len), tail_len, pred_len)
    if variant == "learnable_horizon_gate":
        return LearnableHorizonGate(exporter.build_model(backbone, seq_len, pred_len), exporter.build_model(backbone, seq_len, pred_len), pred_len)
    if variant == "unified_revin_rdn_hybrid":
        return UnifiedRevINRDNHybrid(exporter.build_model(backbone, seq_len, pred_len), tail_len, eps=eps)
    if variant == "robust_unified_hybrid":
        return RobustUnifiedHybrid(exporter.build_model(backbone, seq_len, pred_len), tail_len, eps=eps)
    if variant == "robust_unified_no_scale":
        return RobustUnifiedHybrid(exporter.build_model(backbone, seq_len, pred_len), tail_len, use_scale=False, eps=eps)
    if variant == "fixed_anchor_unified_scale":
        return RobustUnifiedHybrid(exporter.build_model(backbone, seq_len, pred_len), tail_len, fixed_last_anchor=True, eps=eps)
    if variant == "fixed_hybrid_output_blend":
        return FixedHybridOutputBlend(exporter.build_model(backbone, seq_len, pred_len), exporter.build_model(backbone, seq_len, pred_len), tail_len, eps=eps)
    raise ValueError(f"Unknown variant: {variant}")


def train_model(model: nn.Module, scaled: np.ndarray, starts: List[int], seq_len: int, pred_len: int, args: argparse.Namespace, device: torch.device) -> None:
    dataset = RawWindowDataset(scaled, starts, seq_len, pred_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(args.epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()


def export_split(model: nn.Module, raw_series: np.ndarray, scaled: np.ndarray, scaler, starts: List[int], args: argparse.Namespace, job: Job, split: str, device: torch.device) -> List[dict]:
    dataset = RawWindowDataset(scaled, starts, args.seq_len, job.horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    samples = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for x, _ in loader:
            pred = model(x.to(device)).detach().cpu().numpy()[:, :, 0]
            pred = scaler.inverse(pred)
            batch_size = pred.shape[0]
            for b in range(batch_size):
                start = starts[offset + b]
                context = raw_series[start : start + args.seq_len].astype(float)
                target = raw_series[start + args.seq_len : start + args.seq_len + job.horizon].astype(float)
                samples.append(
                    {
                        "sample_id": f"{split}_{offset + b:05d}",
                        "dataset": job.dataset,
                        "model": job.model_label,
                        "split": split,
                        "context": exporter.round_list(context),
                        "prediction": exporter.round_list(pred[b]),
                        "target": exporter.round_list(target),
                        "backbone": job.backbone,
                        "method": "HalluGuard-LRBN",
                        "variant": job.variant,
                        "adapter_mode": "learnable_reversible_boundary_normalization",
                        "normalizer_fit": "train_split_only",
                        "test_threshold_leakage": False,
                    }
                )
            offset += batch_size
    return samples


def context_stats(x: torch.Tensor, tail_len: int, eps: float = 1e-5):
    tail_len = max(4, min(int(tail_len), x.shape[1]))
    tail = x[:, -tail_len:, :]
    last = x[:, -1:, :].detach()
    tail_median = tail.median(dim=1, keepdim=True).values.detach()
    mad = torch.median(torch.abs(tail - tail_median), dim=1, keepdim=True).values
    robust_scale = (1.4826 * mad + eps).detach()
    instance_mean = x.mean(dim=1, keepdim=True).detach()
    instance_std = torch.sqrt(torch.var(x - instance_mean, dim=1, keepdim=True, unbiased=False) + eps).detach()
    return last, tail_median, robust_scale, (instance_mean, instance_std)


def prediction_metric_row(path: Path) -> dict:
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test = [s for s in samples if s.get("split") == "test"]
    if not test:
        raise ValueError(f"{path} has no test rows.")
    preds = np.asarray([s["prediction"] for s in test], dtype=float)
    targets = np.asarray([s["target"] for s in test], dtype=float)
    return {"mse": float(np.mean((preds - targets) ** 2)), "mae": float(np.mean(np.abs(preds - targets))), "n_test": len(test)}


def raw_deltas(metric: dict, raw_metric: dict | None) -> dict:
    if not raw_metric:
        return {"mse_delta_pct_vs_raw": "", "mae_delta_pct_vs_raw": ""}
    return {"mse_delta_pct_vs_raw": pct_delta(float(metric["mse"]), float(raw_metric["mse"])), "mae_delta_pct_vs_raw": pct_delta(float(metric["mae"]), float(raw_metric["mae"]))}


def pct_delta(value: float, baseline: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    return float(100.0 * (value - baseline) / baseline)


def summarize(rows: List[dict]) -> List[dict]:
    out = []
    keys = sorted({(str(r.get("variant", "")), str(r.get("backbone", ""))) for r in rows})
    for variant, backbone in keys:
        selected = [r for r in rows if r.get("variant") == variant and r.get("backbone") == backbone]
        completed = [r for r in selected if r.get("status") == "completed"]
        out.append(summary_row(variant, backbone, selected, completed))
    for variant in sorted({str(r.get("variant", "")) for r in rows}):
        selected = [r for r in rows if r.get("variant") == variant]
        completed = [r for r in selected if r.get("status") == "completed"]
        out.append(summary_row(variant, "ALL", selected, completed))
    return out


def summary_row(variant: str, backbone: str, selected: List[dict], completed: List[dict]) -> dict:
    return {
        "variant": variant,
        "backbone": backbone,
        "completed_rows": len(completed),
        "total_rows": len(selected),
        "mean_mse": mean(float(r["mse"]) for r in completed) if completed else "",
        "mean_mae": mean(float(r["mae"]) for r in completed) if completed else "",
        "mean_mse_delta_pct_vs_raw": safe_mean_delta(completed, "mse_delta_pct_vs_raw"),
        "mean_mae_delta_pct_vs_raw": safe_mean_delta(completed, "mae_delta_pct_vs_raw"),
        "blocked_rows": len(selected) - len(completed),
    }


def summary_md(rows: List[dict], summary: List[dict], args: argparse.Namespace) -> str:
    completed = sum(r.get("status") == "completed" for r in rows)
    lines = [
        "# HalluGuard-LRBN Summary",
        "",
        "Learnable reversible boundary normalization ablations for robust anchor, residual gate, horizon gate, and unified RevIN-RDN hybrid.",
        "",
        f"- Completed rows: {completed} / {len(rows)}",
        "- Test threshold leakage: False",
        f"- seq_len: {args.seq_len}",
        f"- tail_len: {args.tail_len}",
        "",
        "## Variant Summary",
        "",
    ]
    for row in summary:
        lines.append(f"- `{row['variant']}` / `{row['backbone']}`: completed {row['completed_rows']} / {row['total_rows']}, mean MSE {row['mean_mse']}, mean MAE {row['mean_mae']}, mean MSE delta vs raw {row['mean_mse_delta_pct_vs_raw']}, blocked {row['blocked_rows']}")
    blocked = [r for r in rows if r.get("status") != "completed"]
    if blocked:
        lines.extend(["", "## Blocked Rows", ""])
        for row in blocked[:100]:
            lines.append(f"- {row.get('dataset')} {row.get('backbone')} {row.get('horizon')} `{row.get('variant')}`: {row.get('blocker_reason')}")
    return "\n".join(lines) + "\n"


def blocked_record(dataset: str, backbone: str, horizon: int, variant: str, path: Path, method: str, exc: Exception) -> dict:
    return {"dataset": dataset, "backbone": backbone, "horizon": horizon, "variant": variant, "method": method, "model_label": f"{backbone}+{method}-{variant}", "status": "blocked", "mse": "", "mae": "", "mse_delta_pct_vs_raw": "", "mae_delta_pct_vs_raw": "", "prediction_path": str(path), "blocker_reason": f"{type(exc).__name__}: {exc}", "adapter_mode": "learnable_reversible_boundary_normalization", "test_threshold_leakage": False}


def learned_params(model: nn.Module) -> str:
    values = {}
    for name, param in model.named_parameters():
        if name.endswith("alpha_logit"):
            values[name.replace("_logit", "")] = float(torch.sigmoid(param.detach()).mean().cpu())
        elif name.endswith("beta_logit"):
            values[name.replace("_logit", "")] = float(torch.sigmoid(param.detach()).mean().cpu())
        elif name.endswith("gamma_logit"):
            values[name.replace("_logit", "")] = float(torch.sigmoid(param.detach()).mean().cpu())
        elif name.endswith("blend_logit"):
            values[name.replace("_logit", "")] = float(torch.sigmoid(param.detach()).mean().cpu())
        elif name.endswith("horizon_logits"):
            gate = torch.sigmoid(param.detach()).cpu()
            values[name] = {
                "mean": float(gate.mean()),
                "first": float(gate[:, 0, :].mean()),
                "last": float(gate[:, -1, :].mean()),
            }
    return json.dumps(values, sort_keys=True)


def parse_list(raw: str, allowed: Sequence[str], name: str) -> List[str]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise SystemExit(f"Unknown {name}(s): {', '.join(unknown)}. Allowed: {', '.join(map(str, allowed))}")
    return values


def stable_offset(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return float(np.log(value / (1.0 - value)))


def logit_tensor(value: float) -> torch.Tensor:
    return torch.tensor(float(logit(value)), dtype=torch.float32)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def safe_mean_delta(rows: List[dict], key: str):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return mean(values) if values else ""


def write_csv(rows: List[dict], path: Path) -> None:
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


if __name__ == "__main__":
    main()
