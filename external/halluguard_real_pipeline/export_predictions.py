"""Train lightweight real-data forecasters and export HalluGuard JSONL.

This script intentionally lives under external/ so HalluGuard core evaluation stays
model-agnostic. It uses public ETT CSV files and exports only:

sample_id, dataset, model, split, context, prediction, target
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
TSLIB_ROOT = REPO_ROOT / "external" / "Time-Series-Library"
if str(TSLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(TSLIB_ROOT))

STANDARD_BORDERS = {
    "ETTh1": (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24),
    "ETTh2": (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24),
    "ETTm1": (12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4),
    "ETTm2": (12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4),
}
CUSTOM_DATASETS = {
    "Weather": ("weather", "weather.csv"),
    "ECL": ("electricity", "electricity.csv"),
    "Electricity": ("electricity", "electricity.csv"),
    "Traffic": ("traffic", "traffic.csv"),
}
SUPPORTED_DATASETS = tuple(STANDARD_BORDERS) + ("Weather", "ECL", "Traffic")
SUPPORTED_MODELS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "TimeMixer")
DATA_LENGTHS: dict[str, int] = {}


@dataclass
class Scaler:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class WindowDataset(Dataset):
    def __init__(self, series: np.ndarray, starts: Iterable[int], seq_len: int, pred_len: int):
        self.series = series.astype(np.float32)
        self.starts = list(starts)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        x = self.series[start : start + self.seq_len]
        y = self.series[start + self.seq_len : start + self.seq_len + self.pred_len]
        return torch.from_numpy(x[:, None]), torch.from_numpy(y[:, None])


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)


class TinyDLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, moving_avg: int = 25):
        super().__init__()
        self.decomp = MovingAverage(moving_avg)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        nn.init.constant_(self.linear_seasonal.weight, 1.0 / seq_len)
        nn.init.constant_(self.linear_trend.weight, 1.0 / seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        seasonal = x - trend
        seasonal = self.linear_seasonal(seasonal.squeeze(-1))
        trend = self.linear_trend(trend.squeeze(-1))
        return (seasonal + trend).unsqueeze(-1)


class TinyPatchTST(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 32,
        n_heads: int = 4,
        e_layers: int = 1,
        d_ff: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding = int(stride)
        self.patch_num = int((seq_len - patch_len) / stride + 2)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Dropout(dropout), nn.Linear(self.patch_num * d_model, pred_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        z = (centered / stdev).squeeze(-1)
        z = torch.nn.functional.pad(z, (0, self.padding), mode="replicate")
        patches = z.unfold(dimension=1, size=self.patch_len, step=self.stride)
        tokens = self.patch_embedding(patches)
        encoded = self.encoder(tokens)
        out = self.head(encoded).unsqueeze(-1)
        return out * stdev[:, 0:1, :] + means[:, 0:1, :]


class TSLibForecastWrapper(nn.Module):
    """Wrap public Time-Series-Library models into the local x->[y] contract."""

    def __init__(self, model_name: str, seq_len: int, pred_len: int):
        super().__init__()
        module = importlib.import_module(f"models.{model_name}")
        cfg = tslib_config(seq_len, pred_len)
        self.model = module.Model(cfg)
        self.pred_len = int(pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Some official TSLib models use in-place input normalization. That is
        # fine for ordinary forecaster training, but LRBN wraps the backbone
        # with learnable input normalization, making x require gradients. Use a
        # detached backbone input in that case while keeping backbone parameter
        # gradients and LRBN output denormalization gradients intact.
        x_model = x.detach().clone() if x.requires_grad else x
        out = self.model(x_model, None, None, None)
        return out[:, -self.pred_len :, :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export real ETT predictions for HalluGuard.")
    parser.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--horizon", required=True, type=int, choices=[96, 192, 336, 720])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=4096)
    parser.add_argument("--max-eval-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    raw_series, scaler = load_series(args.dataset, args.data_root)
    scaled = scaler.transform(raw_series).astype(np.float32)
    train_starts = select_starts(split_starts(args.dataset, args.seq_len, args.horizon, "train"), args.max_train_windows)
    val_starts = select_starts(split_starts(args.dataset, args.seq_len, args.horizon, "val"), args.max_eval_windows)
    test_starts = select_starts(split_starts(args.dataset, args.seq_len, args.horizon, "test"), args.max_eval_windows)

    model = build_model(args.model, args.seq_len, args.horizon).to(device)
    train_model(model, scaled, train_starts, args.seq_len, args.horizon, args.epochs, args.batch_size, args.learning_rate, device)
    samples = []
    samples.extend(export_split(model, raw_series, scaled, scaler, val_starts, args, "val", device))
    samples.extend(export_split(model, raw_series, scaled, scaler, test_starts, args, "test", device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    manifest = {
        "dataset": args.dataset,
        "model": args.model,
        "horizon": args.horizon,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_windows": args.max_train_windows,
        "max_eval_windows": args.max_eval_windows,
        "device": str(device),
        "n_val": len(val_starts),
        "n_test": len(test_starts),
        "output": str(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_series(dataset: str, data_root: Path) -> Tuple[np.ndarray, Scaler]:
    dataset = normalize_dataset_name(dataset)
    path = dataset_path(dataset, data_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    frame = pd.read_csv(path)
    target = target_column(frame)
    series = frame[target].to_numpy(dtype=np.float32)
    DATA_LENGTHS[dataset] = len(series)
    train_len, _, _ = split_lengths(dataset, len(series))
    train = series[:train_len]
    scaler = Scaler(float(train.mean()), float(train.std() + 1e-6))
    return series, scaler


def split_starts(dataset: str, seq_len: int, pred_len: int, split: str) -> List[int]:
    dataset = normalize_dataset_name(dataset)
    train_len, val_len, test_len = split_lengths(dataset, DATA_LENGTHS.get(dataset))
    if split == "train":
        left, right = 0, train_len
    elif split == "val":
        left, right = train_len - seq_len, train_len + val_len
    elif split == "test":
        left, right = train_len + val_len - seq_len, train_len + val_len + test_len
    else:
        raise ValueError(split)
    last_start = right - seq_len - pred_len
    if last_start < left:
        return []
    return list(range(left, last_start + 1))


def select_starts(starts: List[int], max_count: int) -> List[int]:
    if max_count <= 0 or len(starts) <= max_count:
        return starts
    indices = np.linspace(0, len(starts) - 1, max_count).round().astype(int)
    return [starts[int(i)] for i in indices]


def build_model(model_name: str, seq_len: int, pred_len: int) -> nn.Module:
    if model_name == "DLinear":
        return TinyDLinear(seq_len, pred_len)
    if model_name == "PatchTST":
        return TinyPatchTST(seq_len, pred_len)
    if model_name in {"iTransformer", "TimesNet", "TimeMixer"}:
        return TSLibForecastWrapper(model_name, seq_len, pred_len)
    raise ValueError(model_name)


def normalize_dataset_name(dataset: str) -> str:
    return "ECL" if dataset == "Electricity" else dataset


def dataset_path(dataset: str, data_root: Path) -> Path:
    if dataset in STANDARD_BORDERS:
        return data_root / "ETT-small" / f"{dataset}.csv"
    subdir, filename = CUSTOM_DATASETS[dataset]
    return TSLIB_ROOT / "dataset" / subdir / filename


def target_column(frame: pd.DataFrame) -> str:
    if "OT" in frame.columns:
        return "OT"
    numeric_cols = [c for c in frame.columns if c != "date" and pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric_cols:
        raise ValueError("No numeric target column found")
    return numeric_cols[-1]


def split_lengths(dataset: str, total_len: int | None = None) -> Tuple[int, int, int]:
    if dataset in STANDARD_BORDERS:
        return STANDARD_BORDERS[dataset]
    if total_len is None:
        raise ValueError(f"Need loaded dataset length before splitting {dataset}")
    train_len = int(total_len * 0.7)
    test_len = int(total_len * 0.2)
    val_len = total_len - train_len - test_len
    return train_len, val_len, test_len


def tslib_config(seq_len: int, pred_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=int(seq_len),
        label_len=max(1, int(seq_len) // 2),
        pred_len=int(pred_len),
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=64,
        factor=3,
        dropout=0.05,
        activation="gelu",
        embed="fixed",
        freq="h",
        top_k=2,
        num_kernels=3,
        moving_avg=25,
        down_sampling_layers=1,
        down_sampling_window=2,
        down_sampling_method="avg",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        output_attention=False,
    )


def train_model(
    model: nn.Module,
    scaled: np.ndarray,
    starts: List[int],
    seq_len: int,
    pred_len: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    dataset = WindowDataset(scaled, starts, seq_len, pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()


def export_split(
    model: nn.Module,
    raw_series: np.ndarray,
    scaled: np.ndarray,
    scaler: Scaler,
    starts: List[int],
    args: argparse.Namespace,
    split: str,
    device: torch.device,
) -> List[dict]:
    dataset = WindowDataset(scaled, starts, args.seq_len, args.horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    samples = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).detach().cpu().numpy()[:, :, 0]
            pred = scaler.inverse(pred)
            batch_size = pred.shape[0]
            for b in range(batch_size):
                start = starts[offset + b]
                context = raw_series[start : start + args.seq_len].astype(float)
                target = raw_series[start + args.seq_len : start + args.seq_len + args.horizon].astype(float)
                samples.append(
                    {
                        "sample_id": f"{split}_{offset + b:05d}",
                        "dataset": args.dataset,
                        "model": args.model,
                        "split": split,
                        "context": round_list(context),
                        "prediction": round_list(pred[b]),
                        "target": round_list(target),
                    }
                )
            offset += batch_size
    return samples


def round_list(values: Iterable[float]) -> List[float]:
    return [float(f"{float(v):.8g}") for v in values]


if __name__ == "__main__":
    main()
