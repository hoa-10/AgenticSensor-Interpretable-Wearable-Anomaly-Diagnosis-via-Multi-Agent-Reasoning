import json
import configparser
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis as scipy_kurtosis
from torch import nn

from perception_layer.timing import record_timing

try:
    from classification.train_har import MHealthActivityCNN
except Exception:
    MHealthActivityCNN = None


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HAR_MODEL = PROJECT_ROOT / "classification" / "classification" / "har_model.pt"
DEFAULT_FAULT_MODEL_V2 = PROJECT_ROOT / "classification" / "classification" / "synthetic_fault_detector_v2.pt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "perception_layer" / "perception.config"
DEFAULT_SAMPLE_RATE = 50.0
DEFAULT_LOWPASS_HZ = 20.0

SENSOR_COLUMNS = [
    "chest_acc_x",
    "chest_acc_y",
    "chest_acc_z",
    "ankle_acc_x",
    "ankle_acc_y",
    "ankle_acc_z",
    "arm_acc_x",
    "arm_acc_y",
    "arm_acc_z",
]


class HARModel(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, model_size: str = "large") -> None:
        super().__init__()
        if model_size == "small":
            c1, c2, c3, hidden, dropout = 48, 96, 96, 96, 0.2
        elif model_size == "medium":
            c1, c2, c3, hidden, dropout = 64, 128, 128, 128, 0.25
        else:
            c1, c2, c3, hidden, dropout = 96, 192, 192, 160, 0.3

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=c3, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden * 2, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        weights = torch.softmax(self.attention(x), dim=1)
        return self.classifier((x * weights).sum(dim=1))


class FaultCNN(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 48, kernel_size=9, padding=4),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 96, kernel_size=7, padding=3),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.transpose(1, 2))).squeeze(1)


# ---------------------------------------------------------------------------
# V2 model architecture: MultiScaleCNN + BiLSTM + ChannelAttention + Stats
# ---------------------------------------------------------------------------
_STATS_PER_CHANNEL = 8


class _MultiScaleCNN(nn.Module):
    def __init__(self, in_channels: int, out_per_branch: int = 48, dropout: float = 0.15) -> None:
        super().__init__()
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_per_branch), nn.GELU(),
        )
        self.branch_medium = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_per_branch), nn.GELU(),
        )
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_per_branch), nn.GELU(),
        )
        total = out_per_branch * 3
        self.merge = nn.Sequential(
            nn.Conv1d(total, total, kernel_size=5, padding=2),
            nn.BatchNorm1d(total), nn.GELU(), nn.Dropout(dropout), nn.MaxPool1d(2),
            nn.Conv1d(total, total, kernel_size=3, padding=1),
            nn.BatchNorm1d(total), nn.GELU(), nn.Dropout(dropout), nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat([self.branch_small(x), self.branch_medium(x), self.branch_large(x)], dim=1))


class _ChannelAttention(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1)
        return x * self.attn(avg).unsqueeze(1)


class MultiTaskFaultDetector(nn.Module):
    def __init__(
        self, n_channels: int = 9, stats_dim: int = 72,
        cnn_out: int = 48, lstm_hidden: int = 96, lstm_layers: int = 2,
        fusion_dim: int = 256, n_fault_classes: int = 11, dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cnn = _MultiScaleCNN(n_channels * 2, cnn_out, dropout=dropout)
        cnn_total = cnn_out * 3
        self.lstm = nn.LSTM(
            input_size=cnn_total, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * 2
        self.channel_attn = _ChannelAttention(lstm_out)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64),
        )
        fused = lstm_out + 64
        self.fusion = nn.Sequential(
            nn.Linear(fused, fusion_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.binary_head = nn.Linear(fusion_dim, 1)
        self.type_head = nn.Linear(fusion_dim, n_fault_classes)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nan_mask = torch.isnan(x)
        x_clean = x.clone()
        x_clean[nan_mask] = 0.0
        x_aug = torch.cat([x_clean, nan_mask.float()], dim=2)
        cnn_out = self.cnn(x_aug.transpose(1, 2))
        lstm_out, _ = self.lstm(cnn_out.transpose(1, 2))
        attended = self.channel_attn(lstm_out)
        deep_vec = self.temporal_pool(attended.transpose(1, 2)).squeeze(2)
        stats_vec = self.stats_proj(stats)
        fused = self.fusion(torch.cat([deep_vec, stats_vec], dim=1))
        return self.binary_head(fused).squeeze(1), self.type_head(fused)


def _compute_statistical_features(window: np.ndarray) -> np.ndarray:
    """Compute per-channel stats from a single window [T, C]. Returns [C * 8]."""
    T, C = window.shape
    half = T // 2
    feats = np.zeros(C * _STATS_PER_CHANNEL, dtype=np.float32)
    for c in range(C):
        col = window[:, c]
        finite_mask = np.isfinite(col)
        finite_vals = col[finite_mask]
        n_finite = len(finite_vals)
        offset = c * _STATS_PER_CHANNEL
        feats[offset + 0] = 1.0 - (n_finite / max(T, 1))
        if n_finite < 2:
            continue
        diffs = np.abs(np.diff(finite_vals))
        feats[offset + 1] = float(np.mean(diffs < 1e-9))
        
        first_half = col[:half]
        second_half = col[half:]
        fh_finite = first_half[np.isfinite(first_half)]
        sh_finite = second_half[np.isfinite(second_half)]
        std1 = float(np.std(fh_finite)) if len(fh_finite) > 1 else 0.0
        std2 = float(np.std(sh_finite)) if len(sh_finite) > 1 else 0.0
        feats[offset + 2] = std2 / max(std1, 1e-6)
        
        if n_finite >= 4:
            kurt = float(scipy_kurtosis(finite_vals, fisher=True, nan_policy="omit"))
            feats[offset + 3] = np.clip(kurt, -10.0, 50.0) if np.isfinite(kurt) else 0.0
            
        fmin, fmax = float(np.min(finite_vals)), float(np.max(finite_vals))
        if fmax - fmin > 1e-6:
            at_bounds = np.sum((np.abs(finite_vals - fmin) < 1e-6) | (np.abs(finite_vals - fmax) < 1e-6))
            feats[offset + 4] = float(at_bounds) / n_finite
        else:
            feats[offset + 4] = 1.0
            
        mean_fh = float(np.mean(fh_finite)) if len(fh_finite) > 0 else 0.0
        mean_sh = float(np.mean(sh_finite)) if len(sh_finite) > 0 else 0.0
        feats[offset + 5] = abs(mean_sh - mean_fh)
        
        feats[offset + 6] = float(np.sqrt(np.mean(finite_vals ** 2)))
        feats[offset + 7] = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


_HAR_CACHE: dict[str, tuple[nn.Module, dict[str, Any]]] = {}
_FAULT_CACHE: dict[str, tuple[nn.Module, dict[str, Any]]] = {}


def _to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _default_parquet_path() -> Path:
    config = configparser.ConfigParser()
    config.read(DEFAULT_CONFIG_PATH, encoding="utf-8")
    for key in ("model_parquet_path", "dataset_path", "mhealth_path"):
        value = config.get("paths", key, fallback="").strip()
        if value:
            return _resolve_path(value)
    return PROJECT_ROOT / "mhealth.parquet"


def _load_json(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc


def _window_from_json(window_json: str, sensor_columns: list[str] | None = None) -> np.ndarray:
    expected_channels = len(sensor_columns or SENSOR_COLUMNS)
    data = _load_json(window_json)
    if isinstance(data, dict):
        data = data.get("values", data.get("window", data.get("samples")))
    if data is None:
        raise ValueError("window_json must contain values/window/samples.")

    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Window must be 2D: [time, {expected_channels}] or [{expected_channels}, time].")
    if values.shape[1] == expected_channels:
        return values
    if values.shape[0] == expected_channels:
        return values.T
    raise ValueError(f"Expected {expected_channels} sensor channels, got shape {values.shape}.")


def _column_alias(column: str) -> str:
    aliases = {
        "ankle_acc_x": "left_ankle_acc_x",
        "ankle_acc_y": "left_ankle_acc_y",
        "ankle_acc_z": "left_ankle_acc_z",
        "arm_acc_x": "right_lower_arm_acc_x",
        "arm_acc_y": "right_lower_arm_acc_y",
        "arm_acc_z": "right_lower_arm_acc_z",
    }
    return aliases.get(column, column)


def _window_from_parquet(
    parquet_path: str,
    start_row: int,
    end_row: int,
    sensor_columns: list[str] | None = None,
) -> np.ndarray:
    import pandas as pd

    path = _resolve_path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    requested_columns = list(sensor_columns or SENSOR_COLUMNS)
    df_columns = pd.read_parquet(path).columns
    parquet_columns = [
        column if column in df_columns else _column_alias(column)
        for column in requested_columns
    ]
    missing = [column for column in parquet_columns if column not in df_columns]
    if missing:
        raise ValueError(f"Missing sensor columns in parquet: {missing}")
    df = pd.read_parquet(path, columns=parquet_columns)
    start_row = max(0, int(start_row))
    end_row = len(df) if end_row < 0 else min(len(df), int(end_row))
    if end_row <= start_row:
        raise ValueError("Selected parquet window is empty.")
    return df[parquet_columns].iloc[start_row:end_row].to_numpy(dtype=np.float32)


def _read_window(
    window_json: str,
    parquet_path: str,
    start_row: int,
    end_row: int,
    sensor_columns: list[str] | None = None,
) -> np.ndarray:
    if window_json:
        return _window_from_json(window_json, sensor_columns)
    return _window_from_parquet(
        str(_resolve_path(parquet_path) if parquet_path else _default_parquet_path()),
        start_row,
        end_row,
        sensor_columns,
    )


def _filter(values: np.ndarray, sample_rate: float, lowpass_hz: float) -> np.ndarray:
    values = median_filter(values, size=(3, 1), mode="nearest")
    nyquist = sample_rate / 2.0
    cutoff = min(lowpass_hz, nyquist * 0.95)
    sos = butter(3, cutoff / nyquist, btype="lowpass", output="sos")
    if len(values) <= 18:
        return values.astype(np.float32)
    return sosfiltfilt(sos, values, axis=0).astype(np.float32)


def _fit_length(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) == target_len:
        return values.astype(np.float32)
    if len(values) > target_len:
        return values[:target_len].astype(np.float32)
    pad = np.repeat(values[-1:, :], target_len - len(values), axis=0)
    return np.concatenate([values, pad], axis=0).astype(np.float32)


def _normalize(values: np.ndarray, ckpt: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(ckpt.get("normalization_mean", ckpt.get("mean")), dtype=np.float32)
    std = np.asarray(ckpt.get("normalization_std", ckpt.get("std")), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return ((values - mean) / std).astype(np.float32)


def _load_har(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    key = str(path)
    if key in _HAR_CACHE:
        return _HAR_CACHE[key]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    labels = [int(x) for x in ckpt["label_values"]]
    sensor_columns = list(ckpt.get("sensor_columns", SENSOR_COLUMNS))
    if MHealthActivityCNN is not None and ("sensor_columns" in ckpt or "mean" in ckpt):
        model = MHealthActivityCNN(len(sensor_columns), len(labels))
    else:
        model = HARModel(len(sensor_columns), len(labels), ckpt.get("model_size", "large"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _HAR_CACHE[key] = (model, ckpt)
    return model, ckpt


def _load_fault(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    key = str(path)
    if key in _FAULT_CACHE:
        return _FAULT_CACHE[key]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "FaultCNN")
    if model_type == "MultiTaskFaultDetector_v2":
        n_ch = int(ckpt.get("n_channels", len(SENSOR_COLUMNS)))
        stats_dim = int(ckpt.get("stats_dim", n_ch * _STATS_PER_CHANNEL))
        model = MultiTaskFaultDetector(n_channels=n_ch, stats_dim=stats_dim)
    else:
        model = FaultCNN(len(SENSOR_COLUMNS))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _FAULT_CACHE[key] = (model, ckpt)
    return model, ckpt


def _predict_activity(
    window_json: str = "",
    parquet_path: str = "",
    start_row: int = 0,
    end_row: int = -1,
    model_path: str = "",
    confidence_threshold: float = 0.8,
) -> str:
    """Predict activity for start_row:end_row from the configured HAR parquet."""
    started_at = time.perf_counter()
    path = _resolve_path(model_path) if model_path else DEFAULT_HAR_MODEL
    model, ckpt = _load_har(path)
    window_size = int(ckpt.get("window_size", 128))
    sample_rate = float(ckpt.get("sample_rate", ckpt.get("sample_rate_hz", DEFAULT_SAMPLE_RATE)))
    lowpass_hz = float(ckpt.get("lowpass_hz", DEFAULT_LOWPASS_HZ))
    labels = [int(x) for x in ckpt["label_values"]]
    sensor_columns = list(ckpt.get("sensor_columns", SENSOR_COLUMNS))
    use_signal_filter = bool(ckpt.get("use_signal_filter", True))

    values = _read_window(window_json, parquet_path, start_row, end_row, sensor_columns)
    if use_signal_filter:
        values = _filter(values, sample_rate, lowpass_hz)
    values = _fit_length(values, window_size)
    x = torch.from_numpy(_normalize(values, ckpt)).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).numpy()
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    top3_idx = np.argsort(probs)[::-1][:3]

    result = {
        "task": "activity_classification",
        "predicted_label": labels[pred_idx],
        "confidence": confidence,
        "is_low_confidence": confidence < confidence_threshold,
        "top_predictions": [
            {"label": labels[int(i)], "probability": float(probs[int(i)])}
            for i in top3_idx
        ],
        "window_shape": list(values.shape),
        "model_path": str(path),
    }
    record_timing(
        "thread_1.predict_activity",
        time.perf_counter() - started_at,
        start_row=int(start_row),
        end_row=int(end_row),
        model_path=str(path),
    )
    return _to_json(result)


def _predict_sensor_anomaly(
    window_json: str = "",
    parquet_path: str = "",
    start_row: int = 0,
    end_row: int = -1,
    model_path: str = "",
) -> str:
    """Predict normal/anomaly for start_row:end_row from the configured 9-channel parquet."""
    started_at = time.perf_counter()
    # Prefer v2 model if available, fall back to v1
    if model_path:
        path = _resolve_path(model_path)
    elif DEFAULT_FAULT_MODEL_V2.exists():
        path = DEFAULT_FAULT_MODEL_V2
    else:
        path = DEFAULT_FAULT_MODEL

    model, ckpt = _load_fault(path)
    threshold = min(float(ckpt.get("threshold", 0.6)), 0.6)
    window_size = int(ckpt.get("window_size", 512))
    is_v2 = ckpt.get("model_type") == "MultiTaskFaultDetector_v2"

    values = _read_window(window_json, parquet_path, start_row, end_row)
    values_raw = _fit_length(values, window_size)  # keep NaN for stats
    values_filtered = _fit_length(
        _filter(np.nan_to_num(values, nan=0.0), DEFAULT_SAMPLE_RATE, DEFAULT_LOWPASS_HZ),
        window_size,
    )

    if is_v2:
        # Compute statistical features on raw (pre-filter) data
        stats_raw = _compute_statistical_features(values_raw)
        stats_mean = np.asarray(ckpt["stats_normalization_mean"], dtype=np.float32)
        stats_std = np.asarray(ckpt["stats_normalization_std"], dtype=np.float32)
        stats_std = np.where(stats_std < 1e-6, 1.0, stats_std)
        stats_norm = ((stats_raw - stats_mean) / stats_std).astype(np.float32)
        stats_t = torch.from_numpy(stats_norm).unsqueeze(0)

        # Normalize signal, then restore NaN positions for NaN indicator
        normed = _normalize(values_filtered, ckpt)
        nan_positions = np.isnan(values_raw)
        normed[nan_positions] = np.nan
        x = torch.from_numpy(normed).unsqueeze(0)

        with torch.no_grad():
            bin_logits, _ = model(x, stats_t)
            probability = float(torch.sigmoid(bin_logits).item())
    else:
        # Legacy FaultCNN path
        x = torch.from_numpy(_normalize(values_filtered, ckpt)).unsqueeze(0)
        with torch.no_grad():
            probability = float(torch.sigmoid(model(x)).item())

    result = {
        "task": "sensor_anomaly_detection",
        "predicted_state": "anomaly" if probability >= threshold else "normal",
        "is_anomaly": probability >= threshold,
        "prob_fault": probability,
        "threshold": threshold,
        "window_shape": list(values_filtered.shape),
        "model_path": str(path),
    }
    record_timing(
        "thread_1.predict_sensor_anomaly",
        time.perf_counter() - started_at,
        start_row=int(start_row),
        end_row=int(end_row),
        model_path=str(path),
    )
    return _to_json(result)
