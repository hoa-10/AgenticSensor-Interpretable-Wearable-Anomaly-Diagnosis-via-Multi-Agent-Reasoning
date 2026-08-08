from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .trend_stability import DEFAULT_DATA_PATH, METADATA_COLUMNS, TrendStabilityEvidence


class SpikeStrengthEvidence:
    """Calculate generic spike-strength evidence for a row range.

    This is not an anomaly classifier. It measures sudden extreme values and
    abrupt adjacent-sample jumps across sliding windows.
    """

    def __init__(
        self,
        start_row: int,
        end_row: int,
        data_path: str | Path = DEFAULT_DATA_PATH,
        sample_rate_hz: float = 50.0,
        window_seconds: float = 1.0,
        step_seconds: float = 0.25,
        max_workers: int | None = None,
    ) -> None:
        if end_row <= start_row:
            raise ValueError("end_row must be greater than start_row.")
        self.start_row = int(start_row)
        self.end_row = int(end_row)
        self.data_path = Path(data_path)
        self.sample_rate_hz = float(sample_rate_hz)
        self.window_rows = max(3, int(round(window_seconds * sample_rate_hz)))
        self.step_rows = max(1, int(round(step_seconds * sample_rate_hz)))
        self.max_workers = max_workers

    def analyze(self, channels: list[str] | None = None) -> dict[str, Any]:
        df = self._read_window()
        target_channels = self._channels(df, channels)
        if not target_channels:
            raise ValueError("No numeric channels available for spike analysis.")

        per_raw_channel: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_channel, df[name].to_numpy(dtype=np.float64)): name
                for name in target_channels
            }
            for future in as_completed(futures):
                name = futures[future]
                per_raw_channel[name] = future.result()

        return self._group_raw_channels(per_raw_channel)

    def analyze_toon(self, channels: list[str] | None = None) -> str:
        result = self.analyze(channels)
        channel_fields = [
            "channel",
            "severity",
            "strongest_axis",
            "dominant_feature",
            "spike_strength",
            "max_abs_z_score",
            "max_derivative_z",
            "max_window_abs_z_score",
            "max_window_derivative_z",
            "window_count",
            "finite_fraction",
        ]
        axis_fields = [
            "channel",
            "axis",
            "severity",
            "dominant_feature",
            "spike_strength",
            "max_abs_z_score",
            "max_derivative_z",
            "max_window_abs_z_score",
            "max_window_derivative_z",
            "window_count",
            "finite_fraction",
        ]

        channel_rows = []
        axis_rows = []
        for channel, metrics in result.items():
            channel_rows.append({field: metrics.get(field) for field in channel_fields} | {"channel": channel})
            for axis, axis_metrics in metrics["axes"].items():
                axis_rows.append({field: axis_metrics.get(field) for field in axis_fields} | {"channel": channel, "axis": axis})

        lines = [
            "spike_strength:",
            f"  start_row: {self.start_row}",
            f"  end_row: {self.end_row}",
            f"  start_second: {TrendStabilityEvidence._toon_value(round(self.start_row / self.sample_rate_hz, 6))}",
            f"  end_second: {TrendStabilityEvidence._toon_value(round(self.end_row / self.sample_rate_hz, 6))}",
            f"  sample_rate_hz: {TrendStabilityEvidence._toon_value(self.sample_rate_hz)}",
            "  meaning: higher scores indicate stronger sudden peak or abrupt adjacent-sample jump",
            TrendStabilityEvidence._toon_table("  channels", channel_fields, channel_rows, delimiter="|"),
            TrendStabilityEvidence._toon_table("  axis_evidence", axis_fields, axis_rows, delimiter="|"),
        ]
        return "\n".join(lines)

    def _read_window(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path)
        start = max(0, self.start_row)
        end = min(len(df), self.end_row)
        if end <= start:
            raise ValueError("Selected row range is empty.")
        return df.iloc[start:end].reset_index(drop=True)

    @staticmethod
    def _channels(df: pd.DataFrame, channels: list[str] | None) -> list[str]:
        if channels is not None:
            missing = [channel for channel in channels if channel not in df.columns]
            if missing:
                raise ValueError(f"Missing channels: {missing}")
            return [
                channel
                for channel in channels
                if pd.api.types.is_numeric_dtype(df[channel])
            ]
        return [
            column
            for column in df.columns
            if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
        ]

    @staticmethod
    def _group_raw_channels(per_raw_channel: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        grouped_axes: dict[str, dict[str, dict[str, Any]]] = {}
        for channel, metrics in sorted(per_raw_channel.items()):
            sensor_name, axis_name = SpikeStrengthEvidence._split_sensor_axis(channel)
            grouped_axes.setdefault(sensor_name, {})[axis_name] = metrics

        return {
            sensor_name: SpikeStrengthEvidence._summarize_sensor_channel(axes)
            for sensor_name, axes in sorted(grouped_axes.items())
        }

    @staticmethod
    def _summarize_sensor_channel(axes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        strongest_axis, strongest_metrics = max(
            axes.items(),
            key=lambda item: item[1]["spike_strength"],
        )
        axis_values = list(axes.values())
        return {
            "spike_strength": round(max(item["spike_strength"] for item in axis_values), 6),
            "strongest_axis": strongest_axis,
            "dominant_feature": strongest_metrics["dominant_feature"],
            "severity": strongest_metrics["severity"],
            "max_abs_z_score": round(max(item["max_abs_z_score"] for item in axis_values), 6),
            "max_derivative_z": round(max(item["max_derivative_z"] for item in axis_values), 6),
            "max_window_abs_z_score": round(max(item["max_window_abs_z_score"] for item in axis_values), 6),
            "max_window_derivative_z": round(max(item["max_window_derivative_z"] for item in axis_values), 6),
            "window_count": min(item["window_count"] for item in axis_values),
            "finite_fraction": round(min(item["finite_fraction"] for item in axis_values), 6),
            "axes": dict(sorted(axes.items())),
        }

    @staticmethod
    def _split_sensor_axis(channel: str) -> tuple[str, str]:
        for suffix in ("_x", "_y", "_z"):
            if channel.endswith(suffix):
                return channel[: -len(suffix)], suffix[1:]
        if channel.startswith("ecg_lead_"):
            return "ecg", channel.replace("ecg_", "")
        return channel, "value"

    def _analyze_channel(self, values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        finite_fraction = round(float(finite.size / max(1, len(values))), 6)
        if finite.size < 3:
            return {
                "spike_strength": 0.0,
                "dominant_feature": "insufficient_data",
                "severity": "none",
                "max_abs_z_score": 0.0,
                "max_derivative_z": 0.0,
                "max_window_abs_z_score": 0.0,
                "max_window_derivative_z": 0.0,
                "window_count": 0,
                "finite_fraction": finite_fraction,
            }

        baseline = float(np.nanmedian(finite))
        scale = self._robust_scale(finite)
        max_abs_z_score = float(np.nanmax(np.abs((values - baseline) / scale)))
        max_derivative_z = self._max_derivative_z(values, scale)

        window_abs_z = []
        window_derivative_z = []
        for start in range(0, max(1, len(values) - self.window_rows + 1), self.step_rows):
            chunk = values[start:start + self.window_rows]
            if np.isfinite(chunk).sum() < 3:
                continue
            local_baseline = float(np.nanmedian(chunk))
            window_abs_z.append(float(np.nanmax(np.abs((chunk - local_baseline) / scale))))
            window_derivative_z.append(self._max_derivative_z(chunk, scale))

        max_window_abs_z_score = max(window_abs_z) if window_abs_z else 0.0
        max_window_derivative_z = max(window_derivative_z) if window_derivative_z else 0.0
        spike_strength = max(
            max_abs_z_score,
            max_derivative_z,
            max_window_abs_z_score,
            max_window_derivative_z,
        )
        feature_scores = {
            "max_abs_z_score": max_abs_z_score,
            "max_derivative_z": max_derivative_z,
            "max_window_abs_z_score": max_window_abs_z_score,
            "max_window_derivative_z": max_window_derivative_z,
        }
        dominant_feature = max(feature_scores, key=feature_scores.get)

        return {
            "spike_strength": round(float(spike_strength), 6),
            "dominant_feature": dominant_feature,
            "severity": self._severity(float(spike_strength)),
            "max_abs_z_score": round(max_abs_z_score, 6),
            "max_derivative_z": round(max_derivative_z, 6),
            "max_window_abs_z_score": round(max_window_abs_z_score, 6),
            "max_window_derivative_z": round(max_window_derivative_z, 6),
            "window_count": len(window_abs_z),
            "finite_fraction": finite_fraction,
        }

    @staticmethod
    def _robust_scale(values: np.ndarray) -> float:
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median))) * 1.4826
        std = float(np.nanstd(values))
        scale = mad if mad > 1e-9 else std
        return scale if scale > 1e-9 and np.isfinite(scale) else 1.0

    @staticmethod
    def _max_derivative_z(values: np.ndarray, scale: float) -> float:
        mask = np.isfinite(values)
        if mask.sum() < 2:
            return 0.0
        clean = values[mask]
        return float(np.nanmax(np.abs(np.diff(clean))) / scale)

    @staticmethod
    def _severity(score: float) -> str:
        if score >= 20.0:
            return "very_high"
        if score >= 8.0:
            return "high"
        if score >= 3.0:
            return "moderate"
        return "low"
