from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .trend_stability import DEFAULT_DATA_PATH, METADATA_COLUMNS, TrendStabilityEvidence


class SeasonalityRhythmStabilityEvidence:
    """Calculate generic rhythm-stability evidence for a row range.

    This is not an anomaly classifier. It measures whether the repeating
    frequency pattern stays stable across sliding windows.
    """

    def __init__(
        self,
        start_row: int,
        end_row: int,
        data_path: str | Path = DEFAULT_DATA_PATH,
        sample_rate_hz: float = 50.0,
        window_seconds: float = 2.0,
        step_seconds: float = 0.5,
        max_workers: int | None = None,
    ) -> None:
        if end_row <= start_row:
            raise ValueError("end_row must be greater than start_row.")
        self.start_row = int(start_row)
        self.end_row = int(end_row)
        self.data_path = Path(data_path)
        self.sample_rate_hz = float(sample_rate_hz)
        self.window_rows = max(8, int(round(window_seconds * sample_rate_hz)))
        self.step_rows = max(1, int(round(step_seconds * sample_rate_hz)))
        self.max_workers = max_workers

    def analyze(self, channels: list[str] | None = None) -> dict[str, Any]:
        df = self._read_window()
        target_channels = self._channels(df, channels)
        if not target_channels:
            raise ValueError("No numeric channels available for rhythm analysis.")

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
            "rhythm_instability",
            "dominant_frequency_median_hz",
            "dominant_frequency_jump_hz",
            "dominant_frequency_std_hz",
            "spectral_entropy_mean",
            "spectral_entropy_jump",
            "spectral_distribution_shift",
            "window_count",
            "finite_fraction",
        ]
        axis_fields = [
            "channel",
            "axis",
            "severity",
            "dominant_feature",
            "rhythm_instability",
            "dominant_frequency_median_hz",
            "dominant_frequency_jump_hz",
            "dominant_frequency_std_hz",
            "spectral_entropy_mean",
            "spectral_entropy_jump",
            "spectral_distribution_shift",
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
            "seasonality_rhythm_stability:",
            f"  start_row: {self.start_row}",
            f"  end_row: {self.end_row}",
            f"  start_second: {TrendStabilityEvidence._toon_value(round(self.start_row / self.sample_rate_hz, 6))}",
            f"  end_second: {TrendStabilityEvidence._toon_value(round(self.end_row / self.sample_rate_hz, 6))}",
            f"  sample_rate_hz: {TrendStabilityEvidence._toon_value(self.sample_rate_hz)}",
            "  meaning: higher scores indicate stronger rhythm or frequency-pattern instability",
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
            if column not in METADATA_COLUMNS
            and not column.startswith("ecg_lead_")
            and pd.api.types.is_numeric_dtype(df[column])
        ]

    @staticmethod
    def _group_raw_channels(per_raw_channel: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        grouped_axes: dict[str, dict[str, dict[str, Any]]] = {}
        for channel, metrics in sorted(per_raw_channel.items()):
            sensor_name, axis_name = SeasonalityRhythmStabilityEvidence._split_sensor_axis(channel)
            grouped_axes.setdefault(sensor_name, {})[axis_name] = metrics

        return {
            sensor_name: SeasonalityRhythmStabilityEvidence._summarize_sensor_channel(axes)
            for sensor_name, axes in sorted(grouped_axes.items())
        }

    @staticmethod
    def _summarize_sensor_channel(axes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        strongest_axis, strongest_metrics = max(
            axes.items(),
            key=lambda item: item[1]["rhythm_instability"],
        )
        axis_values = list(axes.values())
        return {
            "rhythm_instability": round(max(item["rhythm_instability"] for item in axis_values), 6),
            "strongest_axis": strongest_axis,
            "dominant_feature": strongest_metrics["dominant_feature"],
            "severity": strongest_metrics["severity"],
            "dominant_frequency_median_hz": round(max(item["dominant_frequency_median_hz"] for item in axis_values), 6),
            "dominant_frequency_jump_hz": round(max(item["dominant_frequency_jump_hz"] for item in axis_values), 6),
            "dominant_frequency_std_hz": round(max(item["dominant_frequency_std_hz"] for item in axis_values), 6),
            "spectral_entropy_mean": round(max(item["spectral_entropy_mean"] for item in axis_values), 6),
            "spectral_entropy_jump": round(max(item["spectral_entropy_jump"] for item in axis_values), 6),
            "spectral_distribution_shift": round(max(item["spectral_distribution_shift"] for item in axis_values), 6),
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
        finite_fraction = round(float(np.isfinite(values).sum() / max(1, len(values))), 6)
        spectra = []
        dominant_frequencies = []
        spectral_entropies = []

        for start in range(0, max(1, len(values) - self.window_rows + 1), self.step_rows):
            chunk = values[start:start + self.window_rows]
            if np.isfinite(chunk).sum() < max(4, self.window_rows // 3):
                continue
            freqs, distribution = self._window_spectrum(chunk)
            if distribution.size == 0:
                continue
            spectra.append(distribution)
            dominant_frequencies.append(float(freqs[int(np.argmax(distribution))]))
            spectral_entropies.append(self._spectral_entropy(distribution))

        if len(spectra) < 2:
            return {
                "rhythm_instability": 0.0,
                "dominant_feature": "insufficient_windows",
                "severity": "none",
                "dominant_frequency_median_hz": 0.0,
                "dominant_frequency_jump_hz": 0.0,
                "dominant_frequency_std_hz": 0.0,
                "spectral_entropy_mean": 0.0,
                "spectral_entropy_jump": 0.0,
                "spectral_distribution_shift": 0.0,
                "window_count": len(spectra),
                "finite_fraction": finite_fraction,
            }

        dom = np.asarray(dominant_frequencies, dtype=float)
        entropy = np.asarray(spectral_entropies, dtype=float)
        spectrum_matrix = np.vstack(spectra)

        freq_resolution = self.sample_rate_hz / self.window_rows
        dominant_frequency_jump_hz = float(np.max(np.abs(np.diff(dom))))
        dominant_frequency_std_hz = float(np.std(dom))
        spectral_entropy_jump = float(np.max(np.abs(np.diff(entropy))))
        spectral_distribution_shift = self._max_distribution_shift(spectrum_matrix)

        feature_scores = {
            "dominant_frequency_jump_hz": dominant_frequency_jump_hz / max(freq_resolution, 1e-9),
            "dominant_frequency_std_hz": dominant_frequency_std_hz / max(freq_resolution, 1e-9),
            "spectral_entropy_jump": spectral_entropy_jump * 10.0,
            "spectral_distribution_shift": spectral_distribution_shift * 10.0,
        }
        dominant_feature = max(feature_scores, key=feature_scores.get)
        rhythm_instability = feature_scores[dominant_feature]

        return {
            "rhythm_instability": round(float(rhythm_instability), 6),
            "dominant_feature": dominant_feature,
            "severity": self._severity(float(rhythm_instability)),
            "dominant_frequency_median_hz": round(float(np.median(dom)), 6),
            "dominant_frequency_jump_hz": round(dominant_frequency_jump_hz, 6),
            "dominant_frequency_std_hz": round(dominant_frequency_std_hz, 6),
            "spectral_entropy_mean": round(float(np.mean(entropy)), 6),
            "spectral_entropy_jump": round(spectral_entropy_jump, 6),
            "spectral_distribution_shift": round(spectral_distribution_shift, 6),
            "window_count": len(spectra),
            "finite_fraction": finite_fraction,
        }

    def _window_spectrum(self, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prepared = self._fill_missing(chunk)
        prepared = prepared - float(np.mean(prepared))
        prepared = prepared * np.hanning(len(prepared))
        power = np.abs(np.fft.rfft(prepared)) ** 2
        freqs = np.fft.rfftfreq(len(prepared), d=1.0 / self.sample_rate_hz)

        if power.size <= 1:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        power = power[1:]
        freqs = freqs[1:]

        total = float(np.sum(power))
        if total <= 1e-12 or not np.isfinite(total):
            return freqs, np.zeros_like(power, dtype=float)
        return freqs, power / total

    @staticmethod
    def _fill_missing(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(values)
        if mask.all():
            return values
        if not mask.any():
            return np.zeros_like(values, dtype=float)
        x = np.arange(len(values), dtype=float)
        return np.interp(x, x[mask], values[mask])

    @staticmethod
    def _spectral_entropy(distribution: np.ndarray) -> float:
        distribution = distribution[distribution > 1e-12]
        if distribution.size <= 1:
            return 0.0
        entropy = -float(np.sum(distribution * np.log2(distribution)))
        return entropy / float(np.log2(len(distribution)))

    @staticmethod
    def _max_distribution_shift(spectrum_matrix: np.ndarray) -> float:
        shifts = []
        for previous, current in zip(spectrum_matrix[:-1], spectrum_matrix[1:]):
            shifts.append(0.5 * float(np.sum(np.abs(current - previous))))
        return max(shifts) if shifts else 0.0

    @staticmethod
    def _severity(score: float) -> str:
        if score >= 10.0:
            return "very_high"
        if score >= 5.0:
            return "high"
        if score >= 2.0:
            return "moderate"
        return "low"
