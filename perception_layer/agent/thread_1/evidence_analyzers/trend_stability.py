from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "mhealth.parquet"

METADATA_COLUMNS = {
    "subject_id",
    "sample_index",
    "scenario_row",
    "time_seconds",
    "label",
    "activity_name",
    "scenario_segment",
    "source_subject_id",
    "source_sample_index",
    "is_impact_anomaly",
    "is_activity_transition",
    "anomaly_type",
}


class TrendStabilityEvidence:
    """Calculate generic trend-stability evidence for a row range.

    This is not an anomaly classifier. It only measures whether each numeric
    channel's baseline changes smoothly or shows unstable trend evidence.
    """

    def __init__(
        self,
        start_row: int,
        end_row: int,
        data_path: str | Path = DEFAULT_DATA_PATH,
        sample_rate_hz: float = 50.0,
        window_seconds: float = 1.0,
        step_seconds: float = 0.5,
        top_k: int = 3,
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
        self.top_k = max(1, int(top_k))
        self.max_workers = max_workers

    def analyze(self, channels: list[str] | None = None) -> dict[str, Any]:
        df = self._read_window()
        target_channels = self._channels(df, channels)
        if not target_channels:
            raise ValueError("No numeric channels available for trend analysis.")

        per_raw_channel: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_channel, name, df[name].to_numpy(dtype=np.float64)): name
                for name in target_channels
            }
            for future in as_completed(futures):
                name = futures[future]
                per_raw_channel[name] = future.result()

        per_channel = self._group_raw_channels(per_raw_channel)
        return dict(sorted(per_channel.items()))

    def analyze_table(self, channels: list[str] | None = None) -> pd.DataFrame:
        result = self.analyze(channels)
        table: dict[str, dict[str, Any]] = {}

        for channel, metrics in result.items():
            axis_name = metrics["strongest_axis"]
            axis = metrics["axes"][axis_name]
            mean_shift = axis["mean_shift"]
            jump = axis["max_window_mean_jump"]
            slope = axis["slope"]
            previous_window = jump.get("previous_window") or {}
            current_window = jump.get("current_window") or {}

            table[channel] = {
                "severity": metrics["severity"],
                "strongest_axis": axis_name,
                "dominant_feature": metrics["dominant_feature"],
                "trend_instability": metrics["trend_instability"],
                "mean_shift_z": metrics["mean_shift_z"],
                "max_window_mean_jump_z": metrics["max_window_mean_jump_z"],
                "slope_z_per_second": metrics["slope_z_per_second"],
                "first_half_mean": mean_shift["first_half_mean"],
                "second_half_mean": mean_shift["second_half_mean"],
                "mean_shift_direction": mean_shift["direction"],
                "jump_from_mean": previous_window.get("mean"),
                "jump_to_mean": current_window.get("mean"),
                "jump_start_second": previous_window.get("start_second"),
                "jump_end_second": current_window.get("end_second"),
                "jump_direction": jump["direction"],
                "raw_slope_per_second": slope["raw_slope_per_second"],
                "slope_direction": slope["direction"],
                "finite_fraction": metrics["finite_fraction"],
                "evidence_summary": metrics["evidence_summary"],
            }

        return pd.DataFrame(table)

    def analyze_toon(self, channels: list[str] | None = None) -> str:
        result = self.analyze(channels)
        channel_fields = [
            "channel",
            "severity",
            "strongest_axis",
            "dominant_feature",
            "trend_instability",
            "mean_shift_z",
            "max_window_mean_jump_z",
            "slope_z_per_second",
            "jump_start_second",
            "jump_end_second",
            "jump_from_mean",
            "jump_to_mean",
        ]
        axis_fields = [
            "channel",
            "axis",
            "severity",
            "dominant_feature",
            "trend_instability",
            "mean_shift_z",
            "max_window_mean_jump_z",
            "slope_z_per_second",
            "first_half_mean",
            "second_half_mean",
            "mean_shift_direction",
            "jump_start_second",
            "jump_end_second",
            "jump_direction",
            "raw_slope_per_second",
            "slope_direction",
        ]

        channel_rows = []
        axis_rows = []

        for channel, metrics in result.items():
            strongest_axis = metrics["strongest_axis"]
            strongest = metrics["axes"][strongest_axis]
            jump = strongest["max_window_mean_jump"]
            previous_window = jump.get("previous_window") or {}
            current_window = jump.get("current_window") or {}
            channel_rows.append(
                {
                    "channel": channel,
                    "severity": metrics["severity"],
                    "strongest_axis": strongest_axis,
                    "dominant_feature": metrics["dominant_feature"],
                    "trend_instability": metrics["trend_instability"],
                    "mean_shift_z": metrics["mean_shift_z"],
                    "max_window_mean_jump_z": metrics["max_window_mean_jump_z"],
                    "slope_z_per_second": metrics["slope_z_per_second"],
                    "jump_start_second": previous_window.get("start_second"),
                    "jump_end_second": current_window.get("end_second"),
                    "jump_from_mean": previous_window.get("mean"),
                    "jump_to_mean": current_window.get("mean"),
                }
            )

            for axis, axis_metrics in metrics["axes"].items():
                axis_jump = axis_metrics["max_window_mean_jump"]
                axis_previous = axis_jump.get("previous_window") or {}
                axis_current = axis_jump.get("current_window") or {}
                mean_shift = axis_metrics["mean_shift"]
                slope = axis_metrics["slope"]
                axis_rows.append(
                    {
                        "channel": channel,
                        "axis": axis,
                        "severity": axis_metrics["severity"],
                        "dominant_feature": axis_metrics["dominant_feature"],
                        "trend_instability": axis_metrics["trend_instability"],
                        "mean_shift_z": axis_metrics["mean_shift_z"],
                        "max_window_mean_jump_z": axis_metrics["max_window_mean_jump_z"],
                        "slope_z_per_second": axis_metrics["slope_z_per_second"],
                        "first_half_mean": mean_shift["first_half_mean"],
                        "second_half_mean": mean_shift["second_half_mean"],
                        "mean_shift_direction": mean_shift["direction"],
                        "jump_start_second": axis_previous.get("start_second"),
                        "jump_end_second": axis_current.get("end_second"),
                        "jump_direction": axis_jump["direction"],
                        "raw_slope_per_second": slope["raw_slope_per_second"],
                        "slope_direction": slope["direction"],
                    }
                )

        lines = [
            "trend_stability:",
            f"  start_row: {self.start_row}",
            f"  end_row: {self.end_row}",
            f"  start_second: {self._toon_value(round(self.start_row / self.sample_rate_hz, 6))}",
            f"  end_second: {self._toon_value(round(self.end_row / self.sample_rate_hz, 6))}",
            f"  sample_rate_hz: {self._toon_value(self.sample_rate_hz)}",
            "  meaning: higher scores indicate stronger baseline or trend instability",
            self._toon_table("  channels", channel_fields, channel_rows, delimiter="|"),
            self._toon_table("  axis_evidence", axis_fields, axis_rows, delimiter="|"),
        ]
        return "\n".join(lines)

    @classmethod
    def _toon_table(
        cls,
        name: str,
        fields: list[str],
        rows: list[dict[str, Any]],
        delimiter: str = "|",
    ) -> str:
        header = f"{name}[{len(rows)}{delimiter}]{{{delimiter.join(fields)}}}:"
        body = [
            "    " + delimiter.join(cls._toon_value(row.get(field), delimiter) for field in fields)
            for row in rows
        ]
        return "\n".join([header, *body])

    @staticmethod
    def _toon_value(value: Any, delimiter: str = "|") -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(value):
                return "null"
            return f"{float(value):.6f}".rstrip("0").rstrip(".")
        text = str(value)
        must_quote = (
            text == ""
            or text.strip() != text
            or text in {"true", "false", "null"}
            or delimiter in text
            or any(char in text for char in [":", '"', "\\", "[", "]", "{", "}", "\n", "\r", "\t"])
        )
        return json.dumps(text) if must_quote else text

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
            sensor_name, axis_name = TrendStabilityEvidence._split_sensor_axis(channel)
            grouped_axes.setdefault(sensor_name, {})[axis_name] = metrics

        return {
            sensor_name: TrendStabilityEvidence._summarize_sensor_channel(axes)
            for sensor_name, axes in sorted(grouped_axes.items())
        }

    @staticmethod
    def _summarize_sensor_channel(axes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        strongest_axis, strongest_metrics = max(
            axes.items(),
            key=lambda item: item[1]["trend_instability"],
        )
        axis_values = list(axes.values())
        return {
            "trend_instability": round(max(item["trend_instability"] for item in axis_values), 6),
            "mean_shift_z": round(max(item["mean_shift_z"] for item in axis_values), 6),
            "max_window_mean_jump_z": round(max(item["max_window_mean_jump_z"] for item in axis_values), 6),
            "slope_z_per_second": round(max(item["slope_z_per_second"] for item in axis_values), 6),
            "strongest_axis": strongest_axis,
            "dominant_feature": strongest_metrics["dominant_feature"],
            "severity": strongest_metrics["severity"],
            "window_mean_count": min(item["window_mean_count"] for item in axis_values),
            "finite_fraction": round(min(item["finite_fraction"] for item in axis_values), 6),
            "evidence_summary": (
                f"Strongest trend evidence is on axis {strongest_axis}: "
                f"{strongest_metrics['evidence_summary']}"
            ),
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

    def _analyze_channel(self, channel: str, values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        if finite.size < 3:
            return {
                "trend_instability": 0.0,
                "dominant_feature": "insufficient_data",
                "severity": "none",
                "mean_shift_z": 0.0,
                "max_window_mean_jump_z": 0.0,
                "slope_z_per_second": 0.0,
                "window_mean_count": 0,
                "finite_fraction": round(float(finite.size / max(1, len(values))), 6),
                "valid_sample_count": int(finite.size),
                "total_sample_count": int(len(values)),
                "evidence_summary": "Not enough finite samples to analyze trend stability.",
            }

        scale = self._robust_scale(finite)
        mean_shift = self._mean_shift(values, scale)
        max_jump = self._max_window_mean_jump(values, scale)
        slope = self._slope_per_second(values, scale)

        feature_scores = {
            "mean_shift_z": mean_shift["mean_shift_z"],
            "max_window_mean_jump_z": max_jump["max_window_mean_jump_z"],
            "slope_z_per_second": slope["slope_z_per_second"],
        }
        dominant_feature = max(feature_scores, key=feature_scores.get)
        instability = float(feature_scores[dominant_feature])
        severity = self._severity(instability)

        return {
            "trend_instability": round(float(instability), 6),
            "dominant_feature": dominant_feature,
            "severity": severity,
            "mean_shift_z": round(float(mean_shift["mean_shift_z"]), 6),
            "max_window_mean_jump_z": round(float(max_jump["max_window_mean_jump_z"]), 6),
            "slope_z_per_second": round(float(slope["slope_z_per_second"]), 6),
            "window_mean_count": int(max_jump["window_mean_count"]),
            "finite_fraction": round(float(finite.size / max(1, len(values))), 6),
            "valid_sample_count": int(finite.size),
            "total_sample_count": int(len(values)),
            "duration_seconds": round(float(len(values) / self.sample_rate_hz), 6),
            "baseline_median": round(float(np.nanmedian(finite)), 6),
            "robust_scale": round(float(scale), 6),
            "mean_shift": mean_shift,
            "max_window_mean_jump": max_jump,
            "slope": slope,
            "evidence_summary": self._evidence_summary(channel, dominant_feature, instability, severity, mean_shift, max_jump, slope),
        }

    @staticmethod
    def _robust_scale(values: np.ndarray) -> float:
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median))) * 1.4826
        std = float(np.nanstd(values))
        scale = mad if mad > 1e-9 else std
        return scale if scale > 1e-9 and np.isfinite(scale) else 1.0

    def _mean_shift(self, values: np.ndarray, scale: float) -> dict[str, Any]:
        midpoint = len(values) // 2
        first = values[:midpoint]
        second = values[midpoint:]
        first_mean = float(np.nanmean(first)) if np.isfinite(first).any() else 0.0
        second_mean = float(np.nanmean(second)) if np.isfinite(second).any() else first_mean
        raw_delta = second_mean - first_mean
        mean_shift_z = abs(raw_delta) / scale
        return {
            "mean_shift_z": round(float(mean_shift_z), 6),
            "first_half_mean": round(first_mean, 6),
            "second_half_mean": round(second_mean, 6),
            "raw_delta": round(float(raw_delta), 6),
            "direction": self._direction(raw_delta, scale),
            "first_half_range": self._range_info(0, midpoint),
            "second_half_range": self._range_info(midpoint, len(values)),
        }

    def _max_window_mean_jump(self, values: np.ndarray, scale: float) -> dict[str, Any]:
        means = []
        windows = []
        for start in range(0, max(1, len(values) - self.window_rows + 1), self.step_rows):
            chunk = values[start:start + self.window_rows]
            if np.isfinite(chunk).any():
                means.append(float(np.nanmean(chunk)))
                windows.append((start, start + len(chunk)))
        if len(means) < 2:
            return {
                "max_window_mean_jump_z": 0.0,
                "raw_jump": 0.0,
                "direction": "stable",
                "window_mean_count": len(means),
                "previous_window": None,
                "current_window": None,
            }

        raw_jumps = np.diff(np.asarray(means, dtype=float))
        jump_index = int(np.nanargmax(np.abs(raw_jumps)))
        raw_jump = float(raw_jumps[jump_index])
        jump_z = abs(raw_jump) / scale
        previous_start, previous_end = windows[jump_index]
        current_start, current_end = windows[jump_index + 1]
        return {
            "max_window_mean_jump_z": round(float(jump_z), 6),
            "raw_jump": round(raw_jump, 6),
            "direction": self._direction(raw_jump, scale),
            "window_mean_count": len(means),
            "previous_window": {
                **self._range_info(previous_start, previous_end),
                "mean": round(float(means[jump_index]), 6),
            },
            "current_window": {
                **self._range_info(current_start, current_end),
                "mean": round(float(means[jump_index + 1]), 6),
            },
        }

    def _slope_per_second(self, values: np.ndarray, scale: float) -> dict[str, Any]:
        mask = np.isfinite(values)
        if mask.sum() < 3:
            return {
                "slope_z_per_second": 0.0,
                "raw_slope_per_second": 0.0,
                "direction": "stable",
                "duration_seconds": round(float(len(values) / self.sample_rate_hz), 6),
            }
        x = np.arange(len(values), dtype=float)[mask] / self.sample_rate_hz
        y = values[mask]
        slope = float(np.polyfit(x, y, 1)[0])
        duration = max(1.0 / self.sample_rate_hz, len(values) / self.sample_rate_hz)
        slope_z = abs(slope) * duration / scale
        return {
            "slope_z_per_second": round(float(slope_z), 6),
            "raw_slope_per_second": round(float(slope), 6),
            "direction": self._direction(slope * duration, scale),
            "duration_seconds": round(float(duration), 6),
        }

    def _range_info(self, local_start: int, local_end: int) -> dict[str, Any]:
        absolute_start = self.start_row + int(local_start)
        absolute_end = self.start_row + int(local_end)
        return {
            "start_row": absolute_start,
            "end_row": absolute_end,
            "start_second": round(float(absolute_start / self.sample_rate_hz), 6),
            "end_second": round(float(absolute_end / self.sample_rate_hz), 6),
        }

    @staticmethod
    def _direction(raw_delta: float, scale: float) -> str:
        normalized = raw_delta / scale
        if normalized > 0.5:
            return "increasing"
        if normalized < -0.5:
            return "decreasing"
        return "stable"

    @staticmethod
    def _severity(score: float) -> str:
        if score >= 6.0:
            return "very_high"
        if score >= 3.0:
            return "high"
        if score >= 1.0:
            return "moderate"
        return "low"

    @staticmethod
    def _evidence_summary(
        channel: str,
        dominant_feature: str,
        score: float,
        severity: str,
        mean_shift: dict[str, Any],
        max_jump: dict[str, Any],
        slope: dict[str, Any],
    ) -> str:
        if dominant_feature == "max_window_mean_jump_z":
            previous_window = max_jump.get("previous_window") or {}
            current_window = max_jump.get("current_window") or {}
            return (
                f"{channel} has {severity} trend instability ({score:.3f}) because the local window mean "
                f"jumps {max_jump['direction']} from {previous_window.get('mean')} to {current_window.get('mean')} "
                f"around {previous_window.get('start_second')}s-{current_window.get('end_second')}s."
            )
        if dominant_feature == "mean_shift_z":
            return (
                f"{channel} has {severity} trend instability ({score:.3f}) because the second-half mean "
                f"moves {mean_shift['direction']} from {mean_shift['first_half_mean']} to {mean_shift['second_half_mean']}."
            )
        return (
            f"{channel} has {severity} trend instability ({score:.3f}) because the overall slope is "
            f"{slope['direction']} at {slope['raw_slope_per_second']} units/second."
        )
