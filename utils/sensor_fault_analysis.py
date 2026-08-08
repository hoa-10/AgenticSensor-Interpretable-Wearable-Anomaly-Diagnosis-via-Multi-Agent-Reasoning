from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SensorFaultAnalyzer:
    """Rule-based analysis for synthetic sensor data-quality faults.

    The analyzer detects the four fault domains used by
    ``sensor_layer.fault_injection.sensor_faults``:
    stuck-at-zero, stuck-at-constant, data dropout, and clipping saturation.
    """

    METADATA_COLUMNS = {
        "sample_index",
        "row",
        "label",
        "activity",
        "activityID",
        "subject_id",
        "batch_id",
        "start_row",
        "end_row",
        "sampling_rate",
        "batch_seconds",
        "time_seconds",
        "phase",
        "is_anomaly",
    }

    def __init__(
        self,
        zero_tolerance: float = 1e-9,
        flatline_fraction_threshold: float = 0.995,
        dropout_fraction_threshold: float = 0.0,
        clipping_fraction_threshold: float = 0.15,
    ) -> None:
        self.zero_tolerance = zero_tolerance
        self.flatline_fraction_threshold = flatline_fraction_threshold
        self.dropout_fraction_threshold = dropout_fraction_threshold
        self.clipping_fraction_threshold = clipping_fraction_threshold

    def detect_stuck_values(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect stuck-at-zero and stuck-at-constant channels."""
        df = self._read_window(data, start_row, end_row)
        per_channel = {}
        stuck_channels = []

        for channel in self._channels(df, channels):
            metrics = self._channel_metrics(df[channel].to_numpy(dtype=np.float64))
            event_mean = metrics["event_mean"]
            is_flatline = (
                metrics["finite_count"] > 0
                and metrics["zero_diff_fraction"] >= self.flatline_fraction_threshold
                and metrics["event_std"] <= self.zero_tolerance
            )
            if is_flatline and abs(event_mean) <= self.zero_tolerance:
                fault_type = "stuck_at_zero"
            elif is_flatline:
                fault_type = "stuck_at_constant"
            else:
                fault_type = "none"

            row = {
                **metrics,
                "is_stuck": fault_type != "none",
                "fault_type": fault_type,
            }
            if row["is_stuck"]:
                stuck_channels.append(channel)
                per_channel[channel] = {
                    "event_mean": row["event_mean"],
                    "event_std": row["event_std"],
                    "zero_diff_fraction": row["zero_diff_fraction"],
                    "fault_type": row["fault_type"],
                }

        return {
            "per_channel": per_channel,
            "stuck_channels": stuck_channels,
            "has_stuck_values": bool(stuck_channels),
        }

    def detect_data_dropouts(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect channels with NaN packet-loss segments."""
        df = self._read_window(data, start_row, end_row)
        per_channel = {}
        dropout_channels = []

        for channel in self._channels(df, channels):
            metrics = self._channel_metrics(df[channel].to_numpy(dtype=np.float64))
            is_dropout = metrics["nan_fraction"] > self.dropout_fraction_threshold
            if is_dropout:
                dropout_channels.append(channel)
                per_channel[channel] = {
                    "nan_fraction": metrics["nan_fraction"],
                    "finite_count": metrics["finite_count"],
                    "length": metrics["length"],
                }

        return {
            "per_channel": per_channel,
            "dropout_channels": dropout_channels,
            "has_data_dropout": bool(dropout_channels),
        }

    def detect_signal_clipping(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect clipping/saturation from repeated event-window bounds."""
        df = self._read_window(data, start_row, end_row)
        per_channel = {}
        clipping_channels = []

        for channel in self._channels(df, channels):
            metrics = self._channel_metrics(df[channel].to_numpy(dtype=np.float64))
            is_clipping = (
                metrics["clipping_fraction"] > self.clipping_fraction_threshold
                and metrics["event_std"] > self.zero_tolerance
                and metrics["zero_diff_fraction"] < self.flatline_fraction_threshold
            )
            if is_clipping:
                clipping_channels.append(channel)
                per_channel[channel] = {
                    "clipping_fraction": metrics["clipping_fraction"],
                    "event_std": metrics["event_std"],
                    "event_min": metrics["event_min"],
                    "event_max": metrics["event_max"],
                }

        return {
            "per_channel": per_channel,
            "clipping_channels": clipping_channels,
            "has_signal_clipping": bool(clipping_channels),
        }

    def check_channel_independence(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Summarize whether detected faults are localized or distributed."""
        stuck = self.detect_stuck_values(data, start_row, end_row, channels)
        dropout = self.detect_data_dropouts(data, start_row, end_row, channels)
        clipping = self.detect_signal_clipping(data, start_row, end_row, channels)

        affected_channels = sorted(
            set(stuck["stuck_channels"])
            | set(dropout["dropout_channels"])
            | set(clipping["clipping_channels"])
        )
        affected_positions = sorted(
            {self._position_from_channel(channel) for channel in affected_channels}
        )
        affected_positions = [
            position for position in affected_positions if position != "unknown"
        ]

        if not affected_channels:
            distribution = "none"
        elif len(affected_channels) == 1 or len(affected_positions) <= 1:
            distribution = "localized"
        else:
            distribution = "distributed"

        return {
            "distribution": distribution,
            "affected_channels": affected_channels,
            "affected_channel_count": len(affected_channels),
            "affected_positions": affected_positions,
            "affected_position_count": len(affected_positions),
            "fault_groups": {
                "stuck_channels": stuck["stuck_channels"],
                "dropout_channels": dropout["dropout_channels"],
                "clipping_channels": clipping["clipping_channels"],
            },
        }

    def analyze_distribution_shift(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare event channels with an equally sized pre-event baseline."""
        df = pd.read_parquet(str(data)) if isinstance(data, (str, Path)) else data
        start = max(0, int(start_row))
        end = len(df) if end_row is None else min(len(df), int(end_row))
        event_rows = max(1, end - start)
        baseline_start = max(0, start - event_rows)
        baseline = df.iloc[baseline_start:start]
        event = df.iloc[start:end]
        selected_channels = self._channels(df, channels)
        per_channel: dict[str, Any] = {}

        for channel in selected_channels:
            pre = baseline[channel].to_numpy(dtype=np.float64)
            current = event[channel].to_numpy(dtype=np.float64)
            pre_finite = pre[np.isfinite(pre)]
            event_finite = current[np.isfinite(current)]
            if pre_finite.size < 8 or event_finite.size < 8:
                continue

            pre_std = max(float(np.std(pre_finite)), 1e-9)
            event_std = float(np.std(event_finite))
            variance_ratio = (event_std * event_std) / (pre_std * pre_std)
            mean_shift_z = abs(float(np.mean(event_finite) - np.mean(pre_finite))) / pre_std
            flatness_delta = self._spectral_flatness(event_finite) - self._spectral_flatness(pre_finite)
            correlation_shift = self._correlation_shift(
                baseline,
                event,
                channel,
                selected_channels,
            )
            rank_score = (
                abs(float(np.log(max(variance_ratio, 1e-9))))
                + min(mean_shift_z, 10.0)
                + 2.0 * correlation_shift
                + abs(flatness_delta)
            )
            is_suspicious = (
                variance_ratio >= 2.0
                or variance_ratio <= 0.5
                or mean_shift_z >= 3.0
                or correlation_shift >= 0.4
                or abs(flatness_delta) >= 0.3
            )
            per_channel[channel] = {
                "variance_ratio": round(variance_ratio, 4),
                "mean_shift_z": round(mean_shift_z, 3),
                "correlation_shift": round(correlation_shift, 4),
                "spectral_flatness_delta": round(flatness_delta, 4),
                "shift_rank_score": round(rank_score, 3),
                "is_suspicious": is_suspicious,
            }

        ranked_channels = sorted(
            per_channel,
            key=lambda channel: per_channel[channel]["shift_rank_score"],
            reverse=True,
        )
        suspicious_channels = [
            channel for channel in ranked_channels
            if per_channel[channel]["is_suspicious"]
        ]
        position_summary: dict[str, dict[str, Any]] = {}
        for channel, values in per_channel.items():
            position = self._position_from_channel(channel)
            if position == "unknown":
                continue
            summary = position_summary.setdefault(
                position,
                {
                    "evaluated_channel_count": 0,
                    "suspicious_channel_count": 0,
                    "max_shift_rank_score": 0.0,
                },
            )
            summary["evaluated_channel_count"] += 1
            summary["suspicious_channel_count"] += int(values["is_suspicious"])
            summary["max_shift_rank_score"] = round(
                max(summary["max_shift_rank_score"], values["shift_rank_score"]),
                3,
            )

        return {
            "baseline_start_row": baseline_start,
            "baseline_end_row": start,
            "baseline_rows": len(baseline),
            "event_rows": len(event),
            "evaluated_channel_count": len(per_channel),
            "suspicious_channel_count": len(suspicious_channels),
            "suspicious_channels": suspicious_channels,
            "affected_positions": sorted(
                {
                    self._position_from_channel(channel)
                    for channel in suspicious_channels
                    if self._position_from_channel(channel) != "unknown"
                }
            ),
            "position_summary": position_summary,
            "top_shifted_channels": ranked_channels[:5],
            "per_channel": per_channel,
            "reliability": "high" if len(baseline) >= event_rows else "low",
        }

    def analyze_faults(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run all sensor-fault checks and return a compact payload."""
        return {
            "input": {
                "start_row": start_row,
                "end_row": end_row,
                "channels": channels,
            },
            "tool_results": {
                "detect_stuck_values": self.detect_stuck_values(
                    data, start_row, end_row, channels
                ),
                "detect_data_dropouts": self.detect_data_dropouts(
                    data, start_row, end_row, channels
                ),
                "detect_signal_clipping": self.detect_signal_clipping(
                    data, start_row, end_row, channels
                ),
                "check_channel_independence": self.check_channel_independence(
                    data, start_row, end_row, channels
                ),
                "analyze_distribution_shift": self.analyze_distribution_shift(
                    data, start_row, end_row, channels
                ),
            },
        }

    def _read_window(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int,
        end_row: int | None,
    ) -> pd.DataFrame:
        df = pd.read_parquet(str(data)) if isinstance(data, (str, Path)) else data
        start = max(0, int(start_row))
        end = len(df) if end_row is None or end_row < 0 else min(len(df), int(end_row))
        if end <= start:
            raise ValueError("Selected sensor fault window is empty.")
        return df.iloc[start:end].reset_index(drop=True)

    def _channels(self, df: pd.DataFrame, channels: list[str] | None) -> list[str]:
        if channels is not None:
            missing = [channel for channel in channels if channel not in df.columns]
            if missing:
                raise ValueError(f"Missing channels: {missing}")
            return channels
        return [
            column
            for column in df.columns
            if column not in self.METADATA_COLUMNS
            and pd.api.types.is_numeric_dtype(df[column])
        ]

    def _channel_metrics(self, values: np.ndarray) -> dict[str, Any]:
        length = int(len(values))
        finite = values[np.isfinite(values)]
        finite_count = int(finite.size)
        nan_fraction = float(np.isnan(values).sum() / length) if length else 0.0

        if finite_count > 1:
            diffs = np.diff(finite)
            zero_diff_fraction = float(
                (np.abs(diffs) <= self.zero_tolerance).sum() / len(diffs)
            )
            event_std = float(np.nanstd(finite))
            event_min = float(np.nanmin(finite))
            event_max = float(np.nanmax(finite))
            at_bounds = (finite == event_min) | (finite == event_max)
            clipping_fraction = float(at_bounds.sum() / finite_count)
        elif finite_count == 1:
            zero_diff_fraction = 1.0
            event_std = 0.0
            event_min = event_max = float(finite[0])
            clipping_fraction = 0.0
        else:
            zero_diff_fraction = 0.0
            event_std = 0.0
            event_min = None
            event_max = None
            clipping_fraction = 0.0

        event_mean = float(np.nanmean(finite)) if finite_count else 0.0
        return {
            "length": length,
            "finite_count": finite_count,
            "event_mean": round(event_mean, 6),
            "event_std": round(event_std, 6),
            "event_min": None if event_min is None else round(event_min, 6),
            "event_max": None if event_max is None else round(event_max, 6),
            "zero_diff_fraction": round(zero_diff_fraction, 6),
            "nan_fraction": round(nan_fraction, 6),
            "clipping_fraction": round(clipping_fraction, 6),
        }

    @staticmethod
    def _spectral_flatness(values: np.ndarray) -> float:
        if values.size < 8:
            return 0.0
        spectrum = np.abs(np.fft.rfft(values - np.mean(values))) ** 2
        spectrum = spectrum[1:] + 1e-12
        if spectrum.size == 0:
            return 0.0
        geometric_mean = float(np.exp(np.mean(np.log(spectrum))))
        arithmetic_mean = float(np.mean(spectrum))
        return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0

    def _correlation_shift(
        self,
        baseline: pd.DataFrame,
        event: pd.DataFrame,
        channel: str,
        channels: list[str],
    ) -> float:
        group = self._channel_group(channel)
        peers = [
            peer for peer in channels
            if peer != channel and self._channel_group(peer) == group
        ]
        shifts = []
        for peer in peers:
            pre_pair = baseline[[channel, peer]].dropna()
            event_pair = event[[channel, peer]].dropna()
            if len(pre_pair) < 8 or len(event_pair) < 8:
                continue
            if (
                pre_pair[channel].std() <= self.zero_tolerance
                or pre_pair[peer].std() <= self.zero_tolerance
                or event_pair[channel].std() <= self.zero_tolerance
                or event_pair[peer].std() <= self.zero_tolerance
            ):
                continue
            pre_corr = float(pre_pair[channel].corr(pre_pair[peer]))
            event_corr = float(event_pair[channel].corr(event_pair[peer]))
            if np.isfinite(pre_corr) and np.isfinite(event_corr):
                shifts.append(abs(event_corr - pre_corr))
        return float(np.mean(shifts)) if shifts else 0.0

    @staticmethod
    def _channel_group(channel: str) -> str:
        for prefix in (
            "chest_acc",
            "ecg_lead",
            "chest_ecg",
            "left_ankle_acc",
            "left_ankle_gyro",
            "left_ankle_mag",
            "right_lower_arm_acc",
            "right_lower_arm_gyro",
            "right_lower_arm_mag",
            "ankle_acc",
            "ankle_gyro",
            "ankle_mag",
            "arm_acc",
            "arm_gyro",
            "arm_mag",
        ):
            if channel.startswith(prefix):
                return prefix
        return channel

    @staticmethod
    def _position_from_channel(channel: str) -> str:
        if channel.startswith("left_ankle_"):
            return "left_ankle"
        if channel.startswith("right_lower_arm_"):
            return "right_lower_arm"
        if channel.startswith("ecg_"):
            return "chest"
        if channel.startswith("chest_"):
            return "chest"
        if channel.startswith("arm_"):
            return "hand"
        if channel.startswith("hand_"):
            return "hand"
        if channel.startswith("ankle_"):
            return "ankle"
        return "unknown"
