from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

class ImpactEventAnalyzer:
    """Acceleration-only analysis for impact-style human events.

    The analyzer expects 3-axis acceleration from chest, ankle, and hand/arm.
    It accepts both legacy names (ankle_acc_*, arm_acc_*) and mHealth names
    (left_ankle_acc_*, right_lower_arm_acc_*).
    """

    POSITIONS = ("chest", "ankle", "hand")
    AXES = ("x", "y", "z")
    PREFIX_ALIASES = {
        "chest": ("chest",),
        "ankle": ("left_ankle", "ankle"),
        "hand": ("right_lower_arm", "hand", "arm"),
    }

    def __init__(
        self,
        sampling_rate: float = 50.0,
        pre_fraction: float = 0.25,
        post_fraction: float = 0.45,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.pre_fraction = pre_fraction
        self.post_fraction = post_fraction

    def compute_acc_magnitude(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, dict[str, float | int]]:
        """Summarize sqrt(x^2+y^2+z^2) for each position."""
        df = self._read_window(data, start_row, end_row)
        results = {}
        for position in self.POSITIONS:
            if not self._has_position(df, position):
                continue
            values = self._magnitude(df, position)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                results[position] = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }
                continue
            results[position] = {
                "count": int(finite.size),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
        return results

    def detect_impact_peak(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Find the strongest acceleration peak across positions."""
        df = self._read_window(data, start_row, end_row)
        per_position = {}
        best_position = None
        best_local_row = 0
        best_zscore = -np.inf
        best_magnitude = 0.0

        for position in self.POSITIONS:
            if not self._has_position(df, position):
                continue
            mag = self._magnitude(df, position)
            peak_index = int(np.nanargmax(mag))
            median = float(np.nanmedian(mag))
            scale = self._robust_scale(mag)
            zscore = float((mag[peak_index] - median) / scale)
            per_position[position] = {
                "local_impact_row": peak_index,
                "impact_row": start_row + peak_index,
                "max_magnitude": float(mag[peak_index]),
                "median_magnitude": median,
                "impact_zscore": zscore,
            }
            if zscore > best_zscore:
                best_position = position
                best_local_row = peak_index
                best_zscore = zscore
                best_magnitude = float(mag[peak_index])

        return {
            "impact_row": start_row + best_local_row,
            "local_impact_row": best_local_row,
            "impact_position": best_position,
            "max_magnitude": best_magnitude,
            "impact_zscore": float(best_zscore if np.isfinite(best_zscore) else 0.0),
        }

    def compare_position_dominance(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Rank positions by acceleration activity inside the anomaly window."""
        df = self._read_window(data, start_row, end_row)
        scores = {}
        for position in self.POSITIONS:
            if not self._has_position(df, position):
                continue
            mag = self._magnitude(df, position)
            scores[position] = float(self._rms(mag - np.nanmedian(mag)))

        total = sum(scores.values()) or 1.0
        normalized = {key: value / total for key, value in scores.items()}
        ranked = sorted(scores, key=scores.get, reverse=True)
        threshold = max(scores.values()) * 0.25 if scores else 0.0
        affected = [position for position in ranked if scores[position] >= threshold]
        return {
            "dominant_position": ranked[0] if ranked else None,
            "normalized_position_scores": normalized,
            "affected_positions": affected,
            "affected_position_count": len(affected),
        }

    def analyze_pre_post_shift(
        self,
        data: pd.DataFrame | str | Path,
        impact_row: int | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Measure pre-vs-post acceleration vector shift for each position."""
        df = self._read_window(data, start_row, end_row)
        impact_local = self._impact_local_row(df, impact_row, start_row)
        pre_slice, post_slice = self._pre_post_slices(len(df), impact_local)
        per_position = {}

        for position in self.POSITIONS:
            if not self._has_position(df, position):
                continue
            vectors = self._vectors(df, position)
            pre = vectors[pre_slice]
            post = vectors[post_slice]
            pre_mean = np.nanmean(pre, axis=0)
            post_mean = np.nanmean(post, axis=0)
            shift = float(np.linalg.norm(post_mean - pre_mean))
            scale = self._robust_scale(np.linalg.norm(pre - pre_mean, axis=1))
            per_position[position] = {
                "shift": shift,
                "shift_score": float(shift / scale),
            }

        return {"impact_row": start_row + impact_local, "per_position": per_position}

    def analyze_posture_angle_change(
        self,
        data: pd.DataFrame | str | Path,
        impact_row: int | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Measure angle change between pre-impact and post-impact mean vectors."""
        df = self._read_window(data, start_row, end_row)
        impact_local = self._impact_local_row(df, impact_row, start_row)
        pre_slice, post_slice = self._pre_post_slices(len(df), impact_local)
        per_position = {}

        for position in self.POSITIONS:
            if not self._has_position(df, position):
                continue
            vectors = self._vectors(df, position)
            pre_mean = np.nanmean(vectors[pre_slice], axis=0)
            post_mean = np.nanmean(vectors[post_slice], axis=0)
            per_position[position] = {
                "angle_change_deg": self._angle_deg(pre_mean, post_mean),
            }

        return {"impact_row": start_row + impact_local, "per_position": per_position}

    def analyze_post_impact_stillness(
        self,
        data: pd.DataFrame | str | Path,
        impact_row: int | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Compare post-impact movement variability against pre-impact variability."""
        df = self._read_window(data, start_row, end_row)
        impact_local = self._impact_local_row(df, impact_row, start_row)
        pre_slice, post_slice = self._pre_post_slices(len(df), impact_local)
        per_position = {}

        for position in self.POSITIONS:
            if not self._has_position(df, position):
                continue
            mag = self._magnitude(df, position)
            pre_std = self._safe_std(mag[pre_slice])
            post_std = self._safe_std(mag[post_slice])
            ratio = float(post_std / pre_std)
            per_position[position] = {
                "post_std_ratio": ratio,
                "stillness_score": float(max(0.0, 1.0 - ratio)),
                "is_stillness_like": ratio <= 0.55,
            }

        return {"impact_row": start_row + impact_local, "per_position": per_position}

    def detect_persistent_offset(
        self,
        data: pd.DataFrame | str | Path,
        impact_row: int | None = None,
        start_row: int = 0,
        end_row: int | None = None,
        threshold_std: float = 1.0,
    ) -> dict[str, Any]:
        """Detect post-impact offset that persists after the impact peak."""
        df = self._read_window(data, start_row, end_row)
        impact_local = self._impact_local_row(df, impact_row, start_row)
        pre_slice, post_slice = self._pre_post_slices(len(df), impact_local)
        per_position = {}

        for position in self.POSITIONS:
            if not self._has_position(df, position):
                continue
            mag = self._magnitude(df, position)
            pre_mean = float(np.nanmean(mag[pre_slice]))
            pre_std = self._safe_std(mag[pre_slice])
            post = mag[post_slice]
            threshold = threshold_std * pre_std
            fraction = float(np.mean(np.abs(post - pre_mean) > threshold)) if len(post) else 0.0
            post_shift = float(abs(np.nanmean(post) - pre_mean)) if len(post) else 0.0
            per_position[position] = {
                "persistent_offset_fraction": fraction,
                "post_mean_shift_score": float(post_shift / pre_std),
                "is_persistent_offset_like": fraction >= 0.65 and post_shift / pre_std >= 1.0,
            }

        offset_positions = [
            position
            for position, result in per_position.items()
            if result["is_persistent_offset_like"]
        ]
        return {
            "impact_row": start_row + impact_local,
            "per_position": per_position,
            "offset_positions": offset_positions,
            "single_sensor_persistent_offset": len(offset_positions) == 1,
        }

    def analyze_recovery(
        self,
        data: pd.DataFrame | str | Path,
        impact_row: int | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Check whether the tail of the anomaly window returns near baseline."""
        df = self._read_window(data, start_row, end_row)
        impact_local = self._impact_local_row(df, impact_row, start_row)
        pre_slice, post_slice = self._pre_post_slices(len(df), impact_local)
        per_position = {}

        for position in self.POSITIONS:
            if not self._has_position(df, position):
                continue
            mag = self._magnitude(df, position)
            post = mag[post_slice]
            tail_len = max(5, int(round(len(post) * 0.25)))
            tail = post[-tail_len:] if len(post) else post
            pre_mean = float(np.nanmean(mag[pre_slice]))
            pre_std = self._safe_std(mag[pre_slice])
            recovery_error = float(abs(np.nanmean(tail) - pre_mean) / pre_std) if len(tail) else 0.0
            per_position[position] = {
                "recovery_error": recovery_error,
                "recovery_good": recovery_error <= 0.75,
            }

        return {"impact_row": start_row + impact_local, "per_position": per_position}

    def analyze_transient_dynamics(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Measure abrupt jerk and short-time high-frequency impact energy."""
        df = self._read_window(data, start_row, end_row)
        per_position: dict[str, Any] = {}

        for position in self.POSITIONS:
            if not self._has_position(df, position):
                continue
            magnitude = self._magnitude(df, position)
            jerk = np.diff(magnitude) * self.sampling_rate
            jerk_abs = np.abs(jerk)
            jerk_median = float(np.nanmedian(jerk_abs)) if jerk_abs.size else 0.0
            jerk_scale = self._robust_scale(jerk_abs)
            peak_jerk = float(np.nanmax(jerk_abs)) if jerk_abs.size else 0.0
            peak_jerk_zscore = float((peak_jerk - jerk_median) / jerk_scale)

            magnitude_median = float(np.nanmedian(magnitude))
            magnitude_scale = self._robust_scale(magnitude)
            magnitude_z = (magnitude - magnitude_median) / magnitude_scale
            impulse_fraction = float(np.mean(magnitude_z >= 4.0))
            spectral = self._short_time_high_frequency_ratio(
                magnitude - magnitude_median
            )

            per_position[position] = {
                "peak_jerk": round(peak_jerk, 4),
                "peak_jerk_zscore": round(peak_jerk_zscore, 3),
                "impulse_fraction": round(impulse_fraction, 4),
                **spectral,
            }

        dominant_position = max(
            per_position,
            key=lambda position: per_position[position]["peak_jerk_zscore"],
            default=None,
        )
        return {
            "dominant_jerk_position": dominant_position,
            "per_position": per_position,
        }

    def classify_impact_event_rules(
        self,
        data: pd.DataFrame | str | Path,
        impact_row: int | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Rule-based first pass for impact event hypotheses."""
        df = self._read_window(data, start_row, end_row)
        # detect_impact_peak must run first to determine local_impact
        peak = self.detect_impact_peak(df)
        local_impact = self._impact_local_row(df, impact_row, start_row)
        if impact_row is None:
            local_impact = int(peak["local_impact_row"])

        # Run remaining 6 analysis functions in parallel
        _task_fns: dict[str, Any] = {
            "dominance": lambda: self.compare_position_dominance(df),
            "shift":     lambda: self.analyze_pre_post_shift(df, local_impact),
            "angle":     lambda: self.analyze_posture_angle_change(df, local_impact),
            "stillness": lambda: self.analyze_post_impact_stillness(df, local_impact),
            "offset":    lambda: self.detect_persistent_offset(df, local_impact),
            "recovery":  lambda: self.analyze_recovery(df, local_impact),
        }
        _results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=len(_task_fns)) as _executor:
            _futures = {_executor.submit(fn): name for name, fn in _task_fns.items()}
            for _future in as_completed(_futures):
                _results[_futures[_future]] = _future.result()

        dominance = _results["dominance"]
        shift     = _results["shift"]
        angle     = _results["angle"]
        stillness = _results["stillness"]
        offset    = _results["offset"]
        recovery  = _results["recovery"]

        dominant = dominance["dominant_position"]
        chest_angle = self._metric(angle, "chest", "angle_change_deg")
        chest_still = self._metric(stillness, "chest", "stillness_score")
        chest_shift = self._metric(shift, "chest", "shift_score")
        ankle_score = dominance["normalized_position_scores"].get("ankle", 0.0)
        all_recovered = all(
            row["recovery_good"] for row in recovery["per_position"].values()
        )

        hypotheses: list[dict[str, Any]] = []
        if offset["single_sensor_persistent_offset"]:
            hypotheses.append(
                {
                    "event_type": "sensor_displacement_after_impact",
                    "score": 0.85,
                    "reason": "one position has a persistent post-impact offset",
                }
            )
        if chest_angle >= 35.0 or chest_still >= 0.45 or chest_shift >= 2.0:
            hypotheses.append(
                {
                    "event_type": "fall_impact_posture_change",
                    "score": min(0.95, 0.45 + chest_angle / 120.0 + chest_still * 0.3),
                    "reason": "chest posture shift or post-impact stillness is present",
                }
            )
        if dominant == "ankle" and ankle_score >= 0.45 and all_recovered:
            hypotheses.append(
                {
                    "event_type": "near_fall_stumble",
                    "score": 0.75,
                    "reason": "ankle-dominant impact with recovery after the event",
                }
            )

        if not hypotheses:
            hypotheses.append(
                {
                    "event_type": "unknown_impact_event",
                    "score": 0.35,
                    "reason": "impact evidence does not match a clean known pattern",
                }
            )

        hypotheses = sorted(hypotheses, key=lambda item: item["score"], reverse=True)
        return {
            "event_type": hypotheses[0]["event_type"],
            "confidence": float(hypotheses[0]["score"]),
            "hypotheses": hypotheses,
        }

    def _read_window(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> pd.DataFrame:
        df = pd.read_parquet(data) if isinstance(data, (str, Path)) else data
        start = max(0, int(start_row))
        end = len(df) if end_row is None or end_row < 0 else min(len(df), int(end_row))
        if end <= start:
            raise ValueError("Selected acceleration window is empty.")
        return df.iloc[start:end].reset_index(drop=True)

    def _has_position(self, df: pd.DataFrame, position: str) -> bool:
        return all(column in df.columns for column in self._acc_columns(df, position))

    def _acc_columns(self, df: pd.DataFrame, position: str) -> list[str]:
        prefixes = self.PREFIX_ALIASES.get(position, (position,))
        for prefix in prefixes:
            columns = [f"{prefix}_acc_{axis}" for axis in self.AXES]
            if all(column in df.columns for column in columns):
                return columns
        return [f"{position}_acc_{axis}" for axis in self.AXES]

    def _vectors(self, df: pd.DataFrame, position: str) -> np.ndarray:
        columns = self._acc_columns(df, position)
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing acceleration columns for {position}: {missing}")
        return df[columns].to_numpy(dtype=np.float64)

    def _magnitude(self, df: pd.DataFrame, position: str) -> np.ndarray:
        vectors = self._vectors(df, position)
        return np.sqrt(np.sum(vectors * vectors, axis=1))

    def _impact_local_row(
        self,
        df: pd.DataFrame,
        impact_row: int | None,
        start_row: int,
    ) -> int:
        if impact_row is None:
            return int(self.detect_impact_peak(df)["local_impact_row"])
        return max(0, min(len(df) - 1, int(impact_row) - int(start_row)))

    def _pre_post_slices(self, row_count: int, impact_local: int) -> tuple[slice, slice]:
        pre_len = max(5, int(round(row_count * self.pre_fraction)))
        post_len = max(5, int(round(row_count * self.post_fraction)))
        pre_start = max(0, impact_local - pre_len)
        pre_end = max(pre_start + 1, impact_local)
        post_start = min(row_count - 1, impact_local + 1)
        post_end = min(row_count, post_start + post_len)
        if post_end <= post_start:
            post_start = max(0, impact_local)
            post_end = row_count
        return slice(pre_start, pre_end), slice(post_start, post_end)

    @staticmethod
    def _rms(values: np.ndarray) -> float:
        clean = values[np.isfinite(values)]
        return float(np.sqrt(np.mean(clean * clean))) if clean.size else 0.0

    def _short_time_high_frequency_ratio(self, signal: np.ndarray) -> dict[str, Any]:
        clean = np.asarray(signal, dtype=np.float64)
        clean = np.where(np.isfinite(clean), clean, 0.0)
        frame_size = min(len(clean), max(16, int(round(self.sampling_rate * 0.5))))
        if frame_size < 16:
            return {
                "stft_high_freq_ratio": 0.0,
                "stft_transient_concentration": 0.0,
                "stft_frame_count": 0,
                "reliability": "low",
            }

        hop = max(1, frame_size // 2)
        starts = list(range(0, len(clean) - frame_size + 1, hop)) or [0]
        ratios = []
        window = np.hanning(frame_size)
        frequencies = np.fft.rfftfreq(frame_size, d=1.0 / self.sampling_rate)
        high_mask = (frequencies >= 5.0) & (
            frequencies <= min(20.0, self.sampling_rate * 0.475)
        )
        for start in starts:
            spectrum = np.abs(np.fft.rfft(clean[start:start + frame_size] * window)) ** 2
            total = float(np.sum(spectrum[1:]))
            ratios.append(float(np.sum(spectrum[high_mask]) / total) if total > 1e-12 else 0.0)

        median_ratio = float(np.median(ratios)) if ratios else 0.0
        max_ratio = float(max(ratios, default=0.0))
        concentration = max_ratio / (median_ratio + 1e-6)
        return {
            "stft_high_freq_ratio": round(max_ratio, 4),
            "stft_transient_concentration": round(concentration, 3),
            "stft_frame_count": len(ratios),
            "reliability": "high" if len(ratios) >= 3 else "medium",
        }

    @staticmethod
    def _safe_std(values: np.ndarray) -> float:
        std = float(np.nanstd(values))
        return std if std > 1e-9 and np.isfinite(std) else 1.0

    @staticmethod
    def _robust_scale(values: np.ndarray) -> float:
        clean = values[np.isfinite(values)]
        if clean.size == 0:
            return 1.0
        median = float(np.nanmedian(clean))
        mad = float(np.nanmedian(np.abs(clean - median))) * 1.4826
        std = float(np.nanstd(clean))
        scale = mad if mad > 1e-9 else std
        return scale if scale > 1e-9 and np.isfinite(scale) else 1.0

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= 1e-9 or not np.isfinite(denominator):
            return 0.0
        cosine = float(np.dot(a, b) / denominator)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    @staticmethod
    def _metric(result: dict[str, Any], position: str, key: str) -> float:
        return float(result.get("per_position", {}).get(position, {}).get(key, 0.0))
