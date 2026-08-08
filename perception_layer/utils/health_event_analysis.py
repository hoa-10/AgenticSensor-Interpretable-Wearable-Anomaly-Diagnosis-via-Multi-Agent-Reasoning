"""Health-event analysis tools for non-impact anomalies.

This module provides ``HealthEventAnalyzer``, a collection of pure-math
tools that extract signal features relevant to **dyspnea / breathing
distress** and **illness / fatigue response** events.

Design mirrors ``ImpactEventAnalyzer`` so that both can be used side-by-side
inside the Thread-2 agent pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class HealthEventAnalyzer:
    """Acceleration + ECG analysis for non-impact health events.

    Health evidence is extracted from chest acceleration and optional ECG
    columns. Other sensor positions may exist in the input but are ignored.
    """

    POSITIONS = ("chest", "ankle", "hand")
    HEALTH_POSITIONS = ("chest",)
    AXES = ("x", "y", "z")
    PREFIX_ALIASES = {
        "chest": ("chest",),
        "ankle": ("left_ankle", "ankle"),
        "hand": ("right_lower_arm", "hand", "arm"),
    }
    ECG_COLUMNS = ("ecg_lead_1", "ecg_lead_2", "chest_ecg_1", "chest_ecg_2")

    # Frequency bands (Hz)
    BREATH_BAND = (0.3, 1.0)
    HR_BAND = (1.5, 2.5)

    def __init__(
        self,
        sampling_rate: float = 50.0,
        pre_fraction: float = 0.45,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.pre_fraction = pre_fraction

    # ------------------------------------------------------------------
    # Tool 1: Motion suppression
    # ------------------------------------------------------------------

    def analyze_motion_suppression(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Detect whether body motion is suppressed inside the event window.

        For each position, compute:
        - ``std_ratio``: std(event_window) / std(pre_event). Values < 0.5
          strongly indicate motion suppression (fatigue / illness).
        - ``energy_decay``: RMS(second_half) / RMS(first_half) of the event
          window. Values < 0.7 indicate progressive energy loss.

        Returns per-position metrics plus a summary flag.
        """
        df = self._read_window(data, start_row, end_row)
        event_start, event_end = self._event_bounds(df)
        per_position: dict[str, Any] = {}
        suppressed_positions: list[str] = []

        for position in self.HEALTH_POSITIONS:
            if not self._has_position(df, position):
                continue
            mag = self._magnitude(df, position)
            pre_std = self._safe_std(mag[:event_start])
            event_std = self._safe_std(mag[event_start:event_end])
            std_ratio = float(event_std / pre_std)

            # Energy decay: second half vs first half of event window
            seg = mag[event_start:event_end]
            mid = len(seg) // 2
            rms_first = self._rms(seg[:mid]) + 1e-9
            rms_second = self._rms(seg[mid:]) + 1e-9
            energy_decay = float(rms_second / rms_first)

            is_suppressed = std_ratio < 0.85
            is_mildly_suppressed = std_ratio < 0.95
            if is_suppressed:
                suppressed_positions.append(position)

            per_position[position] = {
                "std_ratio": round(std_ratio, 4),
                "energy_decay": round(energy_decay, 4),
                "is_suppressed": is_suppressed,
                "is_mildly_suppressed": is_mildly_suppressed,
            }

        all_suppressed = len(suppressed_positions) == len(per_position) and len(suppressed_positions) > 0
        chest_only = suppressed_positions == ["chest"]
        mild_positions = [
            pos for pos, info in per_position.items()
            if info["is_mildly_suppressed"]
        ]
        distributed_mild = len(mild_positions) >= 2

        return {
            "per_position": per_position,
            "suppressed_positions": suppressed_positions,
            "suppressed_count": len(suppressed_positions),
            "all_positions_suppressed": all_suppressed,
            "chest_only_suppressed": chest_only,
            "mildly_suppressed_positions": mild_positions,
            "distributed_mild_suppression": distributed_mild,
        }

    # ------------------------------------------------------------------
    # Tool 2: Respiratory pattern
    # ------------------------------------------------------------------

    def analyze_respiratory_pattern(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Detect abnormal respiratory oscillation in chest accelerometer.

        Computes spectral energy in the breathing band (0.3–1.0 Hz) for
        chest acceleration magnitude, comparing pre-event vs event window.
        Also reports the dominant frequency in the event window.

        A ratio > 1.5 suggests rapid/irregular breathing.
        """
        df = self._read_window(data, start_row, end_row)
        event_start, event_end = self._event_bounds(df)

        result: dict[str, Any] = {"has_chest": False}

        if not self._has_position(df, "chest"):
            return result

        mag = self._magnitude(df, "chest")
        pre_seg = mag[:event_start]
        event_seg = mag[event_start:event_end]

        pre_breath = self._spectral_energy_ratio(pre_seg, *self.BREATH_BAND)
        event_breath = self._spectral_energy_ratio(event_seg, *self.BREATH_BAND)
        ratio = float(event_breath / (pre_breath + 1e-9))

        event_dom_freq = self._dominant_freq(event_seg)
        pre_dom_freq = self._dominant_freq(pre_seg)
        breath_freq_shift_pct = self._frequency_shift_pct(pre_dom_freq, event_dom_freq)

        is_abnormal = ratio > 1.5

        result.update({
            "has_chest": True,
            "pre_breath_energy": round(pre_breath, 4),
            "event_breath_energy": round(event_breath, 4),
            "breath_energy_ratio": round(ratio, 3),
            "pre_dom_freq_hz": round(pre_dom_freq, 3),
            "event_dom_freq_hz": round(event_dom_freq, 3),
            "breath_freq_shift_pct": breath_freq_shift_pct,
            "is_abnormal_breathing": is_abnormal,
        })
        return result

    # ------------------------------------------------------------------
    # Tool 3: ECG heart-rate analysis
    # ------------------------------------------------------------------

    def analyze_ecg_heart_rate(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Detect tachycardia or ECG stress from chest ECG channels.

        Compares spectral energy in the heart-rate band (1.5–2.5 Hz) and
        signal variability (std ratio) between pre-event and event window.

        HR band ratio > 1.3 combined with std_ratio > 1.1 suggests ECG stress.
        """
        df = self._read_window(data, start_row, end_row)
        event_start, event_end = self._event_bounds(df)

        available_ecg = [col for col in self.ECG_COLUMNS if col in df.columns]
        if not available_ecg:
            return {"has_ecg": False, "per_channel": {}}

        per_channel: dict[str, Any] = {}
        overall_stress = False

        for col in available_ecg:
            vals = df[col].to_numpy(dtype=float)
            pre_seg = vals[:event_start]
            event_seg = vals[event_start:event_end]

            pre_hr = self._spectral_energy_ratio(pre_seg, *self.HR_BAND)
            event_hr = self._spectral_energy_ratio(event_seg, *self.HR_BAND)
            hr_ratio = float(event_hr / (pre_hr + 1e-9))

            pre_std = self._safe_std(pre_seg)
            event_std = self._safe_std(event_seg)
            std_ratio = float(event_std / pre_std)

            pre_dom = self._dominant_freq(pre_seg)
            event_dom = self._dominant_freq(event_seg)
            hr_freq_shift_pct = self._frequency_shift_pct(pre_dom, event_dom)

            is_stress = hr_ratio > 1.3 and std_ratio > 1.1

            if is_stress:
                overall_stress = True

            per_channel[col] = {
                "hr_energy_ratio": round(hr_ratio, 3),
                "std_ratio": round(std_ratio, 4),
                "pre_dom_freq_hz": round(pre_dom, 3),
                "event_dom_freq_hz": round(event_dom, 3),
                "hr_freq_shift_pct": hr_freq_shift_pct,
                "is_stress": is_stress,
            }

        return {
            "has_ecg": True,
            "per_channel": per_channel,
            "ecg_stress_detected": overall_stress,
        }

    # ------------------------------------------------------------------
    # Tool 4: Dominant frequency shift
    # ------------------------------------------------------------------

    def analyze_dominant_frequency_shift(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Detect structural frequency change between pre-event and event.

        A large drop in dominant frequency (> 50%) indicates the body
        transitioned from active motion to near-stillness or slow
        rhythmic movement (fatigue / illness signature).
        """
        df = self._read_window(data, start_row, end_row)
        event_start, event_end = self._event_bounds(df)
        per_position: dict[str, Any] = {}

        for position in self.HEALTH_POSITIONS:
            if not self._has_position(df, position):
                continue
            mag = self._magnitude(df, position)
            pre_freq = self._dominant_freq(mag[:event_start])
            event_freq = self._dominant_freq(mag[event_start:event_end])

            if pre_freq > 0.1:
                drop_pct = round((1.0 - event_freq / pre_freq) * 100.0, 1)
            else:
                drop_pct = 0.0

            per_position[position] = {
                "freq_drop_pct": drop_pct,
                "significant_drop": drop_pct > 50.0,
            }

        positions_with_drop = [
            pos for pos, info in per_position.items()
            if info["significant_drop"]
        ]

        return {
            "per_position": per_position,
            "positions_with_significant_drop": positions_with_drop,
            "drop_count": len(positions_with_drop),
        }

    def analyze_time_frequency_evolution(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Compare FFT entropy and short-time band energy before/during an event."""
        df = self._read_window(data, start_row, end_row)
        event_start, event_end = self._event_bounds(df)
        per_position: dict[str, Any] = {}

        for position in self.HEALTH_POSITIONS:
            if not self._has_position(df, position):
                continue
            magnitude = self._magnitude(df, position)
            per_position[position] = self._time_frequency_metrics(
                magnitude[:event_start],
                magnitude[event_start:event_end],
            )

        breathing = {"available": False}
        if self._has_position(df, "chest"):
            chest = self._magnitude(df, "chest")
            breathing = {
                "available": True,
                **self._time_frequency_metrics(
                    chest[:event_start],
                    chest[event_start:event_end],
                    band=self.BREATH_BAND,
                ),
            }

        ecg: dict[str, Any] = {}
        for channel in self.ECG_COLUMNS:
            if channel not in df.columns:
                continue
            values = df[channel].to_numpy(dtype=np.float64)
            ecg[channel] = self._time_frequency_metrics(
                values[:event_start],
                values[event_start:event_end],
                band=self.HR_BAND,
            )

        return {
            "per_position": per_position,
            "breathing": breathing,
            "ecg": ecg,
        }

    # ------------------------------------------------------------------
    # Tool 5: Rule-based classifier
    # ------------------------------------------------------------------

    def classify_non_impact_event_rules(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        """Rule-based baseline hypothesis for non-impact events.

        Uses the other four tools to produce a preliminary classification.
        This serves as a reference for the LLM agent (not final truth).
        """
        suppression = self.analyze_motion_suppression(data, start_row, end_row)
        respiratory = self.analyze_respiratory_pattern(data, start_row, end_row)
        ecg = self.analyze_ecg_heart_rate(data, start_row, end_row)
        freq_shift = self.analyze_dominant_frequency_shift(data, start_row, end_row)

        # Collect evidence
        evidence: list[str] = []
        score_illness = 0.0
        score_dyspnea = 0.0

        # --- Motion suppression ---
        sup_count = suppression["suppressed_count"]
        mild_count = len(suppression.get("mildly_suppressed_positions", []))
        if suppression["all_positions_suppressed"]:
            score_illness += 3.5
            evidence.append(f"All {sup_count} positions show motion suppression (std_ratio<0.85)")
        elif sup_count >= 2:
            score_illness += 3.0
            evidence.append(f"{sup_count} positions show motion suppression (std_ratio<0.85)")
        elif suppression.get("distributed_mild_suppression"):
            score_illness += 2.0
            evidence.append(f"{mild_count} positions show mild motion suppression (std_ratio<0.95)")
        elif sup_count == 1:
            score_illness += 1.0
            evidence.append(f"1 position shows motion suppression")
        elif suppression["chest_only_suppressed"]:
            score_dyspnea += 1.0
            evidence.append("Chest-only motion suppression")

        # --- Respiratory ---
        if respiratory.get("is_abnormal_breathing"):
            breath_ratio = respiratory.get("breath_energy_ratio", 1.0)
            if sup_count >= 2:
                score_illness += 0.5
                score_dyspnea += 0.5
                evidence.append(f"Abnormal breathing during distributed motion suppression (ratio={breath_ratio:.1f}x)")
            elif breath_ratio > 2.0:
                score_dyspnea += 2.5
                evidence.append(f"Strong abnormal breathing (ratio={breath_ratio:.1f}x)")
            else:
                score_dyspnea += 1.5
                score_illness += 0.5
                evidence.append(f"Mild abnormal breathing (ratio={breath_ratio:.1f}x)")

        # --- ECG ---
        if ecg.get("ecg_stress_detected"):
            # Check magnitude: strong ECG = dyspnea, mild = illness
            max_hr_ratio = max(
                (ch.get("hr_energy_ratio", 1.0) for ch in ecg["per_channel"].values()),
                default=1.0,
            )
            if sup_count >= 2:
                score_illness += 1.0
                score_dyspnea += 0.5
                evidence.append(f"ECG stress during distributed motion suppression (HR ratio={max_hr_ratio:.1f}x)")
            elif max_hr_ratio > 1.5:
                score_dyspnea += 2.0
                evidence.append(f"Strong ECG stress (HR ratio={max_hr_ratio:.1f}x)")
            else:
                score_illness += 1.0
                score_dyspnea += 0.5
                evidence.append(f"Mild ECG stress (HR ratio={max_hr_ratio:.1f}x)")

        # --- Frequency shift ---
        drop_count = freq_shift["drop_count"]
        if drop_count >= 3:
            score_illness += 2.0
            evidence.append(f"All {drop_count} positions show significant freq drop")
        elif drop_count >= 1:
            score_illness += 1.0
            evidence.append(f"{drop_count} position(s) show significant freq drop")

        # --- Classification ---
        total = score_illness + score_dyspnea
        if total < 1.5:
            classification = "normal"
            confidence = round(max(0.0, 1.0 - total / 3.0), 2)
        elif score_illness > score_dyspnea:
            classification = "illness_fatigue_response"
            confidence = round(min(1.0, score_illness / (total + 1.0)), 2)
        elif score_illness == score_dyspnea and sup_count >= 2:
            # Tie-break: distributed motion suppression is the strongest
            # indicator of illness/fatigue, outweighing breathing anomaly.
            classification = "illness_fatigue_response"
            confidence = round(min(1.0, score_illness / (total + 1.0)), 2)
        else:
            classification = "dyspnea_breathing_distress"
            confidence = round(min(1.0, score_dyspnea / (total + 1.0)), 2)

        return {
            "classification": classification,
            "confidence": confidence,
            "score_illness": round(score_illness, 2),
            "score_dyspnea": round(score_dyspnea, 2),
            "evidence": evidence,
            "reason": f"Rule-based: illness_score={score_illness:.1f}, dyspnea_score={score_dyspnea:.1f}",
        }

    # ==================================================================
    # Helper methods
    # ==================================================================

    def _event_bounds(self, df: pd.DataFrame) -> tuple[int, int]:
        """Estimate the event window split point using ``pre_fraction``.

        Returns (event_start, event_end) where event_start is the row index
        dividing the pre-event region from the event region.
        """
        n = len(df)
        event_start = max(5, int(round(n * self.pre_fraction)))
        return event_start, n

    def _read_window(
        self,
        data: pd.DataFrame | str | Path,
        start_row: int,
        end_row: int | None,
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.read_parquet(str(data))

        start = max(0, int(start_row))
        end = len(df) if end_row is None else min(len(df), int(end_row))
        if end <= start:
            raise ValueError("Selected window is empty.")
        return df.iloc[start:end].reset_index(drop=True)

    def _has_position(self, df: pd.DataFrame, position: str) -> bool:
        return all(col in df.columns for col in self._acc_columns(df, position))

    def _acc_columns(self, df: pd.DataFrame, position: str) -> list[str]:
        prefixes = self.PREFIX_ALIASES.get(position, (position,))
        for prefix in prefixes:
            columns = [f"{prefix}_acc_{axis}" for axis in self.AXES]
            if all(col in df.columns for col in columns):
                return columns
        return [f"{position}_acc_{axis}" for axis in self.AXES]

    def _resolve_acc_col(self, df: pd.DataFrame, position: str, axis: str) -> str | None:
        prefixes = self.PREFIX_ALIASES.get(position, (position,))
        for prefix in prefixes:
            col = f"{prefix}_acc_{axis}"
            if col in df.columns:
                return col
        return None

    def _vectors(self, df: pd.DataFrame, position: str) -> np.ndarray:
        columns = self._acc_columns(df, position)
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {position}: {missing}")
        return df[columns].to_numpy(dtype=np.float64)

    def _magnitude(self, df: pd.DataFrame, position: str) -> np.ndarray:
        vectors = self._vectors(df, position)
        return np.sqrt(np.sum(vectors * vectors, axis=1))

    def _dominant_freq(self, signal: np.ndarray) -> float:
        clean = signal[np.isfinite(signal)]
        if clean.size < 8:
            return 0.0
        clean = clean - np.mean(clean)
        spectrum = np.abs(np.fft.rfft(clean))
        freqs = np.fft.rfftfreq(clean.size, d=1.0 / self.sampling_rate)
        if spectrum.size <= 1:
            return 0.0
        idx = int(np.argmax(spectrum[1:]) + 1)
        return float(freqs[idx])

    def _spectral_energy_ratio(
        self,
        signal: np.ndarray,
        low_hz: float,
        high_hz: float,
    ) -> float:
        """Fraction of spectral energy in ``[low_hz, high_hz]``."""
        clean = signal[np.isfinite(signal)]
        if clean.size < 8:
            return 0.0
        clean = clean - np.mean(clean)
        spectrum = np.abs(np.fft.rfft(clean)) ** 2
        freqs = np.fft.rfftfreq(clean.size, d=1.0 / self.sampling_rate)
        total = spectrum.sum()
        if total < 1e-12:
            return 0.0
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        return float(spectrum[mask].sum() / total)

    def _time_frequency_metrics(
        self,
        pre_signal: np.ndarray,
        event_signal: np.ndarray,
        band: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        pre_entropy = self._spectral_entropy(pre_signal)
        event_entropy = self._spectral_entropy(event_signal)
        frame_size = min(
            len(event_signal),
            max(32, int(round(self.sampling_rate * 2.0))),
        )
        frame_ratios = self._short_time_energy_ratios(event_signal, frame_size, band)
        if len(frame_ratios) >= 2:
            energy_trend = frame_ratios[-1] / (frame_ratios[0] + 1e-9)
            variability = float(np.std(frame_ratios) / (np.mean(frame_ratios) + 1e-9))
        else:
            energy_trend = 1.0
            variability = 0.0

        min_length = min(len(pre_signal), len(event_signal))
        if min_length >= int(round(self.sampling_rate * 4.0)):
            reliability = "high"
        elif min_length >= int(round(self.sampling_rate * 2.0)):
            reliability = "medium"
        else:
            reliability = "low"

        return {
            "pre_spectral_entropy": round(pre_entropy, 4),
            "event_spectral_entropy": round(event_entropy, 4),
            "spectral_entropy_delta": round(event_entropy - pre_entropy, 4),
            "stft_energy_trend": round(float(energy_trend), 4),
            "stft_energy_variability": round(variability, 4),
            "stft_frame_count": len(frame_ratios),
            "reliability": reliability,
        }

    def _short_time_energy_ratios(
        self,
        signal: np.ndarray,
        frame_size: int,
        band: tuple[float, float] | None,
    ) -> list[float]:
        clean = np.asarray(signal, dtype=np.float64)
        clean = np.where(np.isfinite(clean), clean, 0.0)
        if frame_size < 16 or len(clean) < frame_size:
            return []
        hop = max(1, frame_size // 2)
        frequencies = np.fft.rfftfreq(frame_size, d=1.0 / self.sampling_rate)
        window = np.hanning(frame_size)
        ratios = []
        for start in range(0, len(clean) - frame_size + 1, hop):
            frame = clean[start:start + frame_size]
            frame = (frame - np.mean(frame)) * window
            spectrum = np.abs(np.fft.rfft(frame)) ** 2
            total = float(np.sum(spectrum[1:]))
            if total <= 1e-12:
                ratios.append(0.0)
                continue
            if band is None:
                ratios.append(total / frame_size)
            else:
                mask = (frequencies >= band[0]) & (frequencies <= band[1])
                ratios.append(float(np.sum(spectrum[mask]) / total))
        return ratios

    @staticmethod
    def _spectral_entropy(signal: np.ndarray) -> float:
        clean = np.asarray(signal, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        if clean.size < 8:
            return 0.0
        spectrum = np.abs(np.fft.rfft(clean - np.mean(clean))) ** 2
        spectrum = spectrum[1:]
        total = float(np.sum(spectrum))
        if total <= 1e-12 or spectrum.size <= 1:
            return 0.0
        probabilities = spectrum / total
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        return entropy / float(np.log(spectrum.size))

    @staticmethod
    def _rms(values: np.ndarray) -> float:
        clean = values[np.isfinite(values)]
        return float(np.sqrt(np.mean(clean * clean))) if clean.size else 0.0

    @staticmethod
    def _safe_std(values: np.ndarray) -> float:
        if values.size < 2:
            return 1.0
        std = float(np.nanstd(values))
        return std if std > 1e-9 and np.isfinite(std) else 1.0

    @staticmethod
    def _frequency_shift_pct(pre_freq: float, event_freq: float) -> float:
        if pre_freq <= 0.1:
            return 0.0
        return round((event_freq / pre_freq - 1.0) * 100.0, 1)
