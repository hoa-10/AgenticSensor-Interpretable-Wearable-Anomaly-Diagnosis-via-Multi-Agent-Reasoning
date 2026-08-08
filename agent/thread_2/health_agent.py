from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from agent_sdk import AgentBuilder

from perception_layer.agent.thread_2.common import (
    PROJECT_ROOT,
    add_feature_references,
    clean_text_output,
    extract_tagged_block,
    load_thread_2_impact_config,
)
from perception_layer.agent.utils import extract_anomaly_ranges, extract_candidate_windows
from perception_layer.path_config import DEFAULT_CONFIG_PATH
from perception_layer.utils import HealthEventAnalyzer


PROMPT_PATH = PROJECT_ROOT / "perception_layer" / "prompt" / "thread_2" / "reasoning_prompt_2.md"

HEALTH_FEATURE_REFERENCES = {
    "baseline_rows": "Rows used as the pre-event baseline",
    "event_rows": "Rows used as the candidate health-event window",
    "pre_fraction": "Baseline share of the complete analysis window",
    "motion_std_ratio": "Event/baseline chest variability; below 0.85 indicates suppression",
    "motion_energy_decay": "Second-half/first-half event RMS; below 0.7 indicates energy loss",
    "breath_energy_ratio": "Event/baseline breathing-band energy; above 1.5 supports abnormal breathing",
    "breath_pre_dominant_hz": "Dominant chest breathing frequency before the event",
    "breath_event_dominant_hz": "Dominant chest breathing frequency during the event",
    "breath_frequency_shift_pct": "Percentage change in dominant breathing frequency",
    "ecg_hr_energy_ratio": "Event/baseline ECG heart-rate-band energy; above 1.3 supports stress",
    "ecg_std_ratio": "Event/baseline ECG variability; above 1.1 supports stress",
    "ecg_pre_dominant_hz": "Dominant ECG frequency before the event",
    "ecg_event_dominant_hz": "Dominant ECG frequency during the event",
    "ecg_frequency_shift_pct": "Percentage change in dominant ECG frequency",
    "dominant_frequency_drop_pct": "Chest dominant-frequency drop; above 50 percent is significant",
    "spectral_entropy_delta": "Event minus baseline spectral entropy",
    "stft_energy_trend": "Temporal slope of short-time spectral energy",
    "stft_energy_variability": "Variation of energy across STFT frames",
    "stft_frame_count": "Number of STFT frames used",
}


class Thread2HealthAgent:
    """Run non-impact health-event analysis, then ask an agent to reason over it."""

    MIN_BASELINE_SECONDS = 4.0
    MAX_BASELINE_SECONDS = 8.0

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        model: str | None = None,
        timeout: int = 180,
    ) -> None:
        self.config_path = config_path
        self.config = load_thread_2_impact_config(config_path)
        self.model = model or str(self.config["model"])
        self.timeout = timeout

    def build_health_analysis(
        self,
        parquet_path: str | Path | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        path = Path(parquet_path or self.config["parquet_path"])
        df = pd.read_parquet(path)
        sampling_rate = float(self.config["sampling_rate"])
        event_start = max(0, int(start_row))
        event_end = len(df) if end_row is None else min(len(df), int(end_row))
        event_rows = max(1, event_end - event_start)
        min_baseline_rows = int(round(self.MIN_BASELINE_SECONDS * sampling_rate))
        max_baseline_rows = int(round(self.MAX_BASELINE_SECONDS * sampling_rate))
        baseline_rows = min(max_baseline_rows, max(min_baseline_rows, event_rows))
        analysis_start = max(0, event_start - baseline_rows)
        analysis_end = event_end
        available_baseline_rows = max(0, event_start - analysis_start)
        analysis_rows = max(1, analysis_end - analysis_start)
        pre_fraction = available_baseline_rows / analysis_rows
        pre_fraction = min(0.95, max(0.01, pre_fraction))
        analyzer = HealthEventAnalyzer(
            sampling_rate=sampling_rate,
            pre_fraction=pre_fraction,
        )

        tool_calls = {
            "analyze_motion_suppression": lambda: analyzer.analyze_motion_suppression(
                df,
                start_row=analysis_start,
                end_row=analysis_end,
            ),
            "analyze_respiratory_pattern": lambda: analyzer.analyze_respiratory_pattern(
                df,
                start_row=analysis_start,
                end_row=analysis_end,
            ),
            "analyze_ecg_heart_rate": lambda: analyzer.analyze_ecg_heart_rate(
                df,
                start_row=analysis_start,
                end_row=analysis_end,
            ),
            "analyze_dominant_frequency_shift": lambda: analyzer.analyze_dominant_frequency_shift(
                df,
                start_row=analysis_start,
                end_row=analysis_end,
            ),
            "analyze_time_frequency_evolution": lambda: analyzer.analyze_time_frequency_evolution(
                df,
                start_row=analysis_start,
                end_row=analysis_end,
            ),
        }

        tool_results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            futures = {
                executor.submit(call): name
                for name, call in tool_calls.items()
            }
            for future in as_completed(futures):
                tool_results[futures[future]] = future.result()

        return {
            "input": {
                "parquet_path": str(path),
                "start_row": event_start,
                "end_row": event_end,
                "analysis_start_row": analysis_start,
                "analysis_end_row": analysis_end,
                "baseline_start_row": analysis_start,
                "baseline_end_row": event_start,
                "baseline_rows": available_baseline_rows,
                "event_rows": event_rows,
                "pre_fraction": round(pre_fraction, 4),
                "baseline_limited": available_baseline_rows < baseline_rows,
                "sampling_rate": sampling_rate,
            },
            "tool_results": tool_results,
        }

    def build_health_analyses(
        self,
        thread_1_output: Any,
        parquet_path: str | Path | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        ranges = extract_anomaly_ranges(thread_1_output)
        source = "thread_1_anomaly_ranges" if ranges else "fallback_input_range"
        if not ranges:
            ranges = [{"start_row": start_row, "end_row": end_row}]

        with ThreadPoolExecutor(max_workers=max(1, len(ranges))) as executor:
            futures = {
                executor.submit(
                    self.build_health_analysis,
                    parquet_path=parquet_path,
                    start_row=int(row["start_row"]),
                    end_row=row["end_row"],
                ): index
                for index, row in enumerate(ranges)
            }
            analyses_by_index = {
                futures[future]: future.result()
                for future in as_completed(futures)
            }
            analyses = [
                analyses_by_index[index]
                for index in range(len(ranges))
            ]

        return {
            "source": source,
            "ranges": ranges,
            "analyses": analyses,
        }

    def build_prompt(
        self,
        thread_1_output: Any,
        health_analysis: dict[str, Any],
    ) -> str:
        prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        replacements = {
            "thread_1_output": self._format_thread_1_context(thread_1_output),
            "health_analysis_json": self._format_health_analysis_table(health_analysis),
        }
        for key, value in replacements.items():
            prompt = prompt.replace("{" + key + "}", value)
        return prompt

    def create_agent(self):
        return (
            AgentBuilder()
            .with_model(self.model)
            .with_system_prompt(
                "You are Thread 2 health-event reasoning. Use the provided "
                "health event analysis evidence and analyze it."
            )
            .with_data_analysis_harness()
            .build()
        )

    def run(
        self,
        thread_1_output: Any,
        parquet_path: str | Path | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        analysis_start = time.perf_counter()
        health_analysis = self.build_health_analyses(
            thread_1_output=thread_1_output,
            parquet_path=parquet_path,
            start_row=start_row,
            end_row=end_row,
        )
        analysis_seconds = time.perf_counter() - analysis_start
        prompt_start = time.perf_counter()
        prompt = self.build_prompt(
            thread_1_output=thread_1_output,
            health_analysis=health_analysis,
        )
        prompt_seconds = time.perf_counter() - prompt_start
        self._print_health_agent_input(prompt)
        llm_start = time.perf_counter()
        result = self.create_agent().run_turn(prompt)
        raw_output = result.get("text", "") if isinstance(result, dict) else str(result)
        llm_seconds = time.perf_counter() - llm_start
        return {
            "health_analysis": health_analysis,
            "agent_output": self._clean_text_output(raw_output),
            "timing": {
                "analysis_seconds": round(analysis_seconds, 3),
                "prompt_build_seconds": round(prompt_seconds, 3),
                "llm_seconds": round(llm_seconds, 3),
                "total_seconds": round(time.perf_counter() - total_start, 3),
            },
        }

    @staticmethod
    def _clean_text_output(raw_output: Any) -> str:
        return clean_text_output(raw_output)

    @staticmethod
    def _print_health_agent_input(prompt: str) -> None:
        yellow = "\033[93m"
        reset = "\033[0m"
        print(
            f"{yellow}[HEALTH AGENT INPUT]\n{prompt}\n{reset}",
            flush=True,
        )

    @staticmethod
    def _format_thread_1_context(thread_1_output: Any) -> str:
        text = str(thread_1_output or "")
        timeline_lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "-" in line and ":" in line and (
                " ok" in line
                or "anomaly ->" in line
                or "transition" in line
            ):
                timeline_lines.append(line)

        anomaly_ranges = extract_anomaly_ranges(text)
        candidates = extract_candidate_windows(text)

        lines = ["timeline:"]
        lines.extend(f"- {line}" for line in timeline_lines)
        if anomaly_ranges:
            lines.append("")
            lines.append("target_anomaly_ranges:")
            for item in anomaly_ranges:
                lines.append(f"- {item['start_row']}-{item['end_row']}")
        if candidates:
            lines.append("")
            lines.append("visual_candidates:")
            for item in candidates:
                lines.append(
                    f"- {item['start_row']}-{item['end_row']}: "
                    f"{item.get('candidate_type', '')}, {item.get('visual_state', '')}"
                )
        visual_analysis = extract_tagged_block(text, "visual_anomaly_analysis")
        if visual_analysis:
            lines.extend(["", visual_analysis])
        return "\n".join(lines).strip()

    @classmethod
    def _format_health_analysis_table(cls, health_analysis: dict[str, Any]) -> str:
        rows = []
        analyses = health_analysis.get("analyses", [])
        if "tool_results" in health_analysis:
            analyses = [health_analysis]

        for analysis in analyses:
            input_info = analysis.get("input", {})
            start = input_info.get("start_row")
            end = input_info.get("end_row")
            analysis_start = input_info.get("analysis_start_row", start)
            analysis_end = input_info.get("analysis_end_row", end)
            baseline_start = input_info.get("baseline_start_row", analysis_start)
            baseline_end = input_info.get("baseline_end_row", start)
            tools = analysis.get("tool_results", {})
            rows.append(f"event_range: {start}-{end}")
            rows.append(
                f"health_analysis_window: {analysis_start}-{analysis_end} "
                f"(baseline={baseline_start}-{baseline_end}, event={start}-{end})"
            )
            rows.append("")
            rows.append("| feature | scope | value |")
            rows.append("|---|---|---:|")
            rows.append(f"| baseline_rows | health_window | {input_info.get('baseline_rows', 0)} |")
            rows.append(f"| event_rows | health_window | {input_info.get('event_rows', 0)} |")
            rows.append(f"| pre_fraction | health_window | {input_info.get('pre_fraction')} |")

            suppression = tools.get("analyze_motion_suppression", {})
            for position, values in suppression.get("per_position", {}).items():
                rows.append(f"| motion_std_ratio | {position} | {float(values.get('std_ratio', 0.0) or 0.0):.4f} |")
                rows.append(f"| motion_energy_decay | {position} | {float(values.get('energy_decay', 0.0) or 0.0):.4f} |")

            respiration = tools.get("analyze_respiratory_pattern", {})
            if respiration.get("has_chest"):
                rows.append(f"| breath_energy_ratio | chest | {respiration.get('breath_energy_ratio')} |")
                rows.append(f"| breath_pre_dominant_hz | chest | {respiration.get('pre_dom_freq_hz')} |")
                rows.append(f"| breath_event_dominant_hz | chest | {respiration.get('event_dom_freq_hz')} |")
                rows.append(f"| breath_frequency_shift_pct | chest | {respiration.get('breath_freq_shift_pct')} |")

            ecg = tools.get("analyze_ecg_heart_rate", {})
            for channel, values in ecg.get("per_channel", {}).items():
                rows.append(f"| ecg_hr_energy_ratio | {channel} | {values.get('hr_energy_ratio')} |")
                rows.append(f"| ecg_std_ratio | {channel} | {values.get('std_ratio')} |")
                rows.append(f"| ecg_pre_dominant_hz | {channel} | {values.get('pre_dom_freq_hz')} |")
                rows.append(f"| ecg_event_dominant_hz | {channel} | {values.get('event_dom_freq_hz')} |")
                rows.append(f"| ecg_frequency_shift_pct | {channel} | {values.get('hr_freq_shift_pct')} |")

            freq = tools.get("analyze_dominant_frequency_shift", {})
            for position, values in freq.get("per_position", {}).items():
                rows.append(f"| dominant_frequency_drop_pct | {position} | {values.get('freq_drop_pct')} |")

            time_frequency = tools.get("analyze_time_frequency_evolution", {})
            for position, values in time_frequency.get("per_position", {}).items():
                rows.append(f"| spectral_entropy_delta | {position} | {values.get('spectral_entropy_delta')} |")
                rows.append(f"| stft_energy_trend | {position} | {values.get('stft_energy_trend')} |")
                rows.append(f"| stft_energy_variability | {position} | {values.get('stft_energy_variability')} |")
                rows.append(f"| stft_frame_count | {position} | {values.get('stft_frame_count')} |")
            breathing_tf = time_frequency.get("breathing", {})
            if breathing_tf.get("available"):
                rows.append(f"| spectral_entropy_delta | breathing_band | {breathing_tf.get('spectral_entropy_delta')} |")
                rows.append(f"| stft_energy_trend | breathing_band | {breathing_tf.get('stft_energy_trend')} |")
                rows.append(f"| stft_energy_variability | breathing_band | {breathing_tf.get('stft_energy_variability')} |")
                rows.append(f"| stft_frame_count | breathing_band | {breathing_tf.get('stft_frame_count')} |")
            for channel, values in time_frequency.get("ecg", {}).items():
                rows.append(f"| spectral_entropy_delta | {channel} | {values.get('spectral_entropy_delta')} |")
                rows.append(f"| stft_energy_trend | {channel} | {values.get('stft_energy_trend')} |")
                rows.append(f"| stft_energy_variability | {channel} | {values.get('stft_energy_variability')} |")
                rows.append(f"| stft_frame_count | {channel} | {values.get('stft_frame_count')} |")
            rows.append("")

        return "\n".join(
            add_feature_references(rows, HEALTH_FEATURE_REFERENCES)
        ).strip()


def analyze_health_event(
    thread_1_output: Any,
    model: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    parquet_path: str | Path | None = None,
    start_row: int = 0,
    end_row: int | None = None,
) -> dict[str, Any]:
    return Thread2HealthAgent(config_path=config_path, model=model).run(
        thread_1_output=thread_1_output,
        parquet_path=parquet_path,
        start_row=start_row,
        end_row=end_row,
    )


__all__ = [
    "Thread2HealthAgent",
    "analyze_health_event",
]

