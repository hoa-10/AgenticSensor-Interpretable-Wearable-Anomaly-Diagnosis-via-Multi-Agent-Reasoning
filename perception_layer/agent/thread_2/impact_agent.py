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
from perception_layer.utils import ImpactEventAnalyzer


PROMPT_PATH = PROJECT_ROOT / "perception_layer" / "prompt" / "thread_2" / "reasoning_prompt_1.md"

IMPACT_FEATURE_REFERENCES = {
    "range_rows": "Candidate range length in rows",
    "acc_mean": "Mean acceleration magnitude within the range",
    "acc_max": "Maximum acceleration magnitude within the range",
    "acc_std": "Acceleration variability within the range",
    "impact_peak_row": "Row containing the strongest acceleration peak",
    "impact_zscore": "Robust peak prominence; larger values are more unusual",
    "impact_max_magnitude": "Magnitude of the strongest acceleration peak",
    "peak_jerk": "Maximum rate of acceleration change",
    "peak_jerk_zscore": "Robust jerk prominence; larger values indicate a sharper transient",
    "impulse_fraction": "Fraction of samples with magnitude z-score at least 4",
    "stft_high_freq_ratio": "Share of short-time spectral energy at high frequencies",
    "stft_transient_concentration": "Concentration of transient energy across STFT frames",
    "stft_frame_count": "Number of STFT frames used",
    "normalized_position_energy": "Relative motion-energy share across body positions",
    "pre_post_shift": "Absolute signal shift from before to after the peak",
    "pre_post_shift_score": "Pre/post shift normalized by baseline variation",
    "posture_angle_change_deg": "Change in mean acceleration orientation in degrees",
    "post_pre_std_ratio": "Post/pre variability ratio; at most 0.55 is stillness-like",
    "persistent_offset_fraction": "Post-event fraction outside one baseline standard deviation",
    "post_mean_shift_score": "Post-event mean shift normalized by baseline variation",
    "recovery_error": "Distance from late post-event signal to baseline; at most 0.75 is good recovery",
}


class Thread2ImpactAgent:
    """Run impact-event analysis, then ask an agent to reason over the evidence."""

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
        self.analyzer = ImpactEventAnalyzer(
            sampling_rate=float(self.config["sampling_rate"])
        )

    def build_impact_analysis(
        self,
        parquet_path: str | Path | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        range_start = time.perf_counter()
        path = Path(parquet_path or self.config["parquet_path"])
        read_start = time.perf_counter()
        df = pd.read_parquet(path)
        read_seconds = time.perf_counter() - read_start
        peak_start = time.perf_counter()
        peak = self.analyzer.detect_impact_peak(df, start_row=start_row, end_row=end_row)
        peak_seconds = time.perf_counter() - peak_start
        impact_row = int(peak["impact_row"])

        tool_calls = {
            "compute_acc_magnitude": lambda: self._summarize_magnitudes(
                self.analyzer.compute_acc_magnitude(
                    df,
                    start_row=start_row,
                    end_row=end_row,
                )
            ),
            "analyze_transient_dynamics": lambda: self.analyzer.analyze_transient_dynamics(
                df,
                start_row=start_row,
                end_row=end_row,
            ),
            "compare_position_dominance": lambda: self.analyzer.compare_position_dominance(
                df,
                start_row=start_row,
                end_row=end_row,
            ),
            "analyze_pre_post_shift": lambda: self.analyzer.analyze_pre_post_shift(
                df,
                impact_row=impact_row,
                start_row=start_row,
                end_row=end_row,
            ),
            "analyze_posture_angle_change": lambda: self.analyzer.analyze_posture_angle_change(
                df,
                impact_row=impact_row,
                start_row=start_row,
                end_row=end_row,
            ),
            "analyze_post_impact_stillness": lambda: self.analyzer.analyze_post_impact_stillness(
                df,
                impact_row=impact_row,
                start_row=start_row,
                end_row=end_row,
            ),
            "detect_persistent_offset": lambda: self.analyzer.detect_persistent_offset(
                df,
                impact_row=impact_row,
                start_row=start_row,
                end_row=end_row,
            ),
            "analyze_recovery": lambda: self.analyzer.analyze_recovery(
                df,
                impact_row=impact_row,
                start_row=start_row,
                end_row=end_row,
            ),
        }

        def timed_tool_call(name: str, call: Any, submitted_at: float) -> tuple[str, Any, float, float]:
            worker_started_at = time.perf_counter()
            started_at = time.perf_counter()
            return (
                name,
                call(),
                time.perf_counter() - started_at,
                worker_started_at - submitted_at,
            )

        tool_results: dict[str, Any] = {"detect_impact_peak": peak}
        tool_timings: dict[str, float] = {
            "read_parquet": round(read_seconds, 3),
            "detect_impact_peak": round(peak_seconds, 3),
        }
        queue_timings: dict[str, float] = {}
        executor_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            submit_start = time.perf_counter()
            futures = {}
            for name, call in tool_calls.items():
                submitted_at = time.perf_counter()
                futures[executor.submit(timed_tool_call, name, call, submitted_at)] = name
            submit_seconds = time.perf_counter() - submit_start
            collect_start = time.perf_counter()
            for future in as_completed(futures):
                name, result, seconds, queue_seconds = future.result()
                tool_results[name] = result
                tool_timings[name] = round(seconds, 3)
                queue_timings[name] = round(queue_seconds, 3)
            collect_seconds = time.perf_counter() - collect_start
        executor_seconds = time.perf_counter() - executor_start
        range_total_seconds = time.perf_counter() - range_start
        measured_seconds = read_seconds + peak_seconds + executor_seconds

        return {
            "input": {
                "parquet_path": str(path),
                "start_row": start_row,
                "end_row": end_row,
                "sampling_rate": float(self.config["sampling_rate"]),
            },
            "tool_results": tool_results,
            "timing": {
                "range_total_seconds": round(range_total_seconds, 3),
                "executor_seconds": round(executor_seconds, 3),
                "executor_submit_seconds": round(submit_seconds, 3),
                "executor_collect_seconds": round(collect_seconds, 3),
                "unmeasured_seconds": round(range_total_seconds - measured_seconds, 3),
                "tool_seconds": tool_timings,
                "queue_wait_seconds": queue_timings,
            },
        }

    def build_impact_analyses(
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
                    self.build_impact_analysis,
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
        impact_analysis: dict[str, Any],
    ) -> str:
        prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        replacements = {
            "thread_1_output": self._format_thread_1_context(thread_1_output) if hasattr(self, "_format_thread_1_context") else str(thread_1_output),
            "impact_analysis_json": self._format_impact_analysis_table(impact_analysis),
        }
        for key, value in replacements.items():
            prompt = prompt.replace("{" + key + "}", value)
        return prompt

    def create_agent(self):
        return (
            AgentBuilder()
            .with_model(self.model)
            .with_system_prompt(
                "You are Thread 2 impact-event reasoning. Use the provided "
                "acceleration analysis evidence and analyze it."
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
        impact_analysis = self.build_impact_analyses(
            thread_1_output=thread_1_output,
            parquet_path=parquet_path,
            start_row=start_row,
            end_row=end_row,
        )
        analysis_seconds = time.perf_counter() - analysis_start
        prompt_start = time.perf_counter()
        prompt = self.build_prompt(
            thread_1_output=thread_1_output,
            impact_analysis=impact_analysis,
        )
        prompt_seconds = time.perf_counter() - prompt_start
        self._print_impact_agent_input(prompt)
        llm_start = time.perf_counter()
        result = self.create_agent().run_turn(prompt)
        raw_output = result.get("text", "") if isinstance(result, dict) else str(result)
        llm_seconds = time.perf_counter() - llm_start
        return {
            "impact_analysis": impact_analysis,
            "agent_output": self._clean_text_output(raw_output),
            "timing": {
                "analysis_seconds": round(analysis_seconds, 3),
                "prompt_build_seconds": round(prompt_seconds, 3),
                "llm_seconds": round(llm_seconds, 3),
                "total_seconds": round(time.perf_counter() - total_start, 3),
                "analysis_ranges": self._analysis_timing_summary(impact_analysis),
            },
        }

    @staticmethod
    def _summarize_magnitudes(magnitudes: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
        summary = {}
        for position, values in magnitudes.items():
            if isinstance(values, dict) and {"count", "mean", "std", "min", "max"} <= set(values):
                summary[position] = values
                continue
            finite = [float(value) for value in values if value == value]
            if not finite:
                summary[position] = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }
                continue
            mean = sum(finite) / len(finite)
            variance = sum((value - mean) ** 2 for value in finite) / len(finite)
            summary[position] = {
                "count": len(finite),
                "mean": mean,
                "std": variance ** 0.5,
                "min": min(finite),
                "max": max(finite),
            }
        return summary

    @staticmethod
    def _clean_text_output(raw_output: Any) -> str:
        return clean_text_output(raw_output)

    @staticmethod
    def _print_impact_agent_input(prompt: str) -> None:
        green = "\033[92m"
        reset = "\033[0m"
        print(
            f"{green}[IMPACT AGENT INPUT]\n{prompt}\n{reset}",
            flush=True,
        )

    @staticmethod
    def _analysis_timing_summary(impact_analysis: dict[str, Any]) -> list[dict[str, Any]]:
        summary = []
        analyses = impact_analysis.get("analyses", [])
        if "tool_results" in impact_analysis:
            analyses = [impact_analysis]

        for analysis in analyses:
            input_info = analysis.get("input", {})
            timing = analysis.get("timing", {})
            summary.append(
                {
                    "range": f"{input_info.get('start_row')}-{input_info.get('end_row')}",
                    "total_seconds": timing.get("range_total_seconds"),
                    "executor_seconds": timing.get("executor_seconds"),
                    "executor_submit_seconds": timing.get("executor_submit_seconds"),
                    "executor_collect_seconds": timing.get("executor_collect_seconds"),
                    "unmeasured_seconds": timing.get("unmeasured_seconds"),
                    "tool_seconds": timing.get("tool_seconds", {}),
                    "queue_wait_seconds": timing.get("queue_wait_seconds", {}),
                }
            )
        return summary

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
    def _format_impact_analysis_table(cls, impact_analysis: dict[str, Any]) -> str:
        rows = []
        analyses = impact_analysis.get("analyses", [])
        if "tool_results" in impact_analysis:
            analyses = [impact_analysis]

        for analysis in analyses:
            input_info = analysis.get("input", {})
            start = input_info.get("start_row")
            end = input_info.get("end_row")
            tools = analysis.get("tool_results", {})
            rows.append(f"range: {start}-{end}")
            rows.append("")
            rows.append("| feature | scope | value |")
            rows.append("|---|---|---:|")
            if start is not None and end is not None:
                rows.append(f"| range_rows | full_range | {int(end) - int(start)} |")

            magnitude = tools.get("compute_acc_magnitude", {})
            for position, values in magnitude.items():
                if not isinstance(values, dict):
                    continue
                rows.append(f"| acc_mean | {position} | {float(values.get('mean', 0.0) or 0.0):.4f} |")
                rows.append(f"| acc_max | {position} | {float(values.get('max', 0.0) or 0.0):.4f} |")
                rows.append(f"| acc_std | {position} | {float(values.get('std', 0.0) or 0.0):.4f} |")

            peak = tools.get("detect_impact_peak", {})
            peak_scope = str(peak.get("impact_position") or "unknown")
            rows.append(f"| impact_peak_row | {peak_scope} | {peak.get('impact_row')} |")
            rows.append(f"| impact_zscore | {peak_scope} | {float(peak.get('impact_zscore', 0.0) or 0.0):.4f} |")
            rows.append(f"| impact_max_magnitude | {peak_scope} | {float(peak.get('max_magnitude', 0.0) or 0.0):.4f} |")

            transient = tools.get("analyze_transient_dynamics", {})
            for position, values in transient.get("per_position", {}).items():
                rows.append(f"| peak_jerk | {position} | {float(values.get('peak_jerk', 0.0) or 0.0):.4f} |")
                rows.append(f"| peak_jerk_zscore | {position} | {float(values.get('peak_jerk_zscore', 0.0) or 0.0):.4f} |")
                rows.append(f"| impulse_fraction | {position} | {float(values.get('impulse_fraction', 0.0) or 0.0):.4f} |")
                rows.append(f"| stft_high_freq_ratio | {position} | {float(values.get('stft_high_freq_ratio', 0.0) or 0.0):.4f} |")
                rows.append(f"| stft_transient_concentration | {position} | {float(values.get('stft_transient_concentration', 0.0) or 0.0):.4f} |")
                rows.append(f"| stft_frame_count | {position} | {int(values.get('stft_frame_count', 0) or 0)} |")

            dominance = tools.get("compare_position_dominance", {})
            for position, value in dominance.get("normalized_position_scores", {}).items():
                rows.append(f"| normalized_position_energy | {position} | {float(value):.4f} |")

            shift = tools.get("analyze_pre_post_shift", {}).get("per_position", {})
            for position, values in shift.items():
                rows.append(f"| pre_post_shift | {position} | {float(values.get('shift', 0.0) or 0.0):.4f} |")
                rows.append(f"| pre_post_shift_score | {position} | {float(values.get('shift_score', 0.0) or 0.0):.4f} |")

            angle = tools.get("analyze_posture_angle_change", {}).get("per_position", {})
            for position, values in angle.items():
                rows.append(f"| posture_angle_change_deg | {position} | {float(values.get('angle_change_deg', 0.0) or 0.0):.4f} |")

            stillness = tools.get("analyze_post_impact_stillness", {}).get("per_position", {})
            for position, values in stillness.items():
                rows.append(f"| post_pre_std_ratio | {position} | {float(values.get('post_std_ratio', 0.0) or 0.0):.4f} |")

            offset = tools.get("detect_persistent_offset", {})
            for position, values in offset.get("per_position", {}).items():
                rows.append(f"| persistent_offset_fraction | {position} | {float(values.get('persistent_offset_fraction', 0.0) or 0.0):.4f} |")
                rows.append(f"| post_mean_shift_score | {position} | {float(values.get('post_mean_shift_score', 0.0) or 0.0):.4f} |")

            recovery = tools.get("analyze_recovery", {}).get("per_position", {})
            for position, values in recovery.items():
                rows.append(f"| recovery_error | {position} | {float(values.get('recovery_error', 0.0) or 0.0):.4f} |")
            rows.append("")

        return "\n".join(
            add_feature_references(rows, IMPACT_FEATURE_REFERENCES)
        ).strip()


def analyze_impact_event(
    thread_1_output: Any,
    model: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    parquet_path: str | Path | None = None,
    start_row: int = 0,
    end_row: int | None = None,
) -> dict[str, Any]:
    return Thread2ImpactAgent(config_path=config_path, model=model).run(
        thread_1_output=thread_1_output,
        parquet_path=parquet_path,
        start_row=start_row,
        end_row=end_row,
    )


__all__ = [
    "Thread2ImpactAgent",
    "analyze_impact_event",
]

