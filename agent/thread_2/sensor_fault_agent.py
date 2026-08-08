from __future__ import annotations

import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from agent_sdk import AgentBuilder
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
if os.getenv("OPENROUTER_API_KEY"):
    os.environ.pop("LOCAL_BASE_URL", None)
    os.environ.pop("LOCAL_API_KEY", None)

from perception_layer.agent.thread_2.common import (
    add_feature_references,
    clean_text_output,
    load_thread_2_impact_config,
)
from perception_layer.agent.utils import extract_anomaly_ranges
from perception_layer.path_config import DEFAULT_CONFIG_PATH
from perception_layer.utils import SensorFaultAnalyzer


PROMPT_PATH = PROJECT_ROOT / "perception_layer" / "prompt" / "thread_2" / "reasoning_prompt_3.md"

SENSOR_FEATURE_REFERENCES = {
    "stuck_channel_count": "Channels meeting the stuck-value rule",
    "zero_diff_fraction": "Fraction of unchanged adjacent samples; near 1 indicates a stuck signal",
    "event_std": "Signal standard deviation inside the candidate range",
    "event_mean": "Signal mean inside the candidate range",
    "dropout_channel_count": "Channels containing dropout evidence",
    "nan_fraction": "Fraction of missing samples inside the candidate range",
    "finite_count": "Number of finite samples inside the candidate range",
    "clipping_channel_count": "Channels meeting the clipping rule",
    "clipping_fraction": "Fraction pinned at signal bounds; above 0.15 supports clipping",
    "event_min": "Minimum value inside the candidate range",
    "event_max": "Maximum value inside the candidate range",
    "direct_fault_channel_count": "Channels with direct stuck, dropout, or clipping evidence",
    "direct_fault_position_count": "Body positions containing direct sensor-fault evidence",
    "evaluated_channel_count": "Channels evaluated for distribution shift",
    "suspicious_shift_channel_count": "Channels meeting at least one distribution-shift rule",
    "variance_ratio": "Event/baseline variance; outside 0.5 to 2 is suspicious",
    "mean_shift_z": "Mean shift normalized by baseline standard deviation; at least 3 is suspicious",
    "correlation_shift": "Change in within-position channel correlation",
    "spectral_flatness_delta": "Event minus baseline spectral flatness",
    "shift_rank_score": "Combined ranking score for distribution shift",
    "position_suspicious_channel_count": "Suspicious shifted channels at this body position",
    "position_evaluated_channel_count": "Evaluated channels at this body position",
    "position_max_shift_rank_score": "Largest channel shift score at this body position",
}


class Thread2SensorFaultAgent:
    """Run sensor-fault analysis, then ask an agent to reason over the evidence."""

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
        self.analyzer = SensorFaultAnalyzer()

    def build_fault_analysis(
        self,
        parquet_path: str | Path | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        path = Path(parquet_path or self.config["parquet_path"])
        df = pd.read_parquet(path)

        tool_calls = {
            "detect_stuck_values": lambda: self.analyzer.detect_stuck_values(
                df,
                start_row=start_row,
                end_row=end_row,
            ),
            "detect_data_dropouts": lambda: self.analyzer.detect_data_dropouts(
                df,
                start_row=start_row,
                end_row=end_row,
            ),
            "detect_signal_clipping": lambda: self.analyzer.detect_signal_clipping(
                df,
                start_row=start_row,
                end_row=end_row,
            ),
            "check_channel_independence": lambda: self.analyzer.check_channel_independence(
                df,
                start_row=start_row,
                end_row=end_row,
            ),
            "analyze_distribution_shift": lambda: self.analyzer.analyze_distribution_shift(
                df,
                start_row=start_row,
                end_row=end_row,
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
                "start_row": start_row,
                "end_row": end_row,
            },
            "tool_results": tool_results,
        }

    def build_fault_analyses(
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
                    self.build_fault_analysis,
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
        fault_analysis: dict[str, Any],
    ) -> str:
        prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        replacements = {
            "thread_1_output": str(thread_1_output),
            "fault_analysis_json": self._format_fault_analysis_table(fault_analysis),
        }
        for key, value in replacements.items():
            prompt = prompt.replace("{" + key + "}", value)
        return prompt

    def create_agent(self):
        return (
            AgentBuilder()
            .with_model(self.model)
            .with_system_prompt(
                "You are Thread 2 sensor-fault reasoning. Use the provided "
                "sensor data-quality evidence and analyze it."
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
        fault_analysis = self.build_fault_analyses(
            thread_1_output=thread_1_output,
            parquet_path=parquet_path,
            start_row=start_row,
            end_row=end_row,
        )
        analysis_seconds = time.perf_counter() - analysis_start
        prompt_start = time.perf_counter()
        prompt = self.build_prompt(
            thread_1_output=thread_1_output,
            fault_analysis=fault_analysis,
        )
        prompt_seconds = time.perf_counter() - prompt_start
        llm_start = time.perf_counter()
        result = self.create_agent().run_turn(prompt)
        raw_output = result.get("text", "") if isinstance(result, dict) else str(result)
        llm_seconds = time.perf_counter() - llm_start
        return {
            "fault_analysis": fault_analysis,
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
    def _format_bool(value: Any) -> str:
        if value is True:
            return "yes"
        if value is False:
            return "no"
        return str(value)

    @staticmethod
    def _affected_channels(result: dict[str, Any]) -> list[str]:
        for key in (
            "affected_channels",
            "stuck_channels",
            "dropout_channels",
            "clipping_channels",
            "faulty_channels",
        ):
            channels = result.get(key)
            if isinstance(channels, list):
                return [str(channel) for channel in channels]

        per_channel = result.get("per_channel", {})
        affected = []
        for channel, values in per_channel.items():
            if not isinstance(values, dict):
                continue
            if (
                values.get("is_stuck")
                or values.get("has_dropout")
                or values.get("is_clipped")
                or values.get("is_faulty")
                or values.get("fault_detected")
            ):
                affected.append(str(channel))
        return affected

    @staticmethod
    def _channel_metric_summary(result: dict[str, Any], channels: list[str]) -> str:
        per_channel = result.get("per_channel", {})
        parts = []
        for channel in channels[:5]:
            values = per_channel.get(channel, {})
            if not isinstance(values, dict):
                continue
            metric_parts = []
            for key in (
                "zero_diff_fraction",
                "event_std",
                "event_mean",
                "nan_fraction",
                "finite_fraction",
                "finite_count",
                "length",
                "clipping_fraction",
                "event_min",
                "event_max",
                "fault_type",
            ):
                if key in values:
                    metric_parts.append(f"{key}={values.get(key)}")
            if metric_parts:
                parts.append(f"{channel}: " + ", ".join(metric_parts))
        return "; ".join(parts)

    @classmethod
    def _format_fault_analysis_table(cls, fault_analysis: dict[str, Any]) -> str:
        rows = []
        analyses = fault_analysis.get("analyses", [])
        if "tool_results" in fault_analysis:
            analyses = [fault_analysis]

        for analysis in analyses:
            input_info = analysis.get("input", {})
            start = input_info.get("start_row")
            end = input_info.get("end_row")
            tools = analysis.get("tool_results", {})
            rows.append(f"range: {start}-{end}")
            rows.append("")
            rows.append("| feature | scope | value |")
            rows.append("|---|---|---:|")

            stuck = tools.get("detect_stuck_values", {})
            stuck_channels = cls._affected_channels(stuck)
            rows.append(f"| stuck_channel_count | all_channels | {len(stuck_channels)} |")
            for channel in stuck_channels[:5]:
                values = stuck.get("per_channel", {}).get(channel, {})
                rows.append(f"| zero_diff_fraction | {channel} | {values.get('zero_diff_fraction')} |")
                rows.append(f"| event_std | {channel} | {values.get('event_std')} |")
                rows.append(f"| event_mean | {channel} | {values.get('event_mean')} |")

            dropout = tools.get("detect_data_dropouts", {})
            dropout_channels = cls._affected_channels(dropout)
            rows.append(f"| dropout_channel_count | all_channels | {len(dropout_channels)} |")
            for channel in dropout_channels[:5]:
                values = dropout.get("per_channel", {}).get(channel, {})
                rows.append(f"| nan_fraction | {channel} | {values.get('nan_fraction')} |")
                rows.append(f"| finite_count | {channel} | {values.get('finite_count')} |")

            clipping = tools.get("detect_signal_clipping", {})
            clipping_channels = cls._affected_channels(clipping)
            rows.append(f"| clipping_channel_count | all_channels | {len(clipping_channels)} |")
            for channel in clipping_channels[:5]:
                values = clipping.get("per_channel", {}).get(channel, {})
                rows.append(f"| clipping_fraction | {channel} | {values.get('clipping_fraction')} |")
                rows.append(f"| event_min | {channel} | {values.get('event_min')} |")
                rows.append(f"| event_max | {channel} | {values.get('event_max')} |")

            independence = tools.get("check_channel_independence", {})
            rows.append(f"| direct_fault_channel_count | all_channels | {independence.get('affected_channel_count', 0)} |")
            rows.append(f"| direct_fault_position_count | all_positions | {independence.get('affected_position_count', 0)} |")

            shift = tools.get("analyze_distribution_shift", {})
            shifted_channels = shift.get("top_shifted_channels", [])[:5]
            rows.append(f"| evaluated_channel_count | all_channels | {shift.get('evaluated_channel_count', 0)} |")
            rows.append(f"| suspicious_shift_channel_count | all_channels | {shift.get('suspicious_channel_count', 0)} |")
            for channel in shifted_channels:
                values = shift.get("per_channel", {}).get(channel, {})
                rows.append(f"| variance_ratio | {channel} | {values.get('variance_ratio')} |")
                rows.append(f"| mean_shift_z | {channel} | {values.get('mean_shift_z')} |")
                rows.append(f"| correlation_shift | {channel} | {values.get('correlation_shift')} |")
                rows.append(f"| spectral_flatness_delta | {channel} | {values.get('spectral_flatness_delta')} |")
                rows.append(f"| shift_rank_score | {channel} | {values.get('shift_rank_score')} |")

            for position, values in shift.get("position_summary", {}).items():
                rows.append(f"| position_suspicious_channel_count | {position} | {values.get('suspicious_channel_count')} |")
                rows.append(f"| position_evaluated_channel_count | {position} | {values.get('evaluated_channel_count')} |")
                rows.append(f"| position_max_shift_rank_score | {position} | {values.get('max_shift_rank_score')} |")
            rows.append("")

        return "\n".join(
            add_feature_references(rows, SENSOR_FEATURE_REFERENCES)
        ).strip()


def analyze_sensor_fault_event(
    thread_1_output: Any,
    model: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    parquet_path: str | Path | None = None,
    start_row: int = 0,
    end_row: int | None = None,
) -> dict[str, Any]:
    return Thread2SensorFaultAgent(config_path=config_path, model=model).run(
        thread_1_output=thread_1_output,
        parquet_path=parquet_path,
        start_row=start_row,
        end_row=end_row,
    )

