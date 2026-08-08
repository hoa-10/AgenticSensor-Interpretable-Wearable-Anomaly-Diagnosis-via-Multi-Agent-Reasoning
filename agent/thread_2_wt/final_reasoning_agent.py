from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from perception_layer.agent.thread_2_wt.common import (
    PROJECT_ROOT,
    clean_text_output,
    extract_tagged_block,
    json_for_prompt,
    load_thread_2_impact_config,
    print_phase,
)
from perception_layer.agent.thread_2_wt.health_agent import Thread2HealthAgent
from perception_layer.agent.thread_2_wt.impact_agent import Thread2ImpactAgent
from perception_layer.agent.thread_2_wt.openrouter_client import DEFAULT_MODEL, OpenRouterClient
from perception_layer.agent.thread_2_wt.sensor_fault_agent import Thread2SensorFaultAgent
from perception_layer.path_config import DEFAULT_CONFIG_PATH


PROMPT_PATH = PROJECT_ROOT / "perception_layer" / "prompt" / "thread_2" / "reasoning_prompt_final.md"
MAX_THREAD_1_CHARS = 1200


class Thread2FinalReasoningAgent:
    """Run all Thread 2 specialist agents, then synthesize their outputs."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        model: str | None = None,
        timeout: int = 180,
    ) -> None:
        self.config_path = config_path
        self.config = load_thread_2_impact_config(config_path)
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

    def run_specialist_agents(
        self,
        thread_1_output: Any,
        parquet_path: str | Path | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        started_at = time.perf_counter()
        runners: dict[str, Callable[[], dict[str, Any]]] = {
            "impact_agent_output": lambda: Thread2ImpactAgent(
                config_path=self.config_path,
                model=self.model,
                timeout=self.timeout,
            ).run(
                thread_1_output=thread_1_output,
                parquet_path=parquet_path,
                start_row=start_row,
                end_row=end_row,
            ),
            "health_agent_output": lambda: Thread2HealthAgent(
                config_path=self.config_path,
                model=self.model,
                timeout=self.timeout,
            ).run(
                thread_1_output=thread_1_output,
                parquet_path=parquet_path,
                start_row=start_row,
                end_row=end_row,
            ),
            "sensor_fault_agent_output": lambda: Thread2SensorFaultAgent(
                config_path=self.config_path,
                model=self.model,
                timeout=self.timeout,
            ).run(
                thread_1_output=thread_1_output,
                parquet_path=parquet_path,
                start_row=start_row,
                end_row=end_row,
            ),
        }

        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(runners)) as executor:
            futures = {
                executor.submit(runner): name
                for name, runner in runners.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = {
                        "status": "ok",
                        "output": future.result(),
                    }
                except Exception as exc:
                    results[name] = {
                        "status": "error",
                        "error": str(exc),
                    }
        results["_timing"] = {
            "parallel_wall_seconds": round(time.perf_counter() - started_at, 3)
        }
        return results

    def build_prompt(
        self,
        thread_1_output: Any,
        specialist_outputs: dict[str, dict[str, Any]],
    ) -> str:
        prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        replacements = {
            "thread_1_output": self._compact_thread_1_context(thread_1_output),
            "impact_agent_output": self._specialist_text(
                specialist_outputs.get("impact_agent_output")
            ),
            "impact_numeric_evidence": self._numeric_evidence(
                specialist_outputs.get("impact_agent_output"), "impact"
            ),
            "health_agent_output": self._specialist_text(
                specialist_outputs.get("health_agent_output")
            ),
            "health_numeric_evidence": self._numeric_evidence(
                specialist_outputs.get("health_agent_output"), "health"
            ),
            "sensor_fault_agent_output": self._specialist_text(
                specialist_outputs.get("sensor_fault_agent_output")
            ),
            "sensor_numeric_evidence": self._numeric_evidence(
                specialist_outputs.get("sensor_fault_agent_output"), "sensor"
            ),
        }
        for key, value in replacements.items():
            prompt = prompt.replace("{" + key + "}", value)
        return prompt

    @classmethod
    def _compact_thread_1_context(cls, value: Any) -> str:
        text = str(value or "").strip()
        visual_analysis = extract_tagged_block(text, "visual_anomaly_analysis")
        compact = cls._compact_text(text, MAX_THREAD_1_CHARS)
        if visual_analysis and visual_analysis not in compact:
            compact = f"{compact}\n\n{visual_analysis}"
        return compact

    @staticmethod
    def _specialist_text(result: dict[str, Any] | None) -> str:
        if not result:
            return "SPECIALIST_OUTPUT_MISSING"
        if result.get("status") != "ok":
            return f"SPECIALIST_OUTPUT_MISSING: {result.get('error', 'unknown error')}"

        output = result.get("output", {})
        if isinstance(output, dict):
            agent_output = output.get("agent_output")
            if agent_output:
                return str(agent_output)
            final_output = output.get("final_output")
            if final_output:
                return str(final_output)
        return str(output)

    @classmethod
    def _numeric_evidence(
        cls,
        result: dict[str, Any] | None,
        domain: str,
    ) -> str:
        if not result or result.get("status") != "ok":
            return "NUMERIC_EVIDENCE_MISSING"
        output = result.get("output", {})
        if not isinstance(output, dict):
            return "NUMERIC_EVIDENCE_MISSING"

        analysis_key = {
            "impact": "impact_analysis",
            "health": "health_analysis",
            "sensor": "fault_analysis",
        }[domain]
        analysis = output.get(analysis_key, {})
        items = analysis.get("analyses", []) if isinstance(analysis, dict) else []
        if isinstance(analysis, dict) and "tool_results" in analysis:
            items = [analysis]

        summaries = []
        for item in items:
            input_info = item.get("input", {})
            tools = item.get("tool_results", {})
            summary: dict[str, Any] = {
                "range": f"{input_info.get('start_row')}-{input_info.get('end_row')}"
            }
            if domain == "impact":
                peak = tools.get("detect_impact_peak", {})
                transient = tools.get("analyze_transient_dynamics", {}).get("per_position", {})
                angle = tools.get("analyze_posture_angle_change", {}).get("per_position", {})
                stillness = tools.get("analyze_post_impact_stillness", {}).get("per_position", {})
                recovery = tools.get("analyze_recovery", {}).get("per_position", {})
                offset = tools.get("detect_persistent_offset", {})
                summary.update({
                    "peak_position": peak.get("impact_position"),
                    "peak_zscore": peak.get("impact_zscore"),
                    "peak_magnitude": peak.get("max_magnitude"),
                    "impulse_fraction": {
                        position: values.get("impulse_fraction")
                        for position, values in transient.items()
                    },
                    "peak_jerk_zscore": {
                        position: values.get("peak_jerk_zscore")
                        for position, values in transient.items()
                    },
                    "posture_angle_change_deg": {
                        position: values.get("angle_change_deg")
                        for position, values in angle.items()
                    },
                    "post_pre_std_ratio": {
                        position: values.get("post_std_ratio")
                        for position, values in stillness.items()
                    },
                    "recovery_error": {
                        position: values.get("recovery_error")
                        for position, values in recovery.items()
                    },
                    "persistent_offset_positions": offset.get("offset_positions", []),
                })
            elif domain == "health":
                motion = tools.get("analyze_motion_suppression", {}).get("per_position", {})
                respiration = tools.get("analyze_respiratory_pattern", {})
                ecg = tools.get("analyze_ecg_heart_rate", {}).get("per_channel", {})
                frequency = tools.get("analyze_dominant_frequency_shift", {}).get("per_position", {})
                summary.update({
                    "chest_motion": motion.get("chest", {}),
                    "breath_energy_ratio": respiration.get("breath_energy_ratio"),
                    "breath_frequency_shift_pct": respiration.get("breath_freq_shift_pct"),
                    "ecg": {
                        channel: {
                            "hr_energy_ratio": values.get("hr_energy_ratio"),
                            "std_ratio": values.get("std_ratio"),
                            "frequency_shift_pct": values.get("hr_freq_shift_pct"),
                        }
                        for channel, values in ecg.items()
                    },
                    "chest_frequency_drop_pct": frequency.get("chest", {}).get("freq_drop_pct"),
                })
            else:
                stuck = tools.get("detect_stuck_values", {})
                dropout = tools.get("detect_data_dropouts", {})
                clipping = tools.get("detect_signal_clipping", {})
                shift = tools.get("analyze_distribution_shift", {})
                top_shifted = shift.get("top_shifted_channels", [])[:5]
                summary.update({
                    "stuck_channels": Thread2SensorFaultAgent._affected_channels(stuck),
                    "dropout_channels": Thread2SensorFaultAgent._affected_channels(dropout),
                    "clipping_channels": Thread2SensorFaultAgent._affected_channels(clipping),
                    "suspicious_shift_channel_count": shift.get("suspicious_channel_count"),
                    "top_distribution_shifts": {
                        channel: shift.get("per_channel", {}).get(channel, {})
                        for channel in top_shifted
                    },
                })
            summaries.append(summary)
        return json_for_prompt(summaries) if summaries else "NUMERIC_EVIDENCE_MISSING"

    @staticmethod
    def _compact_text(value: Any, max_chars: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        head = text[: max_chars - 120].rstrip()
        return (
            head
            + "\n...[truncated for final arbiter prompt; specialist should use only strongest evidence]..."
        )

    def create_agent(self):
        return OpenRouterClient(
            model=self.model,
            timeout=self.timeout,
            system_prompt=(
                "You are Thread 2 final anomaly classification. Compare the "
                "specialist outputs and classify each Thread 1 anomaly range."
            ),
        )

    def run(
        self,
        thread_1_output: Any,
        parquet_path: str | Path | None = None,
        start_row: int = 0,
        end_row: int | None = None,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        specialist_outputs = self.run_specialist_agents(
            thread_1_output=thread_1_output,
            parquet_path=parquet_path,
            start_row=start_row,
            end_row=end_row,
        )
        print_phase("THREAD 2 SPECIALIST OUTPUTS", json_for_prompt(specialist_outputs))
        prompt_start = time.perf_counter()
        prompt = self.build_prompt(
            thread_1_output=thread_1_output,
            specialist_outputs=specialist_outputs,
        )
        prompt_seconds = time.perf_counter() - prompt_start
        print_phase("THREAD 2 FINAL AGENT INPUT", prompt)
        llm_start = time.perf_counter()
        result = self.create_agent().run_turn(prompt)
        raw_output = result.get("text", "") if isinstance(result, dict) else str(result)
        final_llm_seconds = time.perf_counter() - llm_start
        final_output = clean_text_output(raw_output)
        print_phase("THREAD 2 FINAL AGENT OUTPUT", final_output)
        timing = {
            "specialists_parallel_wall_seconds": specialist_outputs.get(
                "_timing", {}
            ).get("parallel_wall_seconds"),
            "final_prompt_build_seconds": round(prompt_seconds, 3),
            "final_llm_seconds": round(final_llm_seconds, 3),
            "thread_2_total_seconds": round(time.perf_counter() - total_start, 3),
        }
        return {
            "specialist_outputs": specialist_outputs,
            "final_output": final_output,
            "phase3_output": final_output,
            "timing": timing,
        }


def run_reasoning_pipeline(
    thread_1_output: Any,
    model: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    parquet_path: str | Path | None = None,
    start_row: int = 0,
    end_row: int | None = None,
) -> dict[str, Any]:
    return Thread2FinalReasoningAgent(
        config_path=config_path,
        model=model,
    ).run(
        thread_1_output=thread_1_output,
        parquet_path=parquet_path,
        start_row=start_row,
        end_row=end_row,
    )


__all__ = [
    "Thread2FinalReasoningAgent",
    "run_reasoning_pipeline",
]
