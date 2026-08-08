from __future__ import annotations

import time
from pathlib import Path

from agent_sdk import AgentBuilder

from perception_layer.agent.thread_1.evidence_agent.utils import load_thread_1_model
from perception_layer.path_config import DEFAULT_CONFIG_PATH
from perception_layer.timing import record_timing


PROMPT_FILES = {
    "trend_stability": "trend_stability_prompt.md",
    "seasonality_rhythm_stability": "seasonality_rhythm_stability_prompt.md",
    "spike_strength": "spike_strength_prompt.md",
    "final_window_synthesis": "final_synthesis_prompt.md",
}


def print_colored(title: str, content: str, color_code: str) -> None:
    reset = "\033[0m"
    print(f"\n{color_code}[{title}]\n{content}\n{reset}", flush=True)


class EvidenceSubagent:
    def __init__(
        self,
        model: str | None = None,
        timeout: int = 180,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.prompt_dir = Path(__file__).parent / "prompt"
        self.model = model
        self.timeout = timeout
        self.config_path = config_path

    def prompt_path_for(self, mode: str, group_name: str = "") -> Path:
        prompt_key = group_name if mode == "group_window_analysis" else mode
        prompt_file = PROMPT_FILES.get(prompt_key)
        if not prompt_file:
            raise ValueError(f"No prompt file configured for mode={mode!r}, group={group_name!r}.")
        return self.prompt_dir / prompt_file

    def build_prompt(
        self,
        evidence_inputs: dict[str, str],
        context: str = "",
        mode: str = "group_window_analysis",
        group_name: str = "",
        group_outputs: dict[str, str] | None = None,
    ) -> str:
        prompt = self.prompt_path_for(mode, group_name).read_text(encoding="utf-8").strip()
        group_output_text = ""
        if group_outputs:
            group_output_text = "\n\n".join(
                [
                    f"<{name}_analysis>\n{output.strip()}\n</{name}_analysis>"
                    for name, output in group_outputs.items()
                ]
            )
        replacements = {
            "{{mode}}": mode.strip(),
            "{{group_name}}": group_name.strip() or "all_groups",
            "{{context}}": context.strip() or "No additional context provided.",
            "{{trend_stability_evidence}}": evidence_inputs.get("trend_stability", "").strip(),
            "{{seasonality_rhythm_stability_evidence}}": evidence_inputs.get(
                "seasonality_rhythm_stability",
                "",
            ).strip(),
            "{{spike_strength_evidence}}": evidence_inputs.get("spike_strength", "").strip(),
            "{{group_evidence}}": evidence_inputs.get(group_name, "").strip(),
            "{{group_outputs}}": group_output_text.strip(),
        }
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value or "EMPTY")
        return prompt

    def run(
        self,
        evidence_inputs: dict[str, str],
        context: str = "",
        mode: str = "group_window_analysis",
        group_name: str = "",
        group_outputs: dict[str, str] | None = None,
        debug: bool = False,
    ) -> str:
        started_at = time.perf_counter()
        model = self.model or load_thread_1_model(self.config_path)
        prompt_text = self.build_prompt(
            evidence_inputs=evidence_inputs,
            context=context,
            mode=mode,
            group_name=group_name,
            group_outputs=group_outputs,
        )
        
        if debug:
            if mode == "group_window_analysis":
                print_colored(f"SUBAGENT INPUT (PROMPT): {group_name.upper()}", prompt_text, "\033[93m")  # Yellow
            else:
                print_colored("FINAL AGENT INPUT (PROMPT)", prompt_text, "\033[95m")  # Magenta
        
        agent = (
            AgentBuilder()
            .with_model(model)
            .with_system_prompt("You are a concise wearable-sensor evidence analyst.")
            .build()
        )
        result = agent.run_turn(prompt_text)
        raw_output = result.get("text", "") if isinstance(result, dict) else str(result)
        
        if debug:
            if mode == "group_window_analysis":
                print_colored(f"SUBAGENT OUTPUT: {group_name.upper()}", raw_output, "\033[92m")  # Green
            else:
                print_colored("FINAL AGENT OUTPUT", raw_output, "\033[94m")  # Blue
        
        record_timing(
            "thread_1.evidence_agent",
            time.perf_counter() - started_at,
            model=model,
        )
        return raw_output

    def run_group(self, group_name: str, group_evidence: str, context: str = "", debug: bool = False) -> str:
        return self.run(
            evidence_inputs={group_name: group_evidence},
            context=context,
            mode="group_window_analysis",
            group_name=group_name,
            debug=debug,
        )

    def run_final(self, group_outputs: dict[str, str], context: str = "", debug: bool = False) -> str:
        return self.run(
            evidence_inputs={},
            context=context,
            mode="final_window_synthesis",
            group_outputs=group_outputs,
            debug=debug,
        )
