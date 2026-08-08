from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "perception_layer").exists():
            return parent
    return current.parents[3]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from perception_layer.agent.thread_1_update.pipeline import run_pipeline
from perception_layer.agent.thread_2 import Thread2FinalReasoningAgent


SCENARIO_ROOT = PROJECT_ROOT / "output" / "generated_scenarios"
RESULTS_PATH = SCENARIO_ROOT / "thread_2_outputs.jsonl"
SLEEP_SECONDS = 2

COLORS = {
    "reset": "\033[0m",
    "main": "\033[97m",
    "input": "\033[95m",
    "output": "\033[94m",
    "ok": "\033[92m",
    "time": "\033[96m",
}


def print_color(title: str, content: object, color: str) -> None:
    print(f"\n{color}[{title}]\n{content}\n{COLORS['reset']}", flush=True)


def main() -> None:
    scenario_dirs = sorted(SCENARIO_ROOT.glob("scenario_*"))
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario folders found under {SCENARIO_ROOT}")

    print_color(
        "MAIN PIPELINE",
        f"Running {len(scenario_dirs)} scenario folders under: {SCENARIO_ROOT}",
        COLORS["main"],
    )

    thread_2_agent = Thread2FinalReasoningAgent()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("", encoding="utf-8")
    completed = 0

    for index, scenario_dir in enumerate(scenario_dirs, start=1):
        print_color(
            "PIPELINE PROGRESS",
            f"[{index}/{len(scenario_dirs)}] running {scenario_dir.name}",
            COLORS["time"],
        )

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                thread_1_summaries = run_pipeline(
                    scenario_root=scenario_dir,
                    debug=False,
                    emit_output=False,
                )
            if not thread_1_summaries:
                print_color("PIPELINE WARNING", f"{scenario_dir.name}: no Thread 1 output", COLORS["time"])
                continue

            summary = thread_1_summaries[0]
            thread_1_output = str(summary["final_output"])
            parquet_path = Path(str(summary["data_path"]))

            print_color("PIPELINE PROGRESS", f"{scenario_dir.name}: Thread 1 done, running Thread 2", COLORS["time"])
            with contextlib.redirect_stdout(io.StringIO()):
                thread_2_result = thread_2_agent.run(
                    thread_1_output=thread_1_output,
                    parquet_path=parquet_path,
                )

            record = {
                "scenario": scenario_dir.name,
                "data_path": str(parquet_path),
                "thread_2_output": thread_2_result.get("final_output", ""),
                "timing": thread_2_result.get("timing", {}),
            }
            (scenario_dir / "thread_1_output.txt").write_text(thread_1_output, encoding="utf-8")
            (scenario_dir / "thread_2_output.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with RESULTS_PATH.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

            print_color(
                "PIPELINE PROGRESS",
                f"{scenario_dir.name}: Thread 2 done, saved result",
                COLORS["ok"],
            )
            completed += 1
        except Exception as exc:
            record = {
                "scenario": scenario_dir.name,
                "error": str(exc),
            }
            (scenario_dir / "thread_2_output.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with RESULTS_PATH.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print_color("PIPELINE ERROR", f"{scenario_dir.name}: {exc}", COLORS["time"])

        if index < len(scenario_dirs):
            print_color("PIPELINE PAUSE", f"sleeping {SLEEP_SECONDS}s before next scenario", COLORS["time"])
            time.sleep(SLEEP_SECONDS)

    print_color(
        "MAIN PIPELINE COMPLETE",
        f"completed_scenarios: {completed}/{len(scenario_dirs)}\noutput_file: {RESULTS_PATH}",
        COLORS["ok"],
    )


if __name__ == "__main__":
    main()
