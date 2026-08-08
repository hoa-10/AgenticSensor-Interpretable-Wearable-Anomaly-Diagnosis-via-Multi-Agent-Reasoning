from __future__ import annotations

import argparse
import json
import sys
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

from perception_layer.agent.thread_1_update.vision_agent import VisionOnlyAgent
from perception_layer.agent.thread_2_wt import Thread2WTFinalReasoningAgent


SCENARIO_ROOT = PROJECT_ROOT / "output" / "scenarios_300"
RESULTS_PATH = SCENARIO_ROOT / "thread_2_wt_outputs.jsonl"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vision-only Thread 1 and Thread 2 WT.")
    parser.add_argument(
        "project_id",
        type=str,
        help="Scenario/project id to process, for example 297 or scenario_297.",
    )
    return parser.parse_args()


def resolve_project_dir(value: str) -> Path:
    value = value.strip()
    if value.isdigit():
        value = f"scenario_{int(value):03d}"
    path = Path(value)
    if not path.is_absolute():
        path = SCENARIO_ROOT / value
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Project/scenario folder not found: {path}")
    return path


def load_existing_records_without(scenario_name: str) -> list[dict[str, object]]:
    if not RESULTS_PATH.exists():
        return []

    records: list[dict[str, object]] = []
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("scenario") != scenario_name:
            records.append(record)
    return records


def main() -> None:
    args = parse_args()
    scenario_dirs = [resolve_project_dir(args.project_id)]

    print_color(
        "MAIN PIPELINE",
        f"Running {len(scenario_dirs)} scenario folders under: {SCENARIO_ROOT}",
        COLORS["main"],
    )

    thread_2_agent = Thread2WTFinalReasoningAgent()
    vision_agent = VisionOnlyAgent()
    target_scenario = scenario_dirs[0].name
    existing_records = load_existing_records_without(target_scenario)

    RESULTS_PATH.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in existing_records),
        encoding="utf-8",
    )
    completed = 0
    skipped = 0
    resumed_from_thread_1 = 0

    for index, scenario_dir in enumerate(scenario_dirs, start=1):
        thread_1_path = scenario_dir / "thread_1_output.txt"
        parquet_path = scenario_dir / "data.parquet"
        image_folder = scenario_dir / "image"
        result_path = scenario_dir / "thread_2_wt_output.json"

        print_color(
            "PIPELINE PROGRESS",
            f"[{index}/{len(scenario_dirs)}] running {scenario_dir.name}",
            COLORS["time"],
        )

        try:
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing data file: {parquet_path}")
            if not image_folder.exists():
                raise FileNotFoundError(f"Missing image folder: {image_folder}")

            thread_1_output = vision_agent.run(image_folder)
            if not thread_1_output.strip():
                raise RuntimeError("Vision returned no output")
            thread_1_path.write_text(thread_1_output, encoding="utf-8")
            print_color(
                "PIPELINE PROGRESS",
                f"{scenario_dir.name}: Vision done, running Thread 2",
                COLORS["time"],
            )

            thread_2_result = thread_2_agent.run(
                thread_1_output=thread_1_output,
                parquet_path=parquet_path,
            )

            record = {
                "scenario": scenario_dir.name,
                "data_path": str(parquet_path),
                "thread_2_wt_output": thread_2_result.get("final_output", ""),
                "timing": thread_2_result.get("timing", {}),
            }
            result_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with RESULTS_PATH.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

            print_color(
                "PIPELINE PROGRESS",
                (
                    f"{scenario_dir.name}: Thread 2 done\n"
                    f"output: {result_path}"
                ),
                COLORS["ok"],
            )
            completed += 1
        except Exception as exc:
            print_color("PIPELINE ERROR", f"{scenario_dir.name}: {exc}", COLORS["time"])

    print_color(
        "MAIN PIPELINE COMPLETE",
        (
            f"completed_scenarios: {completed}\n"
            f"skipped_scenarios: {skipped}\n"
            f"resumed_from_thread_1: {resumed_from_thread_1}\n"
            f"total_scenarios: {len(scenario_dirs)}\n"
            f"output_file: {RESULTS_PATH}"
        ),
        COLORS["ok"],
    )


if __name__ == "__main__":
    main()
