from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "perception_layer").exists():
            return parent
    return current.parents[2]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from perception_layer.agent.thread_1_update.vision_agent import VisionOnlyAgent
from perception_layer.agent.thread_2_wt import Thread2WTFinalReasoningAgent


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "output" / "scenarios_300"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "main2_repeat_runs"
CATEGORY_ALIASES = {
    "non_impact": {"non_impact", "non-impact", "health"},
    "sensor": {"sensor", "sensor_fault", "sensor-fault"},
    "impact": {"impact"},
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def ground_truth_label(scenario_dir: Path) -> str:
    data = read_json(scenario_dir / "ground_truth.json")
    anomaly = data.get("anomaly_range") or data.get("anomaly_ranges")
    if isinstance(anomaly, list) and anomaly:
        anomaly = anomaly[0]
    if not isinstance(anomaly, dict):
        return ""
    label = str(
        anomaly.get("label")
        or anomaly.get("anomaly_type")
        or anomaly.get("type")
        or ""
    ).strip().lower()
    if label in {"health", "non-impact", "non_impact"}:
        return "non_impact"
    if "sensor" in label:
        return "sensor"
    if "impact" in label:
        return "impact"
    return label


def scenario_dirs_from_root(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.glob("scenario*") if path.is_dir())


def select_category_scenarios(
    source_root: Path,
    category: str,
    limit: int,
) -> list[Path]:
    category_root_candidates = [
        source_root / category,
        source_root / f"{category}_scenarios",
        PROJECT_ROOT / "output" / f"{category}_scenarios",
        PROJECT_ROOT / "output" / f"{category}_test_scenarios",
    ]
    if category == "non_impact":
        category_root_candidates.append(
            PROJECT_ROOT / "output" / "non_impact_test_scenarios"
        )

    for root in category_root_candidates:
        scenarios = scenario_dirs_from_root(root)
        if scenarios:
            return scenarios[:limit]

    aliases = CATEGORY_ALIASES[category]
    selected = [
        scenario_dir
        for scenario_dir in scenario_dirs_from_root(source_root)
        if ground_truth_label(scenario_dir) in aliases
    ]
    return selected[:limit]


def ensure_thread_1(
    scenario_dir: Path,
    vision_agent: VisionOnlyAgent,
    rerun_vision: bool,
) -> str:
    thread_1_path = scenario_dir / "thread_1_output.txt"
    image_folder = scenario_dir / "image"
    if thread_1_path.exists() and not rerun_vision:
        return thread_1_path.read_text(encoding="utf-8")
    if not image_folder.exists():
        raise FileNotFoundError(f"Missing image folder: {image_folder}")
    with contextlib.redirect_stdout(io.StringIO()):
        output = vision_agent.run(image_folder)
    if not output.strip():
        raise RuntimeError("Vision returned empty Thread 1 output.")
    thread_1_path.write_text(output, encoding="utf-8")
    return output


def run_one_scenario(
    scenario_dir: Path,
    run_index: int,
    category: str,
    thread_2_agent: Thread2WTFinalReasoningAgent,
    vision_agent: VisionOnlyAgent,
    output_root: Path,
    rerun_vision: bool,
) -> dict[str, Any]:
    parquet_path = scenario_dir / "data.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

    started_at = time.perf_counter()
    thread_1_output = ensure_thread_1(
        scenario_dir=scenario_dir,
        vision_agent=vision_agent,
        rerun_vision=rerun_vision,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        thread_2_result = thread_2_agent.run(
            thread_1_output=thread_1_output,
            parquet_path=parquet_path,
        )

    record = {
        "category": category,
        "run_index": run_index,
        "scenario": scenario_dir.name,
        "scenario_path": str(scenario_dir),
        "data_path": str(parquet_path),
        "ground_truth": read_json(scenario_dir / "ground_truth.json"),
        "thread_2_wt_output": thread_2_result.get("final_output", ""),
        "timing": thread_2_result.get("timing", {}),
        "wall_seconds": round(time.perf_counter() - started_at, 3),
    }

    category_dir = output_root / category / f"run_{run_index:02d}"
    category_dir.mkdir(parents=True, exist_ok=True)
    output_path = category_dir / f"{scenario_dir.name}.json"
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    record["output_path"] = str(output_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run main_2-style Thread 1/Thread 2 analysis repeatedly for "
            "non-impact, sensor, and impact scenario groups."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--limit-per-category", type=int, default=10)
    parser.add_argument("--rerun-vision", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root
    if not source_root.is_absolute():
        source_root = PROJECT_ROOT / source_root
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    thread_2_agent = Thread2WTFinalReasoningAgent()
    vision_agent = VisionOnlyAgent()
    all_records: list[dict[str, Any]] = []

    for category in ("non_impact", "sensor", "impact"):
        scenarios = select_category_scenarios(
            source_root=source_root,
            category=category,
            limit=args.limit_per_category,
        )
        if not scenarios:
            print(f"[WARN] {category}: no scenarios found", flush=True)
            continue

        print(
            f"[CATEGORY] {category}: {len(scenarios)} scenarios, "
            f"{args.repeats} repeats",
            flush=True,
        )
        for run_index in range(1, args.repeats + 1):
            run_records = []
            print(f"[RUN] {category} run {run_index:02d}", flush=True)
            for scenario_index, scenario_dir in enumerate(scenarios, start=1):
                try:
                    record = run_one_scenario(
                        scenario_dir=scenario_dir,
                        run_index=run_index,
                        category=category,
                        thread_2_agent=thread_2_agent,
                        vision_agent=vision_agent,
                        output_root=output_root,
                        rerun_vision=args.rerun_vision,
                    )
                    run_records.append(record)
                    all_records.append(record)
                    print(
                        f"  [{scenario_index}/{len(scenarios)}] "
                        f"{scenario_dir.name}: saved {record['output_path']}",
                        flush=True,
                    )
                except Exception as exc:
                    error_record = {
                        "category": category,
                        "run_index": run_index,
                        "scenario": scenario_dir.name,
                        "scenario_path": str(scenario_dir),
                        "error": str(exc),
                    }
                    run_records.append(error_record)
                    all_records.append(error_record)
                    print(
                        f"  [{scenario_index}/{len(scenarios)}] "
                        f"{scenario_dir.name}: ERROR {exc}",
                        flush=True,
                    )

            jsonl_path = output_root / category / f"run_{run_index:02d}.jsonl"
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in run_records
                ),
                encoding="utf-8",
            )

    summary_path = output_root / "all_runs.jsonl"
    summary_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in all_records
        ),
        encoding="utf-8",
    )
    print(f"[DONE] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
