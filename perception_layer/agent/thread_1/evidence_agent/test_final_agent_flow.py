from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "perception_layer").exists():
            return parent
    return current.parents[4]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception_layer.agent.thread_1.evidence_agent.agent import EvidenceSubagent
from perception_layer.agent.thread_1.evidence_agent.utils import (
    EVIDENCE_GROUPS,
    build_windowed_evidence_inputs,
)
from perception_layer.path_config import DEFAULT_CONFIG_PATH, get_config_path, read_config


DEFAULT_INPUT_PATH = Path(__file__).with_name("final_agent_sample_input.txt")

GROUP_NAME_MAP = {
    "TREND_STABILITY": "trend_stability",
    "SEASONALITY_RHYTHM_STABILITY": "seasonality_rhythm_stability",
    "SPIKE_STRENGTH": "spike_strength",
}


def extract_subagent_outputs(text: str) -> dict[str, str]:
    pattern = re.compile(r"^\[SUBAGENT OUTPUT: ([^\]]+)\]\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    outputs: dict[str, str] = {}

    for index, match in enumerate(matches):
        raw_group = match.group(1).strip().upper()
        group = GROUP_NAME_MAP.get(raw_group)
        if group is None:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        final_marker = block.find("[FINAL AGENT OUTPUT]")
        if final_marker >= 0:
            block = block[:final_marker].strip()
        outputs[group] = block

    missing = [group for group in GROUP_NAME_MAP.values() if group not in outputs]
    if missing:
        raise ValueError(f"Missing subagent output block(s): {missing}")
    return outputs


def optional_config_path(config, key: str) -> Path | None:
    return get_config_path(config, key, required=False)


def load_test_config(config_path: Path) -> dict:
    config = read_config(config_path)
    dataset_root = optional_config_path(config, "dataset_root")
    parquet_path = (
        optional_config_path(config, "model_parquet_path")
        or optional_config_path(config, "parquet_path")
        or optional_config_path(config, "data_path")
    )
    if parquet_path is None and dataset_root is not None:
        parquet_path = dataset_root / "data.parquet"
    if parquet_path is None:
        raise ValueError(
            "Missing [paths] model_parquet_path/parquet_path/data_path, "
            "or dataset_root in perception config."
        )

    image_folder_path = (
        optional_config_path(config, "image_folder_path")
        or optional_config_path(config, "image_folder")
    )
    if image_folder_path is None:
        image_path = optional_config_path(config, "image_path")
        if image_path is not None:
            image_folder_path = image_path if image_path.is_dir() else image_path.parent
        elif dataset_root is not None:
            image_folder_path = dataset_root / "image"
        else:
            image_folder_path = parquet_path.parent / "image"

    window_size_rows = config.getint("thread_1", "window_size_rows", fallback=750)
    overlap_rows = config.getint("thread_1", "overlap_rows", fallback=-1)

    return {
        "config_path": config_path,
        "dataset_root": dataset_root,
        "parquet_path": parquet_path,
        "image_folder_path": image_folder_path,
        "model": config.get("thread_1", "model", fallback="").strip() or None,
        "window_size_rows": window_size_rows,
        "overlap_rows": overlap_rows if overlap_rows >= 0 else None,
        "overlap_fraction": config.getfloat("thread_1", "overlap_fraction", fallback=0.0),
        "activity_window_size_rows": config.getint("thread_1", "activity_window_size_rows", fallback=100),
        "activity_overlap_rows": config.getint("thread_1", "activity_overlap_rows", fallback=50),
        "sample_rate_hz": config.getfloat("thread_1", "sampling_rate", fallback=50.0),
    }


def print_section(title: str, body: str = "") -> None:
    print(f"\n[{title}]")
    if body:
        print(body)


def validate_config_paths(test_config: dict) -> None:
    parquet_path = test_config["parquet_path"]
    image_folder_path = test_config["image_folder_path"]
    if not parquet_path.exists():
        raise FileNotFoundError(f"data parquet not found: {parquet_path}")
    if not image_folder_path.exists():
        raise FileNotFoundError(f"image folder not found: {image_folder_path}")
    if not image_folder_path.is_dir():
        raise ValueError(f"image path is not a folder: {image_folder_path}")


def run_from_config(args: argparse.Namespace) -> None:
    test_config = load_test_config(args.config_path)
    validate_config_paths(test_config)
    print_section(
        "CONFIG PATHS",
        "\n".join(
            [
                f"config_path: {test_config['config_path']}",
                f"dataset_root: {test_config['dataset_root']}",
                f"parquet_path: {test_config['parquet_path']}",
                f"image_folder_path: {test_config['image_folder_path']}",
                f"window_size_rows: {test_config['window_size_rows']}",
                f"overlap_rows: {test_config['overlap_rows']}",
                f"activity_window_size_rows: {test_config['activity_window_size_rows']}",
                f"activity_overlap_rows: {test_config['activity_overlap_rows']}",
                f"sample_rate_hz: {test_config['sample_rate_hz']}",
                f"model: {args.model or test_config['model']}",
            ]
        ),
    )

    evidence_inputs = build_windowed_evidence_inputs(
        parquet_path=test_config["parquet_path"],
        window_size_rows=test_config["window_size_rows"],
        sample_rate_hz=test_config["sample_rate_hz"],
        overlap_rows=test_config["overlap_rows"],
        overlap_fraction=test_config["overlap_fraction"],
    )
    if not args.no_tables:
        for group in EVIDENCE_GROUPS:
            print_section(f"TOOL ANALYST OUTPUT (TABLE): {group.upper()}", evidence_inputs[group])

    if args.dry_run:
        print_section("DRY RUN", "Config paths and evidence tables were built. No LLM agents were called.")
        return

    agent = EvidenceSubagent(model=args.model or test_config["model"], config_path=args.config_path)
    group_outputs: dict[str, str] = {}
    for group in EVIDENCE_GROUPS:
        output = agent.run_group(
            group_name=group,
            group_evidence=evidence_inputs[group],
            context=args.context,
            debug=args.debug,
        )
        group_outputs[group] = output
        print_section(f"SUBAGENT OUTPUT: {group.upper()}", output)

    final_output = agent.run_final(
        group_outputs=group_outputs,
        context=args.context,
        debug=args.debug,
    )
    print_section("FINAL AGENT OUTPUT", final_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test only the Thread 1 final evidence synthesis flow.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional text file containing [SUBAGENT OUTPUT: ...] blocks. "
            "When omitted, the script reads perception.config and runs the 3 subagents plus final agent."
        ),
    )
    parser.add_argument("--context", default="", help="Optional final-agent context.")
    parser.add_argument("--model", default=None, help="Override text model.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--run-final",
        action="store_true",
        help="With --input, actually call the text final agent. Without --input, full config flow already runs final agent.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not print the built final prompt in dry-run mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With config flow, validate paths and build evidence tables without calling LLM agents.",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="With config flow, hide the large evidence tables from stdout.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is None:
        run_from_config(args)
        return

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. "
            f"For a sample final-only input, use: {DEFAULT_INPUT_PATH}"
        )
    text = args.input.read_text(encoding="utf-8", errors="replace")
    group_outputs = extract_subagent_outputs(text)
    print("Loaded subagent outputs:")
    for group, output in group_outputs.items():
        print(f"- {group}: {len(output.splitlines())} lines")

    agent = EvidenceSubagent(model=args.model, config_path=args.config_path)
    if args.run_final:
        print("\n[FINAL AGENT OUTPUT]")
        print(agent.run_final(group_outputs=group_outputs, context=args.context, debug=args.debug))
        return

    if not args.no_prompt:
        prompt = agent.build_prompt(
            evidence_inputs={},
            context=args.context,
            mode="final_window_synthesis",
            group_outputs=group_outputs,
        )
        print("\n[FINAL AGENT PROMPT DRY RUN]")
        print(prompt)
        return

    print("Dry run completed. Use --run-final to call only the final text agent.")


if __name__ == "__main__":
    main()
