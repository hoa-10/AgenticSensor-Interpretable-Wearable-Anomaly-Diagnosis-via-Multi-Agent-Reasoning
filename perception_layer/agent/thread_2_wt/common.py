from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
if os.getenv("OPENROUTER_API_KEY"):
    os.environ.pop("LOCAL_BASE_URL", None)
    os.environ.pop("LOCAL_API_KEY", None)

from perception_layer.path_config import DEFAULT_CONFIG_PATH, get_config_path, read_config


def _required_config_value(config, section: str, key: str) -> str:
    value = config.get(section, key, fallback="").strip()
    if not value:
        raise ValueError(f"Missing [{section}] {key} in perception config.")
    return value


def load_thread_2_impact_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = read_config(config_path)
    return {
        "parquet_path": get_config_path(config, "model_parquet_path"),
        "sampling_rate": config.getfloat("thread_2", "sampling_rate", fallback=50.0),
        "model": _required_config_value(config, "thread_2", "model"),
    }


def round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 3)
    if isinstance(obj, dict):
        return {key: round_floats(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_floats(value) for value in obj]
    return obj


def json_for_prompt(value: Any) -> str:
    rounded = round_floats(value)
    return json.dumps(rounded, ensure_ascii=False, separators=(",", ":"))


def json_for_log(value: Any) -> str:
    rounded = round_floats(value)
    return json.dumps(rounded, ensure_ascii=False, indent=2)


def tool_result_for_prompt(analysis: dict[str, Any], tool_name: str) -> Any:
    if "tool_results" in analysis:
        return analysis["tool_results"].get(tool_name)

    rows = []
    for row in analysis.get("analyses", []):
        rows.append(
            {
                "range": row.get("input", {}),
                "result": row.get("tool_results", {}).get(tool_name),
            }
        )
    return rows


def add_feature_references(
    rows: list[str],
    references: dict[str, str],
) -> list[str]:
    """Add a concise interpretation column to generated feature tables."""
    formatted: list[str] = []
    for row in rows:
        if row == "| feature | scope | value |":
            formatted.append("| feature | scope | value | reference |")
            continue
        if row == "|---|---|---:|":
            formatted.append("|---|---|---:|---|")
            continue
        if row.startswith("| ") and row.endswith(" |"):
            cells = [cell.strip() for cell in row[2:-2].split(" | ")]
            if len(cells) == 3:
                feature, scope, value = cells
                reference = references.get(feature, "Observed evidence value")
                formatted.append(
                    f"| {feature} | {scope} | {value} | {reference} |"
                )
                continue
        formatted.append(row)
    return formatted


def extract_tagged_block(value: Any, tag: str) -> str:
    text = str(value or "")
    match = re.search(
        rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(0).strip() if match else ""


def clean_text_output(raw_output: Any) -> str:
    text = str(raw_output)
    text = re.sub(r"\x1b7.*?\x1b8", "", text, flags=re.DOTALL)
    text = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text).strip()
    text = re.sub(
        r"<thinking>.*?</thinking>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    return text


def print_phase(title: str, content: Any) -> None:
    line = "=" * 88
    print(
        f"\n{line}\n[{title}]\n{line}\n{content}\n{line}\n",
        flush=True,
    )
