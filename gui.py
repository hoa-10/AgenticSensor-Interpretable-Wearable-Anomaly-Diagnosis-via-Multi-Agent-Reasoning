from __future__ import annotations

import json
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from perception_layer.agent.thread_1_update.vision_agent import VisionOnlyAgent
from perception_layer.agent.thread_2_wt.common import clean_text_output
from perception_layer.agent.thread_2_wt import Thread2WTFinalReasoningAgent
from perception_layer.agent.thread_2_wt.health_agent import Thread2HealthAgent
from perception_layer.agent.thread_2_wt.impact_agent import Thread2ImpactAgent
from perception_layer.agent.thread_2_wt.sensor_fault_agent import (
    Thread2SensorFaultAgent,
)
from perception_layer.agent.utils import extract_anomaly_ranges


DEFAULT_SCENARIO_ROOT = PROJECT_ROOT / "output" / "scenarios_300"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ACC_GROUPS = {
    "Chest acceleration": ("chest_acc_x", "chest_acc_y", "chest_acc_z"),
    "Left ankle acceleration": (
        "left_ankle_acc_x",
        "left_ankle_acc_y",
        "left_ankle_acc_z",
    ),
    "Right lower-arm acceleration": (
        "right_lower_arm_acc_x",
        "right_lower_arm_acc_y",
        "right_lower_arm_acc_z",
    ),
}


def find_available_port(start_port: int = 7860, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(
        f"No available port found in range {start_port}-{start_port + attempts - 1}."
    )


def scenario_directories(root: str | Path) -> list[Path]:
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return []
    return sorted(
        item for item in path.glob("scenario_*") if item.is_dir()
    )


def scenario_names(root: str | Path) -> list[str]:
    return [path.name for path in scenario_directories(root)]


def resolve_scenario(root: str | Path, scenario_name: str) -> Path:
    matches = {
        path.name: path for path in scenario_directories(root)
    }
    if scenario_name not in matches:
        raise FileNotFoundError(f"Scenario not found: {scenario_name}")
    return matches[scenario_name]


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_parquet(path)


def image_gallery(scenario_dir: Path) -> list[tuple[str, str]]:
    image_dir = scenario_dir / "image"
    if not image_dir.exists():
        return []
    return [
        (str(path), path.stem.replace("_", " ").title())
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def parse_json_text(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {"result": parsed}
            except json.JSONDecodeError:
                pass
    return {"raw_output": text}


def final_anomaly_ranges(final_output: Any) -> list[dict[str, Any]]:
    parsed = parse_json_text(final_output)
    ranges = parsed.get("anomaly_ranges", [])
    if not isinstance(ranges, list):
        return []
    valid = []
    for item in ranges:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            valid.append({**item, "start": start, "end": end})
    return valid


def parse_thread_1_segments(text: str, total_rows: int) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    block_match = re.search(
        r"<normal_activity_ranges>(.*?)</normal_activity_ranges>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if block_match:
        chunks = re.split(
            r"(?=\bstart\s*:)",
            block_match.group(1),
            flags=re.IGNORECASE,
        )
        for chunk in chunks:
            start_match = re.search(r"\bstart\s*:\s*(\d+)", chunk)
            end_match = re.search(r"\bend\s*:\s*(\d+)", chunk)
            activity_match = re.search(r"\bactivity\s*:\s*([^\n]+)", chunk)
            label_match = re.search(r"\blabel\s*:\s*(\d+)", chunk)
            if not start_match or not end_match:
                continue
            segments.append({
                "start": int(start_match.group(1)),
                "end": int(end_match.group(1)),
                "type": "normal",
                "activity": (
                    activity_match.group(1).strip()
                    if activity_match
                    else "normal"
                ),
                "label": (
                    int(label_match.group(1))
                    if label_match
                    else None
                ),
            })

    for item in extract_anomaly_ranges(text):
        segments.append({
            "start": int(item["start_row"]),
            "end": int(item["end_row"]),
            "type": "anomaly",
            "activity": "anomaly candidate",
            "label": None,
        })

    if not segments and total_rows:
        segments.append({
            "start": 0,
            "end": total_rows,
            "type": "unknown",
            "activity": "unclassified",
            "label": None,
        })
    return sorted(segments, key=lambda row: (row["start"], row["end"]))


def signal_figure(
    df: pd.DataFrame,
    anomaly_ranges: list[dict[str, Any]] | None = None,
    title: str = "Multichannel sensor overview",
) -> plt.Figure:
    anomaly_ranges = anomaly_ranges or []
    rows = list(ACC_GROUPS)
    if "ecg_lead_1" in df.columns:
        rows.append("ECG lead 1")

    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(14, 2.25 * len(rows)),
        sharex=True,
    )
    fig.patch.set_facecolor("#080d12")
    axes = np.atleast_1d(axes)
    x = np.arange(len(df))

    for ax, name in zip(axes, rows):
        ax.set_facecolor("#080d12")
        if name == "ECG lead 1":
            values = df["ecg_lead_1"].to_numpy(dtype=float)
            ylabel = "ECG"
        else:
            columns = ACC_GROUPS[name]
            available = [column for column in columns if column in df.columns]
            if len(available) != 3:
                values = np.zeros(len(df))
            else:
                values = np.linalg.norm(
                    df[available].to_numpy(dtype=float),
                    axis=1,
                )
            ylabel = "|a|"

        ax.plot(x, values, color="#22d3ee", linewidth=0.8)
        for item in anomaly_ranges:
            ax.axvspan(
                int(item["start"]),
                int(item["end"]),
                color="#f43f5e",
                alpha=0.28,
            )
        ax.set_ylabel(ylabel, color="#9ccfb2")
        ax.set_title(
            name,
            loc="left",
            fontsize=10,
            fontweight="bold",
            color="#5cff91",
        )
        ax.tick_params(colors="#809d8b")
        for spine in ax.spines.values():
            spine.set_color("#20342b")
        ax.grid(True, color="#254438", alpha=0.35)

    axes[-1].set_xlabel("Data row", color="#9ccfb2")
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#d7ffe8")
    fig.tight_layout()
    return fig


def timeline_figure(
    thread_1_output: str,
    total_rows: int,
    final_output: Any | None = None,
) -> plt.Figure:
    segments = parse_thread_1_segments(thread_1_output, total_rows)
    final_ranges = final_anomaly_ranges(final_output) if final_output else []
    if final_ranges:
        segments = [
            row for row in segments if row["type"] != "anomaly"
        ]
        for item in final_ranges:
            label = str(item.get("label", "anomaly"))
            subtype = str(item.get("subtype", "")).replace("_", " ")
            segments.append({
                "start": item["start"],
                "end": item["end"],
                "type": "anomaly",
                "activity": f"{label}: {subtype}".strip(": "),
            })
        segments.sort(key=lambda row: row["start"])

    fig, ax = plt.subplots(figsize=(14, 2.8))
    fig.patch.set_facecolor("#080d12")
    ax.set_facecolor("#080d12")
    colors = {
        "normal": "#16a34a",
        "anomaly": "#f43f5e",
        "unknown": "#475569",
    }
    for index, item in enumerate(segments):
        start = int(item["start"])
        width = max(1, int(item["end"]) - start)
        ax.barh(
            0,
            width,
            left=start,
            height=0.5,
            color=colors.get(item["type"], "#94a3b8"),
            edgecolor="white",
        )
        if width >= max(100, total_rows * 0.05):
            ax.text(
                start + width / 2,
                0,
                str(item["activity"]).replace("_", " "),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
    ax.set_xlim(0, max(total_rows, 1))
    ax.set_yticks([])
    ax.set_xlabel("Data row", color="#9ccfb2")
    ax.set_title(
        "Before - anomaly event - after timeline",
        color="#d7ffe8",
    )
    ax.tick_params(colors="#809d8b")
    for spine in ax.spines.values():
        spine.set_color("#20342b")
    ax.grid(True, axis="x", color="#254438", alpha=0.35)
    fig.tight_layout()
    return fig


def status_panel(states: dict[str, str]) -> str:
    labels = [
        ("dataset", "Dataset"),
        ("vision", "Thread 1 Vision"),
        ("impact", "Impact Agent"),
        ("health", "Health Agent"),
        ("sensor", "Sensor Agent"),
        ("final", "Final Reasoning"),
    ]
    color = {
        "waiting": ("#64748b", "#0b1118"),
        "running": ("#f59e0b", "#181208"),
        "done": ("#4ade80", "#07140d"),
        "error": ("#fb7185", "#1a090d"),
        "cached": ("#22d3ee", "#07151a"),
    }
    cards = []
    for key, label in labels:
        state = states.get(key, "waiting")
        foreground, background = color.get(state, color["waiting"])
        cards.append(
            "<div class='phase-card' "
            f"style='border-color:{foreground};background:{background}'>"
            f"<div class='phase-name'>{label}</div>"
            f"<div class='phase-state' style='color:{foreground}'>{state.upper()}</div>"
            "</div>"
        )
    return "<div class='phase-grid'>" + "".join(cards) + "</div>"


def specialist_text(result: dict[str, Any] | None) -> str:
    if not result:
        return "No specialist output."
    if result.get("status") != "ok":
        return f"### Error\n\n{result.get('error', 'Unknown error')}"
    output = result.get("output", {})
    if isinstance(output, dict):
        return str(output.get("agent_output") or output.get("final_output") or output)
    return str(output)


def evidence_rows(
    final_agent: Thread2WTFinalReasoningAgent,
    specialist_outputs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    mapping = [
        ("Impact", "impact_agent_output", "impact"),
        ("Health", "health_agent_output", "health"),
        ("Sensor fault", "sensor_fault_agent_output", "sensor"),
    ]
    for display, key, domain in mapping:
        raw = final_agent._numeric_evidence(
            specialist_outputs.get(key),
            domain,
        )
        try:
            items = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            records.append({
                "agent": display,
                "range": "",
                "feature": "evidence",
                "value": str(raw),
            })
            continue
        for item in items if isinstance(items, list) else [items]:
            if not isinstance(item, dict):
                continue
            range_value = str(item.get("range", ""))
            for feature, value in item.items():
                if feature == "range":
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                records.append({
                    "agent": display,
                    "range": range_value,
                    "feature": feature,
                    "value": str(value),
                })
    return pd.DataFrame(
        records,
        columns=["agent", "range", "feature", "value"],
    )


def final_summary_markdown(final_output: Any) -> str:
    result = parse_json_text(final_output)
    description = result.get("person_state_description", "No description")
    lines = [
        "## Final diagnosis",
        "",
        f"**Person state:** {description}",
        "",
        "| Start | End | Type | Subtype |",
        "|---:|---:|---|---|",
    ]
    ranges = result.get("anomaly_ranges", [])
    if not isinstance(ranges, list) or not ranges:
        lines.append("| - | - | No anomaly | - |")
    else:
        for item in ranges:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {item.get('start', '-')} | {item.get('end', '-')} | "
                f"{item.get('label', item.get('anomaly_type', '-'))} | "
                f"{str(item.get('subtype', '-')).replace('_', ' ')} |"
            )
    return "\n".join(lines)


def cached_result(scenario_dir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    thread_1_path = scenario_dir / "thread_1_output.txt"
    result_path = scenario_dir / "thread_2_wt_output.json"
    thread_1 = (
        thread_1_path.read_text(encoding="utf-8")
        if thread_1_path.exists()
        else ""
    )
    record: dict[str, Any] = {}
    final: dict[str, Any] = {}
    if result_path.exists():
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
            final = parse_json_text(record.get("thread_2_wt_output", ""))
        except (OSError, json.JSONDecodeError):
            pass
    return thread_1, record, final


def inspect_scenario(
    root: str,
    scenario_name: str,
) -> tuple[
    list[tuple[str, str]],
    plt.Figure,
    str,
    plt.Figure,
    dict[str, Any],
    str,
    pd.DataFrame,
]:
    scenario_dir = resolve_scenario(root, scenario_name)
    df = load_parquet(scenario_dir / "data.parquet")
    thread_1, record, final = cached_result(scenario_dir)
    ranges = final_anomaly_ranges(final)
    if not ranges and thread_1:
        ranges = [
            {"start": row["start_row"], "end": row["end_row"]}
            for row in extract_anomaly_ranges(thread_1)
        ]
    timing = record.get("timing", {}) if record else {}
    timing_df = pd.DataFrame(
        [{"phase": key, "seconds": value} for key, value in timing.items()],
        columns=["phase", "seconds"],
    )
    return (
        image_gallery(scenario_dir),
        signal_figure(df, ranges),
        thread_1 or "No cached Thread 1 output.",
        timeline_figure(thread_1, len(df), final if final else None),
        final,
        final_summary_markdown(final) if final else "No cached final output.",
        timing_df,
    )


def refresh_scenarios(root: str) -> gr.Dropdown:
    names = scenario_names(root)
    return gr.Dropdown(
        choices=names,
        value=names[0] if names else None,
    )


def run_pipeline(
    root: str,
    scenario_name: str,
    reuse_thread_1: bool,
    save_outputs: bool,
) -> Generator[tuple[Any, ...], None, None]:
    states = {
        key: "waiting"
        for key in ("dataset", "vision", "impact", "health", "sensor", "final")
    }
    empty_evidence = pd.DataFrame(
        columns=["agent", "range", "feature", "value"]
    )
    empty_timing = pd.DataFrame(columns=["phase", "seconds"])
    gallery: list[tuple[str, str]] = []
    overview = None
    thread_1_output = ""
    timeline = None
    impact_md = "Waiting for Impact Agent."
    health_md = "Waiting for Health Agent."
    sensor_md = "Waiting for Sensor Fault Agent."
    final_json: dict[str, Any] = {}
    final_md = "Waiting for final diagnosis."
    evidence = empty_evidence
    timing_df = empty_timing

    def snapshot() -> tuple[Any, ...]:
        return (
            status_panel(states),
            gallery,
            overview,
            thread_1_output,
            timeline,
            impact_md,
            health_md,
            sensor_md,
            evidence,
            final_json,
            final_md,
            timing_df,
        )

    try:
        scenario_dir = resolve_scenario(root, scenario_name)
        parquet_path = scenario_dir / "data.parquet"
        image_dir = scenario_dir / "image"
        df = load_parquet(parquet_path)
        gallery = image_gallery(scenario_dir)
        overview = signal_figure(df)
        states["dataset"] = "done"
        yield snapshot()

        thread_1_path = scenario_dir / "thread_1_output.txt"
        if reuse_thread_1 and thread_1_path.exists():
            states["vision"] = "cached"
            thread_1_output = thread_1_path.read_text(encoding="utf-8")
        else:
            states["vision"] = "running"
            yield snapshot()
            thread_1_output = VisionOnlyAgent().run(image_dir)
            states["vision"] = "done"
            if save_outputs:
                thread_1_path.write_text(thread_1_output, encoding="utf-8")

        thread_1_ranges = [
            {"start": row["start_row"], "end": row["end_row"]}
            for row in extract_anomaly_ranges(thread_1_output)
        ]
        overview = signal_figure(
            df,
            thread_1_ranges,
            "Thread 1 visual anomaly candidates",
        )
        timeline = timeline_figure(thread_1_output, len(df))
        yield snapshot()

        final_agent = Thread2WTFinalReasoningAgent()
        specialist_instances = {
            "impact_agent_output": (
                "impact",
                Thread2ImpactAgent(
                    config_path=final_agent.config_path,
                    model=final_agent.model,
                    timeout=final_agent.timeout,
                ),
            ),
            "health_agent_output": (
                "health",
                Thread2HealthAgent(
                    config_path=final_agent.config_path,
                    model=final_agent.model,
                    timeout=final_agent.timeout,
                ),
            ),
            "sensor_fault_agent_output": (
                "sensor",
                Thread2SensorFaultAgent(
                    config_path=final_agent.config_path,
                    model=final_agent.model,
                    timeout=final_agent.timeout,
                ),
            ),
        }
        for state_key, _ in specialist_instances.values():
            states[state_key] = "running"
        yield snapshot()

        specialists_started = time.perf_counter()
        specialist_outputs: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    agent.run,
                    thread_1_output=thread_1_output,
                    parquet_path=parquet_path,
                ): (result_key, state_key)
                for result_key, (state_key, agent) in specialist_instances.items()
            }
            for future in as_completed(futures):
                result_key, state_key = futures[future]
                try:
                    specialist_outputs[result_key] = {
                        "status": "ok",
                        "output": future.result(),
                    }
                    states[state_key] = "done"
                except Exception as exc:
                    specialist_outputs[result_key] = {
                        "status": "error",
                        "error": str(exc),
                    }
                    states[state_key] = "error"

                impact_md = specialist_text(
                    specialist_outputs.get("impact_agent_output")
                )
                health_md = specialist_text(
                    specialist_outputs.get("health_agent_output")
                )
                sensor_md = specialist_text(
                    specialist_outputs.get("sensor_fault_agent_output")
                )
                evidence = evidence_rows(final_agent, specialist_outputs)
                yield snapshot()

        specialist_seconds = time.perf_counter() - specialists_started
        states["final"] = "running"
        yield snapshot()

        prompt_started = time.perf_counter()
        prompt = final_agent.build_prompt(
            thread_1_output=thread_1_output,
            specialist_outputs=specialist_outputs,
        )
        prompt_seconds = time.perf_counter() - prompt_started
        llm_started = time.perf_counter()
        response = final_agent.create_agent().run_turn(prompt)
        raw_output = (
            response.get("text", "")
            if isinstance(response, dict)
            else str(response)
        )
        final_text = clean_text_output(raw_output)
        final_seconds = time.perf_counter() - llm_started

        final_json = parse_json_text(final_text)
        final_md = final_summary_markdown(final_json)
        final_ranges = final_anomaly_ranges(final_json)
        overview = signal_figure(
            df,
            final_ranges,
            "Final diagnosed anomaly ranges",
        )
        timeline = timeline_figure(
            thread_1_output,
            len(df),
            final_json,
        )
        states["final"] = "done"
        timing = {
            "specialists_parallel_wall_seconds": round(specialist_seconds, 3),
            "final_prompt_build_seconds": round(prompt_seconds, 3),
            "final_llm_seconds": round(final_seconds, 3),
            "total_seconds": round(
                specialist_seconds + prompt_seconds + final_seconds,
                3,
            ),
        }
        timing_df = pd.DataFrame(
            [{"phase": key, "seconds": value} for key, value in timing.items()]
        )

        if save_outputs:
            result_path = scenario_dir / "thread_2_wt_output.json"
            record = {
                "scenario": scenario_dir.name,
                "data_path": str(parquet_path),
                "thread_2_wt_output": final_text,
                "timing": timing,
            }
            result_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        yield snapshot()
    except Exception as exc:
        for key, value in states.items():
            if value == "running":
                states[key] = "error"
        final_md = f"## Pipeline error\n\n```text\n{exc}\n```"
        final_json = {"error": str(exc)}
        yield snapshot()


CSS = """
.gradio-container {
    width: 100% !important;
    max-width: none !important;
    min-height: 100vh;
    margin: 0 !important;
    padding: 14px 20px 24px !important;
    background: #070b10;
    color: #d7ffe8;
}
body {background: #030507 !important;}
.gradio-container .contain,
.gradio-container .panel,
.gradio-container .form,
.gradio-container .block {
    background: #0b1118 !important;
    color: #d7ffe8 !important;
}
.app-hero {
    padding: 15px 21px;
    margin-bottom: 10px;
    border: 1px solid #194e3a;
    border-radius: 7px;
    background: #08130f;
    color: #eafff1;
    border-left: 6px solid #22c55e;
    box-shadow: 0 0 24px rgba(34, 197, 94, 0.10);
}
.app-title {
    font-size: 24px;
    font-weight: 800;
    color: #5cff91;
    font-family: Consolas, "Courier New", monospace;
    text-shadow: 0 0 12px rgba(92, 255, 145, 0.30);
}
.app-subtitle {
    margin-top: 5px;
    color: #8faea0;
    font-size: 13px;
}
.control-panel {
    padding: 9px 11px !important;
    border: 1px solid #20342b !important;
    border-radius: 7px !important;
    background: #0b1118 !important;
    box-shadow: 0 0 16px rgba(34, 197, 94, 0.05);
}
.control-panel label,
.gradio-container label,
.gradio-container .label-wrap {
    color: #9ccfb2 !important;
}
.gradio-container input,
.gradio-container textarea {
    background: #05090d !important;
    border-color: #274536 !important;
    color: #d7ffe8 !important;
}
#inspect-button {
    border-color: #0891b2 !important;
    color: #67e8f9 !important;
    background: #07171d !important;
}
#run-button {
    border-color: #22c55e !important;
    color: #031008 !important;
    background: #4ade80 !important;
    font-weight: 850 !important;
    box-shadow: 0 0 18px rgba(74, 222, 128, 0.20);
}
#run-button:hover {background: #86efac !important;}
.phase-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(120px, 1fr));
    gap: 9px;
    margin: 10px 0 12px;
}
.phase-card {
    border: 1px solid;
    border-left-width: 4px;
    border-radius: 6px;
    padding: 10px 12px;
    min-height: 60px;
    background: #0b1118 !important;
    box-shadow: inset 0 0 16px rgba(0, 0, 0, 0.25);
}
.phase-name {
    font-size: 12px;
    font-weight: 700;
    color: #b8d8c5;
    font-family: Consolas, "Courier New", monospace;
}
.phase-state {
    font-size: 11px;
    font-weight: 800;
    margin-top: 6px;
}
.process-panel {
    width: 100% !important;
    border: 1px solid #20342b !important;
    border-radius: 7px !important;
    background: #080d12 !important;
    box-shadow: 0 0 16px rgba(34, 197, 94, 0.05);
}
.agent-tabs button[role="tab"] {
    color: #809d8b !important;
    font-weight: 700 !important;
    background: #0b1118 !important;
}
.agent-tabs button[role="tab"][aria-selected="true"] {
    color: #5cff91 !important;
    border-color: #22c55e !important;
    background: #0b1a12 !important;
}
.agent-panel {
    min-height: 230px;
    padding: 14px !important;
    border: 1px solid #20342b !important;
    border-left: 4px solid #22c55e !important;
    background: #080d12 !important;
    color: #ccebd8 !important;
}
.final-panel {
    min-height: 230px;
    padding: 15px !important;
    border: 1px solid #176b41 !important;
    border-left: 5px solid #4ade80 !important;
    border-radius: 7px !important;
    background: #07140d !important;
    color: #d7ffe8 !important;
    box-shadow: 0 0 18px rgba(34, 197, 94, 0.08);
}
.data-panel {
    border: 1px solid #20342b !important;
    border-radius: 7px !important;
    background: #080d12 !important;
}
.section-heading {
    margin: 4px 0 8px;
    color: #5cff91;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-family: Consolas, "Courier New", monospace;
}
.gradio-container table,
.gradio-container th,
.gradio-container td {
    background: #080d12 !important;
    border-color: #20342b !important;
    color: #bddbc8 !important;
}
.gradio-container details {
    border-color: #20342b !important;
    background: #0b1118 !important;
}
.gradio-container summary {
    color: #67e8f9 !important;
}
.gradio-container pre,
.gradio-container code {
    color: #86efac !important;
    background: #030705 !important;
    font-family: Consolas, "Courier New", monospace !important;
}
@media (max-width: 900px) {
    .phase-grid {grid-template-columns: repeat(2, minmax(120px, 1fr));}
    .gradio-container {padding: 12px !important;}
    .app-title {font-size: 24px;}
}
@media (min-width: 1600px) {
    .gradio-container {padding-left: 28px !important; padding-right: 28px !important;}
}
"""


def build_app() -> gr.Blocks:
    initial_names = scenario_names(DEFAULT_SCENARIO_ROOT)
    with gr.Blocks(
        title="Agentic Sensor Analysis Monitor",
    ) as app:
        gr.HTML(
            """
            <div class="app-hero">
                <div class="app-title">Agentic Sensor Analysis Monitor</div>
                <div class="app-subtitle">
                    Live process view for anomaly localization, specialist
                    analysis and final diagnosis.
                </div>
            </div>
            """
        )

        with gr.Row(elem_classes=["control-panel"]):
            scenario = gr.Dropdown(
                label="Scenario",
                choices=initial_names,
                value=initial_names[0] if initial_names else None,
                filterable=True,
                scale=4,
            )
            inspect_button = gr.Button(
                "Load existing",
                variant="secondary",
                elem_id="inspect-button",
            )
            run_button = gr.Button(
                "Run analysis",
                variant="primary",
                elem_id="run-button",
            )

        with gr.Accordion("Data source and run options", open=False):
            with gr.Row():
                scenario_root = gr.Textbox(
                    label="Scenario root",
                    value=str(DEFAULT_SCENARIO_ROOT),
                    scale=4,
                )
                refresh_button = gr.Button(
                    "Refresh scenarios",
                    variant="secondary",
                    scale=1,
                )
            with gr.Row():
                reuse_thread_1 = gr.Checkbox(
                    label="Reuse cached Thread 1 output",
                    value=True,
                )
                save_outputs = gr.Checkbox(
                    label="Save outputs in scenario folder",
                    value=False,
                )

        status = gr.HTML(
            status_panel({
                key: "waiting"
                for key in (
                    "dataset",
                    "vision",
                    "impact",
                    "health",
                    "sensor",
                    "final",
                )
            })
        )

        gr.HTML("<div class='section-heading'>Live process view</div>")
        with gr.Row():
            with gr.Column(scale=7, min_width=760):
                overview_plot = gr.Plot(
                    label="Detected ranges on sensor signals",
                    elem_classes=["process-panel"],
                )
            with gr.Column(scale=5, min_width=520):
                timeline_plot = gr.Plot(
                    label="Before - event - after",
                    elem_classes=["process-panel"],
                )

        gr.HTML("<div class='section-heading'>Agent reasoning</div>")
        with gr.Row():
            with gr.Column(scale=8, min_width=780):
                with gr.Tabs(elem_classes=["agent-tabs"]):
                    with gr.Tab("Impact agent"):
                        impact_output = gr.Markdown(
                            "Waiting for Impact Agent.",
                            elem_classes=["agent-panel"],
                        )
                    with gr.Tab("Health agent"):
                        health_output = gr.Markdown(
                            "Waiting for Health Agent.",
                            elem_classes=["agent-panel"],
                        )
                    with gr.Tab("Sensor agent"):
                        sensor_output = gr.Markdown(
                            "Waiting for Sensor Fault Agent.",
                            elem_classes=["agent-panel"],
                        )
            with gr.Column(scale=4, min_width=430):
                final_summary = gr.Markdown(
                    "Waiting for final diagnosis.",
                    elem_classes=["final-panel"],
                )

        with gr.Accordion("Technical details", open=False):
            gallery = gr.Gallery(
                label="Original sensor images",
                columns=3,
                height=300,
                object_fit="contain",
                elem_classes=["data-panel"],
            )
            with gr.Row():
                thread_1_text = gr.Textbox(
                    label="Raw Thread 1 output",
                    lines=16,
                    elem_classes=["data-panel"],
                )
                final_output = gr.JSON(
                    label="Structured final output",
                    elem_classes=["data-panel"],
                )
            evidence_table = gr.Dataframe(
                headers=["agent", "range", "feature", "value"],
                datatype=["str", "str", "str", "str"],
                label="Numeric evidence",
                interactive=False,
                wrap=True,
                elem_classes=["data-panel"],
            )
            timing_table = gr.Dataframe(
                headers=["phase", "seconds"],
                datatype=["str", "number"],
                label="Pipeline timing",
                interactive=False,
                elem_classes=["data-panel"],
            )

        refresh_button.click(
            refresh_scenarios,
            inputs=scenario_root,
            outputs=scenario,
        )
        inspect_button.click(
            inspect_scenario,
            inputs=[scenario_root, scenario],
            outputs=[
                gallery,
                overview_plot,
                thread_1_text,
                timeline_plot,
                final_output,
                final_summary,
                timing_table,
            ],
        )
        run_button.click(
            run_pipeline,
            inputs=[
                scenario_root,
                scenario,
                reuse_thread_1,
                save_outputs,
            ],
            outputs=[
                status,
                gallery,
                overview_plot,
                thread_1_text,
                timeline_plot,
                impact_output,
                health_output,
                sensor_output,
                evidence_table,
                final_output,
                final_summary,
                timing_table,
            ],
        )
    return app


if __name__ == "__main__":
    demo = build_app()
    server_port = find_available_port()
    print(f"Dashboard URL: http://127.0.0.1:{server_port}", flush=True)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=server_port,
        show_error=True,
        css=CSS,
        allowed_paths=[str(PROJECT_ROOT / "output")],
    )
