from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.anomaly_detection import (
    CHEST_MODEL_CONFIG,
    DEFAULT_CALIBRATION_CSV,
    DEFAULT_CHEST_CALIBRATION_CSV,
    DEFAULT_CHEST_MIN_CONSECUTIVE,
    DEFAULT_CHEST_MODEL_PATH,
    DEFAULT_CHEST_Z_THRESHOLD,
    DEFAULT_DATA_PATH,
    DEFAULT_HAR_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_Z_THRESHOLD,
    SENSOR_COLUMNS,
    effective_step_rows,
    load_cscad_model,
    load_har_model,
    load_or_build_activity_calibration,
    merge_ranges,
    read_table,
    scan_data_file,
)
from classification.generate_multi_anomaly_scenarios import OUTPUT_ROOT
from perception_layer.agent.thread_1_update.activity_segments import (
    format_timeline,
    parse_vision_anomaly_ranges,
    parse_visual_anomaly_analysis,
    summarize_normal_activity,
)
from perception_layer.agent.thread_1_update.vision_agent import HighlightVisionAgent


PLOT_GROUPS = [
    ("chest_acc", ["chest_acc_x", "chest_acc_y", "chest_acc_z"]),
    ("ecg", ["ecg_lead_1", "ecg_lead_2"]),
    ("left_ankle_acc", ["left_ankle_acc_x", "left_ankle_acc_y", "left_ankle_acc_z"]),
    ("left_ankle_gyro", ["left_ankle_gyro_x", "left_ankle_gyro_y", "left_ankle_gyro_z"]),
    ("right_lower_arm_acc", ["right_lower_arm_acc_x", "right_lower_arm_acc_y", "right_lower_arm_acc_z"]),
    ("right_lower_arm_gyro", ["right_lower_arm_gyro_x", "right_lower_arm_gyro_y", "right_lower_arm_gyro_z"]),
]


COLORS = {
    "reset": "\033[0m",
    "phase": "\033[97m",
    "input": "\033[95m",
    "output": "\033[92m",
    "warn": "\033[93m",
    "time": "\033[96m",
}


def print_color(title: str, content: object, color: str) -> None:
    print(f"\n{color}[{title}]\n{content}\n{COLORS['reset']}", flush=True)


def format_candidate_ranges(ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return "No candidate ranges."
    return " -- ".join(f"start: {start} | end: {end}" for start, end in ranges)


def format_anomaly_ranges_block(ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return "<anomaly_ranges>\nnone\n</anomaly_ranges>"
    lines = ["<anomaly_ranges>"]
    for start, end in ranges:
        lines.extend([
            f"start: {start}",
            f"end: {end}",
            "candidate_type: anomaly",
            "",
        ])
    lines.append("</anomaly_ranges>")
    return "\n".join(lines)


def format_normal_activity_ranges_block(segments: list[dict[str, object]]) -> str:
    if not segments:
        return "<normal_activity_ranges>\nnone\n</normal_activity_ranges>"
    lines = ["<normal_activity_ranges>"]
    for segment in segments:
        lines.extend([
            f"start: {segment['start']}",
            f"end: {segment['end']}",
            "candidate_type: normal",
            f"activity: {segment['activity']}",
            f"label: {segment['activity_label']}",
            "",
        ])
    lines.append("</normal_activity_ranges>")
    return "\n".join(lines)


def format_thread_1_output(
    scenario_name: str,
    final_anomaly_ranges: list[tuple[int, int]],
    activity_segments: list[dict[str, object]],
    visual_anomaly_analysis: str,
) -> str:
    timeline = format_timeline(activity_segments, final_anomaly_ranges)
    return "\n".join([
        "[THREAD_1_UPDATE_OUTPUT]",
        f"scenario: {scenario_name}",
        format_anomaly_ranges_block(final_anomaly_ranges),
        visual_anomaly_analysis,
        format_normal_activity_ranges_block(activity_segments),
        "<timeline>",
        timeline if timeline else "none",
        "</timeline>",
    ])


def plot_model_candidates(
    data_path: str | Path,
    ranges: list[tuple[int, int]],
    output_path: str | Path,
    highlight_color: str = "#ef4444",
    title: str | None = None,
) -> Path:
    ranges = merge_ranges(ranges)
    data_path = Path(data_path)
    output_path = Path(output_path)
    df = read_table(data_path)
    missing = [col for col in SENSOR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.arange(len(df))
    fig, axes = plt.subplots(len(PLOT_GROUPS), 1, figsize=(16, 2.2 * len(PLOT_GROUPS)), sharex=True)
    for ax, (name, columns) in zip(axes, PLOT_GROUPS):
        for column in columns:
            ax.plot(rows, df[column], linewidth=0.75, label=column)
        for start_row, end_row in ranges:
            ax.axvspan(start_row, end_row, color=highlight_color, alpha=0.24)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8, ncol=min(3, len(columns)))
    axes[-1].set_xlabel("row")
    fig.suptitle(title or f"Thread 1 Update Model Candidate Ranges: {len(ranges)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def load_calibrations(
    window_size: int,
    step_rows: int,
    z_threshold: float,
    chest_z_threshold: float,
    calibration_data_path: Path,
    calibration_csv: Path,
    chest_calibration_csv: Path,
) -> tuple[dict[int, dict[str, float | int]], dict[int, dict[str, float | int]]]:
    del z_threshold, chest_z_threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cscad_model(DEFAULT_MODEL_PATH, device)
    chest_model = load_cscad_model(DEFAULT_CHEST_MODEL_PATH, device, CHEST_MODEL_CONFIG)
    calibration = load_or_build_activity_calibration(
        model,
        calibration_data_path,
        calibration_csv,
        window_size,
        step_rows,
        0.8,
        30,
        False,
    )
    chest_calibration = load_or_build_activity_calibration(
        chest_model,
        calibration_data_path,
        chest_calibration_csv,
        window_size,
        step_rows,
        0.8,
        30,
        False,
        ["chest_acc_x", "chest_acc_y", "chest_acc_z", "ecg_lead_1", "ecg_lead_2"],
    )
    return calibration, chest_calibration


def run_pipeline(
    scenario_root: str | Path = OUTPUT_ROOT,
    window_size: int = 50,
    overlap: float = 0.5,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    chest_z_threshold: float = DEFAULT_CHEST_Z_THRESHOLD,
    chest_min_consecutive: int = DEFAULT_CHEST_MIN_CONSECUTIVE,
    debug: bool = True,
    emit_output: bool = True,
) -> list[dict[str, object]]:
    scenario_root = Path(scenario_root)
    data_files = (
        [scenario_root / "data.parquet"]
        if (scenario_root / "data.parquet").exists()
        else sorted(scenario_root.glob("scenario_*/data.parquet"))
    )
    if not data_files:
        raise FileNotFoundError(f"No scenario data.parquet files found under {scenario_root}")

    step_rows = effective_step_rows(window_size, None, overlap)
    calibration, chest_calibration = load_calibrations(
        window_size,
        step_rows,
        z_threshold,
        chest_z_threshold,
        DEFAULT_DATA_PATH,
        DEFAULT_CALIBRATION_CSV,
        DEFAULT_CHEST_CALIBRATION_CSV,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cscad_model(DEFAULT_MODEL_PATH, device)
    chest_model = load_cscad_model(DEFAULT_CHEST_MODEL_PATH, device, CHEST_MODEL_CONFIG)
    har_model, har_checkpoint = load_har_model(DEFAULT_HAR_MODEL_PATH, device)
    vision_agent = HighlightVisionAgent()

    summaries: list[dict[str, object]] = []
    for data_path in data_files:
        scenario_dir = data_path.parent
        if debug:
            print_color(
                "THREAD 1 / SCENARIO INPUT",
                "\n".join([
                    f"scenario: {scenario_dir.name}",
                    f"data_path: {data_path}",
                    f"window_size: {window_size}",
                    f"overlap: {overlap}",
                    f"z_threshold: {z_threshold}",
                    f"chest_z_threshold: {chest_z_threshold}",
                    f"chest_min_consecutive: {chest_min_consecutive}",
                ]),
                COLORS["phase"],
            )

        rows = scan_data_file(
            data_path=data_path,
            baseline_path=DEFAULT_DATA_PATH,
            window_size=window_size,
            overlap=overlap,
            z_threshold=z_threshold,
            chest_z_threshold=chest_z_threshold,
            chest_min_consecutive=chest_min_consecutive,
            calibration=calibration,
            chest_calibration=chest_calibration,
            model=model,
            chest_model=chest_model,
            har_model=har_model,
            har_checkpoint=har_checkpoint,
        )

        candidate_ranges = merge_ranges([
            (int(row["start"]), int(row["end"]))
            for row in rows
            if row["predicted_state"] == "ANOMALY"
        ])
        ranges_text = format_candidate_ranges(candidate_ranges)
        if debug:
            print_color(
                "THREAD 1 / MODEL SCAN OUTPUT",
                "\n".join([
                    f"candidate_count: {len(candidate_ranges)}",
                    f"candidate_ranges: {ranges_text}",
                ]),
                COLORS["output"],
            )

        vision_output = ""
        if candidate_ranges:
            with tempfile.TemporaryDirectory(prefix="thread1_update_") as temp_dir:
                image_path = Path(temp_dir) / f"{scenario_dir.name}_highlight.png"
                plot_model_candidates(data_path, candidate_ranges, image_path)
                if debug:
                    full_prompt = vision_agent.build_prompt(
                        ranges_text,
                        context="Confirm whether the highlighted model candidate ranges are true anomalies.",
                    )
                    print_color(
                        "THREAD 1 / VISION INPUT",
                        "\n".join([
                            f"temporary_highlight_image: {image_path}",
                            f"candidate_ranges: {ranges_text}",
                            "context: Confirm whether the highlighted model candidate ranges are true anomalies.",
                            "",
                            "full_prompt:",
                            full_prompt,
                        ]),
                        COLORS["input"],
                    )
                vision_output = vision_agent.run(
                    image_path,
                    ranges_text,
                    context="Confirm whether the highlighted model candidate ranges are true anomalies.",
                )
                if debug:
                    print_color("THREAD 1 / VISION OUTPUT", vision_output, COLORS["output"])
        elif debug:
            print_color("THREAD 1 / VISION INPUT", "No candidate ranges, vision skipped.", COLORS["warn"])

        final_anomaly_ranges = parse_vision_anomaly_ranges(vision_output) if vision_output.strip() else candidate_ranges
        visual_anomaly_analysis = parse_visual_anomaly_analysis(vision_output)
        final_highlight_path = scenario_dir / "thread_1_anomaly_highlight.png"
        plot_model_candidates(
            data_path,
            final_anomaly_ranges,
            final_highlight_path,
            highlight_color="#ec4899",
            title=f"Thread 1 Final Anomaly Ranges: {len(final_anomaly_ranges)}",
        )
        if debug:
            print_color(
                "THREAD 1 / FINAL ANOMALY RANGES",
                format_anomaly_ranges_block(final_anomaly_ranges),
                COLORS["output"],
            )

        if debug:
            print_color(
                "THREAD 1 / ACTIVITY INPUT",
                "\n".join([
                    f"data_path: {data_path}",
                    f"excluded_anomaly_ranges: {format_candidate_ranges(final_anomaly_ranges)}",
                    f"activity_window_size: {window_size}",
                ]),
                COLORS["input"],
            )
        activity_windows, activity_segments = summarize_normal_activity(
            data_path,
            final_anomaly_ranges,
            har_model,
            har_checkpoint,
            window_size=window_size,
            min_segment_windows=3,
        )
        del activity_windows
        if debug:
            print_color(
                "THREAD 1 / ACTIVITY OUTPUT",
                format_normal_activity_ranges_block(activity_segments),
                COLORS["output"],
            )

        final_output = format_thread_1_output(
            scenario_dir.name,
            final_anomaly_ranges,
            activity_segments,
            visual_anomaly_analysis,
        )
        if debug:
            print_color("THREAD 1 / FINAL RAW OUTPUT", final_output, COLORS["phase"])
        elif emit_output:
            print(final_output)
            print()

        summaries.append({
            "scenario": scenario_dir.name,
            "data_path": str(data_path),
            "candidate_count": len(candidate_ranges),
            "candidate_ranges": candidate_ranges,
            "final_anomaly_count": len(final_anomaly_ranges),
            "final_anomaly_ranges": final_anomaly_ranges,
            "highlight_image_path": str(final_highlight_path),
            "visual_anomaly_analysis": visual_anomaly_analysis,
            "activity_segment_count": len(activity_segments),
            "final_output": final_output,
            "vision_output": vision_output,
        })

    return summaries


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
