from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perception_layer.agent.thread_1.evidence_analyzers import (
    SeasonalityRhythmStabilityEvidence,
    SpikeStrengthEvidence,
    TrendStabilityEvidence,
)
from perception_layer.agent.thread_1.evidence_analyzers.trend_stability import METADATA_COLUMNS
from perception_layer.path_config import DEFAULT_CONFIG_PATH, read_config


EVIDENCE_GROUPS = (
    "trend_stability",
    "seasonality_rhythm_stability",
    "spike_strength",
)

EXCLUDED_NUMERIC_COLUMNS = METADATA_COLUMNS | {
    "row",
    "is_anomaly",
}

GROUP_FIELDS = {
    "trend_stability": [
        "severity",
        "strongest_axis",
        "trend_instability",
        "mean_shift_z",
        "max_window_mean_jump_z",
        "slope_z_per_second",
    ],
    "seasonality_rhythm_stability": [
        "severity",
        "strongest_axis",
        "rhythm_instability",
        "dominant_frequency_jump_hz",
        "dominant_frequency_std_hz",
        "spectral_entropy_jump",
        "spectral_distribution_shift",
    ],
    "spike_strength": [
        "severity",
        "strongest_axis",
        "spike_strength",
        "max_abs_z_score",
        "max_derivative_z",
        "max_window_abs_z_score",
        "max_window_derivative_z",
    ],
}


def load_thread_1_model(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    config = read_config(config_path)
    model = config.get("thread_1", "model", fallback="").strip()
    if not model:
        model = config.get("thread_1", "vision_model", fallback="").strip()
    if not model:
        raise ValueError("Missing [thread_1] model or vision_model in perception config.")
    return model


def parquet_row_count(parquet_path: str | Path) -> int:
    return int(pd.read_parquet(parquet_path).shape[0])


def sensor_channels(parquet_path: Path | str) -> list[str]:
    df = pd.read_parquet(parquet_path)
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_NUMERIC_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def analyze_window_groups(
    parquet_path: Path | str,
    start_row: int,
    end_row: int,
    sample_rate_hz: float,
    channels: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "trend_stability": TrendStabilityEvidence(
            start_row=start_row,
            end_row=end_row,
            data_path=parquet_path,
            sample_rate_hz=sample_rate_hz,
        ).analyze(channels),
        "seasonality_rhythm_stability": SeasonalityRhythmStabilityEvidence(
            start_row=start_row,
            end_row=end_row,
            data_path=parquet_path,
            sample_rate_hz=sample_rate_hz,
        ).analyze(channels),
        "spike_strength": SpikeStrengthEvidence(
            start_row=start_row,
            end_row=end_row,
            data_path=parquet_path,
            sample_rate_hz=sample_rate_hz,
        ).analyze(channels),
    }


def build_matrix_rows(
    group_name: str,
    window_results: list[dict[str, dict[str, dict[str, Any]]]],
) -> list[tuple[str, list[str]]]:
    fields = GROUP_FIELDS[group_name]
    channels = sorted(
        {
            channel
            for result in window_results
            for channel in result[group_name]
        }
    )
    rows: list[tuple[str, list[str]]] = []
    for channel in channels:
        for field in fields:
            row_name = f"{channel}.{field}"
            values = [
                format_value(result[group_name].get(channel, {}).get(field))
                for result in window_results
            ]
            rows.append((row_name, values))
    return rows


def matrix_text(title: str, window_labels: list[str], rows: list[tuple[str, list[str]]]) -> str:
    first_width = max([len("metric"), *(len(row_name) for row_name, _ in rows)])
    col_widths = [
        max(len(label), *(len(values[index]) for _, values in rows))
        for index, label in enumerate(window_labels)
    ]
    lines = [
        "",
        "=" * 120,
        title.upper(),
        "=" * 120,
    ]
    header = "metric".ljust(first_width) + " | " + " | ".join(
        label.ljust(col_widths[index])
        for index, label in enumerate(window_labels)
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row_name, values in rows:
        lines.append(
            row_name.ljust(first_width)
            + " | "
            + " | ".join(
                values[index].ljust(col_widths[index])
                for index in range(len(window_labels))
            )
        )
    return "\n".join(lines)


def build_row_windows(
    total_rows: int,
    start_row: int = 0,
    end_row: int | None = None,
    window_size_rows: int | None = None,
    window_size_seconds: float = 5.0,
    sample_rate_hz: float = 50.0,
    overlap_rows: int | None = None,
    overlap_fraction: float = 0.0,
) -> list[tuple[int, int]]:
    if end_row is None:
        end_row = total_rows
    start_row = max(0, int(start_row))
    end_row = min(int(end_row), int(total_rows))
    if end_row <= start_row:
        raise ValueError("end_row must be greater than start_row.")

    if window_size_rows is None:
        window_size_rows = int(round(window_size_seconds * sample_rate_hz))
    window_size_rows = max(1, int(window_size_rows))
    if overlap_rows is not None:
        overlap_rows = max(0, min(int(overlap_rows), window_size_rows - 1))
        step_rows = max(1, window_size_rows - overlap_rows)
    else:
        if not 0.0 <= overlap_fraction < 1.0:
            raise ValueError("overlap_fraction must be >= 0.0 and < 1.0.")
        step_rows = max(1, int(round(window_size_rows * (1.0 - overlap_fraction))))

    windows: list[tuple[int, int]] = []
    current = start_row
    while current < end_row:
        window_end = min(current + window_size_rows, end_row)
        windows.append((current, window_end))
        if window_end >= end_row:
            break
        current += step_rows
    return windows


def build_windowed_evidence_inputs(
    parquet_path: str | Path,
    window_size_rows: int | None = None,
    window_side: int | None = None,
    window_size_seconds: float = 5.0,
    sample_rate_hz: float = 50.0,
    overlap_rows: int | None = None,
    overlap_fraction: float = 0.0,
    start_row: int = 0,
    end_row: int | None = None,
) -> dict[str, str]:
    if window_size_rows is None and window_side is not None:
        window_size_rows = window_side
    total_rows = parquet_row_count(parquet_path)
    windows = build_row_windows(
        total_rows=total_rows,
        start_row=start_row,
        end_row=end_row,
        window_size_rows=window_size_rows,
        window_size_seconds=window_size_seconds,
        sample_rate_hz=sample_rate_hz,
        overlap_rows=overlap_rows,
        overlap_fraction=overlap_fraction,
    )
    
    channels = sensor_channels(parquet_path)
    
    window_labels = [
        f"w{index:02d}[{start}-{end}]"
        for index, (start, end) in enumerate(windows, start=1)
    ]
    
    window_results = [
        analyze_window_groups(
            parquet_path=parquet_path,
            start_row=start_row,
            end_row=end_row,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )
        for start_row, end_row in windows
    ]
    
    group_blocks = {}
    for group in EVIDENCE_GROUPS:
        rows = build_matrix_rows(group, window_results)
        matrix = matrix_text(title=group, window_labels=window_labels, rows=rows)
        group_blocks[group] = matrix

    return group_blocks


def extract_anomaly_ranges(text: str) -> str:
    """Extracts only the start/end rows where candidate_type is anomaly."""
    blocks = text.split("start: ")[1:]
    ranges = []
    for block in blocks:
        if "candidate_type: anomaly" in block.lower():
            try:
                start_val = block.split("\n")[0].strip()
                end_val = block.split("end:")[1].split("\n")[0].strip()
                ranges.append(f"- Row {start_val} to {end_val}")
            except Exception:
                pass
    
    if not ranges:
        return "No specific anomaly ranges identified."
    return "\n".join(ranges)


def generate_highlighted_images(parquet_path: Path, anomaly_ranges_str: str) -> tuple[Path, str]:
    """Generates 3 images with distinct colored backgrounds over the candidate anomaly ranges."""
    ranges = []
    for line in anomaly_ranges_str.split("\n"):
        if "Row" in line and "to" in line:
            parts = line.strip().split()
            try:
                start = int(parts[2])
                end = int(parts[4])
                ranges.append((start, end, line.strip()))
            except Exception:
                pass
                
    colors = ['#ef4444', '#3b82f6', '#22c55e', '#a855f7', '#f97316', '#06b6d4', '#ec4899', '#eab308']
    color_names = ['Red', 'Blue', 'Green', 'Purple', 'Orange', 'Cyan', 'Pink', 'Yellow']
    
    out_dir = parquet_path.parent / "image_highlighted"
    out_dir.mkdir(exist_ok=True)
    
    df = pd.read_parquet(parquet_path)
    if "row" not in df.columns:
        df["row"] = range(len(df))
    
    overview_groups = [
        ("chest_acc", ["chest_acc_x", "chest_acc_y", "chest_acc_z"]),
        ("ecg", ["ecg_lead_1", "ecg_lead_2"]),
        ("left_ankle_acc", ["left_ankle_acc_x", "left_ankle_acc_y", "left_ankle_acc_z"]),
        ("left_ankle_gyro", ["left_ankle_gyro_x", "left_ankle_gyro_y", "left_ankle_gyro_z"]),
        ("left_ankle_mag", ["left_ankle_mag_x", "left_ankle_mag_y", "left_ankle_mag_z"]),
        ("right_lower_arm_acc", ["right_lower_arm_acc_x", "right_lower_arm_acc_y", "right_lower_arm_acc_z"]),
        ("right_lower_arm_gyro", ["right_lower_arm_gyro_x", "right_lower_arm_gyro_y", "right_lower_arm_gyro_z"]),
        ("right_lower_arm_mag", ["right_lower_arm_mag_x", "right_lower_arm_mag_y", "right_lower_arm_mag_z"]),
    ]
    
    def plot_group(groups, path, title):
        fig, axes = plt.subplots(len(groups), 1, figsize=(16, 2.4 * len(groups)), sharex=True)
        if len(groups) == 1: axes = [axes]
        for ax, (name, columns) in zip(axes, groups):
            for col in columns:
                if col in df.columns:
                    ax.plot(df["row"], df[col], linewidth=0.75, label=col)
            for idx, (st, en, _) in enumerate(ranges):
                c_idx = idx % len(colors)
                ax.axvspan(st, en, color=colors[c_idx], alpha=0.25)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.22)
            ax.legend(loc="upper right", fontsize=8, ncol=min(3, len(columns)))
        axes[-1].set_xlabel("row")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        
    plot_group(overview_groups[:2], out_dir / "chest.png", "Chest position, row x-axis")
    plot_group(overview_groups[2:5], out_dir / "ankle.png", "Ankle position, row x-axis")
    plot_group(overview_groups[5:], out_dir / "arm.png", "Arm position, row x-axis")
    
    updated_lines = []
    for idx, (st, en, orig_line) in enumerate(ranges):
        c_name = color_names[idx % len(colors)]
        updated_lines.append(f"{orig_line} (Highlighted in {c_name})")
        
    updated_ranges_str = "\n".join(updated_lines) if updated_lines else anomaly_ranges_str
    return out_dir, updated_ranges_str
