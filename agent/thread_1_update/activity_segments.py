from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from classification.anomaly_detection import SENSOR_COLUMNS, predict_activity_labels, read_table


ACTIVITY_NAMES = {
    1: "standing_still",
    2: "sitting_relaxing",
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_elevation_arms",
    8: "knees_bending",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jumping",
}


def parse_visual_anomaly_analysis(text: str) -> str:
    match = re.search(
        r"<visual_anomaly_analysis>.*?</visual_anomaly_analysis>",
        text or "",
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return "<visual_anomaly_analysis>\nnone\n</visual_anomaly_analysis>"
    return match.group(0).strip()


def parse_vision_anomaly_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    pattern = re.compile(
        r"start:\s*(\d+)\s+end:\s*(\d+)\s+candidate_type:\s*(normal|anomaly)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        if match.group(3).lower() == "anomaly":
            ranges.append((int(match.group(1)), int(match.group(2))))
    return merge_ranges(ranges)


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def complement_ranges(total_rows: int, blocked_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normal: list[tuple[int, int]] = []
    cursor = 0
    for start, end in merge_ranges(blocked_ranges):
        start = max(0, min(total_rows, start))
        end = max(0, min(total_rows, end))
        if cursor < start:
            normal.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_rows:
        normal.append((cursor, total_rows))
    return normal


def smooth_short_runs(labels: list[int], min_segment_windows: int = 3) -> list[int]:
    if len(labels) < 3:
        return labels[:]
    smoothed = labels[:]
    for _ in range(2):
        runs = label_runs(smoothed)
        updated = smoothed[:]
        for run_index, (start, end, label) in enumerate(runs):
            if end - start >= min_segment_windows:
                continue
            prev_run = runs[run_index - 1] if run_index > 0 else None
            next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
            replacement = label
            if prev_run and next_run and prev_run[2] == next_run[2]:
                replacement = prev_run[2]
            elif prev_run and next_run:
                replacement = prev_run[2] if prev_run[1] - prev_run[0] >= next_run[1] - next_run[0] else next_run[2]
            elif prev_run:
                replacement = prev_run[2]
            elif next_run:
                replacement = next_run[2]
            for index in range(start, end):
                updated[index] = replacement
        smoothed = updated
    return smoothed


def label_runs(labels: list[int]) -> list[tuple[int, int, int]]:
    if not labels:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    current = labels[0]
    for index, label in enumerate(labels[1:], start=1):
        if label != current:
            runs.append((start, index, current))
            start = index
            current = label
    runs.append((start, len(labels), current))
    return runs


def summarize_normal_activity(
    data_path: str | Path,
    anomaly_ranges: list[tuple[int, int]],
    har_model: torch.nn.Module,
    har_checkpoint: dict[str, object],
    window_size: int = 50,
    min_segment_windows: int = 3,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    df = read_table(Path(data_path))
    total_rows = len(df)
    normal_blocks = complement_ranges(total_rows, anomaly_ranges)
    window_records: list[dict[str, object]] = []
    segment_records: list[dict[str, object]] = []

    for block_index, (block_start, block_end) in enumerate(normal_blocks):
        if block_end - block_start < window_size:
            continue
        windows: list[tuple[int, int]] = [
            (start, start + window_size)
            for start in range(block_start, block_end - window_size + 1, window_size)
        ]
        if not windows:
            continue
        x = np.stack([
            df.iloc[start:end][SENSOR_COLUMNS].to_numpy(dtype=np.float32)
            for start, end in windows
        ])
        predictions = predict_activity_labels(har_model, har_checkpoint, x)
        raw_labels = [int(prediction["activity_label"]) for prediction in predictions]
        smoothed_labels = smooth_short_runs(raw_labels, min_segment_windows=min_segment_windows)

        block_window_records: list[dict[str, object]] = []
        for (start, end), raw_label, smooth_label, prediction in zip(windows, raw_labels, smoothed_labels, predictions):
            record = {
                "block_index": block_index,
                "start": start,
                "end": end,
                "raw_activity_label": raw_label,
                "raw_activity": ACTIVITY_NAMES.get(raw_label, str(raw_label)),
                "activity_label": smooth_label,
                "activity": ACTIVITY_NAMES.get(smooth_label, str(smooth_label)),
                "confidence": float(prediction["activity_confidence"]),
            }
            block_window_records.append(record)
            window_records.append(record)

        for run_start, run_end, label in label_runs(smoothed_labels):
            run_windows = block_window_records[run_start:run_end]
            raw_votes = sum(1 for record in run_windows if int(record["raw_activity_label"]) == label)
            segment_records.append({
                "start": int(run_windows[0]["start"]),
                "end": int(run_windows[-1]["end"]),
                "activity_label": label,
                "activity": ACTIVITY_NAMES.get(label, str(label)),
                "votes": raw_votes,
                "window_count": len(run_windows),
                "mean_confidence": float(np.mean([float(record["confidence"]) for record in run_windows])),
                "block_index": block_index,
            })

    return window_records, segment_records


def format_activity_summary(
    segments: list[dict[str, object]],
    anomaly_ranges: list[tuple[int, int]],
    window_size: int,
) -> str:
    parts = [
        "normal_activity_summary",
        f"activity_window_size_rows: {window_size}",
        f"anomaly_range_count: {len(anomaly_ranges)}",
        f"normal_activity_segment_count: {len(segments)}",
    ]
    for segment in segments:
        confidence = 100.0 * float(segment["mean_confidence"])
        parts.append(
            " | ".join([
                f"start: {segment['start']}",
                f"end: {segment['end']}",
                f"activity: {segment['activity']}",
                f"label: {segment['activity_label']}",
                f"votes: {segment['votes']}/{segment['window_count']}",
                f"mean_confidence: {confidence:.1f}%",
            ])
        )
    return " -- ".join(parts)


def format_timeline(
    activity_segments: list[dict[str, object]],
    anomaly_ranges: list[tuple[int, int]],
) -> str:
    items: list[tuple[int, str]] = []
    for start, end in anomaly_ranges:
        items.append((start, f"start: {start} | end: {end} | type: anomaly"))
    for segment in activity_segments:
        items.append((
            int(segment["start"]),
            (
                f"start: {segment['start']} | "
                f"end: {segment['end']} | "
                f"type: normal | "
                f"activity: {segment['activity']} | "
                f"label: {segment['activity_label']}"
            ),
        ))
    return " -- ".join(text for _, text in sorted(items, key=lambda item: item[0]))
