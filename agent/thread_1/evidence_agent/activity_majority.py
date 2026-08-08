from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from perception_layer.agent.thread_1.model_tools.model_prediction_tools import _predict_activity


ACTIVITY_LABEL_NAMES = {
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
    12: "jump_front_back",
}


def parse_vision_candidate_ranges(vision_output: str) -> list[dict[str, Any]]:
    blocks = str(vision_output or "").split("start: ")[1:]
    ranges: list[dict[str, Any]] = []
    for block in blocks:
        try:
            start_row = int(block.splitlines()[0].strip())
            end_row = int(block.split("end:")[1].splitlines()[0].strip())
            candidate_type = block.split("candidate_type:")[1].splitlines()[0].strip().lower()
        except Exception:
            continue
        if end_row > start_row and candidate_type in {"normal", "anomaly"}:
            ranges.append(
                {
                    "start_row": start_row,
                    "end_row": end_row,
                    "candidate_type": candidate_type,
                }
            )
    return ranges


def parse_evidence_candidate_ranges(candidate_ranges_str: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for line in str(candidate_ranges_str or "").splitlines():
        if "Row" not in line or "to" not in line:
            continue
        parts = line.strip().split()
        try:
            start_row = int(parts[2])
            end_row = int(parts[4])
        except Exception:
            continue
        if end_row > start_row:
            ranges.append((start_row, end_row))
    return ranges


def build_activity_subwindows(
    start_row: int,
    end_row: int,
    window_size_rows: int = 100,
    overlap_rows: int = 50,
) -> list[tuple[int, int]]:
    if end_row <= start_row:
        return []

    window_size_rows = max(1, int(window_size_rows))
    overlap_rows = max(0, min(int(overlap_rows), window_size_rows - 1))
    step_rows = max(1, window_size_rows - overlap_rows)
    windows: list[tuple[int, int]] = []
    current = start_row
    while current + window_size_rows <= end_row:
        windows.append((current, current + window_size_rows))
        current += step_rows

    if windows and windows[-1][1] < end_row:
        tail_start = max(start_row, end_row - window_size_rows)
        tail_window = (tail_start, end_row)
        if tail_window != windows[-1]:
            windows.append(tail_window)
    elif not windows and end_row - start_row > 0:
        windows.append((start_row, end_row))
    return windows


def _parse_prediction(payload: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def predict_activity_majority_for_range(
    parquet_path: str | Path,
    start_row: int,
    end_row: int,
    activity_window_size_rows: int = 100,
    activity_overlap_rows: int = 50,
    debug: bool = False,
) -> dict[str, Any]:
    subwindows = build_activity_subwindows(
        start_row,
        end_row,
        window_size_rows=activity_window_size_rows,
        overlap_rows=activity_overlap_rows,
    )
    predictions: list[dict[str, Any]] = []

    if debug:
        print("\n[ACTIVITY MODEL INPUT RANGE]", flush=True)
        print(f"range: {start_row}-{end_row}", flush=True)
        print(f"activity_window_size_rows: {activity_window_size_rows}", flush=True)
        print(f"activity_overlap_rows: {activity_overlap_rows}", flush=True)
        print(f"subwindows: {subwindows}", flush=True)

    for sub_start, sub_end in subwindows:
        if debug:
            print("\n[ACTIVITY MODEL INPUT WINDOW]", flush=True)
            print(f"parquet_path: {parquet_path}", flush=True)
            print(f"start_row: {sub_start}", flush=True)
            print(f"end_row: {sub_end}", flush=True)

        result = _parse_prediction(
            _predict_activity(
                parquet_path=str(parquet_path),
                start_row=sub_start,
                end_row=sub_end,
            )
        )
        if not result:
            continue
        if debug:
            print("[ACTIVITY MODEL OUTPUT WINDOW]", flush=True)
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

        label = int(result.get("predicted_label", -1))
        predictions.append(
            {
                "start_row": sub_start,
                "end_row": sub_end,
                "label": label,
                "activity": ACTIVITY_LABEL_NAMES.get(label, "unknown"),
                "confidence": float(result.get("confidence", 0.0)),
            }
        )

    label_counts = Counter(item["label"] for item in predictions)
    if not label_counts:
        return {
            "start_row": int(start_row),
            "end_row": int(end_row),
            "activity": "unknown",
            "label": None,
            "vote_count": 0,
            "total_windows": 0,
            "mean_confidence": 0.0,
            "activity_overlap_rows": int(activity_overlap_rows),
            "subwindows": [],
        }

    majority_label, vote_count = label_counts.most_common(1)[0]
    majority_confidences = [
        item["confidence"]
        for item in predictions
        if item["label"] == majority_label
    ]
    return {
        "start_row": int(start_row),
        "end_row": int(end_row),
        "activity": ACTIVITY_LABEL_NAMES.get(int(majority_label), "unknown"),
        "label": int(majority_label),
        "vote_count": int(vote_count),
        "total_windows": len(predictions),
        "mean_confidence": float(sum(majority_confidences) / max(1, len(majority_confidences))),
        "activity_overlap_rows": int(activity_overlap_rows),
        "subwindows": predictions,
    }


def summarize_final_normal_activities(
    parquet_path: str | Path,
    candidate_ranges_str: str,
    vision_output: str,
    activity_window_size_rows: int = 100,
    activity_overlap_rows: int = 50,
    debug: bool = False,
) -> dict[str, Any]:
    evidence_candidates = parse_evidence_candidate_ranges(candidate_ranges_str)
    vision_ranges = parse_vision_candidate_ranges(vision_output)
    final_anomaly_ranges = {
        (int(item["start_row"]), int(item["end_row"]))
        for item in vision_ranges
        if item["candidate_type"] == "anomaly"
    }
    normal_ranges = [
        candidate
        for candidate in evidence_candidates
        if candidate not in final_anomaly_ranges
    ]

    if debug:
        print("\n[FINAL NORMAL ACTIVITY INPUT]", flush=True)
        print("candidate_ranges_from_evidence:", flush=True)
        print(candidate_ranges_str or "EMPTY", flush=True)
        print("\nvision_output:", flush=True)
        print(vision_output or "EMPTY", flush=True)
        print("\nparsed_evidence_candidates:", evidence_candidates, flush=True)
        print("parsed_vision_ranges:", vision_ranges, flush=True)
        print("final_anomaly_ranges:", sorted(final_anomaly_ranges), flush=True)
        print("final_normal_ranges:", normal_ranges, flush=True)

    summaries = [
        predict_activity_majority_for_range(
            parquet_path=parquet_path,
            start_row=start,
            end_row=end,
            activity_window_size_rows=activity_window_size_rows,
            activity_overlap_rows=activity_overlap_rows,
            debug=debug,
        )
        for start, end in normal_ranges
    ]
    summary = {
        "activity_window_size_rows": int(activity_window_size_rows),
        "activity_overlap_rows": int(activity_overlap_rows),
        "candidate_range_count": len(evidence_candidates),
        "final_anomaly_range_count": len(final_anomaly_ranges),
        "normal_range_count": len(normal_ranges),
        "normal_ranges": summaries,
    }

    if debug:
        print("\n[FINAL NORMAL ACTIVITY OUTPUT]", flush=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    return summary


def summarize_vision_normal_activities(
    parquet_path: str | Path,
    vision_output: str,
    activity_window_size_rows: int = 100,
    activity_overlap_rows: int = 50,
) -> dict[str, Any]:
    vision_ranges = parse_vision_candidate_ranges(vision_output)
    normal_candidate_ranges = "\n".join(
        f"- Row {item['start_row']} to {item['end_row']}"
        for item in vision_ranges
    )
    return summarize_final_normal_activities(
        parquet_path=parquet_path,
        candidate_ranges_str=normal_candidate_ranges,
        vision_output=vision_output,
        activity_window_size_rows=activity_window_size_rows,
        activity_overlap_rows=activity_overlap_rows,
    )


def format_activity_summary(summary: dict[str, Any]) -> str:
    lines = [
        "<normal_activity_summary>",
        f"activity_window_size_rows: {summary.get('activity_window_size_rows', 100)}",
        f"activity_overlap_rows: {summary.get('activity_overlap_rows', 50)}",
        f"normal_range_count: {summary.get('normal_range_count', 0)}",
    ]
    normal_ranges = summary.get("normal_ranges", [])
    if not normal_ranges:
        lines.append("none")
    for item in normal_ranges:
        confidence = float(item.get("mean_confidence", 0.0)) * 100.0
        lines.append(f"start: {item['start_row']}")
        lines.append(f"end: {item['end_row']}")
        lines.append(f"activity: {item['activity']}")
        lines.append(f"label: {item['label']}")
        lines.append(f"votes: {item['vote_count']}/{item['total_windows']}")
        lines.append(f"mean_confidence: {confidence:.1f}%")
    lines.append("</normal_activity_summary>")
    return "\n".join(lines)
