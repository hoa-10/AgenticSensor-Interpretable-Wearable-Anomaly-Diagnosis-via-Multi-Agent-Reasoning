from __future__ import annotations

import re
from typing import Any


def extract_anomaly_ranges(thread_1_output: Any) -> list[dict[str, int]]:
    """Extract start/end row pairs from Thread 1 anomaly outputs."""
    text = str(thread_1_output or "")
    blocks = re.findall(
        r"<anomaly_ranges>\s*(.*?)\s*</anomaly_ranges>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    ranges: list[dict[str, int]] = []

    for block in blocks:
        if block.strip().lower() == "none":
            continue
        for match in re.finditer(
            r"start\s*:\s*(\d+)\s*end\s*:\s*(\d+)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            start_row = int(match.group(1))
            end_row = int(match.group(2))
            if end_row > start_row:
                ranges.append({"start_row": start_row, "end_row": end_row})

    if ranges:
        return ranges

    chunks = re.split(r"(?=\bstart\s*:)", text, flags=re.IGNORECASE)
    for chunk in chunks:
        if not re.search(
            r"\bcandidate_type\s*:\s*anomaly\b",
            chunk,
            flags=re.IGNORECASE,
        ):
            continue
        start_match = re.search(r"\bstart\s*:\s*(\d+)", chunk, flags=re.IGNORECASE)
        end_match = re.search(r"\bend\s*:\s*(\d+)", chunk, flags=re.IGNORECASE)
        if not start_match or not end_match:
            continue
        start_row = int(start_match.group(1))
        end_row = int(end_match.group(1))
        if end_row > start_row:
            ranges.append({"start_row": start_row, "end_row": end_row})

    return ranges


def extract_candidate_windows(thread_1_candidate_output: Any) -> list[dict[str, Any]]:
    """Extract candidate windows from Thread 1 <candidate_windows> blocks."""
    text = str(thread_1_candidate_output or "")
    blocks = re.findall(
        r"<candidate_windows>\s*(.*?)\s*</candidate_windows>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates: list[dict[str, Any]] = []

    for block in blocks:
        chunks = re.split(r"(?=\bstart\s*:)", block, flags=re.IGNORECASE)
        for chunk in chunks:
            start_match = re.search(r"\bstart\s*:\s*(\d+)", chunk, flags=re.IGNORECASE)
            end_match = re.search(r"\bend\s*:\s*(\d+)", chunk, flags=re.IGNORECASE)
            if not start_match or not end_match:
                continue
            start_row = int(start_match.group(1))
            end_row = int(end_match.group(1))
            if end_row <= start_row:
                continue

            visual_match = re.search(
                r"\bvisual_state\s*:\s*(.*?)(?=\n\s*candidate_type\s*:|\n\s*start\s*:|\Z)",
                chunk,
                flags=re.IGNORECASE | re.DOTALL,
            )
            type_match = re.search(
                r"\bcandidate_type\s*:\s*(normal|suspicious)",
                chunk,
                flags=re.IGNORECASE,
            )
            visual_state = " ".join((visual_match.group(1) if visual_match else "").split())
            candidate_type = (
                type_match.group(1).lower()
                if type_match
                else ""
            )
            candidates.append(
                {
                    "start_row": start_row,
                    "end_row": end_row,
                    "visual_state": visual_state,
                    "candidate_type": candidate_type,
                }
            )

    candidates.sort(key=lambda item: (item["start_row"], item["end_row"]))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for item in candidates:
        key = (int(item["start_row"]), int(item["end_row"]))
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


