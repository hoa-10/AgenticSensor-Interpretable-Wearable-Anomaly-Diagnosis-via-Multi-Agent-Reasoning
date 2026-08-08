from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


_LOCK = threading.Lock()
_EVENTS: list[dict[str, Any]] = []


def reset_timing() -> None:
    with _LOCK:
        _EVENTS.clear()


def record_timing(name: str, seconds: float, **metadata: Any) -> None:
    event = {
        "name": name,
        "seconds": round(float(seconds), 3),
        **metadata,
    }
    with _LOCK:
        _EVENTS.append(event)


def snapshot_timing() -> dict[str, Any]:
    with _LOCK:
        events = list(_EVENTS)

    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_seconds": 0.0, "calls": []}
    )
    for event in events:
        item = grouped[str(event["name"])]
        item["count"] += 1
        item["total_seconds"] += float(event["seconds"])
        item["calls"].append(event)

    return {
        name: {
            "count": value["count"],
            "total_seconds": round(value["total_seconds"], 3),
            "calls": value["calls"],
        }
        for name, value in grouped.items()
    }
