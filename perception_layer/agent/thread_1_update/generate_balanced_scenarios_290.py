from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.generate_multi_anomaly_scenarios import (
    ACTIVITY_LABELS,
    SAMPLE_RATE_HZ,
    SENSOR_COLUMNS,
    TOTAL_ROWS,
    add_sensor_fault,
    load_mhealth_source,
)
from perception_layer.agent.thread_1_update.generate_human_event_scenarios import (
    ACTIVITY_TEXT,
    inject_event,
)


OUTPUT_ROOT = PROJECT_ROOT / "output" / "balanced_scenarios_290"
SEED = 290_2026

EVENTS = {
    "fall_impact_posture_change": {
        "label": "impact",
        "text": "falls with a clear body impact and posture change",
        "duration": (180, 250),
        "count": 49,
    },
    "near_fall_stumble": {
        "label": "impact",
        "text": "stumbles hard, nearly falls, and recovers",
        "duration": (150, 220),
        "count": 48,
    },
    "sensor_displacement_after_impact": {
        "label": "impact",
        "text": "has a clear impact that displaces one wearable sensor",
        "duration": (170, 240),
        "count": 48,
    },
    "dyspnea_breathing_distress": {
        "label": "non_impact",
        "text": "shows breathing distress",
        "duration": (300, 420),
        "count": 44,
    },
    "illness_fatigue_response": {
        "label": "non_impact",
        "text": "becomes unusually fatigued",
        "duration": (300, 420),
        "count": 43,
    },
    "sensor_fault_chest_dropout": {
        "label": "sensor",
        "text": "has a chest sensor dropout",
        "duration": (240, 380),
        "count": 20,
    },
    "sensor_fault_loose_ankle": {
        "label": "sensor",
        "text": "has a loose ankle sensor signal",
        "duration": (260, 400),
        "count": 19,
    },
    "sensor_fault_arm_mag_saturation": {
        "label": "sensor",
        "text": "has an arm magnetic sensor saturation fault",
        "duration": (240, 380),
        "count": 19,
    },
}

ACTIVITIES = tuple(ACTIVITY_LABELS)


def _balanced_values(values: tuple[str, ...], count: int, rng: np.random.Generator) -> list[str]:
    result = [values[index % len(values)] for index in range(count)]
    rng.shuffle(result)
    return result


def build_specs(count: int = 290, seed: int = SEED) -> list[dict[str, object]]:
    if count != sum(int(event["count"]) for event in EVENTS.values()):
        raise ValueError("This balanced manifest is defined for exactly 290 scenarios.")

    rng = np.random.default_rng(seed)
    event_types = [
        event_type
        for event_type, event in EVENTS.items()
        for _ in range(int(event["count"]))
    ]
    rng.shuffle(event_types)
    before_activities = _balanced_values(ACTIVITIES, count, rng)
    after_activities = _balanced_values(ACTIVITIES, count, rng)
    start_bands = np.tile(np.arange(350, 2201, 150), int(np.ceil(count / 13)))[:count]
    rng.shuffle(start_bands)

    specs: list[dict[str, object]] = []
    for index, event_type in enumerate(event_types, start=1):
        event = EVENTS[event_type]
        before = before_activities[index - 1]
        after = after_activities[index - 1]
        if after == before:
            after = ACTIVITIES[(ACTIVITIES.index(after) + 1 + index % 5) % len(ACTIVITIES)]

        min_duration, max_duration = event["duration"]
        duration = int(rng.integers(int(min_duration), int(max_duration) + 1))
        start = int(start_bands[index - 1] + rng.integers(-45, 46))
        start = max(300, min(start, TOTAL_ROWS - duration - 350))
        end = start + duration
        impact = start + max(35, duration // 3)
        before_subject = 1 + ((index - 1) % 10)
        after_subject = 1 + ((index + 3) % 10)

        specs.append({
            "name": f"scenario_{index:03d}",
            "before": before,
            "after": after,
            "before_subject": before_subject,
            "after_subject": after_subject,
            "event_type": event_type,
            "event_label": str(event["label"]),
            "event_text": str(event["text"]),
            "start": start,
            "impact": impact,
            "end": end,
        })
    return specs


def _sample_segment(
    source_df: pd.DataFrame,
    activity: str,
    subject_id: int,
    length: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, int]]:
    label = ACTIVITY_LABELS[activity]
    group = source_df[
        (source_df["label"] == label) & (source_df["subject_id"] == subject_id)
    ].sort_values("sample_index")
    if len(group) < length:
        raise ValueError(
            f"Insufficient source rows for activity={activity}, subject={subject_id}, length={length}"
        )
    offset = int(rng.integers(0, len(group) - length + 1))
    segment = group.iloc[offset:offset + length].copy()
    source = {
        "subject_id": subject_id,
        "sample_start": int(segment["sample_index"].iloc[0]),
        "sample_end": int(segment["sample_index"].iloc[-1]),
    }
    return segment[[*SENSOR_COLUMNS, "label"]].reset_index(drop=True), source


def make_base_signal(
    source_df: pd.DataFrame,
    spec: dict[str, object],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    split = int(spec["end"])
    segment_specs = [
        (0, split, str(spec["before"]), int(spec["before_subject"])),
        (split, TOTAL_ROWS, str(spec["after"]), int(spec["after_subject"])),
    ]
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for start, end, activity, subject_id in segment_specs:
        frame, source = _sample_segment(source_df, activity, subject_id, end - start, rng)
        frame["activity_name"] = activity
        frames.append(frame)
        sources.append({"start": start, "end": end, "activity": activity, **source})

    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "row", np.arange(TOTAL_ROWS, dtype=np.int32))
    df["time_seconds"] = df["row"] / SAMPLE_RATE_HZ
    df["is_anomaly"] = False
    df["anomaly_type"] = "normal"
    for column in SENSOR_COLUMNS:
        df[column] = df[column].astype(np.float32)
    df["label"] = df["label"].astype(np.int16)
    return df, sources


def inject_anomaly(df: pd.DataFrame, spec: dict[str, object], rng: np.random.Generator) -> None:
    event_type = str(spec["event_type"])
    if str(spec["event_label"]) == "sensor":
        add_sensor_fault(
            df,
            int(spec["start"]),
            int(spec["end"]),
            rng,
            event_type,
        )
        return
    inject_event(df, spec, rng)


def plot_sensor_images(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {
        "chest.png": [
            ("chest_acc", ["chest_acc_x", "chest_acc_y", "chest_acc_z"]),
            ("ecg", ["ecg_lead_1", "ecg_lead_2"]),
        ],
        "left_ankle.png": [
            ("left_ankle_acc", ["left_ankle_acc_x", "left_ankle_acc_y", "left_ankle_acc_z"]),
            ("left_ankle_gyro", ["left_ankle_gyro_x", "left_ankle_gyro_y", "left_ankle_gyro_z"]),
            ("left_ankle_mag", ["left_ankle_mag_x", "left_ankle_mag_y", "left_ankle_mag_z"]),
        ],
        "right_lower_arm.png": [
            ("right_lower_arm_acc", ["right_lower_arm_acc_x", "right_lower_arm_acc_y", "right_lower_arm_acc_z"]),
            ("right_lower_arm_gyro", ["right_lower_arm_gyro_x", "right_lower_arm_gyro_y", "right_lower_arm_gyro_z"]),
            ("right_lower_arm_mag", ["right_lower_arm_mag_x", "right_lower_arm_mag_y", "right_lower_arm_mag_z"]),
        ],
    }
    for filename, panels in plots.items():
        fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.6 * len(panels)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (title, columns) in zip(axes, panels):
            for column in columns:
                ax.plot(df["row"], df[column], linewidth=0.7, label=column)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.22)
            ax.legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("row")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=130)
        plt.close(fig)


def write_scenario(
    index: int,
    spec: dict[str, object],
    source_df: pd.DataFrame,
    output_root: Path,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + index)
    scenario_dir = output_root / str(spec["name"])
    scenario_dir.mkdir(parents=True, exist_ok=True)
    df, sources = make_base_signal(source_df, spec, rng)
    inject_anomaly(df, spec, rng)
    df.to_parquet(scenario_dir / "data.parquet", index=False)

    ground_truth = {
        "person_state_description": (
            f"person is {ACTIVITY_TEXT[str(spec['before'])]}, then {spec['event_text']}, "
            f"then {ACTIVITY_TEXT[str(spec['after'])]}."
        ),
        "anomaly_range": {
            "start": int(spec["start"]),
            "end": int(spec["end"]),
            "label": str(spec["event_label"]),
        },
    }
    (scenario_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_sensor_images(df, scenario_dir / "image")
    return {**spec, "sources": sources}


def generate_batch(
    output_root: Path,
    start_index: int,
    end_index: int,
    seed: int = SEED,
) -> None:
    specs = build_specs(seed=seed)
    source_df = load_mhealth_source()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "scenario_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    for index in range(start_index, end_index + 1):
        record = write_scenario(index, specs[index - 1], source_df, output_root, seed)
        manifest[str(record["name"])] = record
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the balanced 290-scenario benchmark.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=290)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not 1 <= args.start <= args.end <= 290:
        raise ValueError("Expected 1 <= start <= end <= 290")
    generate_batch(args.output_root, args.start, args.end, args.seed)
    specs = build_specs(seed=args.seed)
    labels = Counter(str(spec["event_label"]) for spec in specs)
    print(f"output_root: {args.output_root}")
    print(f"generated_batch: {args.start}-{args.end}")
    print(f"target_distribution: {dict(labels)}")


if __name__ == "__main__":
    main()
