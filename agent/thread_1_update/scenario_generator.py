from __future__ import annotations

import json
import os
import shutil
import stat
import sys
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
    MHEALTH_PARQUET,
    OUTPUT_ROOT,
    SAMPLE_RATE_HZ,
    SENSOR_COLUMNS,
    TOTAL_ROWS,
    add_event,
    load_mhealth_source,
    make_base_signal,
)


ACTIVITY_TEXT = {
    "standing": "standing still",
    "sitting": "sitting and relaxing",
    "walking": "walking",
    "stairs": "climbing stairs",
    "cycling": "cycling",
    "jogging": "jogging",
    "running": "running",
}

EVENT_LIBRARY = [
    {"group": "impact", "type": "impact_trip", "duration": 170, "text": "trips and briefly loses balance"},
    {"group": "impact", "type": "impact_fall", "duration": 180, "text": "falls suddenly"},
    {"group": "impact", "type": "impact_stair_trip", "duration": 180, "text": "stumbles on the stairs"},
    {"group": "impact", "type": "impact_collision", "duration": 160, "text": "has a sudden collision"},
    {"group": "impact", "type": "impact_near_fall", "duration": 180, "text": "nearly falls and recovers"},
    {"group": "impact", "type": "impact_collapse", "duration": 170, "text": "collapses suddenly"},
    {"group": "impact", "type": "impact_stumble", "duration": 170, "text": "stumbles hard"},
    {"group": "impact", "type": "impact_bike_jolt", "duration": 170, "text": "gets a strong jolt while cycling"},
    {"group": "health", "type": "health_event_dyspnea", "duration": 360, "text": "shows breathing distress"},
    {"group": "health", "type": "health_event_tachycardia", "duration": 340, "text": "shows a fast heart rhythm"},
    {"group": "health", "type": "health_event_arrhythmia", "duration": 330, "text": "shows irregular heart rhythm"},
    {"group": "health", "type": "health_event_fatigue", "duration": 340, "text": "becomes unusually fatigued"},
    {"group": "health", "type": "health_event_syncope", "duration": 360, "text": "has a fainting-like episode"},
    {"group": "health", "type": "health_event_recovery_distress", "duration": 360, "text": "shows distress during recovery"},
    {"group": "sensor_fault", "type": "sensor_fault_chest_dropout", "duration": 320, "text": "has a chest sensor dropout"},
    {"group": "sensor_fault", "type": "sensor_fault_loose_ankle", "duration": 380, "text": "has a loose ankle sensor signal"},
    {"group": "sensor_fault", "type": "sensor_fault_arm_mag_saturation", "duration": 360, "text": "has an arm magnetic sensor fault"},
]

ACTIVITY_PAIRS = [
    ("walking", "standing"),
    ("running", "walking"),
    ("jogging", "standing"),
    ("stairs", "walking"),
    ("cycling", "standing"),
    ("sitting", "walking"),
    ("standing", "walking"),
    ("walking", "sitting"),
    ("running", "standing"),
    ("cycling", "walking"),
    ("stairs", "sitting"),
    ("jogging", "walking"),
    ("standing", "cycling"),
    ("walking", "stairs"),
    ("sitting", "standing"),
]

EVENT_STARTS = [
    420, 560, 700, 840, 980,
    1120, 1260, 1400, 1540, 1680,
    1820, 1960, 2100, 2240, 2380,
]


def clean_output_root(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    def force_remove(func, path, _exc_info) -> None:
        os.chmod(path, stat.S_IWRITE)
        func(path)

    for child in output_root.iterdir():
        if child.is_dir() and (child.name.startswith("scenario_") or child.name.startswith("cscad_plots")):
            shutil.rmtree(child, onexc=force_remove)
        elif child.is_file() and child.suffix.lower() in {".csv", ".png", ".json", ".txt"}:
            child.unlink()


def slug(text: str) -> str:
    return text.replace("health_event_", "health_").replace("sensor_fault_", "sensor_").replace("impact_", "impact_")


def build_scenario_specs(count: int = 50) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    events_by_group = {
        group: [event for event in EVENT_LIBRARY if event["group"] == group]
        for group in ["impact", "health", "sensor_fault"]
    }
    group_order = ["impact", "health", "sensor_fault"]
    group_counts = {group: 0 for group in group_order}

    for index in range(1, count + 1):
        group = group_order[(index - 1) % len(group_order)]
        event_template = events_by_group[group][group_counts[group] % len(events_by_group[group])]
        group_counts[group] += 1
        before_activity, after_activity = ACTIVITY_PAIRS[(index - 1) % len(ACTIVITY_PAIRS)]
        start = EVENT_STARTS[(index - 1) % len(EVENT_STARTS)] + 10 * ((index - 1) // len(EVENT_STARTS))
        duration = int(event_template["duration"])
        end = min(start + duration, TOTAL_ROWS - 160)
        split = min(TOTAL_ROWS - 500, max(900, end + 260 + 80 * (index % 4)))

        event = {
            "start_row": int(start),
            "end_row": int(end),
            "group": str(event_template["group"]),
            "type": str(event_template["type"]),
        }
        description = (
            f"person is {ACTIVITY_TEXT[before_activity]}, then {event_template['text']}, "
            f"then {ACTIVITY_TEXT[after_activity]}."
        )
        name = (
            f"scenario_{index:03d}_"
            f"{before_activity}_to_{after_activity}_"
            f"{slug(str(event_template['type']))}"
        )
        specs.append({
            "name": name,
            "segments": [(0, split, before_activity), (split, TOTAL_ROWS, after_activity)],
            "events": [event],
            "description": description,
        })
    return specs


def plot_three_images(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        (
            "chest.png",
            [
                ("chest_acc", ["chest_acc_x", "chest_acc_y", "chest_acc_z"]),
                ("ecg", ["ecg_lead_1", "ecg_lead_2"]),
            ],
        ),
        (
            "left_ankle.png",
            [
                ("left_ankle_acc", ["left_ankle_acc_x", "left_ankle_acc_y", "left_ankle_acc_z"]),
                ("left_ankle_gyro", ["left_ankle_gyro_x", "left_ankle_gyro_y", "left_ankle_gyro_z"]),
            ],
        ),
        (
            "right_lower_arm.png",
            [
                ("right_lower_arm_acc", ["right_lower_arm_acc_x", "right_lower_arm_acc_y", "right_lower_arm_acc_z"]),
                ("right_lower_arm_gyro", ["right_lower_arm_gyro_x", "right_lower_arm_gyro_y", "right_lower_arm_gyro_z"]),
            ],
        ),
    ]

    for filename, axes_spec in groups:
        fig, axes = plt.subplots(len(axes_spec), 1, figsize=(14, 2.8 * len(axes_spec)), sharex=True)
        if len(axes_spec) == 1:
            axes = [axes]
        for ax, (name, columns) in zip(axes, axes_spec):
            for column in columns:
                ax.plot(df["row"], df[column], linewidth=0.75, label=column)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.22)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("row")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def smooth_activity_boundaries(
    df: pd.DataFrame,
    segments: list[tuple[int, int, str]],
    blend_rows: int,
) -> None:
    if blend_rows <= 0:
        return
    for _start, boundary, _activity in segments[:-1]:
        blend_end = min(len(df), boundary + blend_rows)
        pre_start = max(0, boundary - blend_rows)
        if boundary <= pre_start or blend_end <= boundary:
            continue
        weights = np.linspace(1.0, 0.0, blend_end - boundary, dtype=np.float32)
        for column in SENSOR_COLUMNS:
            pre_center = float(df.loc[pre_start:boundary - 1, column].median())
            post_center = float(df.loc[boundary:blend_end - 1, column].median())
            current = df.loc[boundary:blend_end - 1, column].to_numpy(dtype=np.float32)
            df.loc[boundary:blend_end - 1, column] = current + (pre_center - post_center) * weights


def write_scenario(
    index: int,
    spec: dict[str, object],
    source_df: pd.DataFrame,
    output_root: Path,
) -> None:
    rng = np.random.default_rng(20000 + index)
    scenario_dir = output_root / str(spec["name"])
    image_dir = scenario_dir / "image"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    df, _activity_segments = make_base_signal(source_df, spec["segments"], rng)
    smooth_activity_boundaries(
        df,
        spec["segments"],
        int(spec.get("transition_blend_rows", 0)),
    )
    events = list(spec["events"])
    for event in events:
        add_event(df, event, rng)

    event = events[0]
    anomaly_label = {
        "impact": "impact",
        "health": "non_impact",
        "sensor_fault": "sensor",
    }[str(event["group"])]
    df.to_parquet(scenario_dir / "data.parquet", index=False)
    ground_truth = {
        "person_state_description": str(spec["description"]),
        "anomaly_range": {
            "start": int(event["start_row"]),
            "end": int(event["end_row"]),
            "label": anomaly_label,
        },
    }
    (scenario_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    plot_three_images(df, image_dir)


def generate_scenarios(output_root: str | Path = OUTPUT_ROOT, count: int = 50) -> Path:
    output_root = Path(output_root)
    clean_output_root(output_root)
    source_df = load_mhealth_source()
    specs = build_scenario_specs(count)
    for index, spec in enumerate(specs, start=1):
        write_scenario(index, spec, source_df, output_root)
    return output_root


def main() -> None:
    output_root = generate_scenarios()
    print(f"generated_scenarios: {output_root}")
    print("scenario_count: 50")
    print("events_per_scenario: 1")
    print("activity_segments_per_scenario: 2")
    print("images_per_scenario: 3")
    print(f"base_source: {MHEALTH_PARQUET}")


if __name__ == "__main__":
    main()
