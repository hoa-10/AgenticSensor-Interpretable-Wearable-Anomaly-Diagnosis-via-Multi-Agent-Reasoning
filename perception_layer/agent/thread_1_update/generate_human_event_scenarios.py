from __future__ import annotations

import json
import shutil
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
    SAMPLE_RATE_HZ,
    SENSOR_COLUMNS,
    TOTAL_ROWS,
    load_mhealth_source,
    make_base_signal,
)
from sensor_layer.fault_injection.human_events import INJECTORS


OUTPUT_ROOT = PROJECT_ROOT / "output" / "human_event_scenarios"

ACTIVITY_TEXT = {
    "standing": "standing still",
    "sitting": "sitting and relaxing",
    "walking": "walking",
    "stairs": "climbing stairs",
    "cycling": "cycling",
    "jogging": "jogging",
    "running": "running",
}

EVENTS = [
    {
        "type": "fall_impact_posture_change",
        "label": "impact",
        "text": "falls with a clear body impact and posture change",
        "duration": 220,
    },
    {
        "type": "near_fall_stumble",
        "label": "impact",
        "text": "stumbles hard, nearly falls, and recovers",
        "duration": 190,
    },
    {
        "type": "sensor_displacement_after_impact",
        "label": "impact",
        "text": "has a clear impact that displaces one wearable sensor",
        "duration": 210,
    },
    {
        "type": "dyspnea_breathing_distress",
        "label": "non_impact",
        "text": "shows breathing distress",
        "duration": 360,
    },
    {
        "type": "illness_fatigue_response",
        "label": "non_impact",
        "text": "becomes unusually fatigued",
        "duration": 360,
    },
]

ACTIVITY_PAIRS = [
    ("walking", "standing"),
    ("running", "walking"),
    ("jogging", "standing"),
    ("stairs", "sitting"),
    ("cycling", "standing"),
    ("sitting", "walking"),
    ("standing", "walking"),
    ("walking", "sitting"),
    ("running", "standing"),
    ("cycling", "walking"),
]

EVENT_STARTS = [
    420, 540, 660, 780, 900,
    1020, 1140, 1260, 1380, 1500,
    1620, 1740, 1860, 1980, 2100,
]


def clean_output_root(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def build_scenario_specs(count: int = 50) -> list[dict[str, object]]:
    specs = []
    for index in range(1, count + 1):
        event = EVENTS[(index - 1) % len(EVENTS)]
        before, after = ACTIVITY_PAIRS[(index - 1) % len(ACTIVITY_PAIRS)]
        start = EVENT_STARTS[(index - 1) % len(EVENT_STARTS)] + 5 * ((index - 1) // len(EVENT_STARTS))
        end = min(start + int(event["duration"]), TOTAL_ROWS - 350)
        split = end
        specs.append({
            "name": f"scenario_{index:03d}",
            "before": before,
            "after": after,
            "segments": [(0, split, before), (split, TOTAL_ROWS, after)],
            "event_type": str(event["type"]),
            "event_label": str(event["label"]),
            "event_text": str(event["text"]),
            "start": int(start),
            "impact": int(start + max(45, (end - start) // 3)),
            "end": int(end),
        })
    return specs


def _human_event_frames(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    mappings = {
        "chest": {
            "chest_acc_x": "chest_acc_x",
            "chest_acc_y": "chest_acc_y",
            "chest_acc_z": "chest_acc_z",
            "chest_ecg_1": "ecg_lead_1",
            "chest_ecg_2": "ecg_lead_2",
        },
        "ankle": {
            "ankle_acc_x": "left_ankle_acc_x",
            "ankle_acc_y": "left_ankle_acc_y",
            "ankle_acc_z": "left_ankle_acc_z",
            "ankle_gyro_x": "left_ankle_gyro_x",
            "ankle_gyro_y": "left_ankle_gyro_y",
            "ankle_gyro_z": "left_ankle_gyro_z",
        },
        "hand": {
            "hand_acc_x": "right_lower_arm_acc_x",
            "hand_acc_y": "right_lower_arm_acc_y",
            "hand_acc_z": "right_lower_arm_acc_z",
            "hand_gyro_x": "right_lower_arm_gyro_x",
            "hand_gyro_y": "right_lower_arm_gyro_y",
            "hand_gyro_z": "right_lower_arm_gyro_z",
        },
    }
    frames: dict[str, pd.DataFrame] = {}
    for position, mapping in mappings.items():
        frame = pd.DataFrame({
            target: df[source].to_numpy(dtype=np.float64, copy=True)
            for target, source in mapping.items()
        })
        frame["sampling_rate"] = SAMPLE_RATE_HZ
        frames[position] = frame
    return frames, mappings


def _copy_human_event_frames(
    df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    mappings: dict[str, dict[str, str]],
) -> None:
    for position, mapping in mappings.items():
        for target, source in mapping.items():
            df[source] = frames[position][target].to_numpy(dtype=np.float32)


def _robust_scale(values: np.ndarray, minimum: float) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return minimum
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median))) * 1.4826
    return max(mad, float(np.std(clean)), minimum)


def strengthen_impact_signature(
    df: pd.DataFrame,
    event_type: str,
    impact: int,
    rng: np.random.Generator,
) -> None:
    if event_type not in {
        "fall_impact_posture_change",
        "near_fall_stumble",
        "sensor_displacement_after_impact",
    }:
        return

    rows = np.arange(len(df), dtype=float)
    primary = np.exp(-0.5 * ((rows - impact) / 2.0) ** 2)
    rebound = np.exp(-0.5 * ((rows - (impact + 7)) / 3.2) ** 2)
    pulse = primary - 0.45 * rebound

    if event_type == "near_fall_stumble":
        gains = {"chest": 5.0, "ankle": 13.0, "arm": 6.0}
    elif event_type == "sensor_displacement_after_impact":
        gains = {"chest": 7.0, "ankle": 10.0, "arm": 8.0}
    else:
        gains = {"chest": 12.0, "ankle": 15.0, "arm": 10.0}

    channel_groups = {
        "chest": ["chest_acc_x", "chest_acc_y", "chest_acc_z"],
        "ankle": ["left_ankle_acc_x", "left_ankle_acc_y", "left_ankle_acc_z"],
        "arm": ["right_lower_arm_acc_x", "right_lower_arm_acc_y", "right_lower_arm_acc_z"],
    }
    for group, columns in channel_groups.items():
        for column in columns:
            values = df[column].to_numpy(dtype=np.float64)
            scale = _robust_scale(values[max(0, impact - 150):impact], minimum=0.35)
            sign = float(rng.choice([-1.0, 1.0]))
            df[column] = (values + sign * gains[group] * scale * pulse).astype(np.float32)

    gyro_columns = [
        "left_ankle_gyro_x", "left_ankle_gyro_y", "left_ankle_gyro_z",
        "right_lower_arm_gyro_x", "right_lower_arm_gyro_y", "right_lower_arm_gyro_z",
    ]
    for column in gyro_columns:
        values = df[column].to_numpy(dtype=np.float64)
        scale = _robust_scale(values[max(0, impact - 150):impact], minimum=0.08)
        sign = float(rng.choice([-1.0, 1.0]))
        df[column] = (values + sign * 9.0 * scale * pulse).astype(np.float32)


def inject_event(df: pd.DataFrame, spec: dict[str, object], rng: np.random.Generator) -> None:
    start = int(spec["start"])
    impact = int(spec["impact"])
    end = int(spec["end"])
    event_type = str(spec["event_type"])
    frames, mappings = _human_event_frames(df)
    originals = {position: frame.copy(deep=True) for position, frame in frames.items()}

    INJECTORS[event_type](frames, start, impact, end, rng)
    for position, frame in frames.items():
        frame.loc[:start - 1] = originals[position].loc[:start - 1]
        frame.loc[end:] = originals[position].loc[end:]
    _copy_human_event_frames(df, frames, mappings)
    strengthen_impact_signature(df, event_type, impact, rng)

    df.loc[start:end - 1, "is_anomaly"] = True
    df.loc[start:end - 1, "anomaly_type"] = event_type


def plot_three_images(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        ("chest.png", [("chest_acc", ["chest_acc_x", "chest_acc_y", "chest_acc_z"]), ("ecg", ["ecg_lead_1", "ecg_lead_2"])]),
        ("left_ankle.png", [("left_ankle_acc", ["left_ankle_acc_x", "left_ankle_acc_y", "left_ankle_acc_z"]), ("left_ankle_gyro", ["left_ankle_gyro_x", "left_ankle_gyro_y", "left_ankle_gyro_z"])]),
        ("right_lower_arm.png", [("right_lower_arm_acc", ["right_lower_arm_acc_x", "right_lower_arm_acc_y", "right_lower_arm_acc_z"]), ("right_lower_arm_gyro", ["right_lower_arm_gyro_x", "right_lower_arm_gyro_y", "right_lower_arm_gyro_z"])]),
    ]
    for filename, axes_spec in groups:
        fig, axes = plt.subplots(2, 1, figsize=(14, 5.6), sharex=True)
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


def write_scenario(
    index: int,
    spec: dict[str, object],
    source_df: pd.DataFrame,
    output_root: Path,
) -> None:
    rng = np.random.default_rng(31000 + index)
    scenario_dir = output_root / str(spec["name"])
    df, _segments = make_base_signal(source_df, list(spec["segments"]), rng)
    inject_event(df, spec, rng)

    scenario_dir.mkdir(parents=True, exist_ok=True)
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
    plot_three_images(df, scenario_dir / "image")


def generate_scenarios(output_root: str | Path = OUTPUT_ROOT, count: int = 50) -> Path:
    output_root = Path(output_root)
    clean_output_root(output_root)
    source_df = load_mhealth_source()
    for index, spec in enumerate(build_scenario_specs(count), start=1):
        write_scenario(index, spec, source_df, output_root)
    return output_root


def main() -> None:
    output_root = generate_scenarios()
    print(f"generated_scenarios: {output_root}")
    print("scenario_count: 50")
    print("impact_count: 30")
    print("non_impact_count: 20")
    print("events_per_scenario: 1")
    print("images_per_scenario: 3")
    print(f"base_source: {MHEALTH_PARQUET}")


if __name__ == "__main__":
    main()
