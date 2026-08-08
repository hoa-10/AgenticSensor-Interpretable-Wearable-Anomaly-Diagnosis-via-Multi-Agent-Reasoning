from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception_layer.agent.thread_1_update.scenario_generator import (
    ACTIVITY_TEXT,
    clean_output_root,
    write_scenario,
)
from classification.generate_multi_anomaly_scenarios import (
    TOTAL_ROWS,
    load_mhealth_source,
)


OUTPUT_ROOT = PROJECT_ROOT / "output" / "non_impact_test_scenarios"


NON_IMPACT_CASES = [
    ("walking", "standing", "health_event_dyspnea", "shows breathing distress", 700, 1060),
    ("running", "walking", "health_event_tachycardia", "shows a fast heart rhythm", 850, 1190),
    ("jogging", "standing", "health_event_arrhythmia", "shows irregular heart rhythm", 980, 1310),
    ("stairs", "sitting", "health_event_fatigue", "becomes unusually fatigued", 1200, 1540),
    ("cycling", "standing", "health_event_recovery_distress", "shows distress during recovery", 1000, 1360),
    ("walking", "sitting", "health_event_syncope", "has a fainting-like episode", 1450, 1810),
    ("running", "standing", "health_event_dyspnea", "shows breathing distress", 1700, 2060),
    ("cycling", "walking", "health_event_tachycardia", "shows a fast heart rhythm", 1900, 2240),
    ("stairs", "walking", "health_event_arrhythmia", "shows irregular heart rhythm", 2100, 2430),
    ("jogging", "walking", "health_event_fatigue", "becomes unusually fatigued", 2250, 2590),
]


def build_specs() -> list[dict[str, object]]:
    specs = []
    for index, (before, after, event_type, event_text, start, end) in enumerate(
        NON_IMPACT_CASES,
        start=1,
    ):
        split = min(TOTAL_ROWS - 400, end + 300)
        specs.append({
            "name": (
                f"scenario_non_impact_{index:03d}_{before}_to_{after}_"
                f"{event_type.replace('health_event_', '')}"
            ),
            "segments": [(0, split, before), (split, TOTAL_ROWS, after)],
            "transition_blend_rows": 100,
            "events": [{
                "start_row": start,
                "end_row": end,
                "group": "health",
                "type": event_type,
            }],
            "description": (
                f"person is {ACTIVITY_TEXT[before]}, then {event_text}, "
                f"then {ACTIVITY_TEXT[after]}."
            ),
        })
    return specs


def generate_non_impact_scenarios() -> Path:
    clean_output_root(OUTPUT_ROOT)
    source_df = load_mhealth_source()
    for index, spec in enumerate(build_specs(), start=1):
        write_scenario(index, spec, source_df, OUTPUT_ROOT)
    return OUTPUT_ROOT


def main() -> None:
    output_root = generate_non_impact_scenarios()
    print(f"generated_scenarios: {output_root}")
    print(f"scenario_count: {len(NON_IMPACT_CASES)}")
    print("anomaly_label: non_impact")


if __name__ == "__main__":
    main()
