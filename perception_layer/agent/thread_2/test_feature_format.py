from __future__ import annotations

import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception_layer.agent.thread_2.health_agent import Thread2HealthAgent
from perception_layer.agent.thread_2.impact_agent import Thread2ImpactAgent
from perception_layer.agent.thread_2.sensor_fault_agent import Thread2SensorFaultAgent
from perception_layer.agent.utils import extract_anomaly_ranges


SCENARIO_NAME = "scenario_003_jogging_to_standing_sensor_chest_dropout"
SCENARIO_DIR = PROJECT_ROOT / "output" / "generated_scenarios" / SCENARIO_NAME


def main() -> None:
    parquet_path = SCENARIO_DIR / "data.parquet"
    thread_1_path = SCENARIO_DIR / "thread_1_output.txt"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing scenario data: {parquet_path}")
    if thread_1_path.exists():
        thread_1_output = thread_1_path.read_text(encoding="utf-8")
        anomaly_ranges = extract_anomaly_ranges(thread_1_output)
    else:
        ground_truth = json.loads((SCENARIO_DIR / "ground_truth.json").read_text(encoding="utf-8"))
        anomaly = ground_truth["anomaly_range"]
        anomaly_ranges = [{"start_row": anomaly["start"], "end_row": anomaly["end"]}]
    if not anomaly_ranges:
        raise ValueError(f"Thread 1 returned no anomaly range for {SCENARIO_NAME}")

    impact_agent = Thread2ImpactAgent()
    health_agent = Thread2HealthAgent()
    sensor_agent = Thread2SensorFaultAgent()

    print(f"scenario: {SCENARIO_NAME}")
    print(f"thread_1_anomaly_ranges: {anomaly_ranges}")

    for index, anomaly_range in enumerate(anomaly_ranges, start=1):
        start_row = anomaly_range["start_row"]
        end_row = anomaly_range["end_row"]
        print(f"\n######## RANGE {index}: {start_row}-{end_row} ########")

        tables = {
            "IMPACT FEATURE TABLE": impact_agent._format_impact_analysis_table(
                impact_agent.build_impact_analysis(parquet_path, start_row, end_row)
            ),
            "HEALTH FEATURE TABLE": health_agent._format_health_analysis_table(
                health_agent.build_health_analysis(parquet_path, start_row, end_row)
            ),
            "SENSOR FEATURE TABLE": sensor_agent._format_fault_analysis_table(
                sensor_agent.build_fault_analysis(parquet_path, start_row, end_row)
            ),
        }

        for title, table in tables.items():
            if str(parquet_path) in table:
                raise AssertionError(f"Path leaked into {title}")
            print(f"\n=== {title} ===")
            print(table)


if __name__ == "__main__":
    main()
