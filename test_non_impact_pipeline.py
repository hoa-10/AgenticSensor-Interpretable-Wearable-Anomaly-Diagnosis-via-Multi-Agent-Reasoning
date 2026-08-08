from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception_layer.agent.thread_1_update.pipeline import run_pipeline
from perception_layer.agent.thread_2 import Thread2FinalReasoningAgent


SCENARIO_ROOT = PROJECT_ROOT / "output" / "non_impact_test_scenarios"
SUMMARY_PATH = SCENARIO_ROOT / "non_impact_test_summary.json"


def parse_final_output(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(cleaned)


def best_overlapping_prediction(
    predictions: list[dict[str, object]],
    start: int,
    end: int,
) -> dict[str, object] | None:
    best = None
    best_overlap = 0
    for prediction in predictions:
        overlap = max(
            0,
            min(end, int(prediction["end"])) - max(start, int(prediction["start"])),
        )
        if overlap > best_overlap:
            best = prediction
            best_overlap = overlap
    return best


def main() -> None:
    scenario_dirs = sorted(SCENARIO_ROOT.glob("scenario_*"))
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenarios found under {SCENARIO_ROOT}")

    agent = Thread2FinalReasoningAgent()
    results = []
    for index, scenario_dir in enumerate(scenario_dirs, start=1):
        print(f"[{index}/{len(scenario_dirs)}] {scenario_dir.name}", flush=True)
        ground_truth = json.loads(
            (scenario_dir / "ground_truth.json").read_text(encoding="utf-8")
        )
        with contextlib.redirect_stdout(io.StringIO()):
            thread_1 = run_pipeline(
                scenario_root=scenario_dir,
                debug=False,
                emit_output=False,
            )[0]
            thread_2 = agent.run(
                thread_1_output=thread_1["final_output"],
                parquet_path=thread_1["data_path"],
            )

        final_output = str(thread_2.get("final_output", ""))
        parsed = parse_final_output(final_output)
        truth_range = ground_truth["anomaly_range"]
        prediction = best_overlapping_prediction(
            list(parsed.get("anomaly_ranges", [])),
            int(truth_range["start"]),
            int(truth_range["end"]),
        )
        predicted_label = str(prediction.get("label")) if prediction else "none"
        record = {
            "scenario": scenario_dir.name,
            "ground_truth_label": "non_impact",
            "predicted_label": predicted_label,
            "correct": predicted_label == "non_impact",
            "prediction": prediction,
        }
        results.append(record)
        (scenario_dir / "thread_1_output.txt").write_text(
            str(thread_1["final_output"]),
            encoding="utf-8",
        )
        (scenario_dir / "thread_2_output.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  predicted={predicted_label} correct={record['correct']}", flush=True)

    correct = sum(bool(result["correct"]) for result in results)
    summary = {
        "scenario_count": len(results),
        "correct_non_impact": correct,
        "label_accuracy": correct / len(results),
        "results": results,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"accuracy={summary['label_accuracy']:.4f}")
    print(f"summary={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
