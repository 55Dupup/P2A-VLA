import importlib.util
import json
from pathlib import Path


def load_aggregator_module():
    script_path = Path(__file__).resolve().parents[1] / "vla-scripts" / "aggregate_libero_plus_results.py"
    spec = importlib.util.spec_from_file_location("aggregate_libero_plus_results", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_suite(path, camera_success, camera_total, robot_success, robot_total):
    path.write_text(
        json.dumps(
            {
                "Camera Viewpoints": {"success_count": camera_success, "total_count": camera_total},
                "Robot Initial States": {"success_count": robot_success, "total_count": robot_total},
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_outputs_starvla_style_table(tmp_path):
    module = load_aggregator_module()
    for suite in module.TASK_SUITES:
        write_suite(tmp_path / f"{suite}.json", 1, 2, 3, 4)

    results, table = module.aggregate_results(tmp_path)

    assert results["Camera Viewpoints"]["success_rate"] == 0.5
    assert results["Robot Initial States"]["success_rate"] == 0.75
    assert results["overall"]["success_rate"] == 4 / 6
    assert "| Camera | Robot | Language | Light | Background | Noise | Layout | Total |" in table
    assert "| 50.0 | 75.0 | N/A | N/A | N/A | N/A | N/A | 66.7 |" in table
