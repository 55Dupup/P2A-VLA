"""Aggregate LIBERO-plus suite JSON files into a StarVLA-style summary table."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

CATEGORY_COLUMNS = (
    ("Camera", "Camera Viewpoints"),
    ("Robot", "Robot Initial States"),
    ("Language", "Language Instructions"),
    ("Light", "Light Conditions"),
    ("Background", "Background Textures"),
    ("Noise", "Sensor Noise"),
    ("Layout", "Objects Layout"),
)


def _empty_counter() -> Dict[str, int]:
    return {"total_count": 0, "success_count": 0}


def _add_counts(target: Dict[str, int], source: Dict[str, int]) -> None:
    target["total_count"] += int(source.get("total_count", 0))
    target["success_count"] += int(source.get("success_count", 0))


def _finalize_rates(results: Dict[str, Dict[str, float]]) -> None:
    for counts in results.values():
        total = counts["total_count"]
        counts["success_rate"] = float(counts["success_count"]) / float(total) if total else None


def _format_rate(rate) -> str:
    return "N/A" if rate is None else f"{rate * 100:.1f}"


def build_summary_table(results: Dict[str, Dict[str, float]]) -> str:
    headers = [label for label, _ in CATEGORY_COLUMNS] + ["Total"]
    values = []
    values.extend(_format_rate(results.get(category, {}).get("success_rate")) for _, category in CATEGORY_COLUMNS)
    values.append(_format_rate(results["overall"]["success_rate"]))

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    value_line = "| " + " | ".join(values) + " |"
    return "\n".join([header_line, separator, value_line])


def aggregate_results(log_dir: Path) -> Tuple[Dict[str, Dict[str, float]], str]:
    log_dir = Path(log_dir)
    results: Dict[str, Dict[str, float]] = {"overall": _empty_counter()}

    for suite in TASK_SUITES:
        suite_path = log_dir / f"{suite}.json"
        if not suite_path.exists():
            continue

        with suite_path.open("r", encoding="utf-8") as f:
            suite_results = json.load(f)

        for category, counts in suite_results.items():
            _add_counts(results["overall"], counts)
            if category not in results:
                results[category] = _empty_counter()
            _add_counts(results[category], counts)

    _finalize_rates(results)
    return results, build_summary_table(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LIBERO-plus suite category results.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory containing libero_*.json files.")
    args = parser.parse_args()

    results, table = aggregate_results(args.log_dir)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    with (args.log_dir / "overall_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    (args.log_dir / "summary.md").write_text(table + "\n", encoding="utf-8")

    print(table)
    print(f"Wrote aggregate results to: {args.log_dir / 'overall_results.json'}")
    print(f"Wrote summary table to: {args.log_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
