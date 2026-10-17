#!/usr/bin/env python
import json
import sys
from pathlib import Path


DIFFICULTIES = ("easy", "medium", "hard", "very_hard")


def find_summaries(batch_dir: Path) -> list[Path]:
    summaries = list(batch_dir.glob("*/summary.json"))
    summaries.extend(batch_dir.glob("*/*/summary.json"))
    return sorted(set(summaries))


def checkpoint_name_for(summary_path: Path, batch_dir: Path) -> str:
    rel = summary_path.relative_to(batch_dir)
    if len(rel.parts) >= 3:
        return rel.parts[0]
    return summary_path.parent.name


def pct(value: float) -> str:
    return f"{100.0 * float(value):6.2f}"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <metaworld_batch_dir>", file=sys.stderr)
        return 2

    batch_dir = Path(sys.argv[1]).expanduser().resolve()
    summaries = find_summaries(batch_dir)
    if not summaries:
        print(f"No summary.json files found under {batch_dir}", file=sys.stderr)
        return 1

    rows = []
    for summary_path in summaries:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        per_difficulty = summary.get("per_difficulty", {})
        rows.append(
            {
                "checkpoint": checkpoint_name_for(summary_path, batch_dir),
                "summary_path": str(summary_path),
                "overall_success_rate": float(summary.get("overall_success_rate", 0.0)),
                "total_successes": int(summary.get("total_successes", 0)),
                "total_episodes": int(summary.get("total_episodes", 0)),
                "difficulty_success_rates": {
                    difficulty: float(per_difficulty.get(difficulty, {}).get("success_rate", 0.0))
                    for difficulty in DIFFICULTIES
                },
            }
        )

    rows.sort(key=lambda row: (row["overall_success_rate"], row["total_successes"]), reverse=True)

    leaderboard_json = batch_dir / "leaderboard.json"
    with leaderboard_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Leaderboard saved to: {leaderboard_json}")
    print(
        "rank overall% successes/episodes easy% medium% hard% very_hard% checkpoint"
    )
    for idx, row in enumerate(rows, start=1):
        diff = row["difficulty_success_rates"]
        print(
            f"{idx:>4} "
            f"{pct(row['overall_success_rate'])} "
            f"{row['total_successes']:>5}/{row['total_episodes']:<5} "
            f"{pct(diff['easy'])} "
            f"{pct(diff['medium'])} "
            f"{pct(diff['hard'])} "
            f"{pct(diff['very_hard'])} "
            f"{row['checkpoint']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
