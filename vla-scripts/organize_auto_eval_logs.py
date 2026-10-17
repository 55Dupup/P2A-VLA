#!/usr/bin/env python3
"""Clean auto_eval wrapper logs and make internal eval results easy to scan."""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


AUTO_EVAL = Path("/path/to/P2A-VLA/experiments/libero_eval_logs/auto_eval")
INTERNAL = AUTO_EVAL / "internal"
SUMMARY = INTERNAL / "SUCCESS_RATE_SUMMARY.tsv"

FINAL_RE = re.compile(r"Overall success rate:\s*(?P<fraction>[0-9.]+)\s*\((?P<percent>[0-9.]+)%\)")
TOTAL_EPISODES_RE = re.compile(r"Total episodes:\s*(?P<episodes>\d+)")
TOTAL_SUCCESSES_RE = re.compile(r"Total successes:\s*(?P<successes>\d+)")
LOG_RE = re.compile(
    r"^EVAL-(?P<suite>.+?)-auto-(?P<checkpoint>.+--(?P<step>\d+)_chkpt)-"
    r"(?P<timestamp>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})(?:--sr[0-9.]+)?\.txt$"
)


@dataclass(frozen=True)
class InternalResult:
    path: Path
    suite: str
    checkpoint: str
    step: int
    timestamp: str
    rate: float
    episodes: int
    successes: int

    @property
    def normalized_name(self) -> str:
        return (
            f"EVAL-{self.suite}-auto-{self.checkpoint}-"
            f"{self.timestamp}--sr{self.rate:.1f}.txt"
        )


def parse_internal_log(path: Path) -> InternalResult | None:
    match = LOG_RE.match(path.name)
    if not match:
        return None
    text = path.read_text(errors="replace")
    final = FINAL_RE.search(text)
    episodes = TOTAL_EPISODES_RE.search(text)
    successes = TOTAL_SUCCESSES_RE.search(text)
    if final is None or episodes is None or successes is None:
        return None
    return InternalResult(
        path=path,
        suite=match.group("suite"),
        checkpoint=match.group("checkpoint"),
        step=int(match.group("step")),
        timestamp=match.group("timestamp"),
        rate=float(final.group("percent")),
        episodes=int(episodes.group("episodes")),
        successes=int(successes.group("successes")),
    )


def root_garbage_paths() -> list[Path]:
    paths: set[Path] = set()
    for pattern in ("*.log", "best_by_series_*.txt"):
        paths.update(path for path in AUTO_EVAL.glob(pattern) if path.is_file())
    lock = AUTO_EVAL / ".scheduler.lock"
    if lock.exists():
        paths.add(lock)
    staging = AUTO_EVAL / "staging"
    if staging.exists():
        for path in staging.iterdir():
            paths.add(path)
    return sorted(paths)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def write_summary(results: list[InternalResult], apply: bool) -> None:
    lines = [
        "success_rate_percent\tsuite\tstep\tcheckpoint\tepisodes\tsuccesses\tlog_file",
    ]
    for result in sorted(results, key=lambda item: (item.suite, -item.rate, item.checkpoint)):
        filename = result.normalized_name
        lines.append(
            f"{result.rate:.1f}\t{result.suite}\t{result.step}\t{result.checkpoint}"
            f"\t{result.episodes}\t{result.successes}\t{filename}"
        )
    if apply:
        SUMMARY.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="modify files instead of dry-run")
    args = parser.parse_args()

    results = [result for path in sorted(INTERNAL.glob("EVAL-*.txt")) if (result := parse_internal_log(path))]
    renames: list[tuple[Path, Path]] = []
    for result in results:
        target = result.path.with_name(result.normalized_name)
        if target != result.path:
            renames.append((result.path, target))

    garbage = root_garbage_paths()

    print(f"mode={'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"complete_internal_logs={len(results)}")
    print(f"internal_logs_to_rename={len(renames)}")
    print(f"root_garbage_paths_to_delete={len(garbage)}")
    print(f"summary={SUMMARY}")

    print("\nRENAMES")
    for old, new in renames:
        print(f"{old} -> {new.name}")

    print("\nDELETE_GARBAGE")
    for path in garbage:
        print(path)

    if args.apply:
        for old, new in renames:
            if new.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {new}")
            old.rename(new)
        write_summary(results, apply=True)
        for path in garbage:
            remove_path(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
