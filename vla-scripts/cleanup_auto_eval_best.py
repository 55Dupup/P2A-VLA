#!/usr/bin/env python3
"""Keep only the best evaluated checkpoint per series and remove stale logs."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/path/to/P2A-VLA")
OUTPUTS = ROOT / "outputs"
AUTO_EVAL = ROOT / "experiments/libero_eval_logs/auto_eval"
INTERNAL = AUTO_EVAL / "internal"
STATUS = AUTO_EVAL / "status"
STAGING = AUTO_EVAL / "staging"

CKPT_RE = re.compile(r"^(?P<series>.+)--(?P<step>\d+)_chkpt$")
FINAL_RE = re.compile(
    r"Overall success rate:\s*(?P<fraction>[0-9.]+)(?:\s*\((?P<percent>[0-9.]+)%\)|\s*%)"
)


@dataclass(frozen=True)
class EvalResult:
    checkpoint: str
    series: str
    step: int
    rate: float
    log: Path


def checkpoint_from_eval_log(path: Path) -> str | None:
    name = path.name
    if not name.startswith("EVAL-") or not name.endswith(".txt"):
        return None
    stem = name[:-4]
    marker = "-auto-"
    if marker not in stem:
        return None
    rest = stem.split(marker, 1)[1]
    match = re.match(r"(?P<ckpt>.+--\d+_chkpt)-\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}", rest)
    if not match:
        return None
    return match.group("ckpt")


def parse_final_rate(path: Path) -> float | None:
    rate = None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    for match in FINAL_RE.finditer(text):
        if match.group("percent") is not None:
            rate = float(match.group("percent"))
        else:
            value = float(match.group("fraction"))
            rate = value * 100.0 if value <= 1.0 else value
    return rate


def load_eval_results(existing_outputs: set[str]) -> dict[str, EvalResult]:
    best_by_checkpoint: dict[str, EvalResult] = {}
    for path in sorted(INTERNAL.glob("EVAL-*.txt")):
        checkpoint = checkpoint_from_eval_log(path)
        if checkpoint is None or checkpoint not in existing_outputs:
            continue
        ckpt_match = CKPT_RE.match(checkpoint)
        if not ckpt_match:
            continue
        rate = parse_final_rate(path)
        if rate is None:
            continue
        result = EvalResult(
            checkpoint=checkpoint,
            series=ckpt_match.group("series"),
            step=int(ckpt_match.group("step")),
            rate=rate,
            log=path,
        )
        old = best_by_checkpoint.get(checkpoint)
        if old is None or (result.rate, result.log.stat().st_mtime) >= (old.rate, old.log.stat().st_mtime):
            best_by_checkpoint[checkpoint] = result
    return best_by_checkpoint


def grouped(results: dict[str, EvalResult]) -> dict[str, list[EvalResult]]:
    groups: dict[str, list[EvalResult]] = {}
    for result in results.values():
        groups.setdefault(result.series, []).append(result)
    for values in groups.values():
        values.sort(key=lambda item: (item.rate, item.step), reverse=True)
    return groups


def running_markers() -> set[str]:
    markers: set[str] = set()
    for path in AUTO_EVAL.glob("*.running.log"):
        markers.add(path.name.removesuffix(".running.log"))
    for path in AUTO_EVAL.glob("*.manual-running.log"):
        markers.add(path.name.removesuffix(".manual-running.log"))
    if STAGING.exists():
        for path in STAGING.iterdir():
            if path.is_dir():
                match = re.match(r"(?P<ckpt>.+--\d+_chkpt)\.", path.name)
                if match:
                    markers.add(match.group("ckpt"))
    return markers


def collect_logs_for_checkpoint(checkpoint: str) -> list[Path]:
    paths: list[Path] = []
    if INTERNAL.exists():
        paths.extend(path for path in INTERNAL.glob(f"*{checkpoint}*") if path.is_file())
    if STATUS.exists():
        paths.extend(path for path in STATUS.glob(f"*{checkpoint}*") if path.is_file())
    paths.extend(path for path in AUTO_EVAL.glob(f"*{checkpoint}*") if path.is_file())
    if STAGING.exists():
        paths.extend(path for path in STAGING.glob(f"{checkpoint}.*") if path.exists())
    return sorted(set(paths))


def collect_garbage_logs(existing_outputs: set[str], retained: set[str], running: set[str]) -> list[Path]:
    garbage: set[Path] = set()

    for path in AUTO_EVAL.glob("__*.log"):
        garbage.add(path)

    for path in AUTO_EVAL.glob("*.manual-running.log"):
        ckpt = path.name.removesuffix(".manual-running.log")
        if ckpt not in running:
            garbage.add(path)

    if INTERNAL.exists():
        for path in INTERNAL.glob("EVAL-*.txt"):
            checkpoint = checkpoint_from_eval_log(path)
            if checkpoint is None:
                continue
            if checkpoint not in existing_outputs or checkpoint not in retained:
                garbage.add(path)

    if STATUS.exists():
        for path in STATUS.glob("*.state"):
            checkpoint = path.name.removesuffix(".state")
            if checkpoint not in existing_outputs or checkpoint not in retained:
                garbage.add(path)

    if STAGING.exists():
        for path in STAGING.iterdir():
            match = re.match(r"(?P<ckpt>.+--\d+_chkpt)\.", path.name)
            checkpoint = match.group("ckpt") if match else None
            if checkpoint is None or checkpoint not in running:
                garbage.add(path)

    return sorted(garbage)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="delete files instead of dry-run")
    args = parser.parse_args()

    existing = {path.name for path in OUTPUTS.iterdir() if path.is_dir() and CKPT_RE.match(path.name)}
    results = load_eval_results(existing)
    groups = grouped(results)
    running = running_markers()

    keep: set[str] = set()
    delete_outputs: list[Path] = []
    skipped_running: set[str] = set()

    for series, values in sorted(groups.items()):
        if not values:
            continue
        winner = values[0]
        keep.add(winner.checkpoint)
        for result in values[1:]:
            if result.checkpoint in running:
                skipped_running.add(result.checkpoint)
                keep.add(result.checkpoint)
                continue
            delete_outputs.append(OUTPUTS / result.checkpoint)

    retained = keep | skipped_running
    delete_logs: set[Path] = set()
    for output in delete_outputs:
        delete_logs.update(collect_logs_for_checkpoint(output.name))
    delete_logs.update(collect_garbage_logs(existing, retained, running))

    delete_outputs = sorted(set(delete_outputs))
    delete_logs_sorted = sorted(path for path in delete_logs if path.exists())

    print(f"mode={'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"evaluated_existing_checkpoints={len(results)}")
    print(f"series_with_results={len(groups)}")
    print(f"output_dirs_to_delete={len(delete_outputs)}")
    print(f"log_paths_to_delete={len(delete_logs_sorted)}")
    if skipped_running:
        print("skipped_running=" + ",".join(sorted(skipped_running)))

    print("\nKEEP_BY_SERIES")
    for series, values in sorted(groups.items()):
        winner = values[0]
        print(f"{series}\tKEEP\t{winner.checkpoint}\t{winner.rate:.2f}\tstep={winner.step}")
        for result in values[1:]:
            action = "SKIP_RUNNING" if result.checkpoint in skipped_running else "DELETE"
            print(f"{series}\t{action}\t{result.checkpoint}\t{result.rate:.2f}\tstep={result.step}")

    print("\nDELETE_OUTPUTS")
    for path in delete_outputs:
        print(path)

    print("\nDELETE_LOGS")
    for path in delete_logs_sorted:
        print(path)

    if args.apply:
        for path in delete_outputs:
            remove_path(path)
        for path in delete_logs_sorted:
            remove_path(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
