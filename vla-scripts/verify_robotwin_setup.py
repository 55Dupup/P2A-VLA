#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_CONDA_PREFIX = Path("/path/to/conda_envs/vla-adapter-robotwin")
DEFAULT_ROBOTWIN_PATH = Path("/path/to/RoboTwin")
DEFAULT_SOURCE_DIR = Path("/path/to/datasets/robotwin_lerobot")
DEFAULT_DATA_ROOT = Path("/path/to/datasets/robotwin_rlds")
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_ROOT / "robotwin_manifest.json"
DEFAULT_CONVERTER_PATH = Path(
    "/path/to/P2A-VLA/vla-scripts/convert_robotwin_lerobot_to_rlds.py"
)
REQUIRED_CAMERAS = ["head_camera", "left_camera", "right_camera"]
SIMULATOR_IMPORTS = (
    "import sapien, mplib, transforms3d, toppra, open3d, curobo; "
    "from curobo.curobolib import geom; assert geom.geom_cu is not None"
)


class SetupCheck(NamedTuple):
    name: str
    ok: bool
    message: str


def _exists_check(name: str, path: Path, description: str) -> SetupCheck:
    if path.exists():
        return SetupCheck(name, True, f"{description}: {path}")
    return SetupCheck(name, False, f"Missing {description}: {path}")


def check_lerobot_source_dir(source_dir: Path | None) -> SetupCheck:
    if source_dir is None:
        return SetupCheck("lerobot_source", True, "LeRobot source dir check skipped")
    if not source_dir.exists():
        return SetupCheck("lerobot_source", False, f"Missing LeRobot source dir: {source_dir}")
    if not (source_dir / "meta").exists():
        return SetupCheck("lerobot_source", False, f"LeRobot source dir must contain a meta directory: {source_dir}")
    return SetupCheck("lerobot_source", True, f"LeRobot source dir: {source_dir}")


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"__manifest_error__": str(exc)}


def check_manifest_contract(manifest_path: Path) -> SetupCheck:
    payload = _load_manifest(manifest_path)
    if payload is None:
        return SetupCheck("manifest_contract", False, f"Cannot inspect missing manifest: {manifest_path}")
    if "__manifest_error__" in payload:
        return SetupCheck("manifest_contract", False, f"Invalid manifest JSON: {payload['__manifest_error__']}")

    cameras = payload.get("camera_keys")
    action_dim = payload.get("action_dim")
    state_dim = payload.get("state_dim")
    normalization = payload.get("action_normalization")
    action_order = payload.get("action_order")
    robotwin_order = payload.get("robotwin_action_order")

    issues: list[str] = []
    if cameras != REQUIRED_CAMERAS:
        issues.append(f"expected three cameras {REQUIRED_CAMERAS}, got {cameras}")
    if action_dim != 14:
        issues.append(f"expected action_dim=14, got {action_dim}")
    if state_dim != 14:
        issues.append(f"expected state_dim=14, got {state_dim}")
    if normalization != "bounds":
        issues.append(f"expected action_normalization=bounds, got {normalization}")
    if action_order != "model_left6_right6_left_gripper_right_gripper":
        issues.append(f"unexpected model action_order={action_order}")
    if robotwin_order != "left6_left_gripper_right6_right_gripper":
        issues.append(f"unexpected robotwin_action_order={robotwin_order}")

    if issues:
        return SetupCheck("manifest_contract", False, "; ".join(issues))
    return SetupCheck("manifest_contract", True, "Manifest matches RoboTwin three-camera 14D bounds contract")


def check_simulator_dependencies(conda_prefix: Path, run_fn=None) -> SetupCheck:
    python_path = conda_prefix / "bin" / "python"
    if not python_path.is_file():
        return SetupCheck("simulator_dependencies", False, f"Missing RoboTwin Python: {python_path}")
    run_fn = subprocess.run if run_fn is None else run_fn
    completed = run_fn(
        [str(python_path), "-c", SIMULATOR_IMPORTS],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "import failed").strip().splitlines()[-1]
        return SetupCheck("simulator_dependencies", False, f"RoboTwin simulator import failed: {detail}")
    return SetupCheck("simulator_dependencies", True, f"RoboTwin simulator imports pass in {python_path}")


def check_simulator_assets(robotwin_path: Path) -> SetupCheck:
    assets_dir = robotwin_path / "assets"
    required_paths = (
        assets_dir / "embodiments" / "aloha-agilex" / "config.yml",
        assets_dir / "embodiments" / "aloha-agilex" / "curobo_left.yml",
        assets_dir / "embodiments" / "aloha-agilex" / "curobo_right.yml",
        assets_dir / "objects" / "001_bottle",
        assets_dir / "background_texture" / "seen",
        assets_dir / "background_texture" / "unseen",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return SetupCheck("simulator_assets", False, f"Missing RoboTwin simulator assets: {missing}")
    return SetupCheck("simulator_assets", True, f"RoboTwin clean/randomized assets: {assets_dir}")


def collect_setup_checks(
    *,
    robotwin_path: Path,
    data_root: Path,
    manifest_path: Path,
    checkpoint_path: Path | None,
    conda_prefix: Path,
    converter_path: Path,
    source_dir: Path | None = None,
    require_checkpoint: bool = False,
    require_simulator: bool = False,
    require_assets: bool = False,
) -> list[SetupCheck]:
    checks = [
        _exists_check("conda_env", conda_prefix, "RoboTwin conda env"),
        _exists_check("robotwin_eval_policy", robotwin_path / "script" / "eval_policy.py", "RoboTwin eval_policy.py"),
        _exists_check("rlds_data_root", data_root, "RoboTwin RLDS data root"),
        _exists_check("robotwin_manifest", manifest_path, "RoboTwin manifest"),
    ]

    if require_checkpoint:
        if checkpoint_path is None:
            checks.append(SetupCheck("vla_checkpoint", False, "Missing required --checkpoint-path"))
        else:
            checks.append(_exists_check("vla_checkpoint", checkpoint_path, "VLA-Adapter checkpoint"))

    checks.append(_exists_check("lerobot_converter", converter_path, "LeRobot to RLDS converter"))
    checks.append(check_lerobot_source_dir(source_dir))
    checks.append(check_manifest_contract(manifest_path))
    if require_simulator:
        checks.append(check_simulator_dependencies(conda_prefix))
    if require_assets:
        checks.append(check_simulator_assets(robotwin_path))
    return checks


def exit_code_for_checks(checks: list[SetupCheck]) -> int:
    return 0 if all(check.ok for check in checks) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify RoboTwin VLA-Adapter setup before training or eval")
    parser.add_argument("--robotwin-path", type=Path, default=DEFAULT_ROBOTWIN_PATH)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--conda-prefix", type=Path, default=DEFAULT_CONDA_PREFIX)
    parser.add_argument("--converter-path", type=Path, default=DEFAULT_CONVERTER_PATH)
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--require-simulator", action="store_true")
    parser.add_argument("--require-assets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = collect_setup_checks(
        robotwin_path=args.robotwin_path,
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        checkpoint_path=args.checkpoint_path,
        conda_prefix=args.conda_prefix,
        converter_path=args.converter_path,
        source_dir=args.source_dir,
        require_checkpoint=args.require_checkpoint,
        require_simulator=args.require_simulator,
        require_assets=args.require_assets,
    )
    for check in checks:
        status = "OK" if check.ok else "MISSING"
        print(f"[{status}] {check.name}: {check.message}")
    return exit_code_for_checks(checks)


if __name__ == "__main__":
    raise SystemExit(main())
