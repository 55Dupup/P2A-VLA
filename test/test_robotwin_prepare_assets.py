import hashlib
import subprocess
from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

import pytest

from vla_scripts_import import load_script_module


prepare = load_script_module("prepare_robotwin_assets", "vla-scripts/prepare_robotwin_assets.py")


def test_robotwin_asset_specs_match_official_release_metadata():
    assert prepare.ASSET_SPECS["embodiments"].size == 219_859_313
    assert prepare.ASSET_SPECS["objects"].size == 3_737_778_549
    assert prepare.ASSET_SPECS["background_texture"].size == 10_970_687_027
    assert prepare.ASSET_SPECS["embodiments"].sha256 == (
        "6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6"
    )


def test_validate_asset_archive_requires_exact_size_and_sha256(tmp_path):
    payload = b"official-asset"
    archive = tmp_path / "asset.zip"
    archive.write_bytes(payload)
    spec = prepare.AssetSpec(
        name="asset",
        filename="asset.zip",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        extracted_dir="asset",
    )

    prepare.validate_asset_archive(archive, spec)
    archive.write_bytes(payload + b"corrupt")
    with pytest.raises(ValueError, match="size"):
        prepare.validate_asset_archive(archive, spec)


def test_extract_asset_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "unsafe")

    with pytest.raises(ValueError, match="Unsafe path"):
        prepare.extract_asset_archive(archive, tmp_path / "assets")

    assert not (tmp_path / "escape.txt").exists()


def test_extract_asset_archive_writes_valid_tree(tmp_path):
    archive = tmp_path / "objects.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("objects/001_bottle/model.txt", "model")

    prepare.extract_asset_archive(archive, tmp_path / "assets")

    assert (tmp_path / "assets" / "objects" / "001_bottle" / "model.txt").read_text() == "model"


def test_curl_asset_download_retries_transient_disconnects(tmp_path):
    return_codes = iter([18, 18, 0])
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        return Namespace(returncode=next(return_codes))

    prepare.run_curl_with_retries(
        "https://huggingface.co/asset.zip",
        tmp_path / "asset.zip",
        max_attempts=3,
        retry_delay_s=0,
        run_fn=fake_run,
        sleep_fn=lambda _: None,
    )

    assert len(calls) == 3
    assert "--continue-at" in calls[0][0]


def test_curl_asset_download_stops_on_http_error(tmp_path):
    def fake_run(command, check):
        return Namespace(returncode=22)

    with pytest.raises(subprocess.CalledProcessError):
        prepare.run_curl_with_retries(
            "https://huggingface.co/missing.zip",
            tmp_path / "missing.zip",
            run_fn=fake_run,
            sleep_fn=lambda _: None,
        )


def test_update_embodiment_configs_runs_official_script_and_validates_outputs(tmp_path):
    robotwin_path = tmp_path / "RoboTwin"
    update_script = robotwin_path / "script" / "update_embodiment_config_path.py"
    update_script.parent.mkdir(parents=True)
    update_script.write_text("# official updater\n", encoding="ascii")
    embodiment_dir = robotwin_path / "assets" / "embodiments" / "aloha-agilex"
    embodiment_dir.mkdir(parents=True)
    calls = []

    def fake_run(command, cwd, check):
        calls.append((command, cwd, check))
        for filename in ("curobo_left.yml", "curobo_right.yml"):
            (embodiment_dir / filename).write_text(str(robotwin_path), encoding="utf-8")

    prepare.update_embodiment_configs(robotwin_path, run_fn=fake_run)

    assert calls == [([prepare.sys.executable, str(update_script)], robotwin_path, True)]
