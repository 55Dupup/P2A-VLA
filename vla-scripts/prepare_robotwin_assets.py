#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple
from zipfile import ZipFile, ZipInfo


DEFAULT_ROBOTWIN_PATH = Path("/path/to/RoboTwin")
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HF_REPO_ID = "TianxingChen/RoboTwin2.0"
DEFAULT_HTTP_PROXY = "http://localhost:7895"
DEFAULT_HTTPS_PROXY = "http://localhost:7895"
DEFAULT_MAX_ATTEMPTS = 1000
DEFAULT_RETRY_DELAY_S = 2.0


class AssetSpec(NamedTuple):
    name: str
    filename: str
    size: int
    sha256: str
    extracted_dir: str


ASSET_SPECS = {
    "embodiments": AssetSpec(
        name="embodiments",
        filename="embodiments.zip",
        size=219_859_313,
        sha256="6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6",
        extracted_dir="embodiments",
    ),
    "objects": AssetSpec(
        name="objects",
        filename="objects.zip",
        size=3_737_778_549,
        sha256="6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27",
        extracted_dir="objects",
    ),
    "background_texture": AssetSpec(
        name="background_texture",
        filename="background_texture.zip",
        size=10_970_687_027,
        sha256="54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070",
        extracted_dir="background_texture",
    ),
}


def configure_download_env(*, http_proxy: str | None, https_proxy: str | None) -> None:
    if http_proxy:
        os.environ["http_proxy"] = http_proxy
    if https_proxy:
        os.environ["https_proxy"] = https_proxy


def build_asset_url(spec: AssetSpec, *, endpoint: str, repo_id: str) -> str:
    return f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main/{spec.filename}?download=true"


def build_curl_resume_command(url: str, output_path: Path) -> list[str]:
    return [
        "curl",
        "--fail",
        "--location",
        "--http1.1",
        "--silent",
        "--show-error",
        "--continue-at",
        "-",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "--output",
        str(output_path),
        url,
    ]


def run_curl_with_retries(
    url: str,
    output_path: Path,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
    run_fn=None,
    sleep_fn=None,
) -> None:
    if max_attempts < 1:
        raise ValueError("asset download max attempts must be at least 1")
    run_fn = subprocess.run if run_fn is None else run_fn
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    command = build_curl_resume_command(url, output_path)
    for attempt in range(1, max_attempts + 1):
        completed = run_fn(command, check=False)
        if completed.returncode == 0:
            return
        if completed.returncode == 22 or attempt == max_attempts:
            raise subprocess.CalledProcessError(completed.returncode, command)
        print(f"RoboTwin asset download failed for {output_path.name} ({attempt}/{max_attempts}); resuming")
        sleep_fn(retry_delay_s)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset_file:
        for chunk in iter(lambda: asset_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_asset_archive(path: Path, spec: AssetSpec) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing RoboTwin asset archive: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        raise ValueError(f"RoboTwin asset {spec.filename} has size {actual_size}, expected {spec.size}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != spec.sha256:
        raise ValueError(
            f"RoboTwin asset {spec.filename} SHA-256 mismatch: {actual_sha256} != {spec.sha256}"
        )


def _validated_zip_members(archive: ZipFile, destination: Path) -> list[ZipInfo]:
    destination = destination.resolve()
    members = archive.infolist()
    for member in members:
        member_path = Path(member.filename)
        if member_path.is_absolute() or not (destination / member_path).resolve().is_relative_to(destination):
            raise ValueError(f"Unsafe path in RoboTwin asset archive: {member.filename}")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"RoboTwin asset archive contains a symbolic link: {member.filename}")
    return members


def extract_asset_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        archive.extractall(destination, members=_validated_zip_members(archive, destination))


def update_embodiment_configs(robotwin_path: Path, run_fn=None) -> None:
    update_script = robotwin_path / "script" / "update_embodiment_config_path.py"
    embodiments_dir = robotwin_path / "assets" / "embodiments"
    if not update_script.is_file():
        raise FileNotFoundError(f"Missing official RoboTwin embodiment config updater: {update_script}")
    if not embodiments_dir.is_dir():
        raise FileNotFoundError(f"Missing extracted RoboTwin embodiments: {embodiments_dir}")

    run_fn = subprocess.run if run_fn is None else run_fn
    run_fn([sys.executable, str(update_script)], cwd=robotwin_path, check=True)

    generated_configs = (
        embodiments_dir / "aloha-agilex" / "curobo_left.yml",
        embodiments_dir / "aloha-agilex" / "curobo_right.yml",
    )
    for config_path in generated_configs:
        if not config_path.is_file():
            raise FileNotFoundError(f"RoboTwin did not generate CuRobo config: {config_path}")
        config_text = config_path.read_text(encoding="utf-8")
        if "${ASSETS_PATH}" in config_text or "$ASSETS_PATH" in config_text:
            raise ValueError(f"Unresolved asset path in generated CuRobo config: {config_path}")
    print(f"Updated RoboTwin embodiment configs for: {robotwin_path}")


def prepare_asset(
    spec: AssetSpec,
    *,
    assets_dir: Path,
    endpoint: str,
    repo_id: str,
    download_only: bool,
    skip_download: bool,
) -> None:
    archive_path = assets_dir / spec.filename
    extracted_path = assets_dir / spec.extracted_dir
    marker_path = assets_dir / f".vla_adapter_{spec.name}_extract_complete"
    if marker_path.is_file() and extracted_path.is_dir() and not download_only:
        print(f"RoboTwin asset already extracted: {spec.name}")
        return

    if not skip_download and (not archive_path.exists() or archive_path.stat().st_size < spec.size):
        if archive_path.exists() and archive_path.stat().st_size > spec.size:
            raise ValueError(f"RoboTwin asset {spec.filename} is larger than its official release size")
        run_curl_with_retries(
            build_asset_url(spec, endpoint=endpoint, repo_id=repo_id),
            archive_path,
        )
    validate_asset_archive(archive_path, spec)
    print(f"Validated RoboTwin asset archive: {spec.filename}")
    if download_only:
        return

    extract_asset_archive(archive_path, assets_dir)
    if not extracted_path.is_dir():
        raise ValueError(f"RoboTwin asset {spec.filename} did not create {extracted_path}")
    marker_path.write_text(f"{spec.sha256}\n", encoding="ascii")
    print(f"Extracted RoboTwin asset: {spec.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify official RoboTwin simulator assets")
    parser.add_argument("--robotwin-path", type=Path, default=DEFAULT_ROBOTWIN_PATH)
    parser.add_argument("--asset", action="append", choices=tuple(ASSET_SPECS), default=None)
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--http-proxy", default=DEFAULT_HTTP_PROXY)
    parser.add_argument("--https-proxy", default=DEFAULT_HTTPS_PROXY)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets_dir = args.robotwin_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    configure_download_env(http_proxy=args.http_proxy, https_proxy=args.https_proxy)
    asset_names = args.asset or list(ASSET_SPECS)
    for asset_name in asset_names:
        prepare_asset(
            ASSET_SPECS[asset_name],
            assets_dir=assets_dir,
            endpoint=args.hf_endpoint,
            repo_id=args.hf_repo_id,
            download_only=args.download_only,
            skip_download=args.skip_download,
        )
    if not args.download_only and (assets_dir / "embodiments").is_dir():
        update_embodiment_configs(args.robotwin_path)


if __name__ == "__main__":
    main()
