import argparse
import random
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
REPO_TYPE = "dataset"
DEFAULT_MAX_WORKERS = 16
DEFAULT_MAX_RETRIES = 20

FOLDERS = (
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/path/to/datasets/robocasa")
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    return parser.parse_args()


def download_with_retry(filename: str, local_dir: str, max_retries: int) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            return True
        except Exception as exc:
            wait_time = random.uniform(0.5, 2.0) * attempt
            print(f"[{attempt}/{max_retries}] failed: {filename} ({exc}); retrying in {wait_time:.1f}s", flush=True)
            time.sleep(wait_time)
    return False


def _list_target_files(api: HfApi) -> list[str]:
    target_files = []
    for folder in FOLDERS:
        print(f"Listing {folder} ...", flush=True)
        items = api.list_repo_tree(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo=folder,
            recursive=True,
        )
        folder_files = [item.path for item in items if item.__class__.__name__ == "RepoFile"]
        print(f"Found {len(folder_files)} files in {folder}", flush=True)
        target_files.extend(folder_files)
    return target_files


def main():
    args = parse_args()
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=args.endpoint) if args.endpoint else HfApi()

    print(f"Listing files from {REPO_ID} ...", flush=True)
    if args.endpoint:
        print(f"Using endpoint: {args.endpoint}", flush=True)
    target_files = _list_target_files(api)
    print(f"Found {len(target_files)} files across {len(FOLDERS)} RoboCasa task folders.", flush=True)

    failed_files = []
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(download_with_retry, filename, args.local_dir, args.max_retries): filename
            for filename in target_files
        }
        for future in as_completed(futures):
            filename = futures[future]
            if future.result():
                completed += 1
                if completed % 50 == 0 or completed == len(target_files):
                    elapsed = time.time() - start_time
                    print(f"Downloaded {completed}/{len(target_files)} files in {elapsed:.1f}s", flush=True)
            else:
                failed_files.append(filename)
                print(f"FAILED: {filename}", flush=True)

    if failed_files:
        print("Download finished with failures:", flush=True)
        for filename in failed_files:
            print(filename, flush=True)
        raise SystemExit(1)

    print(f"Download completed successfully into {args.local_dir}", flush=True)


if __name__ == "__main__":
    main()
