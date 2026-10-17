import subprocess
import sys
from pathlib import Path

from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds.oxe.robocasa_registry import ROBOCASA_MERGED_PANDA_DATASETS
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS


def test_robocasa_merged_panda_100demos_is_registered():
    dataset_name = "robocasa_merged_24_tasks_100demos_v1"

    assert dataset_name in ROBOCASA_MERGED_PANDA_DATASETS
    assert dataset_name in OXE_DATASET_CONFIGS
    assert dataset_name in OXE_STANDARDIZATION_TRANSFORMS
    assert OXE_NAMED_MIXTURES["robocasa_merged_panda_100demos"] == [(dataset_name, 1.0)]

    config = OXE_DATASET_CONFIGS[dataset_name]
    assert config["image_obs_keys"] == {
        "primary": "right_image",
        "left_wrist": "left_image",
        "right_wrist": "wrist_image",
    }
    assert config["state_obs_keys"] == ["state"]
    assert config["action_dim"] == 12
    assert config["proprio_dim"] == 9


def test_robocasa_merged_panda_30demos_is_registered():
    dataset_name = "robocasa_merged_24_tasks_30demos_v3"

    assert dataset_name in ROBOCASA_MERGED_PANDA_DATASETS
    assert dataset_name in OXE_DATASET_CONFIGS
    assert dataset_name in OXE_STANDARDIZATION_TRANSFORMS
    assert OXE_NAMED_MIXTURES["robocasa_merged_panda_30demos"] == [(dataset_name, 1.0)]


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_constants_for_marker(marker):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from prismatic.vla.constants import ROBOT_PLATFORM, ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK; "
            "print(f'{ROBOT_PLATFORM} {ACTION_DIM} {PROPRIO_DIM} {NUM_ACTIONS_CHUNK}')",
            marker,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_generic_robocasa_marker_selects_panda_constants():
    assert _read_constants_for_marker("robocasa") == "ROBOCASA_PANDA 12 9 16"


def test_explicit_legacy_marker_selects_legacy_robocasa_constants():
    assert _read_constants_for_marker("gr1_unified") == "ROBOCASA_GR1 29 29 16"
