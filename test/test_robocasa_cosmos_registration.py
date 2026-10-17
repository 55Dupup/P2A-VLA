import subprocess
import sys
from pathlib import Path

from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds.oxe.robocasa_registry import (
    ROBOCASA_COSMOS_ACTION_DIM,
    ROBOCASA_COSMOS_DATASETS,
    ROBOCASA_COSMOS_IMAGE_KEYS,
    ROBOCASA_COSMOS_MIXTURE_ALIASES,
    ROBOCASA_COSMOS_PROPRIO_DIM,
)
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS


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


def test_robocasa_cosmos_dataset_is_registered_separately_from_merged_panda():
    dataset_name = "robocasa_cosmos_success_only"

    assert ROBOCASA_COSMOS_DATASETS == (dataset_name,)
    assert ROBOCASA_COSMOS_MIXTURE_ALIASES["robocasa_cosmos"] == dataset_name
    assert dataset_name in OXE_DATASET_CONFIGS
    assert dataset_name in OXE_STANDARDIZATION_TRANSFORMS
    assert OXE_NAMED_MIXTURES["robocasa_cosmos"] == [(dataset_name, 1.0)]
    assert OXE_NAMED_MIXTURES["robocasa_cosmos_success_only"] == [(dataset_name, 1.0)]

    config = OXE_DATASET_CONFIGS[dataset_name]
    assert config["image_obs_keys"] == {
        "primary": "left_image",
        "left_wrist": "right_image",
        "right_wrist": "wrist_image",
    }
    assert config["image_obs_keys"] == ROBOCASA_COSMOS_IMAGE_KEYS
    assert config["state_obs_keys"] == ["state"]
    assert config["action_dim"] == ROBOCASA_COSMOS_ACTION_DIM == 7
    assert config["proprio_dim"] == ROBOCASA_COSMOS_PROPRIO_DIM == 9


def test_robocasa_cosmos_marker_selects_7d_cosmos_constants_before_generic_robocasa():
    assert _read_constants_for_marker("robocasa_cosmos") == "ROBOCASA_COSMOS 7 9 32"
    assert _read_constants_for_marker("robocasa-cosmos") == "ROBOCASA_COSMOS 7 9 32"
    assert _read_constants_for_marker("robocasa") == "ROBOCASA_PANDA 12 9 16"
