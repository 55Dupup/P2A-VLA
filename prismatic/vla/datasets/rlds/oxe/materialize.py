"""
materialize.py

Factory class for initializing Open-X Embodiment dataset kwargs and other parameters; provides and exports functions for
clear control flow.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prismatic.overwatch import initialize_overwatch
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, ActionEncoding
from prismatic.vla.datasets.rlds.oxe.robotwin_registry import ROBOTWIN_DATASET_STATISTICS_FILENAME
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def _dimension_mismatch_hint(dataset_name: str) -> str:
    dataset_name_lower = dataset_name.lower()
    if "metaworld" in dataset_name_lower:
        return "Make sure the command line includes `metaworld` so the METAWORLD constants are selected."
    if "robotwin" in dataset_name_lower:
        return "Make sure the command line includes `robotwin` so the ROBOTWIN constants are selected."
    if "gr1" in dataset_name_lower:
        return "Make sure the command line includes `gr1_unified` or `fourier_gr1` so legacy RoboCasa constants are selected."
    if "robocasa_cosmos" in dataset_name_lower or "robocasa-cosmos" in dataset_name_lower:
        return "Make sure the command line includes `robocasa_cosmos` so 7D Cosmos RoboCasa constants are selected."
    if "robocasa_merged" in dataset_name_lower or "merged_panda" in dataset_name_lower or "robocasa" in dataset_name_lower:
        return "Make sure the command line includes `robocasa` or `merged_panda` so Panda RoboCasa constants are selected."
    return "Make sure the selected robot constants match this dataset's action/proprio dimensions."


def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> Dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    expected_action_dim = dataset_kwargs.pop("action_dim", None)
    expected_proprio_dim = dataset_kwargs.pop("proprio_dim", None)

    if expected_action_dim is not None and ACTION_DIM != expected_action_dim:
        raise ValueError(
            f"`{dataset_name}` expects action_dim={expected_action_dim}, but current constants use ACTION_DIM={ACTION_DIM}. "
            f"{_dimension_mismatch_hint(dataset_name)}"
        )
    if load_proprio and expected_proprio_dim is not None and PROPRIO_DIM != expected_proprio_dim:
        raise ValueError(
            f"`{dataset_name}` expects proprio_dim={expected_proprio_dim}, but current constants use PROPRIO_DIM={PROPRIO_DIM}. "
            f"{_dimension_mismatch_hint(dataset_name)}"
        )

    # FIXME 修改这部分使得能够支持joint
    if dataset_kwargs["action_encoding"] not in [
        ActionEncoding.EEF_POS,
        ActionEncoding.EEF_R6,
        ActionEncoding.JOINT_POS_BIMANUAL,
        ActionEncoding.JOINT_POS,
        ActionEncoding.JOINT_POS_ROBOCASA,
    ]:
        raise ValueError(
            f"Cannot load `{dataset_name}`; only EEF_POS, EEF_R6, JOINT_POS_BIMANUAL, JOINT_POS, "
            "and JOINT_POS_ROBOCASA actions are supported!"
        )

    # import ipdb; ipdb.set_trace()
    # 默认使用相对动作来训练， 只有夹爪使用绝对动作
    # Normalize all action dimensions *except* the gripper
    if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
        dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:
        dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS_BIMANUAL:
        dataset_kwargs["absolute_action_mask"] = [True] * 14
        dataset_kwargs["action_normalization_mask"] = [True] * 14
    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS:
        dataset_kwargs["absolute_action_mask"] = [True] * ACTION_DIM
        dataset_kwargs["action_normalization_mask"] = [True] * ACTION_DIM
    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS_ROBOCASA:
        dataset_kwargs["absolute_action_mask"] = [True] * expected_action_dim
        dataset_kwargs["action_normalization_mask"] = [True] * expected_action_dim
    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type

    # Adjust Loaded Camera Views
    requested_camera_views = set(load_camera_views)
    available_camera_views = set(dataset_kwargs["image_obs_keys"])
    if len(missing_keys := (requested_camera_views - available_camera_views)) > 0:
        raise ValueError(
            f"Cannot load `{dataset_name}`; requested camera views `{sorted(missing_keys)}` "
            f"are not defined. Available views: `{sorted(available_camera_views)}`."
        )
    if "robotwin" in dataset_name.lower():
        missing_required_views = available_camera_views - requested_camera_views
        if missing_required_views:
            raise ValueError(
                f"Cannot load `{dataset_name}`; missing required RoboTwin camera views "
                f"`{sorted(missing_required_views)}`. Requested views: `{list(load_camera_views)}`."
            )

    # Filter
    dataset_kwargs["image_obs_keys"] = {
        k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
    }
    dataset_kwargs["depth_obs_keys"] = {
        k: v for k, v in dataset_kwargs["depth_obs_keys"].items() if k in load_camera_views
    }

    # Eliminate Unnecessary Keys
    dataset_kwargs.pop("state_encoding")
    dataset_kwargs.pop("action_encoding")
    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if not load_proprio:
        dataset_kwargs.pop("state_obs_keys")

    # Load Language
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    # Specify Standardization Transform
    dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    if "robotwin" in dataset_name.lower():
        statistics_path = Path(data_root_dir) / ROBOTWIN_DATASET_STATISTICS_FILENAME
        if statistics_path.is_file():
            dataset_kwargs["dataset_statistics"] = str(statistics_path)

    # FIXME DUBUG :{'image_obs_keys': {'primary': 'image', 'wrist': 'wrist_image'}, 'state_obs_keys': ['state'], 'absolute_action_mask': [False, False, False, False, False, False, True], 'action_normalization_mask': [True, True, True, True, True, True, False], 'action_proprio_normalization_type': <NormalizationType.BOUNDS_Q99: 'bounds_q99'>, 'language_key': 'language_instruction', 'standardize_fn': <function real_world_dataset_transform at 0x72bc72f27b50>}
    return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}


def get_oxe_dataset_kwargs_and_weights(
    data_root_dir: Path,
    mixture_spec: List[Tuple[str, float]],
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
    (per-dataset configs) and weights can be passed directly to `make_interleaved_dataset`.

    :param data_root_dir: Base directory containing RLDS/TFDS-formatted datasets (from Open-X)
    :param mixture_spec: List of (dataset_name, sampling_weight) from `oxe.mixtures.OXE_NAMED_MIXTURES`
    :param load_camera_views: Camera views to load; see `oxe.dataset_configs.py` for available views.
    :param load_depth: Load depth information in addition to camera RGB.
    :param load_proprio: Load proprioceptive state.
    :param load_language: Load language instructions.
    :param action_proprio_normalization_type: Normalization scheme to use for proprioceptive actions.

    return: Tuple of (per_dataset_kwargs, sampling_weights)
    """
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            overwatch.warning(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
            continue

        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))

    # Assemble Dataset Config (kwargs) and Weights
    per_dataset_kwargs, sampling_weights = [], []
    for d_name, d_weight in filtered_mixture_spec:
        try:
            per_dataset_kwargs.append(
                make_oxe_dataset_kwargs(
                    d_name,
                    data_root_dir,
                    load_camera_views,
                    load_depth,
                    load_proprio,
                    load_language,
                    action_proprio_normalization_type,
                )
            )
            sampling_weights.append(d_weight)

        except ValueError as e:
            overwatch.warning(f"Skipping `{d_name}` due to Error: {e}")

    return per_dataset_kwargs, sampling_weights
