import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_NAME = "prismatic.vla.constants"
PACKAGE_NAME = "prismatic.vla"
MATERIALIZE_MODULE_NAME = "prismatic.vla.datasets.rlds.oxe.materialize"
OXE_PACKAGE_NAME = "prismatic.vla.datasets.rlds.oxe"


def snapshot_constants(constants):
    return SimpleNamespace(
        ROBOT_PLATFORM=constants.ROBOT_PLATFORM,
        ACTION_DIM=constants.ACTION_DIM,
        PROPRIO_DIM=constants.PROPRIO_DIM,
        NUM_ACTIONS_CHUNK=constants.NUM_ACTIONS_CHUNK,
        ACTION_PROPRIO_NORMALIZATION_TYPE=constants.ACTION_PROPRIO_NORMALIZATION_TYPE,
    )


def reload_constants_with_argv(argv):
    original_argv = sys.argv[:]
    module_was_loaded = MODULE_NAME in sys.modules
    previous_module = sys.modules.get(MODULE_NAME)
    package = sys.modules.get(PACKAGE_NAME)
    package_had_constants = package is not None and hasattr(package, "constants")
    previous_package_constants = getattr(package, "constants", None)

    sys.modules.pop(MODULE_NAME, None)
    if package is not None and hasattr(package, "constants"):
        delattr(package, "constants")

    try:
        sys.argv = argv
        constants = importlib.import_module(MODULE_NAME)
        return snapshot_constants(constants)
    finally:
        sys.argv = original_argv
        if module_was_loaded:
            sys.modules[MODULE_NAME] = previous_module
        else:
            sys.modules.pop(MODULE_NAME, None)

        package = sys.modules.get(PACKAGE_NAME)
        if package is not None:
            if package_had_constants:
                setattr(package, "constants", previous_package_constants)
            elif hasattr(package, "constants"):
                delattr(package, "constants")


@contextmanager
def robotwin_materialize_module():
    original_argv = sys.argv[:]
    module_specs = (
        (MATERIALIZE_MODULE_NAME, OXE_PACKAGE_NAME, "materialize"),
        (MODULE_NAME, PACKAGE_NAME, "constants"),
    )
    saved_modules = {}
    saved_package_attrs = {}

    for module_name, package_name, attribute_name in module_specs:
        saved_modules[module_name] = (module_name in sys.modules, sys.modules.get(module_name))
        package = sys.modules.get(package_name)
        saved_package_attrs[module_name] = (
            package_name,
            attribute_name,
            package is not None and hasattr(package, attribute_name),
            getattr(package, attribute_name, None),
        )

        sys.modules.pop(module_name, None)
        if package is not None and hasattr(package, attribute_name):
            delattr(package, attribute_name)

    try:
        sys.argv = ["vla-scripts/finetune_robotwin.sh"]
        yield importlib.import_module(MATERIALIZE_MODULE_NAME)
    finally:
        sys.argv = original_argv

        for module_name, (module_was_loaded, previous_module) in saved_modules.items():
            if module_was_loaded:
                sys.modules[module_name] = previous_module
            else:
                sys.modules.pop(module_name, None)

        for package_name, attribute_name, package_had_attribute, previous_attribute in saved_package_attrs.values():
            package = sys.modules.get(package_name)
            if package is None:
                continue
            if package_had_attribute:
                setattr(package, attribute_name, previous_attribute)
            elif hasattr(package, attribute_name):
                delattr(package, attribute_name)


def test_robotwin_constants_selected_by_script_name():
    constants = reload_constants_with_argv(["vla-scripts/finetune_robotwin.sh"])
    assert constants.ROBOT_PLATFORM == "ROBOTWIN"
    assert constants.ACTION_DIM == 14
    assert constants.PROPRIO_DIM == 14
    assert constants.NUM_ACTIONS_CHUNK == 16
    assert constants.ACTION_PROPRIO_NORMALIZATION_TYPE.value == "bounds"


def test_reload_constants_with_argv_does_not_introduce_constants_module():
    module_name = "prismatic.vla.constants"
    package_name = "prismatic.vla"
    original_argv = sys.argv[:]
    module_was_loaded = module_name in sys.modules
    previous_module = sys.modules.pop(module_name, None)
    package = sys.modules.get(package_name)
    package_had_constants = package is not None and hasattr(package, "constants")
    previous_package_constants = getattr(package, "constants", None)
    if package_had_constants:
        delattr(package, "constants")

    try:
        sys.argv = ["pytest", "test/test_robotwin_rlds_registration.py"]
        robotwin_constants = reload_constants_with_argv(["vla-scripts/finetune_robotwin.sh"])
        assert robotwin_constants.ROBOT_PLATFORM == "ROBOTWIN"
        assert module_name not in sys.modules

        package = sys.modules.get(package_name)
        if package is not None:
            assert not hasattr(package, "constants")
    finally:
        sys.argv = original_argv
        if module_was_loaded:
            sys.modules[module_name] = previous_module
        else:
            sys.modules.pop(module_name, None)

        package = sys.modules.get(package_name)
        if package is not None:
            if package_had_constants:
                setattr(package, "constants", previous_package_constants)
            elif hasattr(package, "constants"):
                delattr(package, "constants")


def test_robotwin_dataset_config_has_three_cameras_and_14d_shapes():
    from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS

    cfg = OXE_DATASET_CONFIGS["robotwin_lerobot_unified"]
    assert cfg["image_obs_keys"] == {
        "primary": "cam_high",
        "left_wrist": "cam_left_wrist",
        "right_wrist": "cam_right_wrist",
    }
    assert cfg["state_obs_keys"] == ["state"]
    assert cfg["action_dim"] == 14
    assert cfg["proprio_dim"] == 14


def test_robotwin_mixture_aliases_registered():
    from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES

    assert OXE_NAMED_MIXTURES["robotwin_lerobot_unified"] == [("robotwin_lerobot_unified", 1.0)]
    assert OXE_NAMED_MIXTURES["robotwin_mixed_clean_randomized"] == [("robotwin_mixed_clean_randomized", 1.0)]


def test_robotwin_materialize_kwargs_require_three_requested_views():
    with robotwin_materialize_module() as materialize:
        cfg = materialize.make_oxe_dataset_kwargs(
            "robotwin_lerobot_unified",
            Path("/path/to/datasets/robotwin_rlds"),
            load_camera_views=("primary", "left_wrist", "right_wrist"),
            load_proprio=False,
        )
    assert cfg["name"] == "robotwin_lerobot_unified"
    assert cfg["image_obs_keys"]["primary"] == "cam_high"
    assert cfg["image_obs_keys"]["left_wrist"] == "cam_left_wrist"
    assert cfg["image_obs_keys"]["right_wrist"] == "cam_right_wrist"
    assert "state_obs_keys" not in cfg or cfg["state_obs_keys"] == []
    assert cfg["absolute_action_mask"] == [True] * 14
    assert cfg["action_normalization_mask"] == [True] * 14


def test_robotwin_materialize_rejects_missing_right_wrist_view():
    with robotwin_materialize_module() as materialize:
        with pytest.raises(ValueError, match="right_wrist"):
            materialize.make_oxe_dataset_kwargs(
                "robotwin_lerobot_unified",
                Path("/path/to/datasets/robotwin_rlds"),
                load_camera_views=("primary", "left_wrist"),
                load_proprio=False,
            )


def test_robotwin_transform_normalizes_aliases_shapes_and_language():
    import tensorflow as tf

    from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS

    transform = OXE_STANDARDIZATION_TRANSFORMS["robotwin_adjust_bottle"]
    trajectory = {
        "observation": {
            "head_camera": tf.zeros([2, 4, 4, 3], dtype=tf.uint8),
            "left_camera": tf.ones([2, 4, 4, 3], dtype=tf.uint8),
            "video": {"cam_right_wrist": tf.fill([2, 4, 4, 3], 2)},
            "state": tf.tile(tf.range(14, dtype=tf.float64)[None], [2, 1]),
        },
        "action": tf.tile(tf.range(14, dtype=tf.float64)[None], [2, 1]),
        "task": {"language_instruction": tf.constant(["adjust", "adjust"])},
    }

    transformed = transform(trajectory)

    assert transformed["observation"]["cam_high"].shape.as_list() == [2, 4, 4, 3]
    assert transformed["observation"]["cam_left_wrist"].shape.as_list() == [2, 4, 4, 3]
    assert transformed["observation"]["cam_right_wrist"].shape.as_list() == [2, 4, 4, 3]
    assert transformed["action"].dtype == tf.float32
    assert transformed["action"].shape.as_list() == [2, 14]
    assert transformed["action"][0].numpy().tolist() == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    assert transformed["observation"]["state"].dtype == tf.float32
    assert transformed["observation"]["state"].shape.as_list() == [2, 14]
    assert transformed["observation"]["state"][0].numpy().tolist() == [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13
    ]
    assert transformed["language_instruction"].dtype == tf.string
