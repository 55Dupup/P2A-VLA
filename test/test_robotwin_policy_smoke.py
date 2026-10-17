import numpy as np
import pytest

from vla_scripts_import import load_script_module


smoke = load_script_module(
    "smoke_test_robotwin_policy",
    "vla-scripts/smoke_test_robotwin_policy.py",
)


def test_validate_robotwin_action_accepts_finite_14d_vector():
    action = smoke.validate_robotwin_action(np.arange(14, dtype=np.float32))

    assert action.shape == (14,)
    assert action.dtype == np.float32


def test_validate_robotwin_action_rejects_wrong_shape_and_nonfinite_values():
    with pytest.raises(ValueError, match="14D"):
        smoke.validate_robotwin_action(np.zeros(13))
    action = np.zeros(14)
    action[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        smoke.validate_robotwin_action(action)
