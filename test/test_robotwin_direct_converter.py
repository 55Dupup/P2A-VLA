import json

import numpy as np
import pandas as pd
import pytest

from vla_scripts_import import load_script_module


converter = load_script_module(
    "convert_robotwin_lerobot_to_rlds",
    "vla-scripts/convert_robotwin_lerobot_to_rlds.py",
)


def test_resolve_episode_path_uses_official_chunk_layout(tmp_path):
    path = converter.resolve_episode_path(
        tmp_path,
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        episode_index=1234,
        chunks_size=1000,
        video_key="observation.images.cam_high",
    )

    assert path == (
        tmp_path
        / "videos/chunk-001/observation.images.cam_high/episode_001234.mp4"
    )


def test_load_task_map_preserves_task_index(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 7, "task": "pick the bottle"}) + "\n",
        encoding="utf-8",
    )

    assert converter.load_task_map(tmp_path) == {7: "pick the bottle"}


def test_build_episode_steps_preserves_14d_values_and_per_step_language():
    dataframe = pd.DataFrame(
        {
            "observation.state": [np.arange(14), np.arange(14) + 1],
            "action": [np.arange(14) + 2, np.arange(14) + 3],
            "task_index": [5, 6],
        }
    )
    frames = {
        name: [np.zeros((4, 5, 3), dtype=np.uint8), np.ones((4, 5, 3), dtype=np.uint8)]
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }

    steps = converter.build_episode_steps(dataframe, frames, {5: "first", 6: "second"})

    assert len(steps) == 2
    assert steps[0]["observation"]["state"].shape == (14,)
    assert steps[0]["action"].shape == (14,)
    assert steps[0]["language_instruction"] == "first"
    assert steps[1]["language_instruction"] == "second"
    assert steps[0]["is_first"] and not steps[0]["is_last"]
    assert steps[1]["is_last"] and steps[1]["is_terminal"]


def test_validate_episode_dataframe_rejects_noncontiguous_frames():
    dataframe = pd.DataFrame(
        {
            "frame_index": [0, 2],
            "episode_index": [4, 4],
        }
    )

    with pytest.raises(ValueError, match="non-contiguous"):
        converter.validate_episode_dataframe(dataframe, episode_index=4, expected_length=2)


def test_bounded_parallel_map_preserves_order_and_limits_submission():
    seen = []

    def record(value):
        seen.append(value)
        return value * 2

    results = list(converter.bounded_parallel_map(record, range(7), workers=3))

    assert results == [0, 2, 4, 6, 8, 10, 12]
    assert sorted(seen) == list(range(7))


def test_bounded_parallel_map_rejects_zero_workers():
    with pytest.raises(ValueError, match="at least 1"):
        list(converter.bounded_parallel_map(lambda value: value, [1], workers=0))
