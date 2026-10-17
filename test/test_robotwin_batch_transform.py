from pathlib import Path

import numpy as np
import torch

import prismatic.vla.datasets.datasets as datasets_module
from prismatic.vla.datasets.datasets import RLDSBatchTransform, RLDSDataset


class ArrayImageTransform:
    def __call__(self, image):
        return torch.as_tensor(np.array(image)).permute(2, 0, 1).float()


class FakeActionTokenizer:
    def __call__(self, action, use_minivlm=False):
        return "A"


class FakePromptBuilder:
    def __init__(self, model_family):
        self.turns = []

    def add_turn(self, role, value):
        self.turns.append((role, value))

    def get_prompt(self):
        return "\n".join(f"{role}: {value}" for role, value in self.turns)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True):
        return type("Tokenized", (), {"input_ids": [1, 2, 3, 4]})()


def make_batch():
    return {
        "dataset_name": "robotwin_lerobot_unified",
        "observation": {
            "image_primary": np.ones((1, 4, 4, 3), dtype=np.uint8),
            "image_left_wrist": np.full((1, 4, 4, 3), 2, dtype=np.uint8),
            "image_right_wrist": np.full((1, 4, 4, 3), 3, dtype=np.uint8),
            "proprio": torch.arange(14, dtype=torch.float32),
        },
        "task": {"language_instruction": b"pick the block"},
        "action": np.zeros((16, 14), dtype=np.float32),
    }


def make_transform(*, use_wrist_image=True, use_proprio=False):
    return RLDSBatchTransform(
        action_tokenizer=FakeActionTokenizer(),
        base_tokenizer=FakeTokenizer(),
        image_transform=ArrayImageTransform(),
        prompt_builder_fn=FakePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=use_proprio,
        use_minivlm=False,
    )


def test_robotwin_three_views_are_concatenated_in_left_then_right_order():
    output = make_transform(use_wrist_image=True, use_proprio=False)(make_batch())

    wrist = output["pixel_values_wrist"]
    assert wrist.shape == (6, 4, 4)
    assert torch.all(wrist[:3] == 2.0)
    assert torch.all(wrist[3:] == 3.0)
    assert "proprio" not in output


def test_robotwin_proprio_is_present_only_when_enabled():
    output = make_transform(use_wrist_image=True, use_proprio=True)(make_batch())

    assert output["proprio"].shape == (14,)


def test_robotwin_dataset_requests_three_views_and_respects_no_proprio(monkeypatch):
    captured = {}

    def fake_get_kwargs(data_root_dir, mixture_spec, **kwargs):
        captured["data_root_dir"] = data_root_dir
        captured["mixture_spec"] = mixture_spec
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(datasets_module, "get_oxe_dataset_kwargs_and_weights", fake_get_kwargs)
    monkeypatch.setattr(RLDSDataset, "make_dataset", lambda self, rlds_config: ([], 0, {}))

    batch_transform = make_transform(use_wrist_image=True, use_proprio=False)
    RLDSDataset(
        data_root_dir=Path("/tmp/robotwin_rlds"),
        data_mix="robotwin_lerobot_unified",
        batch_transform=batch_transform,
        resize_resolution=(224, 224),
    )

    assert captured["load_camera_views"] == ("primary", "left_wrist", "right_wrist")
    assert captured["load_proprio"] is False
