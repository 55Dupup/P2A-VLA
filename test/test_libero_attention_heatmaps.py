import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "libero_attention_heatmaps.py"
SPEC = importlib.util.spec_from_file_location("libero_attention_heatmaps", MODULE_PATH)
heatmaps = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = heatmaps
SPEC.loader.exec_module(heatmaps)


def test_infer_patch_grid_requires_square_patch_count():
    assert heatmaps.infer_patch_grid(256) == (16, 16)

    try:
        heatmaps.infer_patch_grid(255)
    except ValueError as exc:
        assert "square" in str(exc)
    else:
        raise AssertionError("Expected non-square patch count to fail")


def test_attention_vector_to_heatmap_normalizes_to_unit_range():
    values = np.arange(16, dtype=np.float32)

    heatmap = heatmaps.attention_vector_to_heatmap(values, output_size=(8, 8))

    assert heatmap.shape == (8, 8)
    assert np.isclose(float(heatmap.min()), 0.0)
    assert np.isclose(float(heatmap.max()), 1.0)


def test_overlay_heatmap_preserves_image_size_and_dtype():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[..., 1] = 120
    heatmap = np.linspace(0, 1, 80, dtype=np.float32).reshape(8, 10)

    overlay = heatmaps.overlay_heatmap_on_image(image, heatmap, alpha=0.5)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, image)


def test_gradcam_vector_uses_positive_weighted_features():
    features = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=np.float32,
    )
    gradients = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    cam = heatmaps.gradcam_vector_from_features(features, gradients)

    assert cam.shape == (4,)
    assert np.allclose(cam, np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))


def test_select_action_target_supports_mean_abs_and_named_dims():
    actions = np.array(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0],
            [-2.0, 4.0, -6.0, 8.0, -10.0, 12.0, -14.0],
        ],
        dtype=np.float32,
    )

    assert np.isclose(heatmaps.select_action_target_value(actions, "mean_abs_action", 0), 6.0)
    assert np.isclose(heatmaps.select_action_target_value(actions, "gripper", 0), 7.0)
    assert np.isclose(heatmaps.select_action_target_value(actions, "y", 1), 4.0)


def test_resolve_unnorm_key_falls_back_to_no_noops():
    assert heatmaps.resolve_unnorm_key("libero_object", {"libero_object_no_noops": {}}) == "libero_object_no_noops"


def test_grad_x_activation_vector_uses_absolute_elementwise_relevance():
    features = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    gradients = np.array([[2.0, 3.0], [-1.0, 0.5]], dtype=np.float32)

    cam = heatmaps.grad_x_activation_vector_from_features(features, gradients)

    assert np.allclose(cam, np.array([8.0, 5.0], dtype=np.float32))


def test_cam_vector_dispatch_uses_requested_method():
    features = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    gradients = np.array([[2.0, 3.0], [-1.0, 0.5]], dtype=np.float32)

    dispatched = heatmaps.cam_vector_from_features(features, gradients, "grad_x_activation")

    assert np.allclose(dispatched, heatmaps.grad_x_activation_vector_from_features(features, gradients))


def test_signed_delta_to_rgb_marks_positive_and_negative_differently():
    delta = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

    rgb = heatmaps.signed_delta_to_rgb(delta)

    assert rgb.shape == (1, 3, 3)
    assert rgb.dtype == np.uint8
    assert not np.array_equal(rgb[0, 0], rgb[0, 2])


def test_source_layer_name_for_visual_sources():
    assert heatmaps.source_layer_name("vision_backbone", 0) == "vision_backbone"
    assert heatmaps.source_layer_name("projected_patch", 0) == "projected_patch"
    assert heatmaps.source_layer_name("vlm_layers", 3) == "layer_03"
