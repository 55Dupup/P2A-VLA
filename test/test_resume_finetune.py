from pathlib import Path

from prismatic.training.resume_finetune import (
    find_component_checkpoint,
    infer_resume_finetune_step,
    resolve_resume_finetune_path,
)


def test_infer_resume_finetune_step_from_directory_name():
    path = Path("/tmp/best-pro-libero_goal--40000_chkpt")

    assert infer_resume_finetune_step(path, None) == 40000


def test_find_component_checkpoint_uses_explicit_step(tmp_path):
    ckpt = tmp_path / "action_head--40000_checkpoint.pt"
    ckpt.write_bytes(b"checkpoint")

    assert find_component_checkpoint(tmp_path, "action_head", 40000) == ckpt


def test_resolve_resume_finetune_path_rejects_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    try:
        resolve_resume_finetune_path(True, missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing resume_finetune_path should fail")
