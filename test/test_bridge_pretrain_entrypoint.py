from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bridge_pretrain_script_uses_requested_rlds_root_and_bridge_dataset():
    script = REPO_ROOT / "vla-scripts" / "pretrain_bridge.sh"

    text = script.read_text()

    assert "/path/to/datasets/OXE-data/bridge-rlds" in text
    assert "data_name=bridge_dataset" in text
    assert "--dataset_name \"${data_name}\"" in text
    assert "--data_root_dir \"${data_root_dir}\"" in text


def test_bridge_pretrain_script_saves_finetune_compatible_checkpoints():
    script = REPO_ROOT / "vla-scripts" / "pretrain_bridge.sh"

    text = script.read_text()

    assert "--use_lora True" in text
    assert "--use_l1_regression True" in text
    assert "--merge_lora_during_training True" in text
    assert "--save_latest_checkpoint_only False" in text


def test_libero_plus_finetune_keeps_resume_finetune_optional():
    script = REPO_ROOT / "vla-scripts" / "finetune_libero_plus.sh"

    text = script.read_text()

    assert "resume_finetune=${RESUME_FINETUNE:-False}" in text
    assert "resume_finetune_path=${RESUME_FINETUNE_PATH:-}" in text
    assert "--resume_finetune_path" in text
