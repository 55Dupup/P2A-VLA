"""Utilities for warm-starting a new finetune run from a completed checkpoint."""

import re
from pathlib import Path
from typing import Optional

import torch


def resolve_resume_finetune_path(resume_finetune: bool, resume_finetune_path: Optional[Path]) -> Optional[Path]:
    """Validate and normalize the optional warm-start checkpoint directory."""
    if not resume_finetune:
        return None
    if resume_finetune_path is None:
        raise ValueError("`resume_finetune_path` must be set when `resume_finetune=True`.")

    checkpoint_dir = Path(resume_finetune_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"`resume_finetune_path` does not exist or is not a directory: {checkpoint_dir}")
    return checkpoint_dir


def infer_resume_finetune_step(checkpoint_dir: Path, explicit_step: Optional[int]) -> Optional[int]:
    """Infer the training step from an explicit value, component filenames, or the checkpoint directory name."""
    if explicit_step is not None:
        return explicit_step

    for pattern in (r"--(\d+)_chkpt$", r"-(\d+)_chkpt$"):
        match = re.search(pattern, checkpoint_dir.name)
        if match:
            return int(match.group(1))

    component_steps = []
    for path in checkpoint_dir.glob("*--*_checkpoint.pt"):
        match = re.match(r".+--(\d+)_checkpoint\.pt$", path.name)
        if match:
            component_steps.append(int(match.group(1)))

    if component_steps:
        return max(component_steps)
    return None


def find_component_checkpoint(checkpoint_dir: Path, module_name: str, step: Optional[int]) -> Optional[Path]:
    """Find a component checkpoint saved as `<module_name>--<step>_checkpoint.pt`."""
    checkpoint_dir = Path(checkpoint_dir)
    if step is not None:
        checkpoint_path = checkpoint_dir / f"{module_name}--{step}_checkpoint.pt"
        return checkpoint_path if checkpoint_path.exists() else None

    matches = sorted(
        checkpoint_dir.glob(f"{module_name}--*_checkpoint.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def load_component_state_dict(
    checkpoint_dir: Path,
    module_name: str,
    step: Optional[int],
    device: str = "cpu",
) -> Optional[dict]:
    """Load a warm-start component state dict if it exists."""
    checkpoint_path = find_component_checkpoint(checkpoint_dir, module_name, step)
    if checkpoint_path is None:
        return None

    print(f"Loading resume-finetune component checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, weights_only=True, map_location=device)
