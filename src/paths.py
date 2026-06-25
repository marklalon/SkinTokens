"""Filesystem locations shared by local and container execution."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = Path(os.environ.get("SKINTOKENS_MODEL_ROOT", PROJECT_ROOT)).expanduser()
EXPERIMENTS_DIR = Path(
    os.environ.get("SKINTOKENS_EXPERIMENTS_DIR", MODEL_ROOT / "experiments")
).expanduser()
DATA_ROOT = Path(os.environ.get("SKINTOKENS_DATA_ROOT", PROJECT_ROOT)).expanduser()


def resolve_model_path(path: str | os.PathLike[str]) -> Path:
    """Resolve project-relative model paths against the external model root.

    Checkpoints contain paths such as ``experiments/...`` in their saved model
    configuration.  This function keeps those checkpoints portable without
    rewriting the checkpoint files themselves.
    """

    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw

    # Normalize paths saved on the other host OS before examining their prefix.
    parts = Path(str(path).replace("\\", "/")).parts
    candidates: list[Path] = []
    if parts and parts[0] == "experiments":
        candidates.append(EXPERIMENTS_DIR.joinpath(*parts[1:]))
    elif parts and parts[0] == "models":
        candidates.append(MODEL_ROOT.joinpath(*parts))
    else:
        candidates.append(MODEL_ROOT.joinpath(*parts))

    candidates.append(PROJECT_ROOT.joinpath(*parts))
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])
