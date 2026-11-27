"""Compatibility shims for transformers API moves used by legacy dependencies."""

from __future__ import annotations

from loguru import logger


def patch_transformers_modeling_utils() -> None:
    """Backfill symbols moved out of transformers.modeling_utils in newer transformers."""
    try:
        import transformers.modeling_utils as modeling_utils
        from transformers import pytorch_utils
    except Exception as err:
        logger.debug(f"transformers compat: skip ({err})")
        return

    moved_names = (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    )
    patched = []
    for name in moved_names:
        if hasattr(modeling_utils, name):
            continue
        value = getattr(pytorch_utils, name, None)
        if value is None:
            continue
        setattr(modeling_utils, name, value)
        patched.append(name)

    if patched:
        logger.debug(f"transformers compat: patched modeling_utils symbols: {', '.join(patched)}")
