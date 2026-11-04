"""Muse edit adapter backed by the existing policy-aware image editor."""

from PIL import Image

from backend.legacy.app_1 import muse_edit as _muse_edit


def muse_edit(
    crop: Image.Image,
    mask: Image.Image,
    prompt: str,
    skin_mask: Image.Image | None = None,
) -> Image.Image:
    """Call the legacy Muse editor with its skin-only policy retry."""
    return _muse_edit(crop, mask, prompt, skin_mask=skin_mask)
