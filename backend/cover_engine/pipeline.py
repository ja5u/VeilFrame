"""Processes an anchor frame with the existing SAM -> Spark -> Muse logic."""

import inspect
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from backend.cover_engine import muse_client, sam_client
from backend.legacy.app_1 import DEFAULT_PROMPT, Region, build_dynamic_targets, clothe_image


def _build_full_frame_alpha(
    size: tuple[int, int],
    approved: list[tuple[Region, str]],
    feather_frac: float = 0.008,
) -> np.ndarray:
    """Construct the full-frame float32 alpha mask from approved regions.

    Fix 3: feather radius is now proportional to the image diagonal (feather_frac *
    diagonal) rather than a hard-coded pixel value.  This keeps the edit boundary
    visually consistent regardless of resolution, preventing edge shimmer when the
    alpha mask composites differently across frames.
    """
    width, height = size
    # Scale feather with the image diagonal so proportions stay constant at any resolution.
    feather = max(2.0, feather_frac * (width ** 2 + height ** 2) ** 0.5)

    full_mask = np.zeros((height, width), dtype=np.uint8)
    for region, _ in approved:
        left, top, right, bottom = region.box
        left_c, top_c = max(0, min(width, left)), max(0, min(height, top))
        right_c, bottom_c = max(0, min(width, right)), max(0, min(height, bottom))
        if right_c <= left_c or bottom_c <= top_c:
            continue
        mask = region.mask[top_c - top : bottom_c - top, left_c - left : right_c - left].copy()
        if region.protect is not None:
            protect = region.protect[top_c - top : bottom_c - top, left_c - left : right_c - left]
            mask &= 1 - protect
        full_mask[top_c:bottom_c, left_c:right_c] |= mask

    if not full_mask.any():
        return np.zeros((height, width), dtype=np.float32)

    blurred = Image.fromarray(full_mask * 255, mode="L").filter(ImageFilter.GaussianBlur(radius=feather))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def process_anchor_frame(
    img: Image.Image,
    cached_hint: dict[int, tuple[str, str]] | None = None,
    debug_dir: Path | None = None,
) -> tuple[Image.Image, np.ndarray, list[dict], dict[int, tuple[str, str]]]:
    """Edit an anchor and return its feathered alpha mask and cached per-person plans."""
    source = img.convert("RGB")
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    segmented = sam_client.run_sam_parallel(source, debug_dir)
    plan = cached_hint if cached_hint is not None else {}

    if "spark_plan" in inspect.signature(build_dynamic_targets).parameters:
        approved, skipped = build_dynamic_targets(
            source,
            segmented["person"],
            segmented["clothing"],
            segmented["torso"],
            [],
            spark_plan=plan,
        )
    else:
        approved, skipped = build_dynamic_targets(
            source,
            segmented["person"],
            segmented["clothing"],
            segmented["torso"],
            [],
        )

    alpha = np.zeros((source.height, source.width), dtype=np.float32)
    raw_edit = np.zeros((source.height, source.width, 3), dtype=np.uint8)
    clothe_parameters = inspect.signature(clothe_image).parameters
    clothe_kwargs = {"debug_dir": debug_dir}
    if "alpha_mask_out" in clothe_parameters:
        clothe_kwargs["alpha_mask_out"] = alpha
    if "raw_edit_out" in clothe_parameters:
        clothe_kwargs["raw_edit_out"] = raw_edit

    edited, report = clothe_image(
        source,
        approved,
        muse_client.muse_edit,
        DEFAULT_PROMPT,
        **clothe_kwargs,
    )
    if "alpha_mask_out" not in clothe_parameters:
        alpha = _build_full_frame_alpha(source.size, approved)
    edited.info["raw_edit_rgb"] = raw_edit
    return edited, alpha, skipped + report, plan
