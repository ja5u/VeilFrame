"""SAM segmentation adapter backed by the existing single-image implementation."""

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

import numpy as np
from PIL import Image

from backend.legacy.app_1 import (
    CLOTHING_PHRASE,
    PERSON_PHRASE,
    TORSO_PHRASE,
    Region,
    dedupe_preserve_order,
    segment,
    to_data_url,
)


def _scale_region(region: Region, source_size: tuple[int, int], target_size: tuple[int, int]) -> Region:
    """Scale a region and its box-local masks back to the original image size."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    left, top, right, bottom = region.box
    scaled_box = (
        max(0, min(target_width, math.floor(left * target_width / source_width))),
        max(0, min(target_height, math.floor(top * target_height / source_height))),
        max(0, min(target_width, math.ceil(right * target_width / source_width))),
        max(0, min(target_height, math.ceil(bottom * target_height / source_height))),
    )
    box_width = scaled_box[2] - scaled_box[0]
    box_height = scaled_box[3] - scaled_box[1]

    def scale_mask(mask: np.ndarray) -> np.ndarray:
        if box_width <= 0 or box_height <= 0:
            return np.zeros((max(0, box_height), max(0, box_width)), dtype=mask.dtype)
        return np.asarray(
            Image.fromarray(mask).resize((box_width, box_height), Image.Resampling.NEAREST)
        ).copy()

    return Region(
        object_id=region.object_id,
        box=scaled_box,
        confidence=region.confidence,
        mask=scale_mask(region.mask),
        protect=None if region.protect is None else scale_mask(region.protect),
        person_mask=None if region.person_mask is None else scale_mask(region.person_mask),
        skin_mask=None if region.skin_mask is None else scale_mask(region.skin_mask),
    )


def run_sam_parallel(img: Image.Image, debug_dir: Path | None = None) -> dict[str, list[Region]]:
    """Run person, clothing, and torso segmentation concurrently using compressed JPEG."""
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    send_img = img
    if max(img.size) > 1536:
        scale = 1536 / max(img.size)
        send_img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

    image_url = to_data_url(send_img, fmt="JPEG", quality=85)
    phrases = dedupe_preserve_order([PERSON_PHRASE, CLOTHING_PHRASE, TORSO_PHRASE])
    with ThreadPoolExecutor(max_workers=len(phrases)) as pool:
        futures = {
            phrase: pool.submit(segment, image_url, phrase, debug_dir, image_size=send_img.size)
            for phrase in phrases
        }
        results = {phrase: futures[phrase].result() for phrase in phrases}
    if send_img.size != img.size:
        results = {
            phrase: [_scale_region(region, send_img.size, img.size) for region in regions]
            for phrase, regions in results.items()
        }
    return {
        "person": results[PERSON_PHRASE],
        "clothing": results[CLOTHING_PHRASE],
        "torso": results[TORSO_PHRASE],
    }
