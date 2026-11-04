"""Propagates edited garment patches through dense optical flow."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _read_bgr(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Unable to read frame: {path}")
    return frame


def _alpha_float(alpha: np.ndarray) -> np.ndarray:
    result = np.array(alpha, dtype=np.float32, copy=True)
    if result.ndim == 3:
        result = result[:, :, 0]
    if result.size and result.max() > 1.0:
        result /= 255.0
    result = np.clip(result, 0.0, 1.0)
    result[result < 0.03] = 0.0
    return result


def compute_backward_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Compute dense flow from curr_gray to prev_gray at half resolution."""
    height, width = curr_gray.shape
    small_height, small_width = max(1, height // 2), max(1, width // 2)
    prev_small = cv2.resize(
        prev_gray, (small_width, small_height), interpolation=cv2.INTER_AREA
    )
    curr_small = cv2.resize(
        curr_gray, (small_width, small_height), interpolation=cv2.INTER_AREA
    )
    flow_small = cv2.calcOpticalFlowFarneback(
        curr_small, prev_small, None, 0.5, 3, 31, 3, 7, 1.5, 0
    )
    flow = cv2.resize(flow_small, (width, height), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] *= width / small_width
    flow[:, :, 1] *= height / small_height
    flow = cv2.GaussianBlur(flow, (15, 15), 0)
    return flow


def _consistent_backward_flow(
    prev_gray: np.ndarray, curr_gray: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = curr_gray.shape
    backward_flow = compute_backward_flow(prev_gray, curr_gray)
    forward_flow = compute_backward_flow(curr_gray, prev_gray)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x + backward_flow[:, :, 0]
    map_y = grid_y + backward_flow[:, :, 1]
    forward_x = cv2.remap(
        forward_flow[:, :, 0],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    forward_y = cv2.remap(
        forward_flow[:, :, 1],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    fb_error = np.hypot(
        backward_flow[:, :, 0] + forward_x,
        backward_flow[:, :, 1] + forward_y,
    )
    valid_flow = (fb_error <= 3.0).astype(np.float32)
    valid_flow = cv2.GaussianBlur(valid_flow, (7, 7), 0)
    return backward_flow, map_x, map_y, valid_flow


def _keep_connected_alpha(
    warped_alpha: np.ndarray,
    previous_alpha: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    previous_support = cv2.remap(
        previous_alpha,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    previous_support = cv2.dilate(
        (previous_support > 0.05).astype(np.uint8),
        np.ones((11, 11), dtype=np.uint8),
    ) > 0
    _, labels = cv2.connectedComponents(
        (warped_alpha > 0.05).astype(np.uint8), connectivity=8
    )
    overlapping_labels = np.unique(labels[previous_support])
    keep = np.isin(labels, overlapping_labels[overlapping_labels > 0])
    warped_alpha[~keep] = 0.0
    return warped_alpha


def warp_step(
    edited_rgb: np.ndarray,
    alpha: np.ndarray,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp an edited patch one frame, returning its warped previous grayscale frame."""
    height, width = curr_gray.shape
    if prev_gray.shape != curr_gray.shape:
        raise ValueError("prev_gray and curr_gray must have matching dimensions")
    if edited_rgb.shape[:2] != (height, width) or alpha.shape[:2] != (height, width):
        raise ValueError("edited_rgb and alpha must match the frame dimensions")

    _, map_x, map_y, valid_flow = _consistent_backward_flow(prev_gray, curr_gray)

    warped_rgb = cv2.remap(
        edited_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    warped_alpha = cv2.remap(
        alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    warped_prev_gray = cv2.remap(
        prev_gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    in_bounds = (map_x >= 0) & (map_x < width - 1) & (map_y >= 0) & (map_y < height - 1)
    warped_alpha *= in_bounds.astype(np.float32)
    warped_alpha *= valid_flow
    warped_alpha = _keep_connected_alpha(warped_alpha, alpha, map_x, map_y)
    warped_alpha[warped_alpha < 0.03] = 0.0
    return warped_rgb, warped_alpha, warped_prev_gray


def warp_map_step(
    map_x: np.ndarray,
    map_y: np.ndarray,
    anchor_rgb: np.ndarray,
    anchor_alpha: np.ndarray,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update cumulative maps and sample the original anchor in one remap."""
    height, width = curr_gray.shape
    if prev_gray.shape != curr_gray.shape:
        raise ValueError("prev_gray and curr_gray must have matching dimensions")
    if map_x.shape != curr_gray.shape or map_y.shape != curr_gray.shape:
        raise ValueError("map_x and map_y must match the frame dimensions")
    if anchor_rgb.shape[:2] != curr_gray.shape or anchor_alpha.shape[:2] != curr_gray.shape:
        raise ValueError("anchor_rgb and anchor_alpha must match the frame dimensions")
    if prev_alpha is not None and prev_alpha.shape != curr_gray.shape:
        raise ValueError("prev_alpha must match the frame dimensions")

    _, step_x, step_y, valid_flow = _consistent_backward_flow(prev_gray, curr_gray)

    new_map_x = cv2.remap(
        map_x, step_x, step_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    new_map_y = cv2.remap(
        map_y, step_x, step_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    warped_rgb = cv2.remap(
        anchor_rgb, new_map_x, new_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    warped_alpha = cv2.remap(
        anchor_alpha, new_map_x, new_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    warped_prev_gray = cv2.remap(
        prev_gray, step_x, step_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    in_bounds = (
        (new_map_x >= 0)
        & (new_map_x < width - 1)
        & (new_map_y >= 0)
        & (new_map_y < height - 1)
    )
    step_in_bounds = (
        (step_x >= 0)
        & (step_x < width - 1)
        & (step_y >= 0)
        & (step_y < height - 1)
    )
    warped_alpha *= in_bounds.astype(np.float32) * step_in_bounds.astype(np.float32)

    if prev_alpha is not None:
        prev_warped_alpha = cv2.remap(
            prev_alpha, step_x, step_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        warped_alpha = 0.7 * warped_alpha + 0.3 * prev_warped_alpha

    warped_alpha *= valid_flow
    warped_alpha *= in_bounds.astype(np.float32) * step_in_bounds.astype(np.float32)
    if prev_alpha is not None:
        warped_alpha = _keep_connected_alpha(warped_alpha, prev_alpha, step_x, step_y)
    warped_alpha[warped_alpha < 0.03] = 0.0
    return new_map_x, new_map_y, warped_rgb, warped_alpha, warped_prev_gray


def composite_patch(
    target_rgb: np.ndarray, patch_rgb: np.ndarray, alpha: np.ndarray
) -> Image.Image:
    """Composite a warped garment patch over the pristine target frame."""
    weight = alpha[:, :, None]
    output = np.clip(
        target_rgb.astype(np.float32) * (1.0 - weight)
        + patch_rgb.astype(np.float32) * weight
        + 0.5,
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(output, "RGB")


def interpolate_anchor_span(
    left_idx: int,
    right_idx: int,
    frame_paths: list[Path],
    left_edited_rgb: np.ndarray,
    left_alpha: np.ndarray,
    right_edited_rgb: np.ndarray,
    right_alpha: np.ndarray,
    forward_cache: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, Image.Image]:
    """Blend forward and backward patch propagation across an anchor span."""
    if left_idx < 0 or right_idx >= len(frame_paths) or right_idx <= left_idx:
        raise ValueError("Anchor indices must define an increasing span in frame_paths")
    span = right_idx - left_idx
    if span <= 1:
        return {}

    backward_patches: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    patch_rgb = right_edited_rgb.copy()
    patch_alpha = _alpha_float(right_alpha)
    previous_gray = cv2.cvtColor(
        _read_bgr(frame_paths[right_idx]), cv2.COLOR_BGR2GRAY
    )
    for idx in range(right_idx - 1, left_idx, -1):
        target_gray = cv2.cvtColor(_read_bgr(frame_paths[idx]), cv2.COLOR_BGR2GRAY)
        patch_rgb, patch_alpha, _ = warp_step(
            patch_rgb, patch_alpha, previous_gray, target_gray
        )
        backward_patches[idx] = (patch_rgb, patch_alpha)
        previous_gray = target_gray

    blended_frames: dict[int, Image.Image] = {}
    for idx in range(left_idx + 1, right_idx):
        if idx not in forward_cache:
            raise ValueError(f"Forward-propagated patch is missing for frame {idx}")
        orig_bgr = _read_bgr(frame_paths[idx])
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        forward_rgb, forward_alpha = forward_cache[idx]
        backward_rgb, backward_alpha = backward_patches[idx]

        t = (idx - left_idx) / float(span)
        weight = float(0.5 - 0.5 * np.cos(np.pi * t))
        blended_rgb = (
            (1.0 - weight) * forward_rgb.astype(np.float32)
            + weight * backward_rgb.astype(np.float32)
        )
        blended_alpha = (1.0 - weight) * forward_alpha + weight * backward_alpha
        blended_alpha[blended_alpha < 0.03] = 0.0
        blended_frames[idx] = composite_patch(orig_rgb, blended_rgb, blended_alpha)
    return blended_frames


def propagate_between_anchors(
    anchor_idx: int,
    target_indices: list[int],
    frame_paths: list[Path],
    anchor_orig: Image.Image,
    anchor_edited: Image.Image,
    anchor_alpha: np.ndarray,
) -> dict[int, tuple[Image.Image, np.ndarray]]:
    """Compatibility wrapper that propagates the anchor patch to requested frames."""
    if not 0 <= anchor_idx < len(frame_paths):
        raise IndexError("anchor_idx is outside frame_paths")
    if anchor_orig.size != anchor_edited.size:
        raise ValueError("anchor_orig and anchor_edited must have identical dimensions")

    width, height = anchor_orig.size
    source_alpha = _alpha_float(anchor_alpha)
    if source_alpha.shape != (height, width):
        raise ValueError("anchor_alpha must match the anchor image dimensions")

    results: dict[int, tuple[Image.Image, np.ndarray]] = {}
    for target_idx in sorted(set(target_indices), key=lambda idx: abs(idx - anchor_idx)):
        if not 0 <= target_idx < len(frame_paths):
            raise IndexError(f"target frame index is out of range: {target_idx}")
        if target_idx == anchor_idx:
            continue

        direction = 1 if target_idx > anchor_idx else -1
        patch_rgb = np.asarray(anchor_edited.convert("RGB"), dtype=np.uint8).copy()
        alpha = source_alpha.copy()
        previous_idx = anchor_idx
        for next_idx in range(anchor_idx + direction, target_idx + direction, direction):
            previous_gray = cv2.cvtColor(
                _read_bgr(frame_paths[previous_idx]), cv2.COLOR_BGR2GRAY
            )
            target_bgr = _read_bgr(frame_paths[next_idx])
            target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
            patch_rgb, alpha, _ = warp_step(
                patch_rgb, alpha, previous_gray, target_gray
            )
            previous_idx = next_idx

        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        results[target_idx] = (composite_patch(target_rgb, patch_rgb, alpha), alpha)
    return results


def propagate_masks(anchor_frame, target_frame, mask):
    """Compatibility wrapper retained for the original scaffold API."""
    raise NotImplementedError("Use propagate_between_anchors with numbered frame paths.")
