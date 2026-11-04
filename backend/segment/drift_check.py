"""Detects mask expansion or flow tracking failure."""

import cv2
import numpy as np


def check_mask_drift(
    warped_alpha: np.ndarray,
    current_frame_bgr: np.ndarray,
    prev_frame_bgr: np.ndarray,
    max_mse: float = 450.0,
    reference_area: float | None = None,
    warped_prev_gray: np.ndarray | None = None,
    current_gray: np.ndarray | None = None,
) -> bool:
    """Return true when mask area or flow-aligned boundary appearance drifts."""
    alpha = np.asarray(warped_alpha, dtype=np.float32)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    if alpha.shape != current_frame_bgr.shape[:2] or prev_frame_bgr.shape != current_frame_bgr.shape:
        raise ValueError("Frames and warped alpha must have matching dimensions")

    smoothed_alpha = cv2.GaussianBlur(alpha, (3, 3), 0.8)
    current_area = float(np.count_nonzero(smoothed_alpha > 0.1))
    if reference_area is not None and reference_area > 0:
        if current_area < 0.55 * reference_area:
            return True
    if current_area == 0:
        return reference_area is not None and reference_area > 0

    if current_gray is None:
        current_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
    elif current_gray.shape != current_frame_bgr.shape[:2]:
        raise ValueError("current_gray must match the frame dimensions")
    if warped_prev_gray is None:
        prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
        backward_flow = cv2.calcOpticalFlowFarneback(
            current_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        height, width = current_gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = (grid_x + backward_flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y + backward_flow[:, :, 1]).astype(np.float32)
        warped_prev_gray = cv2.remap(
            prev_gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
    elif warped_prev_gray.shape != current_frame_bgr.shape[:2]:
        raise ValueError("warped_prev_gray must match the frame dimensions")

    binary = (smoothed_alpha > 0.1).astype(np.uint8)
    boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    if not boundary.any():
        return False
    residual = current_gray.astype(np.float32) - warped_prev_gray.astype(np.float32)
    mse = float(np.mean(np.square(residual[boundary])))
    return mse > max_mse


def check_drift(local_mask, warped_mask):
    """Compatibility wrapper that compares two same-sized binary masks."""
    local = np.asarray(local_mask, dtype=bool)
    warped = np.asarray(warped_mask, dtype=bool)
    if local.shape != warped.shape:
        raise ValueError("Masks must have matching dimensions")
    intersection = np.count_nonzero(local & warped)
    union = np.count_nonzero(local | warped)
    return union > 0 and 1.0 - intersection / union > 0.35
