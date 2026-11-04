"""Fast non-AI skin-tone pre-check for video frames."""

import cv2
import numpy as np


def has_skin_candidate(frame_bgr: np.ndarray, min_skin_ratio: float = 0.015) -> bool:
    """Return whether the central 80 percent has a sufficiently large skin-tone region."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a three-channel BGR image")
    if not 0 <= min_skin_ratio <= 1:
        raise ValueError("min_skin_ratio must be between 0 and 1")

    height, width = frame_bgr.shape[:2]
    x1, x2 = int(width * 0.1), max(int(width * 0.9), 1)
    y1, y2 = int(height * 0.1), max(int(height * 0.9), 1)
    central = frame_bgr[y1:y2, x1:x2]
    if central.size == 0:
        return False

    ycrcb = cv2.cvtColor(central, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    candidate = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    return float(np.count_nonzero(candidate)) / candidate.size >= min_skin_ratio


def detect_skin_candidates(frame: np.ndarray) -> bool:
    """Compatibility wrapper for the original scaffold API."""
    return has_skin_candidate(frame)
