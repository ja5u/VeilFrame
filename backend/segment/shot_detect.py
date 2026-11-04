"""Divides an ordered frame sequence into camera shots."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


def detect_shots(
    frame_paths: list[Path],
    threshold: float = 0.35,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[tuple[int, int]]:
    """Detect shot boundaries with normalized HSV histogram distances."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if not frame_paths:
        return []

    boundaries = [0]
    previous_hist = None
    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Unable to read frame: {frame_path}")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [16, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        if previous_hist is not None:
            distance = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if distance > threshold:
                boundaries.append(idx)
        previous_hist = hist
        if progress_callback is not None:
            progress_callback(idx + 1, len(frame_paths))

    shots = [(start, end - 1) for start, end in zip(boundaries, boundaries[1:])]
    shots.append((boundaries[-1], len(frame_paths) - 1))
    return shots
