"""Chooses representative AI anchors within each video shot."""

from pathlib import Path

def select_anchors_for_shot(
    shot_range: tuple[int, int],
    frame_paths: list[Path],
    skin_flags: list[bool],
    stride: int = 24,
) -> list[int]:
    """Sample AI anchors from skin-candidate frames within a shot."""
    start, end = shot_range
    if start < 0 or end < start or end >= len(frame_paths):
        raise ValueError(f"Invalid shot range: {shot_range}")
    if len(skin_flags) != len(frame_paths):
        raise ValueError("skin_flags must have one entry per frame")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    valid = [idx for idx in range(start, end + 1) if skin_flags[idx]]
    if not valid:
        return []

    first = valid[0]
    anchors = [first]
    next_sample = first + stride
    for idx in valid[1:]:
        if idx >= next_sample:
            anchors.append(idx)
            next_sample = idx + stride
    if valid[-1] - anchors[-1] >= 6:
        anchors.append(valid[-1])
    return anchors


def select_anchor_frames(frames):
    """Compatibility wrapper for callers using the original scaffold API."""
    raise NotImplementedError("Use select_anchors_for_shot with frame paths and skin flags.")
