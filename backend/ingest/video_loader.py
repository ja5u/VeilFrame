"""Extracts video frames and metadata into a per-job working directory."""

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import cv2
from PIL import Image


class VideoMeta(TypedDict):
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float
    frames_dir: Path
    video_path: Path


def probe_and_extract(
    video_path: Path,
    work_dir: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VideoMeta:
    """Extract frames as numbered JPEGs and return source video metadata."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Video has invalid metadata: {video_path}")

    frames_dir = work_dir / "raw_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            frame_file = frames_dir / f"frame_{count:05d}.jpg"
            if not cv2.imwrite(str(frame_file), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Unable to write extracted frame: {frame_file}")
            count += 1
            if progress_callback is not None:
                progress_callback(count, reported_count)
    finally:
        capture.release()

    if count == 0:
        raise ValueError(f"Video contains no readable frames: {video_path}")

    return VideoMeta(
        fps=fps,
        width=width,
        height=height,
        frame_count=count,
        duration_sec=count / fps,
        frames_dir=frames_dir,
        video_path=video_path,
    )


def load_video_metadata(video_path: str) -> dict:
    """Compatibility wrapper that probes metadata without retaining extracted frames."""
    path = Path(video_path)
    meta = probe_and_extract(path, path.parent / f"{path.stem}_work")
    return dict(meta)


def decode_frames(video_path: str) -> list[Image.Image]:
    """Compatibility helper that returns all decoded frames as RGB images."""
    path = Path(video_path)
    meta = probe_and_extract(path, path.parent / f"{path.stem}_work")
    frames: list[Image.Image] = []
    for idx in range(meta["frame_count"]):
        frame_path = meta["frames_dir"] / f"frame_{idx:05d}.jpg"
        if not frame_path.is_file():
            frame_path = meta["frames_dir"] / f"frame_{idx:05d}.png"
        if not frame_path.is_file():
            raise FileNotFoundError(f"Extracted frame is missing: {frame_path}")
        with Image.open(frame_path) as frame:
            frames.append(frame.convert("RGB"))
    return frames
