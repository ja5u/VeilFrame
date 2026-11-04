"""Writes processed and original frames in their original sequence."""

import shutil
from pathlib import Path

from PIL import Image


def write_ordered_frames(
    total_frames: int,
    raw_frames_dir: Path,
    edited_map: dict[int, Image.Image],
    out_frames_dir: Path,
) -> Path:
    """Write exactly one numbered JPEG for each input frame."""
    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    invalid = [idx for idx in edited_map if idx < 0 or idx >= total_frames]
    if invalid:
        raise ValueError(f"edited_map contains out-of-range frame indices: {invalid}")

    out_frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(total_frames):
        destination = out_frames_dir / f"frame_{idx:05d}.jpg"
        edited = edited_map.get(idx)
        if edited is not None:
            edited.convert("RGB").save(destination, format="JPEG", quality=95, subsampling=0)
        else:
            source = raw_frames_dir / f"frame_{idx:05d}.jpg"
            if not source.is_file():
                source = raw_frames_dir / f"frame_{idx:05d}.png"
            if not source.is_file():
                raise FileNotFoundError(f"Original frame is missing: {source}")
            if source.suffix.lower() == ".png":
                with Image.open(source) as original:
                    original.convert("RGB").save(
                        destination, format="JPEG", quality=95, subsampling=0
                    )
            else:
                shutil.copy2(source, destination)
    return out_frames_dir


def write_frames(output_dir, frames):
    """Compatibility wrapper for sequential frames supplied in memory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.convert("RGB").save(
            output_path / f"frame_{idx:05d}.jpg", format="JPEG", quality=95, subsampling=0
        )
    return output_path
