"""Encodes numbered frames and restores the source audio using ffmpeg."""

import subprocess
from pathlib import Path


def encode_video_with_audio(
    frames_dir: Path,
    orig_video_path: Path,
    out_video_path: Path,
    fps: float,
) -> Path:
    """Encode the processed frame sequence and optionally map source audio."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if (frames_dir / "frame_00000.jpg").is_file() else "png"
    command = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / f"frame_%05d.{ext}"),
        "-i",
        str(orig_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        # Force constant frame rate on the output to avoid timestamp jitter.
        "-r",
        str(fps),
        "-fps_mode",
        "cfr",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-shortest",
        str(out_video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg and ensure it is on PATH.") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}: {result.stderr.strip()}")
    return out_video_path


def encode_video(input_video, processed_frames, output_path):
    """Compatibility wrapper for the original scaffold API."""
    raise NotImplementedError("Use encode_video_with_audio with the source frame rate.")
