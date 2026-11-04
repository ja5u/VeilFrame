"""Writes a JSON summary of video processing results."""

import json
from pathlib import Path


def write_video_manifest(job_id: str, meta: dict, frame_log: list[dict], out_path: Path) -> Path:
    """Write per-frame status and aggregate shot, anchor, propagation, and skip counts."""
    statuses = [str(entry.get("status", "")) for entry in frame_log]
    payload = {
        "job_id": job_id,
        "total_frames": int(meta.get("frame_count", len(frame_log))),
        "shots_detected": int(meta.get("shots_detected", 0)),
        "anchors_edited": sum(status == "anchor_edited" for status in statuses),
        "frames_propagated": sum(status == "propagated" for status in statuses),
        "frames_skipped_compliant": sum(
            status in {"compliant_no_skin", "skipped_compliant"} for status in statuses
        ),
        "frames": frame_log,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def write_manifest(output_path, report):
    """Compatibility wrapper for writing arbitrary JSON reports."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
