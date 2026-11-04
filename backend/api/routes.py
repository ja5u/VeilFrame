"""Upload, processing, status, and result endpoints."""

import logging
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from backend.cover_engine.pipeline import process_anchor_frame
from backend.ingest.consent_gate import verify_consent
from backend.ingest.video_loader import probe_and_extract
from backend.output.frame_writer import write_ordered_frames
from backend.output.manifest import write_video_manifest
from backend.output.video_encoder import encode_video_with_audio
from backend.segment.anchor_select import select_anchors_for_shot
from backend.segment.drift_check import check_mask_drift
from backend.segment.flow_propagate import (
    _alpha_float,
    composite_patch,
    interpolate_anchor_span,
    warp_map_step,
)
from backend.segment.local_detector import has_skin_candidate
from backend.segment.shot_detect import detect_shots

logger = logging.getLogger(__name__)
router = APIRouter()

JOBS: dict[str, dict[str, Any]] = {}
STATUS_STORE = JOBS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = PROJECT_ROOT / "jobs"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _job_or_404(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _set_status(job: dict[str, Any], status_text: str, **updates: Any) -> None:
    job.update(status_text=status_text, **updates)


def _read_frame(path: Path) -> tuple[Image.Image, np.ndarray]:
    frame_bgr = cv2.imread(str(path))
    if frame_bgr is None:
        raise ValueError(f"Unable to read frame: {path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb), frame_bgr


def _process_video_job(job: dict[str, Any]) -> None:
    job_dir = Path(job["job_dir"])
    input_path = Path(job["input_path"])
    edited_map: dict[int, Image.Image] = {}
    frame_log: list[dict[str, Any]] = []
    try:
        _set_status(job, "reading frames (0 of N)", current_frame=0, total_frames=0)

        def on_read(count: int, total: int) -> None:
            _set_status(
                job,
                f"reading frames ({count} of {total or 'N'})",
                current_frame=count,
                total_frames=total,
            )

        meta = probe_and_extract(input_path, job_dir, on_read)
        frame_paths = [
            meta["frames_dir"] / f"frame_{idx:05d}.jpg"
            for idx in range(meta["frame_count"])
        ]
        job.update(total_frames=meta["frame_count"], current_frame=0)

        _set_status(job, "detecting shots (0 of {})".format(len(frame_paths)), progress=10)

        def on_detect_shots(count: int, total: int) -> None:
            _set_status(
                job,
                f"detecting shots ({count} of {total})",
                current_frame=count,
                total_frames=total,
                progress=10 + 10 * count / total,
            )

        shots = detect_shots(frame_paths, progress_callback=on_detect_shots)
        _set_status(job, f"checking frames for skin (0 of {len(frame_paths)})", progress=20)
        skin_flags: list[bool] = []
        for idx, frame_path in enumerate(frame_paths):
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                raise ValueError(f"Unable to read frame: {frame_path}")
            skin_flags.append(has_skin_candidate(frame_bgr))
            _set_status(
                job,
                f"checking frames for skin ({idx + 1} of {len(frame_paths)})",
                current_frame=idx + 1,
                total_frames=len(frame_paths),
                progress=20 + 15 * (idx + 1) / len(frame_paths),
            )
        meta["shots_detected"] = len(shots)

        anchor_by_shot: dict[int, list[int]] = {}
        shot_needs_edit: dict[int, bool] = {}
        for shot_idx, shot in enumerate(shots):
            start, end = shot
            mid_idx = (start + end) // 2
            _set_status(
                job,
                f"checking shot {shot_idx + 1} of {len(shots)} for compliance",
                current_frame=mid_idx + 1,
                total_frames=len(frame_paths),
                progress=35 + 15 * shot_idx / max(len(shots), 1),
            )
            anchors = select_anchors_for_shot(
                shot, frame_paths, skin_flags, stride=10
            )
            _set_status(
                job,
                f"checked shot {shot_idx + 1} of {len(shots)} for compliance",
                current_frame=end + 1,
                total_frames=len(frame_paths),
                progress=35 + 15 * (shot_idx + 1) / max(len(shots), 1),
            )
            anchor_by_shot[shot_idx] = anchors
            shot_needs_edit[shot_idx] = bool(anchors)
            if not any(skin_flags[start : end + 1]):
                for idx in range(start, end + 1):
                    frame_log.append({"frame": idx, "status": "compliant_no_skin"})
            elif not anchors:
                for idx in range(start, end + 1):
                    frame_log.append({"frame": idx, "status": "skipped_compliant"})

        selected_anchor_count = sum(len(indices) for indices in anchor_by_shot.values())
        job["total_anchors"] = selected_anchor_count
        completed_anchors = 0

        for shot_idx, (start, end) in enumerate(shots):
            anchors = set(anchor_by_shot[shot_idx])
            if not shot_needs_edit[shot_idx]:
                continue
            cached_hint: dict[int, tuple[str, str]] | None = None
            anchor_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            forward_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            prev_anchor_idx: int | None = None
            curr_patch_rgb: np.ndarray | None = None
            curr_alpha: np.ndarray | None = None
            prev_gray: np.ndarray | None = None
            prev_bgr: np.ndarray | None = None
            anchor_rgb: np.ndarray | None = None
            anchor_alpha: np.ndarray | None = None
            map_x: np.ndarray | None = None
            map_y: np.ndarray | None = None
            anchor_area = 0.0

            for idx in range(start, end + 1):
                job["current_frame"] = idx + 1
                job["progress"] = max(
                    float(job.get("progress", 0)),
                    50 + 35 * (idx + 1) / max(meta["frame_count"], 1),
                )
                source_image, source_bgr = _read_frame(frame_paths[idx])
                make_anchor = idx in anchors
                drift_detected = False
                if not make_anchor and (
                    curr_patch_rgb is None
                    or curr_alpha is None
                    or prev_gray is None
                    or prev_bgr is None
                    or anchor_rgb is None
                    or anchor_alpha is None
                    or map_x is None
                    or map_y is None
                ):
                    status = (
                        "compliant_no_skin"
                        if not skin_flags[idx]
                        else "not_processed_before_anchor"
                    )
                    frame_log.append({"frame": idx, "status": status})
                    continue

                if not make_anchor:
                    _set_status(job, f"propagating motion across shot {shot_idx + 1}")
                    curr_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
                    map_x, map_y, warped_rgb, warped_alpha, warped_prev_gray = warp_map_step(
                        map_x,
                        map_y,
                        anchor_rgb,
                        anchor_alpha,
                        prev_gray,
                        curr_gray,
                        prev_alpha=curr_alpha,
                    )
                    drift_detected = check_mask_drift(
                        warped_alpha,
                        source_bgr,
                        prev_bgr,
                        max_mse=350.0,
                        reference_area=anchor_area,
                        warped_prev_gray=warped_prev_gray,
                        current_gray=curr_gray,
                    )
                    if not drift_detected:
                        curr_patch_rgb = warped_rgb
                        curr_alpha = warped_alpha
                        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
                        edited_map[idx] = composite_patch(
                            source_rgb, warped_rgb, warped_alpha
                        )
                        forward_cache[idx] = (warped_rgb, warped_alpha)
                        prev_gray = curr_gray
                        prev_bgr = source_bgr
                        frame_log.append({"frame": idx, "status": "propagated"})
                        continue

                    make_anchor = True

                if make_anchor:
                    completed_anchors += 1
                    if drift_detected:
                        job["total_anchors"] += 1
                    _set_status(
                        job,
                        f"covering anchor frame {completed_anchors} of {job['total_anchors']} (frame #{idx})",
                    )
                    edited, alpha, report, cached_hint = process_anchor_frame(
                        source_image,
                        cached_hint=cached_hint,
                        debug_dir=job_dir / "debug" / f"frame_{idx:05d}",
                    )
                    edited_map[idx] = edited
                    edited_rgb = np.asarray(edited.convert("RGB"), dtype=np.uint8)
                    raw_rgb = np.asarray(
                        edited.info.get("raw_edit_rgb", edited_rgb), dtype=np.uint8
                    )
                    alpha_float = _alpha_float(alpha)
                    anchor_data[idx] = (raw_rgb, alpha_float)
                    if prev_anchor_idx is not None:
                        left_rgb, left_alpha = anchor_data[prev_anchor_idx]
                        edited_map.update(
                            interpolate_anchor_span(
                                prev_anchor_idx,
                                idx,
                                frame_paths,
                                left_rgb,
                                left_alpha,
                                raw_rgb,
                                alpha_float,
                                forward_cache,
                            )
                        )
                    prev_anchor_idx = idx
                    anchor_rgb = raw_rgb
                    anchor_alpha = alpha_float.copy()
                    curr_patch_rgb = raw_rgb.copy()
                    curr_alpha = alpha_float.copy()
                    prev_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
                    prev_bgr = source_bgr
                    map_x, map_y = np.meshgrid(
                        np.arange(source_bgr.shape[1], dtype=np.float32),
                        np.arange(source_bgr.shape[0], dtype=np.float32),
                    )
                    anchor_area = float(
                        np.count_nonzero(cv2.GaussianBlur(alpha_float, (3, 3), 0.8) > 0.1)
                    )
                    forward_cache.clear()
                    anchor_log = {
                        "frame": idx,
                        "status": "anchor_edited",
                        "report": report,
                    }
                    if drift_detected:
                        anchor_log["reason"] = "flow_drift"
                    frame_log.append(anchor_log)

        out_frames_dir = job_dir / "processed_frames"
        write_ordered_frames(meta["frame_count"], meta["frames_dir"], edited_map, out_frames_dir)
        result_path = job_dir / "result.mp4"
        _set_status(job, "encoding output video", progress=92)
        encode_video_with_audio(out_frames_dir, input_path, result_path, meta["fps"])
        frame_status_counts: dict[str, int] = {}
        for entry in frame_log:
            status = str(entry.get("status", "unknown"))
            frame_status_counts[status] = frame_status_counts.get(status, 0) + 1
        job.update(
            shots_detected=meta["shots_detected"],
            frame_status_counts=frame_status_counts,
            anchors_edited=sum(entry["status"] == "anchor_edited" for entry in frame_log),
            frames_propagated=sum(entry["status"] == "propagated" for entry in frame_log),
            frames_skipped_compliant=sum(
                entry["status"] in {"compliant_no_skin", "skipped_compliant"}
                for entry in frame_log
            ),
        )
        write_video_manifest(job["job_id"], meta, sorted(frame_log, key=lambda row: row["frame"]), job_dir / "manifest.json")
        job.update(
            state="ready",
            progress=100,
            result_path=str(result_path),
            result_url=f"/api/jobs/{job['job_id']}/result",
        )
        _set_status(job, "ready")
    except Exception as exc:
        logger.exception("Video job %s failed", job["job_id"])
        job.update(state="failed", error=str(exc))
        _set_status(job, f"failed: {exc}")


def _process_image_job(job: dict[str, Any]) -> None:
    job_dir = Path(job["job_dir"])
    input_path = Path(job["input_path"])
    try:
        _set_status(job, "covering uploaded image", current_frame=1, total_frames=1)
        with Image.open(input_path) as source:
            edited, _, report, _ = process_anchor_frame(source.convert("RGB"), debug_dir=job_dir / "debug")
        suffix = input_path.suffix.lower()
        result_path = job_dir / f"result{suffix if suffix in IMAGE_SUFFIXES else '.png'}"
        save_format = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG" if suffix == ".png" else "WEBP"
        edited.save(result_path, format=save_format)
        write_video_manifest(
            job["job_id"],
            {"frame_count": 1, "shots_detected": 1},
            [{"frame": 0, "status": "anchor_edited", "report": report}],
            job_dir / "manifest.json",
        )
        job.update(
            state="ready",
            progress=100,
            result_path=str(result_path),
            result_url=f"/api/jobs/{job['job_id']}/result",
        )
        _set_status(job, "ready")
    except Exception as exc:
        logger.exception("Image job %s failed", job["job_id"])
        job.update(state="failed", error=str(exc))
        _set_status(job, f"failed: {exc}")


def _run_job(job_id: str) -> None:
    job = JOBS.get(job_id)
    if job is None:
        logger.error("Queued job %s no longer exists", job_id)
        return
    job["state"] = "running"
    if Path(job["input_path"]).suffix.lower() in IMAGE_SUFFIXES:
        _process_image_job(job)
    else:
        _process_video_job(job)


@router.get("/health")
@router.get("/api/health")
def health_check() -> dict[str, str]:
    """Return a basic backend health status."""
    return {"status": "ok"}


@router.get("/api/status")
def get_status() -> dict[str, Any]:
    """Return the in-memory job status snapshot."""
    return {"status": "idle" if not JOBS else "active", "jobs": JOBS}


@router.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    consent: bool = Form(False),
) -> dict[str, str]:
    """Save an explicitly consented image or video as a new processing job."""
    filename = file.filename or ""
    try:
        verify_consent(consent, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported media type.")

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = job_dir / f"input{suffix}"
    try:
        with input_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                target.write(chunk)
    except Exception:
        shutil.rmtree(job_dir)
        raise
    finally:
        await file.close()

    JOBS[job_id] = {
        "job_id": job_id,
        "filename": Path(filename).name,
        "state": "uploaded",
        "status_text": "uploaded",
        "progress": 0,
        "current_frame": 0,
        "total_frames": 0,
        "job_dir": str(job_dir),
        "input_path": str(input_path),
        "result_path": None,
        "result_url": None,
    }
    return {"job_id": job_id, "filename": Path(filename).name}


@router.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Queue an uploaded media job for background processing."""
    job = _job_or_404(job_id)
    if job["state"] == "running":
        raise HTTPException(status_code=409, detail="Job is already running.")
    if job["state"] == "ready":
        raise HTTPException(status_code=409, detail="Job has already completed.")
    job.update(state="running", progress=0)
    _set_status(job, "queued")
    background_tasks.add_task(_run_job, job_id)
    return {"job_id": job_id, "state": "running"}


@router.get("/api/jobs/{job_id}/status")
def get_job_status(job_id: str) -> dict[str, Any]:
    """Return one job's current processing state."""
    job = _job_or_404(job_id)
    return {
        "job_id": job_id,
        "state": job["state"],
        "status_text": job["status_text"],
        "current_frame": job["current_frame"],
        "total_frames": job["total_frames"],
        "progress": job["progress"],
        "shots_detected": job.get("shots_detected", 0),
        "anchors_edited": job.get("anchors_edited", 0),
        "frames_propagated": job.get("frames_propagated", 0),
        "frames_skipped_compliant": job.get("frames_skipped_compliant", 0),
        "frame_status_counts": job.get("frame_status_counts", {}),
        "result_url": job.get("result_url"),
        "error": job.get("error"),
    }


@router.get("/api/jobs/{job_id}/result")
def get_result(job_id: str) -> FileResponse:
    """Stream a completed video or return the edited image."""
    job = _job_or_404(job_id)
    result_path = job.get("result_path")
    if job["state"] != "ready" or not result_path or not Path(result_path).is_file():
        raise HTTPException(status_code=409, detail="Job result is not ready.")
    path = Path(result_path)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)
