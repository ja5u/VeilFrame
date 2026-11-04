#!/usr/bin/env python3
"""Dynamic Garment-Adjustment Pipeline (Meta API: SAM 3.1 + Spark 1.3 + Muse Image 1.0)."""
from __future__ import annotations

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable

from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import requests
from scipy.ndimage import binary_dilation, label

from meta_sam_parser import (
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)

project_env = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(project_env)
load_dotenv(Path(__file__).resolve().with_name(".env"))

# ---- EDIT THESE LINES ------------------------------------------------------------------------
IMAGE_PATH = r"C:\old stuff\Downloads\files\analysis\frame_000000.png"
PERSON_PHRASE = "person"
CLOTHING_PHRASE = "clothing"
TORSO_PHRASE = "torso"
# -----------------------------------------------------------------------------------------------

API_BASE = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
MUSE_MODEL = "muse-image-1.0"
SPARK_MODEL = "muse-spark-1.3"
HTTP_SESSION = requests.Session()
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Seamlessly extend, stitch, and continue the exact existing garment fabric, color, ribbing, and knit pattern "
    "upward from the current inner neckline/hem to cover the exposed gap. "
    "CRITICAL RULES: "
    "1. NO FLOATING PATCHES: Do not paste a separate piece of clothing; weave and extend the existing shirt fabric upward as a single continuous garment. "
    "2. KEEP OUTER LAYERS UNTOUCHED: Leave open blazers, lapels, jackets, and straps 100% physically unmodified. "
    "3. PRESERVE SKIN ABOVE: Keep all skin from the collarbones upward and outer shoulders completely pristine and untouched."
)

class ApiError(RuntimeError):
    """Any non-success response from the Meta Model API."""


class ContentBlocked(ApiError):
    """Muse Image refused the request (HTTP 400, code content_policy_violation)."""


@dataclass
class Region:
    object_id: str
    box: tuple[int, int, int, int]  # left, top, right, bottom
    confidence: float | None
    mask: np.ndarray  # uint8 0/1, box-local editable canvas (skin + garment band)
    protect: np.ndarray | None = None  # uint8 0/1, box-local protection shield
    person_mask: np.ndarray | None = None  # uint8 0/1, box-local person silhouette
    skin_mask: np.ndarray | None = None  # uint8 0/1, box-local bare skin ONLY (for safe policy retry blur)


# --------------------------------------------------------------------------- HTTP helpers

def _auth_headers() -> dict[str, str]:
    key = (
        os.getenv("MODEL_API_KEY")
        or os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    if not key:
        raise RuntimeError(
            "MODEL_API_KEY, API_KEY, or OPENAI_API_KEY must be set in the environment "
            "or in the .env file next to backend/legacy/app_1.py."
        )
    return {"Authorization": f"Bearer {key}"}


def to_data_url(img: Image.Image, fmt: str = "PNG", quality: int = 90) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- SAM 3.1

def sam_request_events(image_url: str, phrase: str, timeout=(60, 300)) -> list[dict]:
    """Request SSE events from /v1/responses and buffer them before returning."""
    body = {
        "model": SAM_MODEL,
        "stream": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": phrase},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
    }
    headers = {
        **_auth_headers(),
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    resp = requests.post(f"{API_BASE}/responses", headers=headers, json=body, stream=True, timeout=timeout)
    if resp.status_code >= 400:
        raise ApiError(f"SAM request failed for '{phrase}': HTTP {resp.status_code}: {resp.text}")
    events: list[dict] = []
    for raw in resp.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        events.append(json.loads(data))
    return events


async def _aiter(events):
    for event in events:
        yield event


def _raster_to_box_mask(raster: bytes, w: int, h: int, box_w: int, box_h: int) -> np.ndarray:
    arr = np.frombuffer(raster, dtype=np.uint8).reshape(h, w)
    arr = (arr != 0).astype(np.uint8)
    if (w, h) == (box_w, box_h):
        return arr.copy()
    im = Image.fromarray(arr * 255, "L").resize((box_w, box_h), Image.Resampling.BILINEAR)
    return (np.asarray(im) >= 128).astype(np.uint8)


def parse_sam_events(
    events: list[dict],
    log: Callable[[str], None] = logger.info,
    image_size: tuple[int, int] | None = None,
) -> list[Region]:
    for e in events:
        t = e.get("type")
        if t in ("error", "response.failed", "response.refusal.delta", "response.refusal.done"):
            raise ApiError(f"SAM stream reported a problem: {json.dumps(e)[:500]}")

    if not any(e.get("type") == "response.output_text.delta" and e.get("delta") for e in events):
        return []

    async def _run():
        parsed = parse_responses_stream(_aiter(events), image_segmentation_format())
        return await parsed.final_result()

    result = asyncio.run(_run())
    for d in result.diagnostics:
        log(f"  [sam diagnostic] {d.severity}: {d.code}: {d.message} | {d.raw[:120]}")
    if result.outcome.status != "completed":
        raise ApiError(f"SAM stream ended incompletely: {result.outcome}")

    regions: list[Region] = []
    for rec in result.records:
        if rec.kind != "mask":
            continue
        b = rec.bounds
        box = (int(round(b.left)), int(round(b.top)), int(round(b.right)), int(round(b.bottom)))
        if image_size is not None:
            width, height = image_size
            box = (
                max(0, min(width, box[0])),
                max(0, min(height, box[1])),
                max(0, min(width, box[2])),
                max(0, min(height, box[3])),
            )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        raster = decode_mask_to_raster(rec.mask)
        mask = _raster_to_box_mask(raster, rec.mask.width, rec.mask.height, box[2] - box[0], box[3] - box[1])
        regions.append(Region(rec.object_id, box, rec.confidence, mask))
    return regions


def segment(
    image_url: str,
    phrase: str,
    debug_dir: Path | None,
    log: Callable[[str], None] = logger.info,
    image_size: tuple[int, int] | None = None,
) -> list[Region]:
    events = sam_request_events(image_url, phrase)
    if debug_dir is not None:
        raw_text = "".join(e.get("delta", "") for e in events if e.get("type") == "response.output_text.delta")
        tag = re.sub(r"\W+", "_", phrase).strip("_") or "phrase"
        (debug_dir / f"sam_raw_{tag}.txt").write_text(raw_text, encoding="utf-8")
    return parse_sam_events(events, log=log, image_size=image_size)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


# --------------------------------------------------------------------------- Mask & Spark Logic

def dilate_local(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or mask.size == 0 or not mask.any():
        return mask
    im = Image.fromarray(mask * 255, "L").filter(ImageFilter.MaxFilter(2 * r + 1))
    return (np.asarray(im) >= 128).astype(np.uint8)


def union_mask(regions: list[Region], size: tuple[int, int]) -> np.ndarray:
    W, H = size
    full = np.zeros((H, W), dtype=np.uint8)
    for reg in regions:
        l, t, r, b = reg.box
        l_c, t_c = max(0, min(W, l)), max(0, min(H, t))
        r_c, b_c = max(0, min(W, r)), max(0, min(H, b))
        if r_c <= l_c or b_c <= t_c:
            continue
        full[t_c:b_c, l_c:r_c] |= reg.mask[t_c - t : b_c - t, l_c - l : r_c - l]
    return full


def analyze_person_with_spark(person_crop: Image.Image, timeout=(10, 60)) -> tuple[str, str]:
    """Ask Spark 1.3 for ("upper"|"lower"|"both", custom_tailoring_hint) via /v1/responses."""
    prompt_text = (
        "You are an expert fashion retouching director. Inspect the central person in this image crop.\n"
        "Output ONLY valid JSON with two keys:\n"
        '1. "target_zone": Choose "upper" (chest, cleavage, midriff, back, low-cut top/dress), '
        '"lower" (glutes, hips, bikini bottom, high-cut shorts/swimwear), or "both".\n'
        '2. "hint": A concise 1-sentence tailoring instruction naming the exact visible garment color, fabric, and style '
        "to naturally extend, button, or re-drape so the exposed area is modestly covered. "
        "For upper-body edits, explicitly state to raise the neckline/hem to a natural modest chest level while keeping "
        "upper collarbones, neck, bare shoulders, and any outer open blazer/jacket untouched.\n"
        'Respond strictly with raw JSON: {"target_zone": "upper"|"lower"|"both", "hint": "..."}'
    )
    body = {
        "model": SPARK_MODEL,
        "stream": False,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": to_data_url(person_crop, fmt="JPEG", quality=85)},
                ],
            }
        ],
    }
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    try:
        resp = HTTP_SESSION.post(f"{API_BASE}/responses", headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Spark analysis fallback (%s); using default zone='both'.", exc)
        return "both", ""

    text_parts: list[str] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("output_text"), str):
            text_parts.append(payload["output_text"])
        for item in payload.get("output") or []:
            if isinstance(item, dict):
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text_parts.append(part.get("text", ""))

    raw = " ".join(text_parts).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            zone = str(data.get("target_zone", "both")).lower().strip()
            if zone not in {"upper", "lower", "both"}:
                zone = "both"
            hint = str(data.get("hint") or "").strip()
            return zone, hint
        except (ValueError, TypeError):
            pass
    return "both", ""


def build_dynamic_targets(
    img: Image.Image,
    persons: list[Region],
    clothes: list[Region],
    torsos: list[Region],
    extra_exclude: list[Region],
    min_pixels: int = 80,
    log: Callable[[str], None] = logger.info,
    spark_plan: dict[int, tuple[str, str]] | None = None,
) -> tuple[list[tuple[Region, str]], list[dict]]:
    """
    Combines SAM ('person', 'clothing', 'torso') with Spark's dynamic target_zone & hint
    to build a seamless Editable Canvas without boundary-gap trenches.
    """
    from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes

    size = img.size
    cloth_full = union_mask(clothes, size)
    torso_full = union_mask(torsos, size)
    excl_full = union_mask(extra_exclude, size) if extra_exclude else np.zeros((size[1], size[0]), dtype=np.uint8)

    skipped: list[dict] = []
    person_areas = [
        max(0, p.box[2] - p.box[0]) * max(0, p.box[3] - p.box[1])
        for p in persons
    ]
    max_person_area = max(person_areas, default=0)
    foreground_persons = []
    for person, area in zip(persons, person_areas):
        if max_person_area > 0 and area < 0.18 * max_person_area:
            skipped.append(
                {
                    "object_id": person.object_id,
                    "box": list(person.box),
                    "status": "skipped_background_person",
                }
            )
        else:
            foreground_persons.append(person)

    persons_sorted = sorted(foreground_persons, key=lambda p: p.box[0])
    approved: list[tuple[Region, str]] = []

    spark_futures = {}
    with ThreadPoolExecutor() as executor:
        spark_futures = {
            idx: executor.submit(analyze_person_with_spark, img.crop(p.box))
            for idx, p in enumerate(persons_sorted)
            if (p.box[2] - p.box[0]) > 0
            and (p.box[3] - p.box[1]) > 0
            and (spark_plan is None or idx not in spark_plan)
        }

    for idx, p in enumerate(persons_sorted):
        l, t, r, b = p.box
        bh, bw = b - t, r - l
        if bh <= 0 or bw <= 0:
            continue

        if spark_plan is not None and idx in spark_plan:
            target_zone, hint = spark_plan[idx]
        else:
            target_zone, hint = spark_futures[idx].result()
            if spark_plan is not None:
                spark_plan[idx] = (target_zone, hint)
        log(f"  [spark] person {p.object_id}: zone={target_zone} | hint={hint}")

        p_mask = p.mask.astype(bool)
        p_cloth = (cloth_full[t:b, l:r] != 0) & p_mask
        p_torso = (torso_full[t:b, l:r] != 0) & p_mask
        p_excl = (excl_full[t:b, l:r] != 0) & p_mask

        other_persons = [other for j, other in enumerate(persons_sorted) if j != idx]
        others_in_box = (union_mask(other_persons, size)[t:b, l:r] != 0) & (~p_mask)

        has_torso = bool(p_torso.any())
        has_cloth = bool(p_cloth.any())

        # Close 1-5px SAM boundary gaps between torso and clothing so seam pixels are NEVER marked as protected!
        torso_or_cloth_closed = binary_closing(p_torso | p_cloth, iterations=5)
        p_torso_skin = p_torso & (~p_cloth)

        if target_zone in {"upper", "both"}:
            unclothed_person = p_mask & (~p_cloth) & (~p_excl)
            upper_chest_box = np.zeros_like(p_mask)
            y_chest_top, y_chest_bot = round(bh * 0.16), round(bh * 0.52)
            x_chest_left, x_chest_right = round(bw * 0.15), round(bw * 0.85)
            upper_chest_box[
                y_chest_top:y_chest_bot, x_chest_left:x_chest_right
            ] = True
            chest_skin_fallback = unclothed_person & upper_chest_box
            if has_cloth and chest_skin_fallback.any():
                touch_cloth = binary_dilation(p_cloth, iterations=5) & chest_skin_fallback
                lbl_chest, n_chest = label(chest_skin_fallback)
                if n_chest > 0 and touch_cloth.any():
                    valid_c = np.unique(lbl_chest[touch_cloth])
                    valid_c = valid_c[valid_c > 0]
                    chest_skin_fallback = np.isin(lbl_chest, valid_c)
                else:
                    chest_skin_fallback = np.zeros_like(p_mask)
            p_torso_skin |= chest_skin_fallback

        # 1. Limb & Head Protection Shield: Erode away from torso/clothing so no skin trench forms
        if has_torso:
            raw_limb_head = p_mask & (~torso_or_cloth_closed)
            limb_head_shield = binary_erosion(raw_limb_head, iterations=3)
        else:
            limb_head_shield = p_excl.copy()

        # 2. Lower-body skin (connected to lower half of clothing, dropping side hanging hands/arms)
        lower_skin = np.zeros_like(p_mask)
        if has_cloth and (target_zone in {"lower", "both"} or not has_torso):
            ys, xs = np.where(p_cloth)
            if len(ys) > 0:
                y_mid = int(ys.min()) + (int(ys.max()) - int(ys.min())) // 2
                cand_lower = p_mask & (~p_cloth) & (~p_torso) & (~p_excl)
                cand_lower[:y_mid, :] = False

                touching_seed = binary_dilation(p_cloth, iterations=6) & cand_lower
                labeled_arr, num_feats = label(cand_lower)
                if num_feats > 0 and touching_seed.any():
                    valid_ids = np.unique(labeled_arr[touching_seed])
                    valid_ids = valid_ids[valid_ids > 0]
                    lower_skin = np.isin(labeled_arr, valid_ids)

        # 3. Select active bare skin based on Spark's target_zone
        if target_zone == "upper" and has_torso:
            active_skin = p_torso_skin
        elif target_zone == "lower" and lower_skin.any():
            active_skin = lower_skin
        else:
            active_skin = p_torso_skin | lower_skin

        if not active_skin.any():
            active_skin = (p_mask & (~p_cloth) & (~p_excl)) if has_cloth else p_mask

        # 4. Expand primarily vertically into the inner garment hem without bleeding into lapels
        v_band = max(24, round(bh * 0.08))
        h_band = max(6, round(bw * 0.015))
        if has_cloth:
            selem = np.ones((v_band, h_band), dtype=bool)
            dilated_skin = binary_dilation(active_skin, structure=selem)
            active_garment = dilated_skin & p_cloth
        else:
            active_garment = np.zeros_like(p_mask)

        combined_canvas = binary_fill_holes(
            binary_closing(active_skin | active_garment, iterations=4)
        ) & p_mask

        # 5. Build final Protection Shield and Editable Canvas
        if target_zone == "upper" and has_torso:
            protect_bool = limb_head_shield | p_excl | others_in_box
        else:
            protect_bool = (limb_head_shield & (~lower_skin)) | p_excl | others_in_box

        protect_u8 = protect_bool.astype(np.uint8)
        canvas_u8 = (combined_canvas.astype(np.uint8)) & (1 - protect_u8)
        if target_zone == "upper":
            canvas_u8[round(bh * 0.62) :, :] = 0
        skin_u8 = (active_skin.astype(np.uint8)) & (1 - protect_u8)

        area = int(canvas_u8.sum())
        if area < min_pixels:
            skipped.append({
                "object_id": p.object_id,
                "box": list(p.box),
                "exposed_pixels": area,
                "status": "skipped_empty_target",
            })
            continue

        person_mask_dilated = dilate_local(p.mask, 4)
        approved.append((
            Region(
                object_id=p.object_id,
                box=p.box,
                confidence=p.confidence,
                mask=canvas_u8,
                protect=protect_u8,
                person_mask=person_mask_dilated,
                skin_mask=skin_u8,
            ),
            hint,
        ))

    return approved, skipped

# --------------------------------------------------------------------------- Muse Image 1.0

def _obscure_masked_skin(crop: Image.Image, skin_mask: Image.Image) -> Image.Image:
    """Blur ONLY bare skin pixels on policy fallback so surrounding garment folds/textures remain sharp."""
    m_gray = skin_mask.convert("L").resize(crop.size, Image.Resampling.NEAREST)
    heavily_blurred = crop.filter(ImageFilter.GaussianBlur(radius=28))
    soft_alpha = m_gray.filter(ImageFilter.GaussianBlur(radius=6))
    return Image.composite(heavily_blurred, crop, soft_alpha)


def muse_edit(
    crop: Image.Image,
    mask: Image.Image,
    prompt: str,
    skin_mask: Image.Image | None = None,
    retries: int = 3,
    timeout=(10, 300),
) -> Image.Image:
    """POST /v1/images/edits with automatic skin-only blur retry that preserves Spark's custom prompt."""
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    last = "unknown error"
    current_crop = crop
    current_prompt = prompt
    policy_retried = False

    for attempt in range(retries + 1):
        body = {
            "model": MUSE_MODEL,
            "prompt": (
                "The first image is the photo to edit. The second image is a mask: "
                "white marks the editable garment and skin region, black must stay untouched. "
                + current_prompt
            ),
            "images": [
                {"image_url": to_data_url(current_crop, fmt="JPEG", quality=92)},
                {"image_url": to_data_url(mask.convert("RGB"), fmt="PNG")},
            ],
            "n": 1,
            "output_format": "png",
            "size": f"{crop.width}x{crop.height}",
            "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False},
        }
        try:
            r = HTTP_SESSION.post(f"{API_BASE}/images/edits", headers=headers, json=body, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = f"network error: {exc}"
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 503:
            last = f"HTTP 503: {r.text[:300]}"
            time.sleep(2 ** attempt)
            continue
        if r.status_code >= 400:
            if r.status_code == 400:
                try:
                    payload = r.json()
                except ValueError:
                    payload = None
                error = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(error, dict) and error.get("code") == "content_policy_violation":
                    if not policy_retried:
                        policy_retried = True
                        logger.info("  [muse] policy filter triggered; blurring bare skin ONLY and keeping Spark prompt ...")
                        blur_target = skin_mask if skin_mask is not None else mask
                        current_crop = _obscure_masked_skin(crop, blur_target)
                        continue
                    raise ContentBlocked(r.text)
            raise ApiError(f"Muse Image edit failed: HTTP {r.status_code}: {r.text}")
        payload = r.json()
        try:
            b64 = payload["data"][0]["b64_json"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiError(f"Unexpected Muse Image response shape: {json.dumps(payload)[:500]}") from exc
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    raise ApiError(f"Muse Image edit failed after {retries + 1} attempts: {last}")


# --------------------------------------------------------------------------- Compositing

def _grow(lo: int, hi: int, limit: int, minlen: int) -> tuple[int, int]:
    need = minlen - (hi - lo)
    if need <= 0 or limit < minlen:
        return lo, hi
    lo2 = max(0, lo - need // 2)
    hi2 = min(limit, hi + (need - need // 2))
    short = minlen - (hi2 - lo2)
    if short > 0:
        if lo2 == 0:
            hi2 = min(limit, hi2 + short)
        else:
            lo2 = max(0, lo2 - short)
    return lo2, hi2


def fit_to(img: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, bool]:
    tw, th = size
    w, h = img.size
    target = tw / th
    adjusted = False
    if abs(w / h - target) / target > 0.03:
        adjusted = True
        if w / h > target:
            nw = max(1, round(h * target))
            x0 = (w - nw) // 2
            img = img.crop((x0, 0, x0 + nw, h))
        else:
            nh = max(1, round(w / target))
            y0 = (h - nh) // 2
            img = img.crop((0, y0, w, y0 + nh))
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img, adjusted


def clothe_image(
    img: Image.Image,
    approved_regions: list[tuple[Region, str]],
    edit_fn: Callable[..., Image.Image],
    prompt: str,
    *,
    pad: float = 0.15,
    dilate: int = 2,
    feather: float = 4.0,
    min_side: int = 128,
    max_crop: int = 1024,
    on_block: str = "fail",
    debug_dir: Path | None = None,
    log: Callable[[str], None] = logger.info,
    alpha_mask_out: np.ndarray | None = None,
    raw_edit_out: np.ndarray | None = None,
) -> tuple[Image.Image, list[dict]]:
    W, H = img.size
    result = img.copy()
    report: list[dict] = []
    if raw_edit_out is not None:
        if raw_edit_out.shape != (H, W, 3):
            raise ValueError(f"raw_edit_out must have shape {(H, W, 3)}")
        raw_edit_out[:] = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if alpha_mask_out is not None:
        if alpha_mask_out.shape != (H, W):
            raise ValueError(f"alpha_mask_out must have shape {(H, W)}, got {alpha_mask_out.shape}")
        alpha_mask_out.fill(0)

    if not approved_regions:
        return result, report

    jobs = []
    for i, (reg, hint) in enumerate(approved_regions):
        l, t, r, b = reg.box
        eff_dilate = max(dilate, round(min(r - l, b - t) * 0.003))
        eff_feather = max(feather, 8.0, round(min(r - l, b - t) * 0.015))
        margin = eff_dilate + int(math.ceil(3 * eff_feather)) + 2
        pw, ph = max(int((r - l) * pad), margin), max(int((b - t) * pad), margin)
        cl, cr = _grow(max(0, l - pw), min(W, r + pw), W, min_side)
        ct, cb = _grow(max(0, t - ph), min(H, b + ph), H, min_side)

        crop = result.crop((cl, ct, cr, cb))
        job_prompt = f"{prompt} SPECIFIC GARMENT GUIDANCE: {hint}" if hint else prompt
        entry = {
            "object_id": reg.object_id,
            "confidence": reg.confidence,
            "box": list(reg.box),
            "crop_box": [cl, ct, cr, cb],
            "exposed_pixels": int(reg.mask.sum()),
            "spark_hint": hint,
            "status": "covered",
            "aspect_adjusted": False,
        }
        if debug_dir is not None:
            crop.save(debug_dir / f"person{i}_crop_in.png")

        send = crop
        if max(crop.size) > max_crop:
            s = max_crop / max(crop.size)
            send = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.Resampling.LANCZOS)

        win = np.zeros((crop.height, crop.width), dtype=np.uint8)
        win[t - ct : b - ct, l - cl : r - cl] = reg.mask

        skin_win = np.zeros((crop.height, crop.width), dtype=np.uint8)
        if reg.skin_mask is not None:
            skin_win[t - ct : b - ct, l - cl : r - cl] = reg.skin_mask
        else:
            skin_win = win.copy()

        if reg.protect is not None:
            prot_win = np.zeros((crop.height, crop.width), dtype=np.uint8)
            prot_win[t - ct : b - ct, l - cl : r - cl] = reg.protect
            win &= (1 - prot_win)
            skin_win &= (1 - prot_win)

        send_mask = Image.fromarray(win * 255, "L")
        send_skin_mask = Image.fromarray(skin_win * 255, "L")
        if eff_dilate > 0:
            send_mask = send_mask.filter(ImageFilter.MaxFilter(2 * eff_dilate + 1))

        mask_img = send_mask.convert("RGB")
        skin_mask_img = send_skin_mask.convert("RGB")
        if mask_img.size != send.size:
            mask_img = mask_img.resize(send.size, Image.Resampling.NEAREST)
            skin_mask_img = skin_mask_img.resize(send.size, Image.Resampling.NEAREST)
        if debug_dir is not None:
            send_mask.save(debug_dir / f"person{i}_mask_sent.png")

        jobs.append((i, reg, (cl, ct, cr, cb), crop, send, mask_img, skin_mask_img, eff_dilate, eff_feather, job_prompt, entry))

    with ThreadPoolExecutor() as executor:
        futures = [
            (executor.submit(edit_fn, send, mask_img, job_prompt, skin_mask_img), time.time())
            for _, _, _, _, send, mask_img, skin_mask_img, _, _, job_prompt, _ in jobs
        ]
        edited_images: list[Image.Image | None] = []
        for (future, started_at), (_, reg, _, _, _, _, _, _, _, _, entry) in zip(futures, jobs):
            try:
                edited = future.result()
            except ContentBlocked as exc:
                if on_block != "skip":
                    raise
                entry["status"] = "NOT_COVERED_blocked"
                entry["detail"] = str(exc)[:500]
                log(f"  !! person {reg.object_id} was BLOCKED by Muse Image and is NOT covered (--on-block skip)")
                edited_images.append(None)
            else:
                entry["edit_seconds"] = round(time.time() - started_at, 2)
                edited_images.append(edited)

    for (i, reg, (cl, ct, cr, cb), crop, _, _, _, eff_dilate, eff_feather, _, entry), edited in zip(jobs, edited_images):
        if edited is None:
            report.append(entry)
            continue
        edited, adjusted = fit_to(edited, crop.size)
        entry["aspect_adjusted"] = adjusted
        if debug_dir is not None:
            edited.save(debug_dir / f"person{i}_crop_out.png")

        cw, ch = crop.size
        l, t, r, b = reg.box
        win = np.zeros((ch, cw), dtype=np.uint8)
        win[t - ct : b - ct, l - cl : r - cl] = reg.mask
        prot = None
        if reg.protect is not None:
            prot = np.zeros((ch, cw), dtype=np.uint8)
            prot[t - ct : b - ct, l - cl : r - cl] = reg.protect
            win &= 1 - prot
        mask_pil = Image.fromarray(win * 255, "L")
        if eff_dilate > 0:
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(2 * eff_dilate + 1))
        if eff_feather > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(eff_feather))

        alpha = np.asarray(mask_pil, dtype=np.float32) / 255.0
        if prot is not None:
            prot_smooth = np.asarray(
                Image.fromarray(prot * 255, "L").filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32
            ) / 255.0
            alpha *= (1.0 - prot_smooth)

        if reg.person_mask is not None:
            p_win = np.zeros((ch, cw), dtype=np.uint8)
            p_win[t - ct : b - ct, l - cl : r - cl] = reg.person_mask
            p_smooth = np.asarray(
                Image.fromarray(p_win * 255, "L").filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32
            ) / 255.0
            alpha *= p_smooth

        alpha[alpha < 0.03] = 0.0
        if alpha_mask_out is not None:
            region_alpha = alpha[t - ct : b - ct, l - cl : r - cl]
            alpha_mask_out[t:b, l:r] = np.maximum(alpha_mask_out[t:b, l:r], region_alpha)

        if raw_edit_out is not None:
            new_u8 = np.clip(np.asarray(edited, dtype=np.float32) + 0.5, 0, 255).astype(np.uint8)
            raw_crop = raw_edit_out[t:b, l:r]
            active_region = region_alpha[..., None] > 0.01
            raw_edit_out[t:b, l:r] = np.where(
                active_region,
                new_u8[t - ct : b - ct, l - cl : r - cl],
                raw_crop,
            )

        alpha = alpha[..., None]
        base = np.asarray(result.crop((cl, ct, cr, cb)), dtype=np.float32)
        new = np.asarray(edited, dtype=np.float32)
        out = np.clip(base * (1.0 - alpha) + new * alpha + 0.5, 0, 255).astype(np.uint8)
        result.paste(Image.fromarray(out, "RGB"), (cl, ct))
        report.append(entry)

    return result, report


def draw_overlay(img: Image.Image, persons: list[Region], targets: list[Region]) -> Image.Image:
    W, H = img.size
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    for reg in targets:
        l, t, r, b = reg.box
        if reg.protect is not None:
            arr[t:b, l:r][reg.protect == 1] = (0, 120, 255, 110)
    for reg in targets:
        l, t, r, b = reg.box
        arr[t:b, l:r][reg.mask == 1] = (255, 0, 0, 120)
    out = Image.alpha_composite(img.convert("RGBA"), Image.fromarray(arr, "RGBA"))
    d = ImageDraw.Draw(out)
    covered = {t.object_id for t in targets}
    for reg in persons:
        l, t, r, b = reg.box
        color = (0, 255, 0, 255) if reg.object_id in covered else (160, 160, 160, 255)
        d.rectangle((l, t, r - 1, b - 1), outline=color, width=3)
        lbl = f"person {reg.object_id}" + ("" if reg.object_id in covered else " (nothing exposed)")
        d.text((l + 4, t + 4), lbl, fill=(255, 255, 0, 255))
    return out.convert("RGB")


# --------------------------------------------------------------------------- CLI

def load_image(path: str, max_side: int, min_side: int = 1024) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    longest = max(img.size)
    if max_side and longest > max_side:
        scale = max_side / longest
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    elif min_side and longest < min_side:
        scale = min_side / longest
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    return img


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", nargs="?", default=IMAGE_PATH, help=f"image path (default: {IMAGE_PATH})")
    p.add_argument("--person", default=PERSON_PHRASE, help="SAM phrase for people")
    p.add_argument("--exclude", action="append", help="optional extra SAM phrases to protect")
    p.add_argument("--out", help="output path (default: <image>_clothed.png)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="base edit instruction sent to Muse Image")
    p.add_argument("--min-conf", type=float, default=0.0, help="drop SAM clothing regions below this confidence")
    p.add_argument("--min-pixels", type=int, default=80, help="ignore a person with fewer target pixels than this")
    p.add_argument("--max-side", type=int, default=2048, help="downscale so the longest side <= this (0 = off)")
    p.add_argument("--pad", type=float, default=0.15, help="crop padding as a fraction of the person box")
    p.add_argument("--dilate", type=int, default=2, help="grow the mask by N pixels before blending")
    p.add_argument("--feather", type=float, default=4.0, help="gaussian blur radius for the blend edge")
    p.add_argument("--on-block", choices=["fail", "skip"], default="fail",
                   help="when Muse Image blocks a person: fail or skip")
    p.add_argument("--debug-dir", help="save overlay, masks, raw SAM output, crops and manifest here")
    p.add_argument("--dry-run", action="store_true", help="only run SAM + Spark and save the overlay")
    args = p.parse_args(argv)

    extra_exclude_phrases = dedupe_preserve_order(args.exclude or [])

    if not Path(args.image).is_file():
        logger.error("Image not found: %s", args.image)
        return 1

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        img = load_image(args.image, args.max_side)
        url = to_data_url(img, fmt="JPEG")
        sam_phrases = dedupe_preserve_order([args.person, CLOTHING_PHRASE, TORSO_PHRASE] + extra_exclude_phrases)
        logger.info("image %sx%s; firing %s parallel SAM call(s): %s ...",
                    img.width, img.height, len(sam_phrases), sam_phrases)

        with ThreadPoolExecutor() as executor:
            futures = {
                phrase: executor.submit(segment, url, phrase, debug_dir, image_size=img.size)
                for phrase in sam_phrases
            }
            sam_results = {phrase: future.result() for phrase, future in futures.items()}

        persons = sam_results[args.person]
        logger.info("  '%s': %s found", args.person, len(persons))
        if not persons:
            logger.error("SAM found no people.")
            return 2

        clothes = sam_results[CLOTHING_PHRASE]
        if args.min_conf > 0:
            clothes = [r for r in clothes if r.confidence is None or r.confidence >= args.min_conf]
        logger.info("  '%s': %s found", CLOTHING_PHRASE, len(clothes))

        torsos = sam_results[TORSO_PHRASE]
        logger.info("  '%s': %s found", TORSO_PHRASE, len(torsos))

        extra_exclude: list[Region] = []
        for phrase in extra_exclude_phrases:
            extra_exclude.extend(sam_results[phrase])

        approved_regions, skipped = build_dynamic_targets(
            img, persons, clothes, torsos, extra_exclude, min_pixels=args.min_pixels
        )
        targets = [reg for reg, _ in approved_regions]

        for reg, hint in approved_regions:
            logger.info("  person %s: %s editable canvas px | hint: %s", reg.object_id, int(reg.mask.sum()), hint)

        if debug_dir:
            draw_overlay(img, persons, targets).save(debug_dir / "overlay.png")
            Image.fromarray(union_mask(clothes, img.size) * 255, "L").save(debug_dir / "clothing_mask.png")
            Image.fromarray(union_mask(torsos, img.size) * 255, "L").save(debug_dir / "torso_mask.png")

        if not approved_regions:
            logger.error("No target regions built.")
            return 2
        if args.dry_run:
            logger.info("dry run: stopping before Muse Image.")
            return 0

        out_path = Path(args.out) if args.out else Path(args.image).with_name(Path(args.image).stem + "_clothed.png")
        result, report = clothe_image(
            img, approved_regions, muse_edit, args.prompt,
            pad=args.pad, dilate=args.dilate, feather=args.feather,
            on_block=args.on_block, debug_dir=debug_dir,
        )
        result.save(out_path)
        manifest = {
            "input": args.image,
            "output": str(out_path),
            "sam_phrases": sam_phrases,
            "people": report + skipped,
        }
        out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if debug_dir:
            (debug_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("saved %s", out_path)
        return 0
    except ContentBlocked as exc:
        logger.error("Muse Image blocked a person (content policy).\n%s", exc)
        return 3
    except ApiError as exc:
        logger.error("API error: %s", exc)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())