"""Spark compliance and tailoring-plan adapters."""

import json
import re

from PIL import Image

from backend.legacy.app_1 import (
    API_BASE,
    HTTP_SESSION,
    SPARK_MODEL,
    _auth_headers,
    analyze_person_with_spark,
    to_data_url,
)


def _response_text(payload: dict) -> str:
    parts = [payload["output_text"]] if isinstance(payload.get("output_text"), str) else []
    for item in payload.get("output") or []:
        if isinstance(item, dict):
            for content in item.get("content") or []:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    parts.append(str(content.get("text", "")))
    return "\n".join(parts).strip()


def check_shot_needs_edit(frame_img: Image.Image) -> bool:
    """Ask Spark once whether a representative shot frame needs garment editing."""
    prompt = (
        "Inspect this video frame for any person's exposed chest, midriff, or hips that should be modestly "
        'covered by their existing clothing. Respond only as JSON: {"breach": true} or {"breach": false}.'
    )
    body = {
        "model": SPARK_MODEL,
        "stream": False,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": to_data_url(frame_img, fmt="JPEG", quality=85)},
                ],
            }
        ],
    }
    response = HTTP_SESSION.post(
        f"{API_BASE}/responses",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json=body,
        timeout=(10, 60),
    )
    response.raise_for_status()
    text = _response_text(response.json())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise RuntimeError("Spark compliance response did not contain a JSON breach field.")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Spark compliance response contained invalid JSON.") from exc
    breach = data.get("breach")
    if not isinstance(breach, bool):
        raise RuntimeError("Spark compliance response omitted a boolean breach field.")
    return breach


def get_person_tailoring_plan(person_crop: Image.Image) -> tuple[str, str]:
    """Return Spark's target zone and tailoring instruction for a person crop."""
    return analyze_person_with_spark(person_crop)
