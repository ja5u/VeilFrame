import numpy as np
from PIL import Image

from backend.legacy import app_1


def _region(object_id, box, mask):
    return app_1.Region(
        object_id=object_id,
        box=box,
        confidence=1.0,
        mask=mask.astype(np.uint8),
    )


def test_build_dynamic_targets_skips_small_background_and_caches_person_plans(
    monkeypatch,
):
    image = Image.new("RGB", (100, 100))
    main_person = _region("main", (0, 0, 80, 80), np.ones((80, 80), dtype=bool))
    second_person = _region(
        "second", (15, 0, 75, 60), np.ones((60, 60), dtype=bool)
    )
    background_person = _region(
        "background", (85, 0, 100, 15), np.ones((15, 15), dtype=bool)
    )
    analyzed_sizes = []

    def analyze_person(crop):
        analyzed_sizes.append(crop.size)
        return "upper", f"plan-{crop.width}"

    monkeypatch.setattr(app_1, "analyze_person_with_spark", analyze_person)
    plans = {}

    approved, skipped = app_1.build_dynamic_targets(
        image,
        [background_person, second_person, main_person],
        [],
        [],
        [],
        spark_plan=plans,
    )

    assert len(approved) == 2
    assert plans == {0: ("upper", "plan-80"), 1: ("upper", "plan-60")}
    assert sorted(analyzed_sizes) == [(60, 60), (80, 80)]
    assert [entry["status"] for entry in skipped] == [
        "skipped_background_person"
    ]


def test_upper_zone_uses_chest_fallback_and_clips_canvas_below_upper_body(
    monkeypatch,
):
    image = Image.new("RGB", (80, 80))
    person = _region("person", (0, 0, 80, 80), np.ones((80, 80), dtype=bool))
    clothing_mask = np.zeros((80, 80), dtype=bool)
    clothing_mask[40:80, 10:70] = True
    clothing = _region("clothing", (0, 0, 80, 80), clothing_mask)
    monkeypatch.setattr(
        app_1,
        "analyze_person_with_spark",
        lambda crop: ("upper", "upper plan"),
    )

    approved, skipped = app_1.build_dynamic_targets(
        image,
        [person],
        [clothing],
        [],
        [],
    )

    assert not skipped
    assert len(approved) == 1
    canvas = approved[0][0].mask
    assert canvas[25, 40] == 1
    assert not canvas[50:, :].any()
