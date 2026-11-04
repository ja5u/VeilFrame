import cv2
import numpy as np

from backend.segment import drift_check, flow_propagate


def test_alpha_float_scales_and_suppresses_ghost_edges():
    alpha = np.array([[0, 5, 128, 255]], dtype=np.uint8)

    result = flow_propagate._alpha_float(alpha)

    assert result.dtype == np.float32
    assert np.allclose(result, [[0.0, 0.0, 128 / 255, 1.0]])
    assert alpha.tolist() == [[0, 5, 128, 255]]


def test_warp_step_computes_bidirectional_flow_for_patch_and_drift_image(monkeypatch):
    height, width = 8, 8
    patch = np.full((height, width, 3), 100, dtype=np.uint8)
    alpha = np.ones((height, width), dtype=np.float32)
    gray = np.full((height, width), 80, dtype=np.uint8)
    flow_calls = []

    def zero_flow(prev_gray, curr_gray):
        flow_calls.append((prev_gray, curr_gray))
        return np.zeros((height, width, 2), dtype=np.float32)

    monkeypatch.setattr(flow_propagate, "compute_backward_flow", zero_flow)

    warped_rgb, warped_alpha, warped_prev_gray = flow_propagate.warp_step(
        patch, alpha, gray, gray
    )

    assert len(flow_calls) == 2
    assert np.array_equal(warped_rgb, patch)
    assert np.array_equal(warped_prev_gray, gray)
    assert np.all(warped_alpha[:-1, :-1] == 1)
    assert np.all(warped_alpha[-1, :] == 0)


def test_warp_step_drops_forward_backward_inconsistent_alpha(monkeypatch):
    height, width = 32, 32
    patch = np.full((height, width, 3), 100, dtype=np.uint8)
    alpha = np.ones((height, width), dtype=np.float32)
    gray = np.full((height, width), 80, dtype=np.uint8)
    flow_calls = 0

    def inconsistent_flow(prev_gray, curr_gray):
        nonlocal flow_calls
        flow_calls += 1
        flow = np.zeros((height, width, 2), dtype=np.float32)
        if flow_calls == 2:
            flow[:, :, 0] = 8.0
        return flow

    monkeypatch.setattr(flow_propagate, "compute_backward_flow", inconsistent_flow)

    _, warped_alpha, _ = flow_propagate.warp_step(patch, alpha, gray, gray)

    assert flow_calls == 2
    assert np.all(warped_alpha == 0)


def test_keep_connected_alpha_discards_detached_components():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[5:10, 5:10] = 1.0
    alpha[22:26, 22:26] = 1.0
    previous_alpha = np.zeros_like(alpha)
    previous_alpha[5:10, 5:10] = 1.0
    map_x, map_y = np.meshgrid(
        np.arange(32, dtype=np.float32), np.arange(32, dtype=np.float32)
    )

    result = flow_propagate._keep_connected_alpha(
        alpha.copy(), previous_alpha, map_x, map_y
    )

    assert result[7, 7] == 1.0
    assert result[24, 24] == 0.0


def test_warp_map_step_samples_from_anchor_and_smooths_alpha(monkeypatch):
    height, width = 8, 8
    anchor_rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(
        height, width, 3
    )
    anchor_alpha = np.ones((height, width), dtype=np.float32)
    previous_alpha = np.full((height, width), 0.5, dtype=np.float32)
    gray = np.full((height, width), 80, dtype=np.uint8)
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    monkeypatch.setattr(
        flow_propagate,
        "compute_backward_flow",
        lambda prev, curr: np.zeros((height, width, 2), dtype=np.float32),
    )

    next_x, next_y, warped_rgb, warped_alpha, warped_prev_gray = (
        flow_propagate.warp_map_step(
            map_x,
            map_y,
            anchor_rgb,
            anchor_alpha,
            gray,
            gray,
            prev_alpha=previous_alpha,
        )
    )

    assert np.array_equal(next_x, map_x)
    assert np.array_equal(next_y, map_y)
    assert np.array_equal(warped_rgb, anchor_rgb)
    assert np.array_equal(warped_prev_gray, gray)
    assert np.allclose(warped_alpha[:-1, :-1], 0.85)
    assert np.all(warped_alpha[-1, :] == 0)


def test_compute_backward_flow_scales_half_resolution_flow_to_frame_size(monkeypatch):
    shapes = []

    def zero_flow(curr, prev, *args):
        shapes.append((curr.shape, prev.shape))
        return np.zeros((*curr.shape, 2), dtype=np.float32)

    monkeypatch.setattr(flow_propagate.cv2, "calcOpticalFlowFarneback", zero_flow)
    gray = np.zeros((9, 11), dtype=np.uint8)

    flow = flow_propagate.compute_backward_flow(gray, gray)

    assert shapes == [((4, 5), (4, 5))]
    assert flow.shape == (9, 11, 2)


def test_compute_backward_flow_uses_wider_window_and_smooths_flow(monkeypatch):
    observed_args = []

    def noisy_flow(curr, prev, initial, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags):
        observed_args.append(
            (pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        )
        field = np.zeros((*curr.shape, 2), dtype=np.float32)
        field[4, 4, 0] = 10
        return field

    monkeypatch.setattr(flow_propagate.cv2, "calcOpticalFlowFarneback", noisy_flow)
    gray = np.zeros((16, 16), dtype=np.uint8)

    flow = flow_propagate.compute_backward_flow(gray, gray)

    assert observed_args == [(0.5, 3, 31, 3, 7, 1.5, 0)]
    assert flow.shape == (16, 16, 2)
    assert 0 < flow[:, :, 0].max() < 20


def test_drift_check_skips_optical_flow_when_warped_gray_is_supplied(monkeypatch):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    alpha = np.zeros((8, 8), dtype=np.float32)
    alpha[2:6, 2:6] = 1
    gray = np.zeros((8, 8), dtype=np.uint8)

    def unexpected_flow(*args, **kwargs):
        raise AssertionError("drift check must reuse the existing flow")

    monkeypatch.setattr(drift_check.cv2, "calcOpticalFlowFarneback", unexpected_flow)

    assert not drift_check.check_mask_drift(
        alpha,
        frame,
        frame,
        warped_prev_gray=gray,
        current_gray=gray,
    )


def test_drift_check_requires_boundary_mismatch_or_large_area_loss():
    frame = np.full((32, 32, 3), 40, dtype=np.uint8)
    gray = np.zeros((32, 32), dtype=np.uint8)
    current_gray = np.full((32, 32), 40, dtype=np.uint8)
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 1
    smoothed_area = np.count_nonzero(cv2.GaussianBlur(alpha, (3, 3), 0.8) > 0.1)

    assert not drift_check.check_mask_drift(
        alpha,
        frame,
        frame,
        reference_area=float(smoothed_area),
        warped_prev_gray=gray,
        current_gray=gray,
    )
    assert drift_check.check_mask_drift(
        alpha,
        frame,
        frame,
        max_mse=1,
        reference_area=float(smoothed_area) / 1.2,
        warped_prev_gray=gray,
        current_gray=current_gray,
    )


def test_drift_check_triggers_on_boundary_mse_without_area_change():
    frame = np.full((32, 32, 3), 40, dtype=np.uint8)
    previous_gray = np.zeros((32, 32), dtype=np.uint8)
    current_gray = np.full((32, 32), 40, dtype=np.uint8)
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 1
    reference_area = np.count_nonzero(
        cv2.GaussianBlur(alpha, (3, 3), 0.8) > 0.1
    )

    assert drift_check.check_mask_drift(
        alpha,
        frame,
        frame,
        max_mse=350.0,
        reference_area=float(reference_area),
        warped_prev_gray=previous_gray,
        current_gray=current_gray,
    )


def test_interpolate_anchor_span_cosine_blends_forward_and_backward_patches(
    tmp_path, monkeypatch
):
    height, width = 8, 8
    frame_paths = []
    for idx in range(3):
        path = tmp_path / f"frame_{idx:05d}.png"
        assert cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))
        frame_paths.append(path)

    left_rgb = np.full((height, width, 3), (255, 0, 0), dtype=np.uint8)
    right_rgb = np.full((height, width, 3), (0, 0, 255), dtype=np.uint8)
    alpha = np.ones((height, width), dtype=np.float32)
    monkeypatch.setattr(
        flow_propagate,
        "compute_backward_flow",
        lambda prev, curr: np.zeros((height, width, 2), dtype=np.float32),
    )

    result = flow_propagate.interpolate_anchor_span(
        0,
        2,
        frame_paths,
        left_rgb,
        alpha,
        right_rgb,
        alpha,
        {1: (left_rgb, alpha)},
    )

    assert list(result) == [1]
    assert result[1].getpixel((3, 3)) == (128, 0, 128)
