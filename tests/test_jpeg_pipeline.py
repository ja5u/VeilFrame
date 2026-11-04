import cv2
import numpy as np
from PIL import Image

from backend.ingest import video_loader
from backend.output import frame_writer
from backend.segment import anchor_select


def test_video_extraction_writes_jpeg_frames_without_color_conversion(
    tmp_path, monkeypatch
):
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :, 0] = 200

    class Capture:
        def __init__(self, path):
            self.frames = [frame]

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 24
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 6
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 4
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 1
            raise AssertionError(f"Unexpected capture property: {prop}")

        def read(self):
            if self.frames:
                return True, self.frames.pop()
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(video_loader.cv2, "VideoCapture", Capture)

    meta = video_loader.probe_and_extract(tmp_path / "source.mp4", tmp_path / "job")

    jpeg_path = meta["frames_dir"] / "frame_00000.jpg"
    assert jpeg_path.is_file()
    decoded = cv2.imread(str(jpeg_path))
    assert decoded is not None
    assert decoded.shape == frame.shape
    assert decoded[:, :, 0].mean() > decoded[:, :, 2].mean()


def test_decode_frames_falls_back_to_png(tmp_path, monkeypatch):
    frames_dir = tmp_path / "raw_frames"
    frames_dir.mkdir()
    Image.new("RGB", (2, 2), "red").save(frames_dir / "frame_00000.png")
    meta = {"frames_dir": frames_dir, "frame_count": 1}
    monkeypatch.setattr(video_loader, "probe_and_extract", lambda *args: meta)

    frames = video_loader.decode_frames(str(tmp_path / "source.mp4"))

    assert frames[0].mode == "RGB"
    assert frames[0].getpixel((0, 0)) == (255, 0, 0)


def test_frame_writer_emits_jpeg_and_reads_png_fallback(tmp_path):
    raw_frames = tmp_path / "raw"
    output_frames = tmp_path / "processed"
    raw_frames.mkdir()
    Image.new("RGB", (3, 3), "blue").save(raw_frames / "frame_00000.png")
    edited = Image.new("RGBA", (3, 3), (255, 0, 0, 128))

    frame_writer.write_ordered_frames(
        2,
        raw_frames,
        {1: edited},
        output_frames,
    )

    assert (output_frames / "frame_00000.jpg").is_file()
    assert (output_frames / "frame_00001.jpg").is_file()
    with Image.open(output_frames / "frame_00000.jpg") as fallback_frame:
        assert fallback_frame.format == "JPEG"
    with Image.open(output_frames / "frame_00001.jpg") as edited_frame:
        assert edited_frame.format == "JPEG"
        red, green, blue = edited_frame.convert("RGB").getpixel((1, 1))
        assert red >= 245 and green <= 5 and blue <= 5


def test_anchor_selection_uses_default_stride_and_bounds_last_valid_frame(
    tmp_path,
):
    frame_paths = []
    for idx in range(55):
        path = tmp_path / f"frame_{idx:05d}.jpg"
        assert cv2.imwrite(str(path), np.zeros((2, 2, 3), dtype=np.uint8))
        frame_paths.append(path)
    anchors = anchor_select.select_anchors_for_shot(
        (0, 54),
        frame_paths,
        [True] * len(frame_paths),
    )

    assert anchors == [0, 24, 48, 54]


def test_skin_candidates_are_not_vetoed_by_shot_level_compliance_gate(tmp_path):
    frame_paths = [tmp_path / f"frame_{idx:05d}.jpg" for idx in range(9)]

    anchors = anchor_select.select_anchors_for_shot(
        (0, 8),
        frame_paths,
        [False, True, False, True, False, False, False, False, False],
    )

    assert anchors == [1]
