from pathlib import Path
from types import SimpleNamespace

from backend.output import video_encoder


def test_encode_video_uses_supported_cfr_option(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(video_encoder.subprocess, "run", fake_run)

    video_encoder.encode_video_with_audio(
        tmp_path / "frames",
        tmp_path / "source.mp4",
        tmp_path / "output.mp4",
        24,
    )

    command = captured["command"]
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert "-vsync" not in command
    assert captured["kwargs"] == {"capture_output": True, "text": True, "check": False}
    assert Path(command[-1]) == tmp_path / "output.mp4"
    assert command[command.index("-i") + 1] == str(
        tmp_path / "frames" / "frame_%05d.png"
    )


def test_encode_video_prefers_jpeg_sequence_when_present(tmp_path, monkeypatch):
    captured = {}
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_00000.jpg").touch()
    monkeypatch.setattr(
        video_encoder.subprocess,
        "run",
        lambda command, **kwargs: (
            captured.update(command=command) or SimpleNamespace(returncode=0, stderr="")
        ),
    )

    video_encoder.encode_video_with_audio(
        frames_dir,
        tmp_path / "source.mp4",
        tmp_path / "output.mp4",
        24,
    )

    command = captured["command"]
    assert str(frames_dir / "frame_%05d.jpg") == command[command.index("-i") + 1]
