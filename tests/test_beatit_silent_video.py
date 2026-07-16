import importlib
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

beatdrop_nodes = importlib.import_module("beatdrop_nodes")
BeatItNode = beatdrop_nodes.BeatItNode


def test_silent_video_returns_empty_beat_analysis(monkeypatch, tmp_path):
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"placeholder")

    def fail_audio_extract(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            args[0],
            stderr=(
                b"Output file does not contain any stream\n"
                b"Error opening output files: Invalid argument\n"
            ),
        )

    monkeypatch.setattr(beatdrop_nodes.subprocess, "run", fail_audio_extract)

    assert BeatItNode().detect(video=str(video), fps=30.0) == ("[]", 0)
