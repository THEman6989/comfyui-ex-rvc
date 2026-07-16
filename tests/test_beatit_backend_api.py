import importlib
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

beatdrop_nodes = importlib.import_module("beatdrop_nodes")
BeatItNode = beatdrop_nodes.BeatItNode


def test_beatthis_uses_installed_v1_inference_api(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")
    calls = {}

    class FakeFile2Beats:
        def __init__(self, checkpoint_path="final0", device="cpu", float16=False, dbn=False):
            calls["init"] = {
                "checkpoint_path": checkpoint_path,
                "device": str(device),
                "float16": bool(float16),
                "dbn": bool(dbn),
            }

        def __call__(self, path):
            calls["path"] = str(path)
            return np.array([1.0, 2.0]), np.array([2.0])

    monkeypatch.setattr("beat_this.inference.File2Beats", FakeFile2Beats)
    monkeypatch.setattr(
        "beat_this.preprocessing.load_audio",
        lambda path: (np.ones(22050 * 3, dtype=np.float32) * 0.1, 22050),
    )
    monkeypatch.setattr(beatdrop_nodes.torch.cuda, "is_available", lambda: False)

    beats = BeatItNode()._detect_with_beatthis(str(audio_path), fps=30.0)

    assert calls["init"]["checkpoint_path"] == "final0"
    assert calls["init"]["device"] == "cpu"
    assert calls["path"] == str(audio_path)
    assert [entry["method"] for entry in beats] == ["beat_this", "beat_this"]
    assert all(0.0 <= entry["drop_confidence"] <= 1.0 for entry in beats)
    assert [entry["frame_index"] for entry in beats] == [30, 60]
    assert beats[1]["is_downbeat"] is True


def test_detect_does_not_truncate_late_beats_when_max_beats_allows_them(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")
    expected = [
        {
            "time_seconds": round(index * 0.5, 3),
            "frame_index": round(index * 15),
            "confidence": 0.8,
            "is_drop": False,
            "drop_confidence": 0.0,
            "is_downbeat": False,
            "energy_jump": 1.0,
            "method": "beat_this",
        }
        for index in range(20)
    ]
    node = BeatItNode()
    monkeypatch.setattr(node, "_detect_with_beatthis", lambda path, fps: expected)

    beats_json, beat_count = node.detect(
        audio_path=str(audio_path),
        fps=30.0,
        max_beats=64,
    )

    assert beat_count == 20
    assert len(__import__("json").loads(beats_json)) == 20
