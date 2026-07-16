import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

module = importlib.import_module("frame_embedding_change_detector")
DINOv2FrameChangeDetector = module.DINOv2FrameChangeDetector


def test_no_visual_change_marks_dino_as_ignored_and_audio_as_authority(monkeypatch):
    detector = DINOv2FrameChangeDetector()
    embeddings = F.normalize(torch.ones(4, 8), p=2, dim=1)
    monkeypatch.setattr(detector, "_compute_embeddings", lambda *args, **kwargs: embeddings)

    result = detector.detect(
        crops=torch.zeros(4, 16, 16, 3),
        mode="auto",
        model_name="dinov2_vitb14",
        device="cpu",
        change_threshold=0.25,
        fps=30.0,
        beat_frame=2,
        beat_time_sec=1.5,
        search_before_frames=2,
        search_after_frames=2,
        alignment_offset_frames=7,
        manual_offset_frames=3,
    )
    payload = json.loads(result[0])

    assert payload["has_existing_visual_change"] is False
    assert payload["dino_used_for_drop_decision"] is False
    assert payload["drop_decision_source"] == "audio_beat"
    assert payload["confidence"] == 0.0
    assert "planned_drop_frame" not in payload
    assert result[1] == 2


def test_visual_change_keeps_dino_as_drop_authority(monkeypatch):
    detector = DINOv2FrameChangeDetector()
    embeddings = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    monkeypatch.setattr(detector, "_compute_embeddings", lambda *args, **kwargs: embeddings)

    result = detector.detect(
        crops=torch.zeros(4, 16, 16, 3),
        mode="auto",
        model_name="dinov2_vitb14",
        device="cpu",
        change_threshold=0.25,
        fps=30.0,
        beat_frame=2,
        beat_time_sec=1.5,
        search_before_frames=2,
        search_after_frames=2,
    )
    payload = json.loads(result[0])

    assert payload["has_existing_visual_change"] is True
    assert payload["dino_used_for_drop_decision"] is True
    assert payload["drop_decision_source"] == "dinov2_visual_change"
    assert result[1] == 2
