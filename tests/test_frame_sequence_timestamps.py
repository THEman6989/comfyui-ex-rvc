import importlib
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

beatdrop_nodes = importlib.import_module("beatdrop_nodes")
FrameSequenceGenerator = beatdrop_nodes.FrameSequenceGenerator


def test_video_timestamp_mapping_preserves_fractional_seek_origin():
    frames = FrameSequenceGenerator._frame_timestamps(
        range(8, 10),
        batch_offset=0,
        fps=30.0,
        start_time_seconds=0.25,
    )

    assert frames == [
        {"batch_index": 0, "source_frame_index": 8, "time_seconds": 0.25},
        {"batch_index": 1, "source_frame_index": 9, "time_seconds": 0.283333},
    ]


def test_image_mode_records_exact_timestamp_for_every_selected_frame():
    images = torch.zeros((6, 4, 4, 3), dtype=torch.float32)
    beats = json.dumps(
        [
            {
                "time_seconds": 1.0,
                "frame_index": 2,
                "is_drop": True,
                "energy_jump": 0.9,
            }
        ]
    )

    output = FrameSequenceGenerator().generate(
        video="",
        base_url="http://localhost:8188",
        filename_prefix="test",
        fps=2.0,
        window_seconds=0.5,
        images=images,
        beats_json=beats,
    )

    assert output[3] == 3
    windows = json.loads(output[4])
    assert windows[0]["batch_offset"] == 0
    assert windows[0]["batch_frame_count"] == 3
    assert windows[0]["frames"] == [
        {"batch_index": 0, "source_frame_index": 1, "time_seconds": 0.5},
        {"batch_index": 1, "source_frame_index": 2, "time_seconds": 1.0},
        {"batch_index": 2, "source_frame_index": 3, "time_seconds": 1.5},
    ]


def test_image_mode_records_manual_fallback_window_without_beats():
    images = torch.zeros((6, 4, 4, 3), dtype=torch.float32)

    output = FrameSequenceGenerator().generate(
        video="",
        base_url="http://localhost:8188",
        filename_prefix="test",
        fps=2.0,
        window_seconds=0.5,
        start_seconds=0.0,
        end_seconds=2.0,
        images=images,
        beats_json="[]",
    )

    windows = json.loads(output[4])
    assert output[3] == 5
    assert len(windows) == 1
    assert windows[0]["fallback"] == "manual_range"
    assert windows[0]["range_start"] == 0.0
    assert windows[0]["range_end"] == 2.0
    assert windows[0]["batch_offset"] == 0
    assert windows[0]["batch_frame_count"] == 5
    assert windows[0]["frames"][0] == {
        "batch_index": 0,
        "source_frame_index": 0,
        "time_seconds": 0.0,
    }
    assert windows[0]["frames"][-1] == {
        "batch_index": 4,
        "source_frame_index": 4,
        "time_seconds": 2.0,
    }
