import importlib
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

FrameSequenceGenerator = importlib.import_module("beatdrop_nodes").FrameSequenceGenerator


def _beats():
    return [
        {"time_seconds": 0.12, "frame_index": 4, "is_drop": True, "is_downbeat": True, "energy_jump": 2.98},
        {"time_seconds": 0.62, "frame_index": 19, "is_drop": False, "is_downbeat": False, "energy_jump": 0.64},
        {"time_seconds": 1.12, "frame_index": 34, "is_drop": True, "is_downbeat": False, "energy_jump": 1.84},
        {"time_seconds": 2.12, "frame_index": 64, "is_drop": False, "is_downbeat": True, "energy_jump": 1.10},
        {"time_seconds": 3.12, "frame_index": 94, "is_drop": False, "is_downbeat": False, "energy_jump": 0.96},
        {"time_seconds": 4.12, "frame_index": 124, "is_drop": True, "is_downbeat": True, "energy_jump": 2.37},
        {"time_seconds": 4.62, "frame_index": 139, "is_drop": True, "is_downbeat": False, "energy_jump": 1.90},
        {"time_seconds": 5.12, "frame_index": 154, "is_drop": False, "is_downbeat": False, "energy_jump": 1.66},
        {"time_seconds": 9.62, "frame_index": 289, "is_drop": False, "is_downbeat": False, "energy_jump": 0.0},
    ]


def test_beats_before_drop_uses_first_downbeat_drop_after_intro_as_anchor():
    selected, anchor = FrameSequenceGenerator._select_beats_for_mode(
        _beats(),
        mode="beats_before_drop",
        anchor_strategy="first_downbeat_drop",
        ignore_start_seconds=1.0,
    )

    assert anchor["time_seconds"] == pytest.approx(4.12)
    assert [beat["time_seconds"] for beat in selected] == [1.12, 2.12, 3.12]
    assert all(beat["selection_mode"] == "beats_before_drop" for beat in selected)
    assert all(beat["anchor_drop_time_seconds"] == pytest.approx(4.12) for beat in selected)


def test_beats_after_drop_excludes_anchor_and_keeps_every_later_beat():
    selected, anchor = FrameSequenceGenerator._select_beats_for_mode(
        _beats(),
        mode="beats_after_drop",
        anchor_strategy="first_downbeat_drop",
        ignore_start_seconds=1.0,
        max_time_seconds=9.611,
    )

    assert anchor["time_seconds"] == pytest.approx(4.12)
    assert [beat["time_seconds"] for beat in selected] == [4.62, 5.12]
    assert all(beat["selection_mode"] == "beats_after_drop" for beat in selected)
    assert all(beat["relative_to_anchor"] == "after" for beat in selected)
