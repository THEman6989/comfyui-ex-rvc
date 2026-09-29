import importlib
import json
import sys
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

beatdrop_nodes = importlib.import_module("beatdrop_nodes")


def _images(count: int, *, dtype=torch.float32) -> torch.Tensor:
    return torch.arange(count * 2 * 3 * 3, dtype=dtype).reshape(count, 2, 3, 3)


def test_finalizer_without_reset_preserves_visible_frames_and_emits_motion_tail():
    images = _images(8)

    final_images, continuation, masks, metadata_json = (
        beatdrop_nodes.BeatDropWanSegmentFinalizerNode().finalize(
            images=images,
            expected_visible_frames=6,
            motion_tail_frames=3,
            append_black_frame=False,
            black_frame_count=1,
            run_id="run-1",
            attempt_id="attempt-1",
            plan_hash="hash-1",
            segment_index=0,
        )
    )

    assert torch.equal(final_images, images[:6])
    assert torch.equal(continuation, images[3:6])
    assert masks.shape == (3, 2, 3)
    assert masks.dtype == images.dtype
    assert masks.device == images.device
    assert torch.count_nonzero(masks) == 0
    assert json.loads(metadata_json) == {
        "attempt_id": "attempt-1",
        "continuation_frame_count": 3,
        "motion_tail_frame_count": 3,
        "plan_hash": "hash-1",
        "reset_frame_count": 0,
        "run_id": "run-1",
        "segment_index": 0,
        "visible_frame_count": 6,
    }


def test_finalizer_appends_black_reset_to_output_and_continuation():
    images = _images(6, dtype=torch.float64)

    final_images, continuation, masks, metadata_json = (
        beatdrop_nodes.BeatDropWanSegmentFinalizerNode().finalize(
            images=images,
            expected_visible_frames=5,
            motion_tail_frames=2,
            append_black_frame=True,
            black_frame_count=1,
            run_id="run-2",
            attempt_id="attempt-2",
            plan_hash="hash-2",
            segment_index=1,
        )
    )

    assert final_images.shape == (6, 2, 3, 3)
    assert final_images.dtype == images.dtype
    assert final_images.device == images.device
    assert torch.equal(final_images[:5], images[:5])
    assert torch.count_nonzero(final_images[-1]) == 0
    assert torch.equal(continuation[:2], images[3:5])
    assert torch.count_nonzero(continuation[-1]) == 0
    assert masks.shape == (3, 2, 3)
    assert masks.dtype == images.dtype
    assert masks.device == images.device
    assert torch.count_nonzero(masks) == 0
    metadata = json.loads(metadata_json)
    assert metadata["visible_frame_count"] == 5
    assert metadata["reset_frame_count"] == 1
    assert metadata["motion_tail_frame_count"] == 2
    assert metadata["continuation_frame_count"] == 3


def _finalize(images, **overrides):
    values = {
        "expected_visible_frames": 2,
        "motion_tail_frames": 1,
        "append_black_frame": False,
        "black_frame_count": 1,
        "run_id": "run",
        "attempt_id": "attempt",
        "plan_hash": "hash",
        "segment_index": 0,
    }
    values.update(overrides)
    return beatdrop_nodes.BeatDropWanSegmentFinalizerNode().finalize(
        images=images, **values
    )


def test_finalizer_rejects_output_shorter_than_expected_visible_frames():
    with pytest.raises(ValueError, match="shorter"):
        _finalize(_images(2), expected_visible_frames=3)


@pytest.mark.parametrize(
    ("images", "message"),
    [
        (torch.empty((0, 2, 3, 3)), "nonempty"),
        (torch.empty((2, 3, 3)), "4-D"),
        (torch.empty((2, 3, 64, 64)), "BHWC"),
        (torch.empty((2, 64, 64, 1)), "3 channels"),
        (torch.empty((2, 64, 64, 4)), "3 channels"),
        (torch.zeros((2, 2, 3, 3), dtype=torch.int64), "floating-point"),
        (torch.zeros((2, 2, 3, 3), dtype=torch.complex64), "floating-point"),
        ("not-a-tensor", "torch.Tensor"),
    ],
)
def test_finalizer_rejects_invalid_image_batches(images, message):
    with pytest.raises(ValueError, match=message):
        _finalize(images)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_visible_frames": 0}, "expected_visible_frames"),
        ({"motion_tail_frames": 0}, "motion_tail_frames"),
        ({"black_frame_count": 0}, "black_frame_count"),
        ({"expected_visible_frames": True}, "expected_visible_frames"),
        ({"motion_tail_frames": False}, "motion_tail_frames"),
        ({"black_frame_count": True}, "black_frame_count"),
        ({"append_black_frame": 1}, "append_black_frame"),
        ({"expected_visible_frames": 1000001}, "expected_visible_frames"),
        ({"motion_tail_frames": 16385}, "motion_tail_frames"),
        ({"black_frame_count": 65}, "black_frame_count"),
        ({"segment_index": -1}, "segment_index"),
        ({"run_id": ""}, "run_id"),
        ({"attempt_id": "  "}, "attempt_id"),
        ({"plan_hash": None}, "plan_hash"),
    ],
)
def test_finalizer_rejects_invalid_control_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        _finalize(_images(3), **overrides)


def test_finalizer_caps_tail_to_visible_frames_and_supports_multiple_resets():
    images = _images(4)
    final_images, continuation, masks, metadata_json = _finalize(
        images,
        expected_visible_frames=2,
        motion_tail_frames=4,
        append_black_frame=True,
        black_frame_count=2,
    )

    assert final_images.shape[0] == 4
    assert torch.equal(continuation[:2], images[:2])
    assert torch.count_nonzero(continuation[2:]) == 0
    assert continuation.shape[0] == 4
    assert masks.shape == (4, 2, 3)
    metadata = json.loads(metadata_json)
    assert metadata["motion_tail_frame_count"] == 2
    assert metadata["reset_frame_count"] == 2
    assert metadata["continuation_frame_count"] == 4


def test_finalizer_schema_and_registration_are_stable():
    cls = beatdrop_nodes.BeatDropWanSegmentFinalizerNode

    assert cls.RETURN_TYPES == ("IMAGE", "IMAGE", "MASK", "STRING")
    assert cls.RETURN_NAMES == (
        "final_images",
        "continuation_images",
        "continuation_masks",
        "metadata_json",
    )
    assert cls.FUNCTION == "finalize"
    assert cls.CATEGORY == "Amin/Beatdrop/Wan"
    assert beatdrop_nodes.NODE_CLASS_MAPPINGS["BeatDropWanSegmentFinalizerNode"] is cls
    assert (
        beatdrop_nodes.NODE_DISPLAY_NAME_MAPPINGS[
            "BeatDropWanSegmentFinalizerNode"
        ]
        == "🎞️ BeatDrop WAN Segment Finalizer"
    )
