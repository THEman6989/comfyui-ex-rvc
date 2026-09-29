import importlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

beatdrop_nodes = importlib.import_module("beatdrop_nodes")


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(70, 69), (69, 69), (13, 13), (12, 9), (9, 9), (5, 5), (1, 1)],
)
def test_normalize_wan_frame_count_uses_largest_4k_plus_1_not_above_request(
    requested, expected
):
    assert beatdrop_nodes.normalize_wan_frame_count(requested) == expected


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "70", None])
def test_normalize_wan_frame_count_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        beatdrop_nodes.normalize_wan_frame_count(invalid)


def test_initial_mode_uses_full_normalized_window_without_motion_context():
    node = beatdrop_nodes.BeatDropWanContinuationConfigNode()

    result = node.configure(70, 13, "initial", 1)

    assert result[:5] == (69, 13, 69, 0, 0)
    metadata = json.loads(result[5])
    assert metadata == {
        "black_frame_count": 0,
        "effective_context_frames": 0,
        "mode": "initial",
        "motion_context_frames": 0,
        "requested_context_frames": 13,
        "requested_window_frames": 70,
        "temporal_alignment": "4k+1",
        "visible_capacity": 69,
        "wan_length": 69,
    }


@pytest.mark.parametrize("mode", ["internal_continue", "outfit_transition"])
def test_continuation_modes_use_13_context_frames_and_56_visible_frames(mode):
    node = beatdrop_nodes.BeatDropWanContinuationConfigNode()

    result = node.configure(70, 13, mode, 1)

    expected_motion = 12 if mode == "outfit_transition" else 13
    expected_black = 1 if mode == "outfit_transition" else 0
    assert result[:5] == (69, 13, 56, expected_motion, expected_black)
    metadata = json.loads(result[5])
    assert metadata["requested_window_frames"] == 70
    assert metadata["wan_length"] == 69
    assert metadata["requested_context_frames"] == 13
    assert metadata["effective_context_frames"] == 13
    assert metadata["visible_capacity"] == 56
    assert metadata["motion_context_frames"] == expected_motion
    assert metadata["black_frame_count"] == expected_black
    assert metadata["temporal_alignment"] == "4k+1"


def test_config_node_normalizes_requested_context_to_4k_plus_1():
    result = beatdrop_nodes.BeatDropWanContinuationConfigNode().configure(
        70, 12, "internal_continue", 1
    )

    assert result[:5] == (69, 9, 60, 9, 0)
    assert json.loads(result[5])["effective_context_frames"] == 9


@pytest.mark.parametrize(
    ("window", "context", "mode", "black", "message"),
    [
        (True, 13, "initial", 1, "positive integer"),
        (70, False, "internal_continue", 1, "positive integer"),
        (70, 13, "unknown", 1, "mode"),
        (13, 13, "internal_continue", 1, "smaller than"),
        (9, 13, "outfit_transition", 1, "smaller than"),
        (70, 13, "outfit_transition", 0, "black_frame_count"),
        (70, 13, "outfit_transition", 2, "black_frame_count"),
    ],
)
def test_config_node_rejects_invalid_or_ambiguous_frame_plans(
    window, context, mode, black, message
):
    with pytest.raises(ValueError, match=message):
        beatdrop_nodes.BeatDropWanContinuationConfigNode().configure(
            window, context, mode, black
        )


def test_config_node_schema_is_explicit_and_registered():
    cls = beatdrop_nodes.BeatDropWanContinuationConfigNode

    assert cls.INPUT_TYPES() == {
        "required": {
            "requested_window_frames": (
                "INT",
                {"default": 70, "min": 1, "max": 16384, "step": 1},
            ),
            "requested_context_frames": (
                "INT",
                {"default": 13, "min": 1, "max": 16384, "step": 1},
            ),
            "mode": (["initial", "internal_continue", "outfit_transition"],),
            "black_frame_count": (
                "INT",
                {"default": 1, "min": 1, "max": 1, "step": 1},
            ),
        }
    }
    assert cls.RETURN_TYPES == ("INT", "INT", "INT", "INT", "INT", "STRING")
    assert cls.RETURN_NAMES == (
        "wan_length",
        "continue_motion_max_frames",
        "visible_capacity",
        "motion_context_frames",
        "black_frame_count",
        "config_json",
    )
    assert cls.FUNCTION == "configure"
    assert cls.CATEGORY == "Amin/Beatdrop/Wan"
    assert beatdrop_nodes.NODE_CLASS_MAPPINGS["BeatDropWanContinuationConfigNode"] is cls
    assert (
        beatdrop_nodes.NODE_DISPLAY_NAME_MAPPINGS[
            "BeatDropWanContinuationConfigNode"
        ]
        == "🧮 BeatDrop WAN Continuation Config"
    )
