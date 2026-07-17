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


def _item(batch_index, outfit_state=3, beat_frame=42, **extra):
    return {
        "outfit_batch_index": batch_index,
        "outfit_state": outfit_state,
        "beat_frame": beat_frame,
        **extra,
    }


def _schedule(items, run_id="run-a", attempt_id="attempt-a", plan_hash="plan-a"):
    return json.dumps(
        {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "plan_hash": plan_hash,
            "items": items,
        }
    )


def _iterate(node, images, schedule_json, iteration=0, **overrides):
    return node.iterate(
        images=images,
        schedule_json=schedule_json,
        run_id=overrides.get("run_id", "run-a"),
        plan_hash=overrides.get("plan_hash", "plan-a"),
        iteration=iteration,
        attempt_id=overrides.get("attempt_id", "attempt-a"),
    )


def test_iterate_selects_only_scheduled_batch_index_and_returns_canonical_item():
    node = beatdrop_nodes.BeatDropOutfitIteratorNode()
    images = torch.arange(3 * 2 * 2 * 1, dtype=torch.float32).reshape(3, 2, 2, 1)
    item = _item(2, outfit_state=7, beat_frame=99, note="drop")

    result = _iterate(node, images, _schedule([_item(0), item]), iteration=1)

    assert len(result) == 6
    assert torch.equal(result[0], images[2:3])
    assert result[0].shape == (1, 2, 2, 1)
    assert result[1:] == (
        7,
        99,
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        True,
        2,
    )


def test_iterate_returns_finite_nested_extra_as_parseable_canonical_json():
    item = _item(
        0,
        details={
            "label": "café",
            "weights": [0.0, -2.5, 1e100, {"active": True, "offset": None}],
        },
    )

    item_json = _iterate(
        beatdrop_nodes.BeatDropOutfitIteratorNode(),
        torch.zeros((1, 1, 1, 3)),
        _schedule([item]),
    )[3]

    assert json.loads(item_json) == item
    assert item_json == json.dumps(
        item,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize("non_finite_json", ["NaN", "Infinity", "-Infinity", "1e10000"])
def test_iterate_rejects_non_finite_selected_item_extra_before_return(non_finite_json):
    schedule_json = (
        '{"run_id":"run-a","attempt_id":"attempt-a","plan_hash":"plan-a","items":['
        '{"outfit_batch_index":0,"outfit_state":3,"beat_frame":42,"extra":'
        f"{non_finite_json}"
        "}]}"
    )

    with pytest.raises(ValueError, match="finite|JSON"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            torch.zeros((1, 1, 1, 3)),
            schedule_json,
        )


def test_comfy_registration_schema_and_six_outputs_are_exact():
    cls = beatdrop_nodes.BeatDropOutfitIteratorNode

    assert cls.INPUT_TYPES() == {
        "required": {
            "images": ("IMAGE",),
            "schedule_json": ("STRING", {"multiline": True}),
            "run_id": ("STRING",),
            "plan_hash": ("STRING",),
            "iteration": ("INT", {"min": 0}),
            "attempt_id": ("STRING",),
        }
    }
    assert cls.RETURN_TYPES == ("IMAGE", "INT", "INT", "STRING", "BOOLEAN", "INT")
    assert cls.RETURN_NAMES == (
        "current_image",
        "outfit_state",
        "beat_frame",
        "item_json",
        "done",
        "total",
    )
    assert cls.FUNCTION == "iterate"
    assert cls.CATEGORY == "Amin/Beatdrop"
    assert beatdrop_nodes.NODE_CLASS_MAPPINGS["BeatDropOutfitIteratorNode"] is cls
    assert (
        beatdrop_nodes.NODE_DISPLAY_NAME_MAPPINGS["BeatDropOutfitIteratorNode"]
        == "🔁 BeatDrop Outfit Iterator"
    )


@pytest.mark.parametrize(
    ("field", "input_value"),
    [
        ("run_id", "other-run"),
        ("attempt_id", "other-attempt"),
        ("plan_hash", "other-plan"),
    ],
)
def test_iterate_rejects_each_schedule_identity_mismatch(field, input_value):
    node = beatdrop_nodes.BeatDropOutfitIteratorNode()
    images = torch.zeros((1, 1, 1, 3))
    overrides = {field: input_value}

    with pytest.raises(ValueError, match=field):
        _iterate(node, images, _schedule([_item(0)]), **overrides)


@pytest.mark.parametrize("field", ["run_id", "attempt_id", "plan_hash"])
@pytest.mark.parametrize("invalid_value", [None, "", " \t", 17])
def test_iterate_rejects_matching_non_string_or_blank_identity_values(field, invalid_value):
    schedule_kwargs = {field: invalid_value}

    with pytest.raises(ValueError, match=field):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            torch.zeros((1, 1, 1, 3)),
            _schedule([_item(0)], **schedule_kwargs),
            **schedule_kwargs,
        )


@pytest.mark.parametrize(
    ("schedule_json", "message"),
    [
        ("{not-json", "valid JSON object"),
        (json.dumps([]), "JSON object"),
        (json.dumps({"run_id": "run-a", "attempt_id": "attempt-a", "plan_hash": "plan-a"}), "items"),
        (
            json.dumps(
                {
                    "run_id": "run-a",
                    "attempt_id": "attempt-a",
                    "plan_hash": "plan-a",
                    "items": {},
                }
            ),
            "items",
        ),
    ],
)
def test_iterate_rejects_malformed_root_or_items(schedule_json, message):
    node = beatdrop_nodes.BeatDropOutfitIteratorNode()

    with pytest.raises(ValueError, match=message):
        _iterate(node, torch.zeros((1, 1, 1, 3)), schedule_json)


@pytest.mark.parametrize("missing_field", ["run_id", "attempt_id", "plan_hash"])
def test_iterate_rejects_missing_schedule_identity(missing_field):
    schedule = json.loads(_schedule([_item(0)]))
    del schedule[missing_field]

    with pytest.raises(ValueError, match=missing_field):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            torch.zeros((1, 1, 1, 3)),
            json.dumps(schedule),
        )


def test_iterate_rejects_missing_identity_even_when_input_is_none():
    schedule = json.loads(_schedule([_item(0)]))
    del schedule["run_id"]

    with pytest.raises(ValueError, match="run_id"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            torch.zeros((1, 1, 1, 3)),
            json.dumps(schedule),
            run_id=None,
        )


@pytest.mark.parametrize("iteration", [-1, True, False, 1.5, "0"])
def test_iterate_rejects_non_integer_bool_or_negative_iteration(iteration):
    with pytest.raises(ValueError, match="iteration"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            torch.zeros((1, 1, 1, 3)),
            _schedule([_item(0), _item(0)]),
            iteration=iteration,
        )


def test_iterate_rejects_empty_schedule_without_returning_an_image():
    with pytest.raises(ValueError, match="empty"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            torch.zeros((1, 1, 1, 3)),
            _schedule([]),
        )


def test_iterate_rejects_exhausted_schedule_without_clamping_to_last_item():
    images = torch.tensor([[[[10.0]]], [[[20.0]]]])

    with pytest.raises(IndexError, match="iteration"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            images,
            _schedule([_item(1)]),
            iteration=1,
        )


def test_iterate_rejects_non_object_item_without_fallback():
    with pytest.raises(ValueError, match="item"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            torch.zeros((1, 1, 1, 3)),
            _schedule(["not-an-object"]),
        )


@pytest.mark.parametrize(
    "item",
    [
        {"outfit_state": 1, "beat_frame": 2},
        {"candidate_frame": 1, "outfit_state": 1, "beat_frame": 2},
        _item(True),
        _item(1.5),
        _item(-1),
        _item(2),
    ],
)
def test_iterate_rejects_invalid_or_out_of_range_batch_index_without_fallback(item):
    images = torch.tensor([[[[10.0]]], [[[20.0]]]])

    with pytest.raises((ValueError, IndexError), match="outfit_batch_index"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            images,
            _schedule([item]),
        )


@pytest.mark.parametrize(
    "item",
    [
        {"outfit_batch_index": 0, "beat_frame": 2},
        _item(0, outfit_state=True),
        _item(0, outfit_state=1.5),
        _item(0, outfit_state=-1),
    ],
)
def test_iterate_rejects_missing_or_non_integer_outfit_state(item):
    with pytest.raises(ValueError, match="outfit_state"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            torch.zeros((1, 1, 1, 3)),
            _schedule([item]),
        )


@pytest.mark.parametrize(
    "item",
    [
        {"outfit_batch_index": 0, "outfit_state": 1},
        _item(0, beat_frame=True),
        _item(0, beat_frame=2.5),
        _item(0, beat_frame=-1),
    ],
)
def test_iterate_rejects_missing_or_non_integer_beat_frame(item):
    with pytest.raises(ValueError, match="beat_frame"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            torch.zeros((1, 1, 1, 3)),
            _schedule([item]),
        )


@pytest.mark.parametrize(
    "images",
    [
        [[[0.0]]],
        torch.zeros((1, 2, 3)),
        torch.zeros((1, 1, 1, 1, 3)),
        torch.zeros((0, 1, 1, 3)),
        torch.zeros((1, 0, 1, 3)),
    ],
)
def test_iterate_rejects_invalid_rank_type_or_empty_image_batch(images):
    with pytest.raises(ValueError, match="images"):
        _iterate(
            beatdrop_nodes.BeatDropOutfitIteratorNode(),
            images,
            _schedule([_item(0)]),
        )


def _cache_key(
    images,
    schedule_json=None,
    run_id="run-a",
    attempt_id="attempt-a",
    plan_hash="plan-a",
    iteration=0,
):
    if schedule_json is None:
        schedule_json = _schedule([_item(0), _item(0)])
    return beatdrop_nodes.BeatDropOutfitIteratorNode.IS_CHANGED(
        images=images,
        schedule_json=schedule_json,
        run_id=run_id,
        plan_hash=plan_hash,
        iteration=iteration,
        attempt_id=attempt_id,
    )


def test_cache_key_is_deterministic_for_equal_tensor_content_not_object_identity():
    images = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)

    first = _cache_key(images)
    second = _cache_key(images.clone())

    assert isinstance(first, str)
    assert first == second


@pytest.mark.parametrize("field", ["run_id", "attempt_id", "plan_hash"])
@pytest.mark.parametrize("invalid_value", [None, "", " \t", 17])
def test_cache_key_rejects_non_string_or_blank_identity_values(field, invalid_value):
    with pytest.raises(ValueError, match=field):
        _cache_key(torch.zeros((1, 1, 1, 3)), **{field: invalid_value})


@pytest.mark.parametrize("schedule_json", [None, {}])
def test_cache_key_rejects_non_string_schedule_json(schedule_json):
    with pytest.raises(ValueError, match="schedule_json"):
        beatdrop_nodes.BeatDropOutfitIteratorNode.IS_CHANGED(
            images=torch.zeros((1, 1, 1, 3)),
            schedule_json=schedule_json,
            run_id="run-a",
            plan_hash="plan-a",
            iteration=0,
            attempt_id="attempt-a",
        )


@pytest.mark.parametrize(
    "images",
    [
        [[[0.0]]],
        torch.zeros((1, 2, 3)),
        torch.zeros((1, 1, 1, 1, 3)),
        torch.zeros((0, 1, 1, 3)),
        torch.zeros((1, 0, 1, 3)),
    ],
)
def test_cache_key_rejects_invalid_rank_type_or_empty_image_batch(images):
    with pytest.raises(ValueError, match="images"):
        _cache_key(images)


@pytest.mark.parametrize(
    "change",
    ["run_id", "attempt_id", "plan_hash", "iteration", "pixel", "shape", "dtype", "schedule"],
)
def test_cache_key_changes_for_every_bound_input_or_tensor_property(change):
    images = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    kwargs = {}
    changed_images = images.clone()
    if change == "run_id":
        kwargs["run_id"] = "run-b"
    elif change == "attempt_id":
        kwargs["attempt_id"] = "attempt-b"
    elif change == "plan_hash":
        kwargs["plan_hash"] = "plan-b"
    elif change == "iteration":
        kwargs["iteration"] = 1
    elif change == "pixel":
        changed_images[0, 0, 0, 0] += 1
    elif change == "shape":
        changed_images = changed_images.reshape(1, 1, 4, 3)
    elif change == "dtype":
        changed_images = changed_images.to(torch.float64)
    else:
        kwargs["schedule_json"] = _schedule([_item(0), _item(0, note="changed")])

    assert _cache_key(changed_images, **kwargs) != _cache_key(images)


def test_cache_key_hashes_noncontiguous_and_bfloat16_tensor_bytes_safely():
    images = torch.arange(24, dtype=torch.bfloat16).reshape(1, 2, 4, 3).transpose(1, 2)
    assert not images.is_contiguous()

    assert _cache_key(images) == _cache_key(images.contiguous())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cache_key_hashes_cuda_content_identically_to_cpu():
    images = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)

    assert _cache_key(images.cuda()) == _cache_key(images)


def test_iteration_order_does_not_create_hidden_history():
    node = beatdrop_nodes.BeatDropOutfitIteratorNode()
    images = torch.tensor([[[[10.0]]], [[[20.0]]]])
    schedule = _schedule([_item(0), _item(1)])

    later = _iterate(node, images, schedule, iteration=1)
    earlier = _iterate(node, images, schedule, iteration=0)
    repeated_later = _iterate(node, images, schedule, iteration=1)

    assert torch.equal(later[0], images[1:2])
    assert later[4] is True
    assert torch.equal(earlier[0], images[0:1])
    assert earlier[4] is False
    assert torch.equal(repeated_later[0], later[0])
    assert node.__dict__ == {}


def test_new_run_and_attempt_at_iteration_zero_select_first_scheduled_index():
    node = beatdrop_nodes.BeatDropOutfitIteratorNode()
    images = torch.tensor([[[[10.0]]], [[[20.0]]], [[[30.0]]]])

    old_result = _iterate(node, images, _schedule([_item(2)]))
    new_result = _iterate(
        node,
        images,
        _schedule([_item(0)], run_id="run-b", attempt_id="attempt-b", plan_hash="plan-b"),
        run_id="run-b",
        attempt_id="attempt-b",
        plan_hash="plan-b",
    )

    assert torch.equal(old_result[0], images[2:3])
    assert torch.equal(new_result[0], images[0:1])
    assert new_result[4:] == (True, 1)
    assert node.__dict__ == {}
