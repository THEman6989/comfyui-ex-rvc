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


def _images(count: int, height: int = 3, width: int = 4) -> torch.Tensor:
    values = torch.arange(count * height * width * 3, dtype=torch.float32)
    return (values.reshape(count, height, width, 3) % 251) / 250.0


def _masks(count: int, height: int = 3, width: int = 4) -> torch.Tensor:
    values = torch.arange(count * height * width, dtype=torch.float32)
    return (values.reshape(count, height, width) % 17) / 16.0


def _write_package(tmp_path: Path):
    source_video = tmp_path / "source-clip.mp4"
    source_video.write_bytes(b"lossless source bytes")
    return beatdrop_nodes.BeatDropWanAnalysisPackageWriterNode().write_package(
        reference_images=_images(2),
        face_images=_images(3),
        mimic_images=_images(4),
        background_images=_images(4),
        input_video_images=_images(4),
        resolution_reference=_images(1),
        void_joined_images=_images(4),
        character_masks=_masks(4),
        segmentation_mask=_masks(4),
        fps=29.97,
        width=4,
        height=3,
        frame_count=4,
        video_id="clip-001",
        run_id="run-001",
        package_root=str(tmp_path),
        immutable_base_prompt="owhx EXACT BASE",
        scene_prompt=" appended scene",
        plan_json='{"segment":1}',
        source_video=str(source_video),
    )


def test_package_nodes_have_explicit_schema_and_are_registered():
    writer = beatdrop_nodes.BeatDropWanAnalysisPackageWriterNode
    reader = beatdrop_nodes.BeatDropWanAnalysisPackageReaderNode

    writer_inputs = writer.INPUT_TYPES()
    assert set(writer_inputs["required"]) == {
        "reference_images",
        "face_images",
        "mimic_images",
        "background_images",
        "input_video_images",
        "resolution_reference",
        "void_joined_images",
        "character_masks",
        "segmentation_mask",
        "fps",
        "width",
        "height",
        "frame_count",
        "video_id",
        "run_id",
        "package_root",
    }
    assert set(writer_inputs["optional"]) == {
        "immutable_base_prompt",
        "scene_prompt",
        "plan_json",
        "source_video",
    }
    assert writer.RETURN_TYPES == ("STRING", "STRING")
    assert writer.OUTPUT_NODE is True

    assert reader.RETURN_NAMES == (
        "reference_images",
        "face_images",
        "mimic_images",
        "background_images",
        "input_video_images",
        "resolution_reference",
        "void_joined_images",
        "character_masks",
        "segmentation_mask",
        "fps",
        "width",
        "height",
        "frame_count",
        "immutable_base_prompt",
        "scene_prompt",
        "plan_json",
        "source_video",
        "manifest_json",
    )
    assert beatdrop_nodes.NODE_CLASS_MAPPINGS["BeatDropWanAnalysisPackageWriterNode"] is writer
    assert beatdrop_nodes.NODE_CLASS_MAPPINGS["BeatDropWanAnalysisPackageReaderNode"] is reader


def test_package_v2_round_trip_uses_png_only_with_declared_quantization(tmp_path):
    package_dir, manifest_json = _write_package(tmp_path)
    package = Path(package_dir)
    manifest = json.loads(manifest_json)

    assert package == tmp_path / "clip-001" / "run-001"
    assert (package / "manifest.json").is_file()
    assert (package / "READY").read_text(encoding="utf-8") == "beatdrop_wan_analysis_package/v2\n"
    assert not (package / "tensors").exists()
    assert len(list((package / "reference_images").glob("*.png"))) == 2
    assert len(list((package / "face_images").glob("*.png"))) == 3
    assert len(list((package / "mimic_images").glob("*.png"))) == 4
    assert len(list((package / "background_images").glob("*.png"))) == 4
    assert len(list((package / "input_video_images").glob("*.png"))) == 4
    assert len(list((package / "resolution_reference").glob("*.png"))) == 1
    assert len(list((package / "void_joined_images").glob("*.png"))) == 4
    assert len(list((package / "character_masks").glob("*.png"))) == 4
    assert len(list((package / "segmentation_mask").glob("*.png"))) == 4
    assert manifest["schema"] == "beatdrop_wan_analysis_package/v2"
    assert manifest["representations"] == {
        "images": {"format": "png", "bit_depth": 8, "channels": 3},
        "masks": {"format": "png", "bit_depth": 16, "channels": 1},
    }
    assert manifest["video_id"] == "clip-001"
    assert manifest["run_id"] == "run-001"
    assert manifest["frame_count"] == 4
    assert manifest["immutable_base_prompt"] == "owhx EXACT BASE"
    assert manifest["scene_prompt"] == " appended scene"
    assert manifest["plan_json"] == '{"segment":1}'

    loaded = beatdrop_nodes.BeatDropWanAnalysisPackageReaderNode().read_package(
        package_dir=package_dir,
        verify_hashes=True,
    )
    for actual, expected in zip(loaded[:7], (
        _images(2), _images(3), _images(4), _images(4), _images(4), _images(1), _images(4),
    )):
        assert torch.equal(actual, expected.clamp(0, 1).mul(255).round().div(255))
    for actual, expected in zip(loaded[7:9], (_masks(4), _masks(4))):
        assert torch.equal(actual, expected.clamp(0, 1).mul(65535).round().div(65535))
    assert loaded[9:13] == (29.97, 4, 3, 4)
    assert loaded[13:16] == ("owhx EXACT BASE", " appended scene", '{"segment":1}')
    assert Path(loaded[16]).read_bytes() == b"lossless source bytes"
    assert json.loads(loaded[17])["schema"] == "beatdrop_wan_analysis_package/v2"


def test_writer_is_non_destructive_and_rejects_unsafe_job_ids(tmp_path):
    _write_package(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        _write_package(tmp_path)

    with pytest.raises(ValueError, match="video_id"):
        beatdrop_nodes.BeatDropWanAnalysisPackageWriterNode().write_package(
            reference_images=_images(1),
            face_images=_images(1),
            mimic_images=_images(1),
            background_images=_images(1),
            input_video_images=_images(1),
            resolution_reference=_images(1),
            void_joined_images=_images(1),
            character_masks=_masks(1),
            segmentation_mask=_masks(1),
            fps=24.0,
            width=4,
            height=3,
            frame_count=1,
            video_id="../escape",
            run_id="run",
            package_root=str(tmp_path),
        )


def test_reader_detects_corrupted_png_artifact(tmp_path):
    package_dir, _ = _write_package(tmp_path)
    artifact = Path(package_dir) / "mimic_images" / "000000.png"
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="SHA-256"):
        beatdrop_nodes.BeatDropWanAnalysisPackageReaderNode().read_package(
            package_dir=package_dir,
            verify_hashes=True,
        )


def test_reader_rejects_package_without_atomic_ready_marker(tmp_path):
    package_dir, _ = _write_package(tmp_path)
    (Path(package_dir) / "READY").unlink()

    with pytest.raises(ValueError, match="READY"):
        beatdrop_nodes.BeatDropWanAnalysisPackageReaderNode().read_package(
            package_dir=package_dir,
            verify_hashes=True,
        )
