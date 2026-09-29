import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mask_researcher_tools import MaskDiffPreview


def _masks(batch_size: int):
    masks = torch.zeros((batch_size, 16, 16), dtype=torch.float32)
    for frame in range(batch_size):
        start = frame + 1
        masks[frame, start : start + 3, start : start + 3] = 1.0
    return masks


class MaskDiffPreviewTests(unittest.TestCase):
    def test_multi_frame_diff_is_marked_animated_and_keeps_every_frame(self):
        node = MaskDiffPreview.__new__(MaskDiffPreview)
        captured = {}

        def fake_save_images(images, *args, **kwargs):
            captured["images"] = images.clone()
            return {
                "ui": {
                    "images": [
                        {
                            "filename": f"frame_{frame}.png",
                            "type": "temp",
                            "subfolder": "",
                        }
                        for frame in range(images.shape[0])
                    ]
                }
            }

        node.save_images = fake_save_images
        expected = _masks(3)
        output = node.preview(
            torch.zeros((3, 16, 16)), expected, "white_diff"
        )

        self.assertEqual(output["ui"]["animated"], (True,))
        self.assertEqual(len(output["ui"]["images"]), 3)
        self.assertEqual(captured["images"].shape, (3, 16, 16, 3))
        self.assertEqual(output["result"][0].shape, (3, 16, 16))
        self.assertTrue(torch.equal(output["result"][0], expected))

    def test_single_frame_diff_is_not_marked_animated(self):
        node = MaskDiffPreview.__new__(MaskDiffPreview)
        node.save_images = lambda images, *args, **kwargs: {
            "ui": {
                "images": [
                    {"filename": "frame_0.png", "type": "temp", "subfolder": ""}
                ]
            }
        }

        output = node.preview(
            torch.zeros((1, 16, 16)), _masks(1), "white_diff"
        )

        self.assertEqual(output["ui"]["animated"], (False,))
        self.assertEqual(len(output["ui"]["images"]), 1)


if __name__ == "__main__":
    unittest.main()
