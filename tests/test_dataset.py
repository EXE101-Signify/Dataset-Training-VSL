"""Focused unit and integration checks for the Section 2 data contract."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src import config
from src.datasets import SignLanguageVideoDataset, make_dataloader
from src.preprocessing.video_processing import VideoDecodeError, load_video, sample_indices


class DatasetTests(unittest.TestCase):
    def test_split_and_batch(self):
        for csv_path, count, train in ((config.TRAIN_CSV, 3108, True), (config.VAL_CSV, 767, False)):
            with self.subTest(split=csv_path.name):
                dataset = SignLanguageVideoDataset(csv_path, train=train)
                self.assertEqual(len(dataset), count)
                self.assertEqual(dataset.num_classes, 100)
                self.assertEqual({r[2] for r in dataset.rows}, set(range(100)))
                sample = dataset[0]
                self.assertEqual(sample["pixel_values"].shape, (config.NUM_FRAMES, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
                self.assertEqual(sample["pixel_values"].dtype, torch.float32)
                self.assertTrue(torch.isfinite(sample["pixel_values"]).all())
                batch = next(iter(make_dataloader(dataset, batch_size=2, num_workers=0)))
                self.assertEqual(batch["pixel_values"].shape, (2, config.NUM_FRAMES, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
                self.assertTrue(torch.isfinite(batch["pixel_values"]).all())

    def test_sampling(self):
        rng = np.random.default_rng(42)
        first = sample_indices(64, config.NUM_FRAMES, True, rng)
        self.assertTrue(any(not np.array_equal(first, sample_indices(64, config.NUM_FRAMES, True, rng)) for _ in range(5)))
        self.assertTrue(np.array_equal(sample_indices(64, config.NUM_FRAMES, False, rng), sample_indices(64, config.NUM_FRAMES, False, rng)))
        self.assertEqual(sample_indices(3, 5, False, rng).tolist(), [0, 1, 2, 2, 2])

    def test_unreadable_video(self):
        with self.assertRaises(VideoDecodeError):
            load_video(ROOT / "missing_section_2_video.mp4")


if __name__ == "__main__":
    unittest.main()
