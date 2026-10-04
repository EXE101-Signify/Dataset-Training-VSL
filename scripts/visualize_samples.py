"""Save original and normalized-then-denormalized frame strips as PNG."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="backslashreplace")

import cv2
import numpy as np

from src import config
from src.datasets import SignLanguageVideoDataset
from src.preprocessing.video_processing import load_video, sample_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "sample_grid.png")
    parser.add_argument("--dataset-root", type=Path, default=config.DATASET_ROOT)
    args = parser.parse_args()
    dataset = SignLanguageVideoDataset(
        config.TRAIN_CSV if args.split == "train" else config.VAL_CSV,
        args.dataset_root, train=False,
    )
    sample = dataset[args.index]
    original = load_video(sample["video_path"])
    indices = sample_indices(len(original), dataset.num_frames, False, np.random.default_rng(config.RANDOM_SEED))
    chosen = np.linspace(0, dataset.num_frames - 1, min(4, dataset.num_frames), dtype=int)
    mean = np.array(dataset.image_mean, dtype=np.float32)
    std = np.array(dataset.image_std, dtype=np.float32)
    processed = sample["pixel_values"].numpy().transpose(0, 2, 3, 1)
    processed = np.uint8(np.clip((processed * std + mean) * 255, 0, 255))
    tiles = []
    for i in chosen:
        source = original[indices[i]]
        source = cv2.resize(source, (dataset.image_size, dataset.image_size))
        tiles.append(np.concatenate((source, processed[i]), axis=0))
    grid = np.concatenate(tiles, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save {args.output}")
    print(f"Saved original (top) / processed (bottom): {args.output}")


if __name__ == "__main__":
    main()
