"""Use an audited 16-frame source video to test repeat-last-frame padding."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="backslashreplace")

import numpy as np

from src import config
from src.preprocessing.video_processing import load_video, process_frames, sample_indices


def main() -> None:
    with (ROOT / "reports" / "audit_video.csv").open(newline="", encoding="utf-8-sig") as file:
        row = next((r for r in csv.DictReader(file) if r["split"] == "train" and r["full_decode_ok"] == "True" and int(r["frame_count_decoded"]) == 16), None)
    if row is None:
        raise RuntimeError("No audited 16-frame video found")
    path = config.DATASET_ROOT / Path(row["path"])
    frames = load_video(path)
    target = len(frames) + 4
    indices = sample_indices(len(frames), target, False, np.random.default_rng(config.RANDOM_SEED))
    assert len(indices) == target
    assert np.array_equal(indices[:len(frames)], np.arange(len(frames)))
    assert np.all(indices[len(frames):] == len(frames) - 1)
    values = process_frames(frames[indices], config.IMAGE_SIZE, False, np.random.default_rng(config.RANDOM_SEED), config.IMAGE_MEAN, config.IMAGE_STD, resize_shortest_edge=config.RESIZE_SHORTEST_EDGE, rescale_factor=config.RESCALE_FACTOR)
    assert values.shape == (target, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert np.isfinite(values).all()
    print(f"Loaded real short video: {path}")
    print(f"Decoded frames: {len(frames)}, target frames: {target}, output shape: {values.shape}")
    print("Strategy: retain original frames in order, repeat final frame; PASS")


if __name__ == "__main__":
    main()
