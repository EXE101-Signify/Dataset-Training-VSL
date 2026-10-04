"""Check random train and deterministic validation temporal indices."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config
from src.preprocessing.video_processing import sample_indices


def main() -> None:
    length = 64
    rng = np.random.default_rng(config.RANDOM_SEED)
    train_draws = [sample_indices(length, config.NUM_FRAMES, True, rng) for _ in range(8)]
    assert all(len(draw) == config.NUM_FRAMES for draw in train_draws)
    assert all(np.all((draw >= 0) & (draw < length)) for draw in train_draws)
    assert any(not np.array_equal(train_draws[0], draw) for draw in train_draws[1:])
    val_a = sample_indices(length, config.NUM_FRAMES, False, np.random.default_rng(1))
    val_b = sample_indices(length, config.NUM_FRAMES, False, np.random.default_rng(999))
    assert len(val_a) == config.NUM_FRAMES and np.array_equal(val_a, val_b)
    print(f"Training sampling varies: PASS ({len(train_draws)} draws)")
    print(f"Validation sampling deterministic: PASS ({config.NUM_FRAMES} frames)")


if __name__ == "__main__":
    main()
