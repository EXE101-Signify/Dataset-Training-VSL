"""CPU data-loading benchmark; excludes model/GPU work."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.datasets import SignLanguageVideoDataset, make_dataloader, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=config.DATASET_ROOT)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--batches", type=int, default=5)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")
    seed_everything()
    dataset = SignLanguageVideoDataset(config.TRAIN_CSV, args.dataset_root, train=True)
    start = time.perf_counter()
    dataset[0]
    sample_seconds = time.perf_counter() - start
    loader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    start = time.perf_counter()
    batches = 0
    samples = 0
    for batch in loader:
        batches += 1
        samples += len(batch["label"])
        if batches >= args.batches:
            break
    seconds = time.perf_counter() - start
    print(f"One sample: {sample_seconds:.3f} s")
    print(f"Batch size: {args.batch_size}, num_workers: {args.num_workers}")
    print(f"{batches} batches ({samples} samples): {seconds:.3f} s")
    print(f"Batch mean: {seconds / batches:.3f} s")
    print(f"Throughput: {samples / seconds:.2f} samples/s")


if __name__ == "__main__":
    main()
