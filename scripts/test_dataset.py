"""Smoke test both splits and one batch without touching source data."""

import argparse
import pickle
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(errors="backslashreplace")

import torch

from src import config
from src.datasets import SignLanguageVideoDataset, make_dataloader, seed_everything


def inspect(name: str, dataset: SignLanguageVideoDataset, batch_size: int) -> None:
    print(f"{name} dataset size: {len(dataset)}")
    print(f"{name} number of classes: {dataset.num_classes}")
    assert dataset.num_classes == 100
    assert all(0 <= label_id < 100 for _, _, label_id in dataset.rows)
    sample = dataset[0]
    pixels = sample["pixel_values"]
    expected = (dataset.num_frames, 3, dataset.image_size, dataset.image_size)
    assert tuple(pixels.shape) == expected
    assert pixels.dtype == torch.float32 and torch.isfinite(pixels).all()
    assert 0 <= sample["label"] < 100
    print(f"{name} sample shape: {tuple(pixels.shape)}, dtype: {pixels.dtype}")
    print(f"{name} sample min/max: {pixels.min().item():.4f}/{pixels.max().item():.4f}")
    print(f"{name} label: {sample['label']}, label_name: {sample['label_name']}")
    print(f"{name} video_path: {sample['video_path']}")
    loader = make_dataloader(dataset, batch_size=batch_size, num_workers=0)
    batch = next(iter(loader))
    values = batch["pixel_values"]
    assert tuple(values.shape) == (min(batch_size, len(dataset)), *expected)
    assert values.dtype == torch.float32 and torch.isfinite(values).all()
    assert torch.all((batch["label"] >= 0) & (batch["label"] < 100))
    print(f"{name} batch shape: {tuple(values.shape)}, dtype: {values.dtype}")
    print(f"{name} batch min/max: {values.min().item():.4f}/{values.max().item():.4f}")
    print(f"{name} batch test: PASS (no NaN/Inf; labels valid)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=config.DATASET_ROOT)
    parser.add_argument("--num-frames", type=int, default=config.NUM_FRAMES)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()
    seed_everything()
    with (args.dataset_root / "label_mapping.pkl").open("rb") as file:
        local_mapping = pickle.load(file)
    local_mapping = {unicodedata.normalize("NFC", name): int(label_id) for name, label_id in local_mapping.items()}
    for name, csv_path, train in (("train", config.TRAIN_CSV, True), ("validation", config.VAL_CSV, False)):
        dataset = SignLanguageVideoDataset(csv_path, args.dataset_root, args.num_frames, args.image_size, train)
        assert len(dataset) == (3108 if train else 767)
        assert all(local_mapping[unicodedata.normalize("NFC", label_name)] == label_id for _, label_name, label_id in dataset.rows)
        inspect(name, dataset, args.batch_size)
    local_id_to_label = {label_id: name for name, label_id in local_mapping.items()}
    mismatches = sum(
        unicodedata.normalize("NFC", local_id_to_label[i])
        != unicodedata.normalize("NFC", config.MODEL_CONFIG["id2label"][str(i)])
        for i in range(config.NUM_CLASSES)
    )
    print(f"Checkpoint/local label ID mismatches: {mismatches}/{config.NUM_CLASSES}")
    if mismatches:
        print("WARNING: Checkpoint logits IDs cannot be interpreted with the local CSV label mapping without remapping.")


if __name__ == "__main__":
    main()
