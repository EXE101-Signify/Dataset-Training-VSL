"""CSV-backed VideoMAE input dataset."""

import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from src import config
from src.preprocessing.video_processing import load_video, process_frames, sample_indices

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int = config.RANDOM_SEED) -> None:
    """Seed common RNGs without forcing slow deterministic GPU algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class SignLanguageVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: str | Path = config.DATASET_ROOT,
        num_frames: int = config.NUM_FRAMES,
        image_size: int = config.IMAGE_SIZE,
        train: bool = False,
        random_seed: int = config.RANDOM_SEED,
        image_mean: tuple[float, float, float] = config.IMAGE_MEAN,
        image_std: tuple[float, float, float] = config.IMAGE_STD,
        train_crop_scale: tuple[float, float] = (0.9, 1.0),
    ) -> None:
        if num_frames < 1 or image_size < 1:
            raise ValueError("num_frames and image_size must be positive")
        if len(image_mean) != 3 or len(image_std) != 3 or any(s <= 0 for s in image_std):
            raise ValueError("Three channel means and positive standard deviations are required")
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train
        self.random_seed = random_seed
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.train_crop_scale = train_crop_scale
        self._rng = None
        with self.csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {"video_path", "label", "label_id"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{self.csv_path} needs columns {sorted(required)}")
            self.rows = []
            for line, row in enumerate(reader, start=2):
                try:
                    relative = Path(row["video_path"])
                    if relative.is_absolute() or ".." in relative.parts or not row["label"]:
                        raise ValueError("Invalid path or empty label")
                    self.rows.append((relative, row["label"], int(row["label_id"])))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid CSV row {line} in {self.csv_path}: {row}") from exc
        if not self.rows:
            raise ValueError(f"Empty split: {self.csv_path}")
        ids = {row[2] for row in self.rows}
        if ids != set(range(config.NUM_CLASSES)):
            raise ValueError(f"Expected label IDs 0..{config.NUM_CLASSES - 1} in {self.csv_path}")
        self.num_classes = len(ids)

    def __len__(self) -> int:
        return len(self.rows)

    def _generator(self) -> np.random.Generator:
        if self._rng is None:
            worker = get_worker_info()
            seed = (worker.seed if worker else self.random_seed) if self.train else self.random_seed
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __getitem__(self, index: int) -> dict:
        relative, label_name, label_id = self.rows[index]
        path = self.dataset_root / relative
        try:
            frames = load_video(path)
            indices = sample_indices(len(frames), self.num_frames, self.train, self._generator())
            pixels = process_frames(
                frames[indices], self.image_size, self.train, self._generator(),
                self.image_mean, self.image_std, self.train_crop_scale,
                config.RESIZE_SHORTEST_EDGE, config.RESCALE_FACTOR,
            )
            return {
                "pixel_values": torch.from_numpy(pixels),
                "label": label_id,
                "video_path": str(path),
                "label_name": label_name,
            }
        except Exception:
            LOGGER.exception("Failed to process video %s (CSV row %d)", path, index + 2)
            raise


def make_dataloader(
    dataset: SignLanguageVideoDataset,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    shuffle: bool | None = None,
    drop_last: bool = False,
    seed: int = config.RANDOM_SEED,
) -> DataLoader:
    if shuffle is None:
        shuffle = dataset.train
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        drop_last=drop_last, worker_init_fn=_seed_worker, generator=generator,
    )
