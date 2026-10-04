"""OpenCV video decoding and modest, clip-consistent spatial transforms."""

from pathlib import Path
import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class VideoDecodeError(RuntimeError):
    """A video could not provide usable frames."""


def load_video(path: str | Path) -> np.ndarray:
    """Decode every available frame in RGB order, ignoring unreliable frame/FPS metadata.

    Sequential decoding keeps sampling correct when CAP_PROP_FRAME_COUNT disagrees
    with actual frames. The clip is short (~30 frames in this dataset).
    """
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError(f"Cannot open video: {path}")
    frames = []
    try:
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            LOGGER.warning("Unusual FPS %s for %s; sampling by decoded frame index", fps, path)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise VideoDecodeError(f"Empty decoded frame in {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise VideoDecodeError(f"No decoded frames: {path}")
    if reported > 0 and reported != len(frames):
        LOGGER.warning("Frame count mismatch for %s: metadata=%d decoded=%d", path, reported, len(frames))
    try:
        return np.stack(frames)
    except ValueError as exc:
        raise VideoDecodeError(f"Inconsistent frame sizes in {path}") from exc


def sample_indices(length: int, num_frames: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    """One random index per temporal bin in train; bin centers in validation.

    Short clips retain all frames in order and repeat their final frame.
    """
    if length < 1 or num_frames < 1:
        raise ValueError("length and num_frames must be positive")
    if length < num_frames:
        return np.minimum(np.arange(num_frames), length - 1)
    edges = np.linspace(0, length, num_frames + 1, dtype=np.int64)
    if train:
        return np.array([rng.integers(edges[i], edges[i + 1]) for i in range(num_frames)], dtype=np.int64)
    return np.array([(edges[i] + edges[i + 1] - 1) // 2 for i in range(num_frames)], dtype=np.int64)


def process_frames(
    frames: np.ndarray,
    image_size: int,
    train: bool,
    rng: np.random.Generator,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    train_crop_scale: tuple[float, float] = (0.9, 1.0),
    resize_shortest_edge: int | None = None,
    rescale_factor: float = 1.0 / 255.0,
) -> np.ndarray:
    """Resize preserving aspect ratio, then one shared crop for the whole clip.

    Returns float32 [T,C,H,W]. No flip or rotation is used for signed motion.
    """
    if image_size < 1:
        raise ValueError("image_size must be positive")
    h, w = frames.shape[1:3]
    if h < 1 or w < 1:
        raise ValueError("Frames have invalid dimensions")
    resize_shortest_edge = resize_shortest_edge or image_size
    scale = resize_shortest_edge * (1.05 if train else 1.0) / min(h, w)
    resized_h, resized_w = max(image_size, round(h * scale)), max(image_size, round(w * scale))
    resized = np.stack([cv2.resize(f, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR) for f in frames])
    crop_size = image_size
    if train:
        lo, hi = train_crop_scale
        if not 0 < lo <= hi <= 1:
            raise ValueError("train_crop_scale must satisfy 0 < low <= high <= 1")
        # Mild translation within the spare margin. Keep crop at model input size.
        margin_y, margin_x = resized_h - crop_size, resized_w - crop_size
        fraction = float(rng.uniform(lo, hi))
        max_shift_y = round((1 - fraction) * margin_y / 2)
        max_shift_x = round((1 - fraction) * margin_x / 2)
        top = max(0, margin_y // 2 + int(rng.integers(-max_shift_y, max_shift_y + 1)))
        left = max(0, margin_x // 2 + int(rng.integers(-max_shift_x, max_shift_x + 1)))
    else:
        top, left = (resized_h - crop_size) // 2, (resized_w - crop_size) // 2
    cropped = resized[:, top:top + crop_size, left:left + crop_size]
    values = cropped.astype(np.float32) * np.float32(rescale_factor)
    values = (values - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(values.transpose(0, 3, 1, 2))
