# Section 2 — Signify video dataset

## Input and split

The source videos and `label_mapping.pkl` live in `../dataset`. This project reads `splits/train.csv` (3,108 rows) and `splits/val.csv` (767 rows). Both contain `video_path`, `label`, and `label_id`; each covers IDs 0–99. The existing stratified split was created with seed 42. Section 2 reads these files without regenerating or editing them.

The implementation follows the existing `src/datasets` and `src/preprocessing` layout. `SignLanguageVideoDataset` returns `pixel_values`, `label`, `video_path`, and `label_name`. `make_dataloader` exposes batch size, worker count, pinned memory, persistent workers, shuffle, and drop-last. Its default shuffle follows `dataset.train`.

## Target checkpoint contract

Checkpoint: [`star092304/vi-sign-language-videomae-base`](https://huggingface.co/star092304/vi-sign-language-videomae-base). Section 2 stores **only** its small [model config](https://huggingface.co/star092304/vi-sign-language-videomae-base/blob/main/config.json) and [processor config](https://huggingface.co/star092304/vi-sign-language-videomae-base/blob/main/preprocessor_config.json) as `reports/videomae_model_config.json` and `reports/videomae_preprocessor_config.json`. No model weights were downloaded. `src/config.py` reads these snapshots, validates essential fields, and supplies defaults to the dataset.

| Setting | Checkpoint value |
| --- | --- |
| Model | `VideoMAEForVideoClassification`, 100 `id2label` entries |
| Frames / channels | 16 / 3 RGB |
| Model image size | 224×224 |
| Patch / tubelet | 16×16 / 2 frames |
| Processor | `VideoMAEImageProcessor` |
| Resize / crop | shortest edge 224, center crop 224×224 |
| Resample | bilinear (`resample=2`) |
| Rescale | multiply by `1/255` |
| Normalize | mean `(0.485, 0.456, 0.406)`; std `(0.229, 0.224, 0.225)` |

The [VideoMAE patch embedding implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/videomae/modeling_videomae.py) unpacks `pixel_values` as `(batch, frames, channels, height, width)` before permuting for its 3D convolution. The sample tensor is therefore float32 `[T,C,H,W]`, normally `[16,3,224,224]`; DataLoader yields `[B,T,C,H,W]`. The classifier expects label IDs 0–99.

**Label compatibility issue:** the checkpoint's `id2label` ordering differs from this project's `label_mapping.pkl` for **96 of 100 IDs** after NFC Unicode normalization. The CSV remains authoritative for Section 2. Do not interpret checkpoint logits with CSV IDs or calculate checkpoint classification metrics until Section 3 defines an explicit verified mapping. `scripts/test_dataset.py` prints the mismatch count without changing any label data.

## Video preprocessing

OpenCV sequentially decodes all frames and converts BGR to RGB. Sampling uses the number of decoded frames because FPS and frame-count metadata can be unreliable. An unreadable or empty video raises `VideoDecodeError`; the dataset logs path and CSV row, then re-raises. A metadata mismatch or unusual FPS is logged. It never substitutes fake frames for a broken video.

Training divides the clip into 16 temporal bins and randomly samples one frame per bin. Validation takes the center of each bin deterministically. If a clip has fewer than the target frames, all decoded frames are kept in order and its final frame is repeated until the target length is reached. This is tested using an audited 16-frame source video with a temporary target of 20 frames; no video file is created or modified.

Validation resizes the short edge to 224 and center crops at 224×224. Training enlarges the short edge mildly to 105% of the processor target and applies a small crop offset shared by every frame in the clip. This is the only augmentation. It preserves temporal coherence and avoids horizontal flips or rotations. Both paths use the processor's rescale and normalization values.

`seed_everything(42)` seeds Python, NumPy, and PyTorch CPU/CUDA generators. The DataLoader uses a seeded generator and seeds Python/NumPy in each worker. Validation frame selection and spatial processing are deterministic. Training randomness is reproducible for the same access order and worker configuration; strict deterministic GPU algorithms are not forced.

## Commands

Run in the project root with the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe scripts\test_dataset.py
.\.venv\Scripts\python.exe scripts\test_sampling.py
.\.venv\Scripts\python.exe scripts\test_short_video.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\visualize_samples.py --split val --index 0 --output reports\sample_grid.png
.\.venv\Scripts\python.exe scripts\benchmark_dataset.py --batches 5 --batch-size 4 --num-workers 0
```

The visualizer exports four selected source frames above their processed and denormalized counterparts. For train CSV rows it uses deterministic validation transforms so each pair has matching temporal indices.

## Dependencies and limits

The data pipeline requires PyTorch, NumPy, and OpenCV. Tested environment: Python 3.14.6, PyTorch 2.14.1+cpu, OpenCV 5.0.0, NumPy 2.5.3; CUDA unavailable. `transformers` is not installed or required to load the two JSON config files or generate dataset tensors. A future model integration will need a compatible Transformers installation.

Sequential full-video decoding uses memory proportional to video length. The audited clips are short; much longer videos may need a streaming decoder. OpenCV can return readable early frames from a truncated video, in which case frame-count mismatch logging is the available signal. The checkpoint snapshot is from the current `main` branch; refresh and revalidate it if the checkpoint changes. The label-ID mismatch above blocks direct checkpoint score interpretation, although Section 2 dataset and DataLoader tests can run.
