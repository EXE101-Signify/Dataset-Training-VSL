"""Section 2 contract from the selected checkpoint's small JSON files."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
TRAIN_CSV = PROJECT_ROOT / "splits" / "train.csv"
VAL_CSV = PROJECT_ROOT / "splits" / "val.csv"

CHECKPOINT_ID = "star092304/vi-sign-language-videomae-base"
MODEL_CONFIG_PATH = PROJECT_ROOT / "reports" / "videomae_model_config.json"
PROCESSOR_CONFIG_PATH = PROJECT_ROOT / "reports" / "videomae_preprocessor_config.json"
MODEL_CONFIG = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
PROCESSOR_CONFIG = json.loads(PROCESSOR_CONFIG_PATH.read_text(encoding="utf-8"))

NUM_FRAMES = int(MODEL_CONFIG["num_frames"])
IMAGE_SIZE = int(MODEL_CONFIG["image_size"])
PATCH_SIZE = int(MODEL_CONFIG["patch_size"])
TUBELET_SIZE = int(MODEL_CONFIG["tubelet_size"])
NUM_CHANNELS = int(MODEL_CONFIG["num_channels"])
NUM_CLASSES = len(MODEL_CONFIG["id2label"])
RESIZE_SHORTEST_EDGE = int(PROCESSOR_CONFIG["size"]["shortest_edge"])
CROP_SIZE = (int(PROCESSOR_CONFIG["crop_size"]["height"]), int(PROCESSOR_CONFIG["crop_size"]["width"]))
IMAGE_MEAN = tuple(float(x) for x in PROCESSOR_CONFIG["image_mean"])
IMAGE_STD = tuple(float(x) for x in PROCESSOR_CONFIG["image_std"])
RESCALE_FACTOR = float(PROCESSOR_CONFIG["rescale_factor"])

if MODEL_CONFIG["model_type"] != "videomae" or NUM_CHANNELS != 3:
    raise ValueError("Checkpoint is not a three-channel VideoMAE model")
if CROP_SIZE != (IMAGE_SIZE, IMAGE_SIZE):
    raise ValueError("Processor crop size does not match model input size")
if NUM_FRAMES % TUBELET_SIZE or IMAGE_SIZE % PATCH_SIZE:
    raise ValueError("Checkpoint frame/image dimensions are incompatible with patch/tubelet sizes")
if not all(PROCESSOR_CONFIG[k] for k in ("do_resize", "do_center_crop", "do_rescale", "do_normalize")):
    raise ValueError("Unsupported checkpoint processor flags")
if PROCESSOR_CONFIG["image_processor_type"] != "VideoMAEImageProcessor":
    raise ValueError("Unexpected image processor type")
if PROCESSOR_CONFIG["resample"] != 2:
    raise ValueError("Expected bilinear resampling (PIL enum 2)")

RANDOM_SEED = 42
BATCH_SIZE = 4
NUM_WORKERS = 0
VAL_RATIO = 0.20  # Existing split provenance; not used to regenerate CSV files.
