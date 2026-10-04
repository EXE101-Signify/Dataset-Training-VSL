from __future__ import annotations

import csv
import hashlib
import json
import pickle
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"D:\BuiGiaHuy\CN8\EXE202\dataset")

TRAIN_DIR = DATASET_ROOT / "train"
PUBLIC_TEST_DIR = DATASET_ROOT / "public_test"
PRIVATE_TEST_DIR = DATASET_ROOT / "private_test"
LABEL_MAPPING_FILE = DATASET_ROOT / "label_mapping.pkl"

OUTPUT_DIR = Path(__file__).resolve().parent / "reports"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

# Audit thresholds.
# These DO NOT mean a video is automatically invalid.
# They are only used to flag suspicious samples.
FPS_MIN_REASONABLE = 5.0
FPS_MAX_REASONABLE = 35.0

DURATION_MIN_REASONABLE = 0.5
DURATION_MAX_REASONABLE = 10.0

FRAME_COUNT_MIN_REASONABLE = 8
FRAME_COUNT_MAX_REASONABLE = 64

EXPECTED_WIDTH = 224
EXPECTED_HEIGHT = 224

# If True, decode every frame of every train video.
# For 3,875 videos this should still be manageable.
FULL_FRAME_DECODE = True

# Calculate SHA-256 for every train video.
# This reads the entire video file once, so it takes some time.
CALCULATE_SHA256 = True

# Print progress every N videos.
PROGRESS_INTERVAL = 100


# ============================================================
# HELPERS
# ============================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def list_videos(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.rglob("*")
            if is_video(path)
        ]
    )


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"

    if size < 1024 ** 3:
        return f"{size / (1024 ** 2):.3f} MB"

    return f"{size / (1024 ** 3):.3f} GB"


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)

        if value != value:
            return None

        return value
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    index = (len(sorted_values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(sorted_values))
    weight = index - lower

    return (
            sorted_values[lower] * (1 - weight)
            + sorted_values[upper] * weight
    )


def statistics_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATASET_ROOT))
    except ValueError:
        return str(path)


def infer_split(path: Path) -> str:
    try:
        relative_parts = path.relative_to(DATASET_ROOT).parts
    except ValueError:
        return "unknown"

    if not relative_parts:
        return "unknown"

    return relative_parts[0]


def infer_class(path: Path) -> str | None:
    """
    For train/<class>/<video>.mp4:
        returns <class>

    For test files without class directories:
        returns None.

    If nested class directories exist, returns the first directory
    after the split directory.
    """

    try:
        relative_parts = path.relative_to(DATASET_ROOT).parts
    except ValueError:
        return None

    if len(relative_parts) >= 3:
        split = relative_parts[0]

        if split == "train":
            return relative_parts[1]

    return None


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Counter):
        return dict(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    return value


# ============================================================
# LABEL MAPPING
# ============================================================

def inspect_label_mapping() -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": LABEL_MAPPING_FILE.exists(),
        "path": str(LABEL_MAPPING_FILE),
        "type": None,
        "length": None,
        "preview": [],
        "mapping_direction": None,
        "error": None,
    }

    if not LABEL_MAPPING_FILE.exists():
        return result

    try:
        with LABEL_MAPPING_FILE.open("rb") as file:
            mapping = pickle.load(file)

        result["type"] = type(mapping).__name__

        if hasattr(mapping, "__len__"):
            result["length"] = len(mapping)

        if isinstance(mapping, dict):
            items = list(mapping.items())[:20]

            result["preview"] = [
                {
                    "key": str(key),
                    "value": str(value),
                }
                for key, value in items
            ]

            if mapping:
                first_key = next(iter(mapping.keys()))
                first_value = mapping[first_key]

                if isinstance(first_key, int) and isinstance(first_value, str):
                    result["mapping_direction"] = "id_to_label"

                elif isinstance(first_key, str) and isinstance(first_value, int):
                    result["mapping_direction"] = "label_to_id"

                else:
                    result["mapping_direction"] = "unknown"

        elif isinstance(mapping, list):
            result["preview"] = [
                str(item)
                for item in mapping[:20]
            ]

            result["mapping_direction"] = "list"

        else:
            result["mapping_direction"] = "unknown"

    except Exception as exc:
        result["error"] = repr(exc)

    return result


# ============================================================
# VIDEO AUDIT
# ============================================================

def audit_video(
        path: Path,
        full_decode: bool,
        calculate_hash: bool,
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "path": relative_path(path),
        "split": infer_split(path),
        "class_name": infer_class(path),
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "file_size_bytes": None,
        "file_size_mb": None,

        "open_ok": False,
        "first_frame_ok": False,
        "full_decode_ok": None,
        "decode_error": None,

        "fps": None,
        "frame_count_metadata": None,
        "frame_count_decoded": None,
        "frame_count_difference": None,

        "width": None,
        "height": None,
        "duration_metadata": None,

        "codec": None,
        "sha256": None,

        "suspicious": False,
        "suspicious_reasons": [],
    }

    try:
        result["file_size_bytes"] = path.stat().st_size
        result["file_size_mb"] = path.stat().st_size / (1024 ** 2)
    except OSError as exc:
        result["decode_error"] = f"Cannot stat file: {exc}"
        result["suspicious"] = True
        result["suspicious_reasons"].append("file_stat_error")
        return result

    if calculate_hash:
        try:
            result["sha256"] = sha256_file(path)
        except Exception as exc:
            result["decode_error"] = f"SHA256 error: {exc}"
            result["suspicious"] = True
            result["suspicious_reasons"].append("sha256_error")

    capture = None

    try:
        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            result["decode_error"] = "OpenCV could not open video"
            result["suspicious"] = True
            result["suspicious_reasons"].append("cannot_open")

            return result

        result["open_ok"] = True

        fps = safe_float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_metadata = safe_int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        width = safe_int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = safe_int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        result["fps"] = fps
        result["frame_count_metadata"] = frame_count_metadata
        result["width"] = width
        result["height"] = height

        if fps is not None and fps > 0 and frame_count_metadata is not None:
            result["duration_metadata"] = (
                    frame_count_metadata / fps
            )

        # ----------------------------------------------------
        # First frame test
        # ----------------------------------------------------

        ok, frame = capture.read()

        if ok and frame is not None:
            result["first_frame_ok"] = True
        else:
            result["first_frame_ok"] = False
            result["suspicious"] = True
            result["suspicious_reasons"].append(
                "first_frame_decode_failed"
            )

        # ----------------------------------------------------
        # Full decode
        # ----------------------------------------------------

        if full_decode:
            decoded_frames = 1 if result["first_frame_ok"] else 0

            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                if frame is None:
                    break

                decoded_frames += 1

            result["frame_count_decoded"] = decoded_frames
            result["full_decode_ok"] = True

            if frame_count_metadata is not None:
                result["frame_count_difference"] = (
                        decoded_frames - frame_count_metadata
                )

                # OpenCV metadata can occasionally be slightly off.
                # Flag only significant differences.
                if abs(
                        result["frame_count_difference"]
                ) > 2:
                    result["suspicious"] = True
                    result["suspicious_reasons"].append(
                        "frame_count_mismatch"
                    )

    except Exception as exc:
        result["decode_error"] = repr(exc)
        result["full_decode_ok"] = False
        result["suspicious"] = True
        result["suspicious_reasons"].append(
            "decode_exception"
        )

    finally:
        if capture is not None:
            capture.release()

    # ========================================================
    # Metadata checks
    # ========================================================

    fps = result["fps"]
    duration = result["duration_metadata"]
    frame_count = result["frame_count_metadata"]
    width = result["width"]
    height = result["height"]

    if fps is None or fps <= 0:
        result["suspicious"] = True
        result["suspicious_reasons"].append(
            "invalid_fps"
        )

    elif fps < FPS_MIN_REASONABLE:
        result["suspicious"] = True
        result["suspicious_reasons"].append(
            "fps_too_low"
        )

    elif fps > FPS_MAX_REASONABLE:
        result["suspicious"] = True
        result["suspicious_reasons"].append(
            "fps_too_high"
        )

    if frame_count is None or frame_count <= 0:
        result["suspicious"] = True
        result["suspicious_reasons"].append(
            "invalid_frame_count"
        )

    elif frame_count < FRAME_COUNT_MIN_REASONABLE:
        result["suspicious"] = True
        result["suspicious_reasons"].append(
            "too_few_frames"
        )

    elif frame_count > FRAME_COUNT_MAX_REASONABLE:
        result["suspicious"] = True
        result["suspicious_reasons"].append(
            "too_many_frames"
        )

    if duration is not None:
        if duration < DURATION_MIN_REASONABLE:
            result["suspicious"] = True
            result["suspicious_reasons"].append(
                "duration_too_short"
            )

        elif duration > DURATION_MAX_REASONABLE:
            result["suspicious"] = True
            result["suspicious_reasons"].append(
                "duration_too_long"
            )

    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        result["suspicious"] = True
        result["suspicious_reasons"].append(
            "unexpected_resolution"
        )

    return result


# ============================================================
# CSV WRITER
# ============================================================

def write_csv(
        path: Path,
        rows: list[dict[str, Any]],
) -> None:

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            normalized = dict(row)

            if isinstance(
                    normalized.get("suspicious_reasons"),
                    list,
            ):
                normalized["suspicious_reasons"] = ";".join(
                    normalized["suspicious_reasons"]
                )

            writer.writerow(normalized)


# ============================================================
# CLASS AUDIT
# ============================================================

def build_class_distribution(
        video_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    counter: Counter[str] = Counter()

    for result in video_results:
        if result["split"] != "train":
            continue

        class_name = result["class_name"]

        if class_name:
            counter[class_name] += 1

    rows = []

    for class_name, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
    ):
        rows.append(
            {
                "class_name": class_name,
                "video_count": count,
            }
        )

    return rows


# ============================================================
# DUPLICATE AUDIT
# ============================================================

def build_duplicate_report(
        video_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for result in video_results:
        sha256 = result.get("sha256")

        if sha256:
            groups[sha256].append(result)

    rows = []

    for sha256, items in groups.items():
        if len(items) <= 1:
            continue

        for item in items:
            rows.append(
                {
                    "sha256": sha256,
                    "duplicate_count": len(items),
                    "path": item["path"],
                    "class_name": item["class_name"],
                    "split": item["split"],
                }
            )

    return rows


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
        train_videos: list[Path],
        public_test_videos: list[Path],
        private_test_videos: list[Path],
        results: list[dict[str, Any]],
        label_mapping: dict[str, Any],
) -> dict[str, Any]:

    train_results = [
        r for r in results
        if r["split"] == "train"
    ]

    valid_results = [
        r for r in train_results
        if r["open_ok"]
           and r["first_frame_ok"]
           and (
                   r["full_decode_ok"] is not False
           )
    ]

    invalid_results = [
        r for r in train_results
        if not (
                r["open_ok"]
                and r["first_frame_ok"]
        )
    ]

    suspicious_results = [
        r for r in train_results
        if r["suspicious"]
    ]

    fps_values = [
        r["fps"]
        for r in train_results
        if r["fps"] is not None
    ]

    frame_values = [
        r["frame_count_metadata"]
        for r in train_results
        if r["frame_count_metadata"] is not None
    ]

    decoded_frame_values = [
        r["frame_count_decoded"]
        for r in train_results
        if r["frame_count_decoded"] is not None
    ]

    duration_values = [
        r["duration_metadata"]
        for r in train_results
        if r["duration_metadata"] is not None
    ]

    width_values = [
        r["width"]
        for r in train_results
        if r["width"] is not None
    ]

    height_values = [
        r["height"]
        for r in train_results
        if r["height"] is not None
    ]

    resolution_counter = Counter(
        f"{r['width']}x{r['height']}"
        for r in train_results
        if r["width"] is not None
        and r["height"] is not None
    )

    suspicious_counter = Counter()

    for result in suspicious_results:
        for reason in result["suspicious_reasons"]:
            suspicious_counter[reason] += 1

    class_counts = Counter(
        r["class_name"]
        for r in train_results
        if r["class_name"]
    )

    total_train_size = sum(
        r["file_size_bytes"] or 0
        for r in train_results
    )

    summary = {
        "dataset_root": str(DATASET_ROOT),

        "directories": {
            "train": str(TRAIN_DIR),
            "public_test": str(PUBLIC_TEST_DIR),
            "private_test": str(PRIVATE_TEST_DIR),
        },

        "file_counts": {
            "train": len(train_videos),
            "public_test": len(public_test_videos),
            "private_test": len(private_test_videos),
            "all": (
                    len(train_videos)
                    + len(public_test_videos)
                    + len(private_test_videos)
            ),
        },

        "train_audit": {
            "total": len(train_results),
            "valid": len(valid_results),
            "invalid": len(invalid_results),
            "suspicious": len(suspicious_results),
            "valid_percentage": (
                len(valid_results) / len(train_results) * 100
                if train_results
                else 0
            ),
        },

        "fps": statistics_summary(fps_values),

        "frame_count_metadata": statistics_summary(
            [
                float(value)
                for value in frame_values
            ]
        ),

        "frame_count_decoded": statistics_summary(
            [
                float(value)
                for value in decoded_frame_values
            ]
        ),

        "duration_seconds": statistics_summary(
            duration_values
        ),

        "width": statistics_summary(
            [
                float(value)
                for value in width_values
            ]
        ),

        "height": statistics_summary(
            [
                float(value)
                for value in height_values
            ]
        ),

        "resolution_distribution": dict(
            resolution_counter
        ),

        "suspicious_reasons": dict(
            suspicious_counter
        ),

        "class_distribution": {
            "number_of_classes": len(class_counts),
            "minimum": min(class_counts.values())
            if class_counts else None,
            "maximum": max(class_counts.values())
            if class_counts else None,
            "mean": (
                statistics.mean(class_counts.values())
                if class_counts
                else None
            ),
            "median": (
                statistics.median(class_counts.values())
                if class_counts
                else None
            ),
        },

        "storage": {
            "train_size_bytes": total_train_size,
            "train_size_gb": (
                    total_train_size / (1024 ** 3)
            ),
        },

        "label_mapping": label_mapping,

        "audit_configuration": {
            "full_frame_decode": FULL_FRAME_DECODE,
            "calculate_sha256": CALCULATE_SHA256,
            "fps_min_reasonable": FPS_MIN_REASONABLE,
            "fps_max_reasonable": FPS_MAX_REASONABLE,
            "duration_min_reasonable": DURATION_MIN_REASONABLE,
            "duration_max_reasonable": DURATION_MAX_REASONABLE,
            "frame_count_min_reasonable": FRAME_COUNT_MIN_REASONABLE,
            "frame_count_max_reasonable": FRAME_COUNT_MAX_REASONABLE,
            "expected_width": EXPECTED_WIDTH,
            "expected_height": EXPECTED_HEIGHT,
        },
    }

    return summary


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
        summary: dict[str, Any],
        duplicate_rows: list[dict[str, Any]],
) -> None:

    train = summary["train_audit"]
    counts = summary["file_counts"]

    print()
    print("=" * 70)
    print("DATASET AUDIT V2")
    print("=" * 70)

    print()
    print("DATASET")
    print(f"Root                 : {summary['dataset_root']}")
    print(f"Train videos         : {counts['train']}")
    print(f"Public test videos   : {counts['public_test']}")
    print(f"Private test videos  : {counts['private_test']}")

    print()
    print("TRAIN INTEGRITY")
    print(f"Total                : {train['total']}")
    print(f"Valid                : {train['valid']}")
    print(f"Invalid              : {train['invalid']}")
    print(f"Suspicious           : {train['suspicious']}")
    print(
        f"Valid percentage     : "
        f"{train['valid_percentage']:.2f}%"
    )

    fps = summary["fps"]

    print()
    print("FPS")
    print(f"Minimum              : {fps['min']}")
    print(f"Maximum              : {fps['max']}")
    print(f"Mean                 : {fps['mean']}")
    print(f"Median               : {fps['median']}")
    print(f"P25                  : {fps['p25']}")
    print(f"P75                  : {fps['p75']}")

    frames = summary["frame_count_metadata"]

    print()
    print("FRAME COUNT - METADATA")
    print(f"Minimum              : {frames['min']}")
    print(f"Maximum              : {frames['max']}")
    print(f"Mean                 : {frames['mean']}")
    print(f"Median               : {frames['median']}")
    print(f"P25                  : {frames['p25']}")
    print(f"P75                  : {frames['p75']}")

    decoded = summary["frame_count_decoded"]

    print()
    print("FRAME COUNT - ACTUAL DECODE")
    print(f"Minimum              : {decoded['min']}")
    print(f"Maximum              : {decoded['max']}")
    print(f"Mean                 : {decoded['mean']}")
    print(f"Median               : {decoded['median']}")
    print(f"P25                  : {decoded['p25']}")
    print(f"P75                  : {decoded['p75']}")

    duration = summary["duration_seconds"]

    print()
    print("DURATION")
    print(f"Minimum              : {duration['min']} s")
    print(f"Maximum              : {duration['max']} s")
    print(f"Mean                 : {duration['mean']} s")
    print(f"Median               : {duration['median']} s")
    print(f"P25                  : {duration['p25']} s")
    print(f"P75                  : {duration['p75']} s")

    print()
    print("RESOLUTION")

    for resolution, count in sorted(
            summary["resolution_distribution"].items(),
            key=lambda item: (-item[1], item[0]),
    ):
        percentage = (
            count / train["total"] * 100
            if train["total"]
            else 0
        )

        print(
            f"{resolution:20s}: "
            f"{count:5d} "
            f"({percentage:.2f}%)"
        )

    classes = summary["class_distribution"]

    print()
    print("CLASS BALANCE")
    print(
        f"Classes              : "
        f"{classes['number_of_classes']}"
    )
    print(
        f"Minimum videos/class : "
        f"{classes['minimum']}"
    )
    print(
        f"Maximum videos/class : "
        f"{classes['maximum']}"
    )
    print(
        f"Mean videos/class    : "
        f"{classes['mean']}"
    )
    print(
        f"Median videos/class  : "
        f"{classes['median']}"
    )

    print()
    print("SUSPICIOUS REASONS")

    suspicious_reasons = summary["suspicious_reasons"]

    if suspicious_reasons:
        for reason, count in sorted(
                suspicious_reasons.items(),
                key=lambda item: (-item[1], item[0]),
        ):
            print(
                f"{reason:30s}: {count}"
            )
    else:
        print("None")

    print()
    print("DUPLICATES")

    if duplicate_rows:
        duplicate_hashes = len(
            {
                row["sha256"]
                for row in duplicate_rows
            }
        )

        print(
            f"Duplicate SHA-256 groups: "
            f"{duplicate_hashes}"
        )

        print(
            f"Videos involved        : "
            f"{len(duplicate_rows)}"
        )

    else:
        print("No exact SHA-256 duplicates found.")

    print()
    print("STORAGE")

    train_size = summary["storage"]["train_size_gb"]

    print(
        f"Train video size      : "
        f"{train_size:.3f} GB"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    start_time = time.time()

    ensure_output_dir()

    print("=" * 70)
    print("Starting Dataset Audit V2")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    print()
    print("DATASET ROOT")
    print(f"Dataset root : {DATASET_ROOT}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist:\n{DATASET_ROOT}"
        )

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Train directory does not exist:\n{TRAIN_DIR}"
        )

    # --------------------------------------------------------
    # Discover videos
    # --------------------------------------------------------

    print()
    print("DISCOVERING VIDEOS...")

    train_videos = list_videos(TRAIN_DIR)
    public_test_videos = list_videos(PUBLIC_TEST_DIR)
    private_test_videos = list_videos(PRIVATE_TEST_DIR)

    print(
        f"Train videos        : "
        f"{len(train_videos)}"
    )

    print(
        f"Public test videos  : "
        f"{len(public_test_videos)}"
    )

    print(
        f"Private test videos : "
        f"{len(private_test_videos)}"
    )

    if not train_videos:
        raise RuntimeError(
            "No videos found inside train directory."
        )

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    print()
    print("READING LABEL MAPPING...")

    label_mapping = inspect_label_mapping()

    print(
        f"Exists               : "
        f"{label_mapping['exists']}"
    )

    print(
        f"Type                 : "
        f"{label_mapping['type']}"
    )

    print(
        f"Length               : "
        f"{label_mapping['length']}"
    )

    print(
        f"Mapping direction    : "
        f"{label_mapping['mapping_direction']}"
    )

    if label_mapping["error"]:
        print(
            f"Error                : "
            f"{label_mapping['error']}"
        )

    # --------------------------------------------------------
    # Audit train videos
    # --------------------------------------------------------

    print()
    print("AUDITING TRAIN VIDEOS...")
    print(
        f"Full frame decode    : "
        f"{FULL_FRAME_DECODE}"
    )

    print(
        f"SHA-256 calculation  : "
        f"{CALCULATE_SHA256}"
    )

    results: list[dict[str, Any]] = []

    total = len(train_videos)

    for index, video_path in enumerate(
            train_videos,
            start=1,
    ):

        result = audit_video(
            path=video_path,
            full_decode=FULL_FRAME_DECODE,
            calculate_hash=CALCULATE_SHA256,
        )

        results.append(result)

        if (
                index % PROGRESS_INTERVAL == 0
                or index == total
        ):
            elapsed = time.time() - start_time
            speed = index / elapsed if elapsed > 0 else 0
            percentage = index / total * 100

            print(
                f"Processing: "
                f"{index}/{total} "
                f"({percentage:.2f}%) | "
                f"{speed:.2f} videos/s"
            )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_distribution = build_class_distribution(
        results
    )

    # --------------------------------------------------------
    # Duplicate detection
    # --------------------------------------------------------

    print()
    print("CHECKING EXACT DUPLICATES...")

    duplicate_rows = build_duplicate_report(
        results
    )

    if duplicate_rows:
        print(
            f"Duplicate videos found: "
            f"{len(duplicate_rows)}"
        )
    else:
        print(
            "No exact SHA-256 duplicates found."
        )

    # --------------------------------------------------------
    # Suspicious videos
    # --------------------------------------------------------

    suspicious_results = [
        result
        for result in results
        if result["suspicious"]
    ]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        train_videos=train_videos,
        public_test_videos=public_test_videos,
        private_test_videos=private_test_videos,
        results=results,
        label_mapping=label_mapping,
    )

    # --------------------------------------------------------
    # Write reports
    # --------------------------------------------------------

    audit_csv = OUTPUT_DIR / "audit_video.csv"
    suspicious_csv = OUTPUT_DIR / "suspicious_videos.csv"
    duplicate_csv = OUTPUT_DIR / "duplicate_videos.csv"
    class_csv = OUTPUT_DIR / "class_distribution.csv"
    report_json = OUTPUT_DIR / "dataset_audit.json"

    print()
    print("WRITING REPORTS...")

    write_csv(
        audit_csv,
        results,
    )

    write_csv(
        suspicious_csv,
        suspicious_results,
    )

    write_csv(
        duplicate_csv,
        duplicate_rows,
    )

    write_csv(
        class_csv,
        class_distribution,
    )

    with report_json.open(
            "w",
            encoding="utf-8",
    ) as file:

        json.dump(
            json_safe(summary),
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print_summary(
        summary,
        duplicate_rows,
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print()
    print("OUTPUT")
    print(
        f"Audit CSV            : "
        f"{audit_csv}"
    )

    print(
        f"Suspicious CSV       : "
        f"{suspicious_csv}"
    )

    print(
        f"Duplicate CSV        : "
        f"{duplicate_csv}"
    )

    print(
        f"Class CSV            : "
        f"{class_csv}"
    )

    print(
        f"JSON report          : "
        f"{report_json}"
    )

    elapsed = time.time() - start_time

    print()
    print(
        f"Total execution time : "
        f"{elapsed:.2f} seconds"
    )

    print()
    print("Audit completed.")


if __name__ == "__main__":
    main()