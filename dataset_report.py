from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

# Root directory containing the VSL dataset.
DATASET_ROOT = Path(r"D:\BuiGiaHuy\CN8\EXE202\dataset")

# Training dataset.
TRAIN_DIR = DATASET_ROOT / "train"

# Output directory.
OUTPUT_DIR = Path(__file__).resolve().parent / "reports"

# Output files.
REPORT_JSON = OUTPUT_DIR / "dataset_report.json"
VIDEO_CSV = OUTPUT_DIR / "dataset_report.csv"
BAD_VIDEO_CSV = OUTPUT_DIR / "bad_videos.csv"


# ============================================================
# HELPERS
# ============================================================

def safe_mean(values: list[float | int]) -> float | None:
    """Return mean value or None if the list is empty."""
    if not values:
        return None

    return round(statistics.mean(values), 4)


def safe_median(values: list[float | int]) -> float | None:
    """Return median value or None if the list is empty."""
    if not values:
        return None

    return round(statistics.median(values), 4)


def safe_min(values: list[float | int]) -> float | None:
    """Return minimum value or None if the list is empty."""
    if not values:
        return None

    return min(values)


def safe_max(values: list[float | int]) -> float | None:
    """Return maximum value or None if the list is empty."""
    if not values:
        return None

    return max(values)


def percentile(
        values: list[float | int],
        p: float
) -> float | None:
    """
    Calculate a simple percentile.

    p should be between 0 and 100.
    """
    if not values:
        return None

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    index = (len(sorted_values) - 1) * (p / 100)

    lower = int(index)
    upper = min(
        lower + 1,
        len(sorted_values) - 1
    )

    weight = index - lower

    result = (
            sorted_values[lower]
            + (
                    sorted_values[upper]
                    - sorted_values[lower]
            ) * weight
    )

    return round(float(result), 4)


def print_progress(
        current: int,
        total: int,
        start_time: float
) -> None:
    """Print progress information in the console."""

    if total == 0:
        return

    elapsed = time.time() - start_time

    percentage = current / total * 100

    if current > 0 and elapsed > 0:
        speed = current / elapsed
        remaining = (
            (total - current) / speed
            if speed > 0
            else 0
        )
    else:
        speed = 0
        remaining = 0

    print(
        f"\rProcessing: {current}/{total} "
        f"({percentage:6.2f}%) | "
        f"{speed:6.2f} videos/s | "
        f"ETA: {remaining:7.1f}s",
        end="",
        flush=True,
    )


# ============================================================
# VIDEO INSPECTION
# ============================================================

def inspect_video(video_path: Path) -> dict[str, Any]:
    """
    Inspect one MP4 video using OpenCV.

    Returns metadata and validation information.
    """

    result: dict[str, Any] = {
        "file": str(video_path),
        "class": video_path.parent.name,
        "filename": video_path.name,
        "extension": video_path.suffix.lower(),
        "file_size_bytes": video_path.stat().st_size,
        "valid": False,
        "error": None,
        "fps": None,
        "frame_count": None,
        "width": None,
        "height": None,
        "duration_seconds": None,
        "codec": None,
    }

    cap = None

    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            result["error"] = "Cannot open video"
            return result

        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # Try to obtain codec information.
        fourcc_value = int(
            cap.get(cv2.CAP_PROP_FOURCC)
        )

        if fourcc_value:
            codec = "".join(
                [
                    chr(fourcc_value & 0xFF),
                    chr((fourcc_value >> 8) & 0xFF),
                    chr((fourcc_value >> 16) & 0xFF),
                    chr((fourcc_value >> 24) & 0xFF),
                ]
            )
        else:
            codec = None

        # Basic validation.
        if fps <= 0:
            result["error"] = "Invalid FPS"
            return result

        if frame_count <= 0:
            result["error"] = "Invalid frame count"
            return result

        if width <= 0 or height <= 0:
            result["error"] = "Invalid resolution"
            return result

        duration = frame_count / fps

        result.update(
            {
                "valid": True,
                "fps": round(fps, 4),
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": round(
                    duration,
                    4
                ),
                "codec": codec,
            }
        )

        # Read the first frame to verify that the
        # actual video contains decodable image data.
        success, frame = cap.read()

        if not success or frame is None:
            result["valid"] = False
            result["error"] = (
                "Cannot decode first frame"
            )

        return result

    except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return result

    finally:
        if cap is not None:
            cap.release()


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze_dataset() -> None:
    """Analyze the complete VSL training dataset."""

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist:\n"
            f"{DATASET_ROOT}"
        )

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Train directory does not exist:\n"
            f"{TRAIN_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "Signify - Vietnamese Sign Language Dataset Report"
    )
    print("=" * 70)

    print(
        f"Dataset root : {DATASET_ROOT}"
    )

    print(
        f"Train folder : {TRAIN_DIR}"
    )

    print(
        f"Output folder: {OUTPUT_DIR}"
    )

    print()

    # --------------------------------------------------------
    # Discover classes
    # --------------------------------------------------------

    class_dirs = sorted(
        [
            path
            for path in TRAIN_DIR.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name.lower(),
    )

    if not class_dirs:
        raise RuntimeError(
            f"No class directories found in:\n"
            f"{TRAIN_DIR}"
        )

    print(
        f"Classes found: {len(class_dirs)}"
    )

    # --------------------------------------------------------
    # Discover videos
    # --------------------------------------------------------

    videos: list[tuple[Path, str]] = []

    for class_dir in class_dirs:

        class_videos = sorted(
            [
                path
                for path in class_dir.rglob("*")
                if path.is_file()
                   and path.suffix.lower() == ".mp4"
            ],
            key=lambda path: path.name.lower(),
        )

        for video in class_videos:
            videos.append(
                (
                    video,
                    class_dir.name
                )
            )

    total_videos = len(videos)

    print(
        f"Videos found : {total_videos}"
    )

    print()

    if total_videos == 0:
        raise RuntimeError(
            "No MP4 files found."
        )

    # --------------------------------------------------------
    # Statistics containers
    # --------------------------------------------------------

    all_results: list[dict[str, Any]] = []

    bad_videos: list[dict[str, Any]] = []

    class_counts: Counter[str] = Counter()

    valid_class_counts: Counter[str] = Counter()

    invalid_class_counts: Counter[str] = Counter()

    fps_values: list[float] = []

    frame_counts: list[int] = []

    width_values: list[int] = []

    height_values: list[int] = []

    duration_values: list[float] = []

    file_sizes: list[int] = []

    resolution_counter: Counter[str] = Counter()

    codec_counter: Counter[str] = Counter()

    start_time = time.time()

    # --------------------------------------------------------
    # Inspect every video
    # --------------------------------------------------------

    for index, (
            video_path,
            class_name
    ) in enumerate(
        videos,
        start=1
    ):

        class_counts[class_name] += 1

        result = inspect_video(
            video_path
        )

        all_results.append(result)

        if result["valid"]:

            valid_class_counts[
                class_name
            ] += 1

            fps_values.append(
                result["fps"]
            )

            frame_counts.append(
                result["frame_count"]
            )

            width_values.append(
                result["width"]
            )

            height_values.append(
                result["height"]
            )

            duration_values.append(
                result["duration_seconds"]
            )

            file_sizes.append(
                result["file_size_bytes"]
            )

            resolution = (
                f'{result["width"]}x'
                f'{result["height"]}'
            )

            resolution_counter[
                resolution
            ] += 1

            if result["codec"]:
                codec_counter[
                    result["codec"]
                ] += 1

        else:

            invalid_class_counts[
                class_name
            ] += 1

            bad_videos.append(result)

        print_progress(
            index,
            total_videos,
            start_time,
        )

    print("\n")

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    valid_count = (
            len(all_results)
            - len(bad_videos)
    )

    invalid_count = len(bad_videos)

    report = {
        "dataset": {
            "name": "ViSignLanguage-Video",
            "root": str(DATASET_ROOT),
            "train_directory": str(
                TRAIN_DIR
            ),
        },

        "summary": {
            "total_classes": len(
                class_dirs
            ),

            "total_videos": total_videos,

            "valid_videos": valid_count,

            "invalid_videos": invalid_count,

            "valid_percentage": round(
                valid_count
                / total_videos
                * 100,
                4,
                ),

            "invalid_percentage": round(
                invalid_count
                / total_videos
                * 100,
                4,
                ),
        },

        "fps": {
            "min": safe_min(
                fps_values
            ),

            "max": safe_max(
                fps_values
            ),

            "mean": safe_mean(
                fps_values
            ),

            "median": safe_median(
                fps_values
            ),

            "p25": percentile(
                fps_values,
                25
            ),

            "p75": percentile(
                fps_values,
                75
            ),
        },

        "frames": {
            "min": safe_min(
                frame_counts
            ),

            "max": safe_max(
                frame_counts
            ),

            "mean": safe_mean(
                frame_counts
            ),

            "median": safe_median(
                frame_counts
            ),

            "p25": percentile(
                frame_counts,
                25
            ),

            "p75": percentile(
                frame_counts,
                75
            ),
        },

        "duration_seconds": {
            "min": safe_min(
                duration_values
            ),

            "max": safe_max(
                duration_values
            ),

            "mean": safe_mean(
                duration_values
            ),

            "median": safe_median(
                duration_values
            ),

            "p25": percentile(
                duration_values,
                25
            ),

            "p75": percentile(
                duration_values,
                75
            ),
        },

        "resolution": {
            "unique_resolutions": len(
                resolution_counter
            ),

            "distribution": dict(
                resolution_counter.most_common()
            ),
        },

        "codec": {
            "distribution": dict(
                codec_counter.most_common()
            ),
        },

        "file_size_bytes": {
            "min": safe_min(
                file_sizes
            ),

            "max": safe_max(
                file_sizes
            ),

            "mean": safe_mean(
                file_sizes
            ),

            "median": safe_median(
                file_sizes
            ),

            "total": sum(
                file_sizes
            ),
        },

        "classes": {
            "distribution": dict(
                sorted(
                    class_counts.items(),
                    key=lambda item:
                    item[0].lower(),
                )
            ),

            "valid_distribution": dict(
                sorted(
                    valid_class_counts.items(),
                    key=lambda item:
                    item[0].lower(),
                )
            ),

            "invalid_distribution": dict(
                sorted(
                    invalid_class_counts.items(),
                    key=lambda item:
                    item[0].lower(),
                )
            ),
        },

        "class_balance": {
            "minimum_videos_per_class": (
                min(
                    class_counts.values()
                )
                if class_counts
                else None
            ),

            "maximum_videos_per_class": (
                max(
                    class_counts.values()
                )
                if class_counts
                else None
            ),

            "mean_videos_per_class": (
                round(
                    statistics.mean(
                        class_counts.values()
                    ),
                    4,
                )
                if class_counts
                else None
            ),

            "median_videos_per_class": (
                round(
                    statistics.median(
                        class_counts.values()
                    ),
                    4,
                )
                if class_counts
                else None
            ),
        },

        "validation": {
            "bad_videos": len(
                bad_videos
            ),

            "bad_video_files": [
                item["file"]
                for item in bad_videos
            ],
        },

        "generated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with REPORT_JSON.open(
            "w",
            encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Save detailed CSV
    # --------------------------------------------------------

    csv_fields = [
        "file",
        "class",
        "filename",
        "extension",
        "file_size_bytes",
        "valid",
        "error",
        "fps",
        "frame_count",
        "width",
        "height",
        "duration_seconds",
        "codec",
    ]

    with VIDEO_CSV.open(
            "w",
            encoding="utf-8-sig",
            newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # --------------------------------------------------------
    # Save bad video CSV
    # --------------------------------------------------------

    bad_fields = [
        "file",
        "class",
        "filename",
        "error",
        "file_size_bytes",
    ]

    with BAD_VIDEO_CSV.open(
            "w",
            encoding="utf-8-sig",
            newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=bad_fields,
        )

        writer.writeheader()

        for item in bad_videos:

            writer.writerow(
                {
                    field: item.get(field)
                    for field in bad_fields
                }
            )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    total_size_gb = (
        sum(file_sizes)
        / (1024 ** 3)
        if file_sizes
        else 0
    )

    print("=" * 70)
    print("DATASET REPORT")
    print("=" * 70)

    print()
    print("GENERAL")
    print("-" * 70)

    print(
        f"Classes              : "
        f"{len(class_dirs)}"
    )

    print(
        f"Total videos         : "
        f"{total_videos}"
    )

    print(
        f"Valid videos         : "
        f"{valid_count}"
    )

    print(
        f"Invalid videos       : "
        f"{invalid_count}"
    )

    print(
        f"Valid percentage     : "
        f"{valid_count / total_videos * 100:.2f}%"
    )

    print()
    print("FPS")
    print("-" * 70)

    print(
        f"Minimum              : "
        f"{safe_min(fps_values)}"
    )

    print(
        f"Maximum              : "
        f"{safe_max(fps_values)}"
    )

    print(
        f"Mean                 : "
        f"{safe_mean(fps_values)}"
    )

    print(
        f"Median               : "
        f"{safe_median(fps_values)}"
    )

    print(
        f"P25                  : "
        f"{percentile(fps_values, 25)}"
    )

    print(
        f"P75                  : "
        f"{percentile(fps_values, 75)}"
    )

    print()
    print("FRAMES")
    print("-" * 70)

    print(
        f"Minimum              : "
        f"{safe_min(frame_counts)}"
    )

    print(
        f"Maximum              : "
        f"{safe_max(frame_counts)}"
    )

    print(
        f"Mean                 : "
        f"{safe_mean(frame_counts)}"
    )

    print(
        f"Median               : "
        f"{safe_median(frame_counts)}"
    )

    print(
        f"P25                  : "
        f"{percentile(frame_counts, 25)}"
    )

    print(
        f"P75                  : "
        f"{percentile(frame_counts, 75)}"
    )

    print()
    print("DURATION")
    print("-" * 70)

    print(
        f"Minimum              : "
        f"{safe_min(duration_values)} s"
    )

    print(
        f"Maximum              : "
        f"{safe_max(duration_values)} s"
    )

    print(
        f"Mean                 : "
        f"{safe_mean(duration_values)} s"
    )

    print(
        f"Median               : "
        f"{safe_median(duration_values)} s"
    )

    print(
        f"P25                  : "
        f"{percentile(duration_values, 25)} s"
    )

    print(
        f"P75                  : "
        f"{percentile(duration_values, 75)} s"
    )

    print()
    print("RESOLUTION")
    print("-" * 70)

    for resolution, count in (
            resolution_counter.most_common()
    ):

        percentage = (
                count
                / valid_count
                * 100
        )

        print(
            f"{resolution:15} : "
            f"{count:5} "
            f"({percentage:.2f}%)"
        )

    print()
    print("CLASS BALANCE")
    print("-" * 70)

    print(
        f"Minimum videos/class : "
        f"{min(class_counts.values())}"
    )

    print(
        f"Maximum videos/class : "
        f"{max(class_counts.values())}"
    )

    print(
        f"Mean videos/class    : "
        f"{statistics.mean(class_counts.values()):.2f}"
    )

    print(
        f"Median videos/class  : "
        f"{statistics.median(class_counts.values()):.2f}"
    )

    print()
    print("SMALLEST CLASSES")
    print("-" * 70)

    for class_name, count in sorted(
            class_counts.items(),
            key=lambda item: item[1],
    )[:10]:

        print(
            f"{count:5}  {class_name}"
        )

    print()
    print("LARGEST CLASSES")
    print("-" * 70)

    for class_name, count in sorted(
            class_counts.items(),
            key=lambda item: item[1],
            reverse=True,
    )[:10]:

        print(
            f"{count:5}  {class_name}"
        )

    print()
    print("STORAGE")
    print("-" * 70)

    print(
        f"Valid video size     : "
        f"{total_size_gb:.3f} GB"
    )

    print()
    print("OUTPUT")
    print("-" * 70)

    print(
        f"JSON report          : "
        f"{REPORT_JSON}"
    )

    print(
        f"Video CSV            : "
        f"{VIDEO_CSV}"
    )

    print(
        f"Bad video CSV        : "
        f"{BAD_VIDEO_CSV}"
    )

    print()
    print("=" * 70)
    print("Analysis completed.")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    analyze_dataset()