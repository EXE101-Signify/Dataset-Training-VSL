from pathlib import Path
import csv
import pickle
import random
import unicodedata
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"D:\BuiGiaHuy\CN8\EXE202\dataset"
)

TRAIN_DIR = DATASET_ROOT / "train"

LABEL_MAPPING_FILE = (
        DATASET_ROOT / "label_mapping.pkl"
)

OUTPUT_DIR = (
        Path(__file__).resolve().parent / "splits"
)

TRAIN_CSV = OUTPUT_DIR / "train.csv"
VAL_CSV = OUTPUT_DIR / "val.csv"
SUMMARY_CSV = OUTPUT_DIR / "split_summary.csv"

VAL_RATIO = 0.20
RANDOM_SEED = 42

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# UNICODE
# ============================================================

def normalize_label(value: str) -> str:
    """
    Normalize Vietnamese Unicode về NFC.

    Ví dụ:

        'An ủi'

    dù được lưu bằng Unicode representation khác nhau
    vẫn được chuẩn hóa thành cùng một representation.
    """

    if not isinstance(value, str):
        raise TypeError(
            f"Label phải là str, nhận được: {type(value)}"
        )

    return unicodedata.normalize(
        "NFC",
        value.strip()
    )


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

def load_label_mapping():

    if not LABEL_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy label_mapping.pkl:\n"
            f"{LABEL_MAPPING_FILE}"
        )

    with LABEL_MAPPING_FILE.open(
            "rb"
    ) as f:

        raw_mapping = pickle.load(f)

    if not isinstance(raw_mapping, dict):
        raise ValueError(
            "label_mapping.pkl không chứa dictionary."
        )

    normalized_mapping = {}

    for raw_label, label_id in raw_mapping.items():

        label = normalize_label(raw_label)

        if label in normalized_mapping:
            raise ValueError(
                "Phát hiện duplicate label sau Unicode normalization:\n"
                f"{label!r}"
            )

        normalized_mapping[label] = int(label_id)

    return normalized_mapping


# ============================================================
# COLLECT VIDEOS
# ============================================================

def collect_videos():

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy train directory:\n"
            f"{TRAIN_DIR}"
        )

    label_mapping = load_label_mapping()

    videos = []

    class_dirs = sorted(
        [
            path
            for path in TRAIN_DIR.iterdir()
            if path.is_dir()
        ],
        key=lambda p: normalize_label(p.name)
    )

    if not class_dirs:
        raise RuntimeError(
            f"Không tìm thấy class directory trong:\n"
            f"{TRAIN_DIR}"
        )

    print(
        f"Found class directories: {len(class_dirs)}"
    )

    # --------------------------------------------------------
    # Check class mapping
    # --------------------------------------------------------

    for class_dir in class_dirs:

        raw_label = class_dir.name
        label = normalize_label(raw_label)

        if label not in label_mapping:

            print()
            print("LABEL MAPPING ERROR")
            print("-" * 60)

            print(
                f"Directory label : {raw_label!r}"
            )

            print(
                f"Normalized label: {label!r}"
            )

            print()
            print(
                "Closest mapping labels:"
            )

            for mapping_label in label_mapping:
                print(
                    f"  {mapping_label!r}"
                )

            raise ValueError(
                f"Class '{raw_label}' không tồn tại "
                f"trong label_mapping.pkl sau Unicode normalization."
            )

        label_id = label_mapping[label]

        video_files = sorted(
            [
                path
                for path in class_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in VIDEO_EXTENSIONS
            )
            ]
        )

        for video_path in video_files:

            relative_path = (
                video_path.relative_to(
                    DATASET_ROOT
                )
            )

            videos.append(
                {
                    "video_path": relative_path.as_posix(),
                    "label": label,
                    "label_id": label_id,
                }
            )

    return videos, label_mapping


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(videos):

    rng = random.Random(
        RANDOM_SEED
    )

    grouped = {}

    for item in videos:

        label = item["label"]

        grouped.setdefault(
            label,
            []
        ).append(item)

    train_items = []
    val_items = []

    for label in sorted(grouped.keys()):

        items = grouped[label].copy()

        rng.shuffle(items)

        total = len(items)

        if total < 2:

            raise ValueError(
                f"Class '{label}' chỉ có {total} video. "
                "Không thể tạo train/validation split."
            )

        val_count = max(
            1,
            round(
                total * VAL_RATIO
            )
        )

        val_count = min(
            val_count,
            total - 1
        )

        class_val = items[
            :val_count
        ]

        class_train = items[
            val_count:
        ]

        train_items.extend(
            class_train
        )

        val_items.extend(
            class_val
        )

    rng.shuffle(train_items)
    rng.shuffle(val_items)

    return train_items, val_items


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(
        path,
        rows
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "video_path",
        "label",
        "label_id",
    ]

    with path.open(
            "w",
            newline="",
            encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
        train_items,
        val_items
):

    train_counter = Counter(
        item["label"]
        for item in train_items
    )

    val_counter = Counter(
        item["label"]
        for item in val_items
    )

    labels = sorted(
        set(train_counter)
        | set(val_counter)
    )

    rows = []

    for label in labels:

        train_count = (
            train_counter[label]
        )

        val_count = (
            val_counter[label]
        )

        rows.append(
            {
                "label": label,
                "train_count": train_count,
                "val_count": val_count,
                "total": (
                        train_count
                        + val_count
                ),
            }
        )

    with SUMMARY_CSV.open(
            "w",
            newline="",
            encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "train_count",
                "val_count",
                "total",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# VALIDATE SPLIT
# ============================================================

def validate_split(
        original_videos,
        train_items,
        val_items
):

    original_paths = {
        item["video_path"]
        for item in original_videos
    }

    train_paths = {
        item["video_path"]
        for item in train_items
    }

    val_paths = {
        item["video_path"]
        for item in val_items
    }

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    if (
            len(train_paths)
            + len(val_paths)
            != len(original_paths)
    ):

        raise RuntimeError(
            "Số lượng video sau split "
            "không khớp dataset gốc."
        )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    missing = (
            original_paths
            - train_paths
            - val_paths
    )

    if missing:

        raise RuntimeError(
            f"Có {len(missing)} video "
            "bị mất khỏi split."
        )

    # --------------------------------------------------------
    # Unexpected
    # --------------------------------------------------------

    unexpected = (
                         train_paths
                         | val_paths
                 ) - original_paths

    if unexpected:

        raise RuntimeError(
            f"Có {len(unexpected)} video "
            "không tồn tại trong dataset gốc."
        )

    # --------------------------------------------------------
    # Train / Val overlap
    # --------------------------------------------------------

    overlap = (
            train_paths
            & val_paths
    )

    if overlap:

        raise RuntimeError(
            f"Có {len(overlap)} video "
            "xuất hiện cả train và validation."
        )

    # --------------------------------------------------------
    # Duplicate path
    # --------------------------------------------------------

    if len(train_paths) != len(train_items):

        raise RuntimeError(
            "Có duplicate video path "
            "trong train split."
        )

    if len(val_paths) != len(val_items):

        raise RuntimeError(
            "Có duplicate video path "
            "trong validation split."
        )

    # --------------------------------------------------------
    # Class coverage
    # --------------------------------------------------------

    original_classes = {
        item["label"]
        for item in original_videos
    }

    train_classes = {
        item["label"]
        for item in train_items
    }

    val_classes = {
        item["label"]
        for item in val_items
    }

    missing_train = (
            original_classes
            - train_classes
    )

    missing_val = (
            original_classes
            - val_classes
    )

    if missing_train:

        raise RuntimeError(
            "Class bị thiếu trong train:\n"
            + "\n".join(
                sorted(missing_train)
            )
        )

    if missing_val:

        raise RuntimeError(
            "Class bị thiếu trong validation:\n"
            + "\n".join(
                sorted(missing_val)
            )
        )

    print(
        "Split validation: PASSED"
    )


# ============================================================
# REPORT
# ============================================================

def print_report(
        videos,
        train_items,
        val_items,
        label_mapping
):

    total = len(videos)

    train_count = len(
        train_items
    )

    val_count = len(
        val_items
    )

    print()
    print("=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    print(
        f"Dataset root        : {DATASET_ROOT}"
    )

    print(
        f"Classes             : "
        f"{len(label_mapping)}"
    )

    print(
        f"Original videos     : {total}"
    )

    print(
        f"Train split         : "
        f"{train_count}"
    )

    print(
        f"Validation split    : "
        f"{val_count}"
    )

    print(
        f"Validation ratio    : "
        f"{val_count / total:.2%}"
    )

    print(
        f"Random seed         : "
        f"{RANDOM_SEED}"
    )

    print()
    print("OUTPUT")
    print("-" * 60)

    print(
        f"Train CSV           : "
        f"{TRAIN_CSV}"
    )

    print(
        f"Validation CSV      : "
        f"{VAL_CSV}"
    )

    print(
        f"Summary CSV         : "
        f"{SUMMARY_CSV}"
    )

    print()
    print("CLASS DISTRIBUTION")
    print("-" * 60)

    train_counter = Counter(
        item["label"]
        for item in train_items
    )

    val_counter = Counter(
        item["label"]
        for item in val_items
    )

    for label in sorted(
            label_mapping.keys()
    ):

        print(
            f"{label:<30} "
            f"train={train_counter[label]:>3} "
            f"val={val_counter[label]:>3}"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CREATE STRATIFIED DATASET SPLIT")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Scan
    # --------------------------------------------------------

    print()
    print("[1/5] Scanning dataset...")

    videos, label_mapping = (
        collect_videos()
    )

    print(
        f"Found videos : {len(videos)}"
    )

    print(
        f"Found classes: "
        f"{len(label_mapping)}"
    )

    # --------------------------------------------------------
    # 2. Validate
    # --------------------------------------------------------

    print()
    print(
        "[2/5] Validating dataset structure..."
    )

    dataset_labels = {
        item["label"]
        for item in videos
    }

    mapping_labels = set(
        label_mapping.keys()
    )

    missing_mapping = (
            dataset_labels
            - mapping_labels
    )

    extra_mapping = (
            mapping_labels
            - dataset_labels
    )

    if missing_mapping:

        raise RuntimeError(
            "Các class không có trong "
            "label_mapping.pkl:\n"
            + "\n".join(
                sorted(missing_mapping)
            )
        )

    if extra_mapping:

        print(
            f"WARNING: mapping có "
            f"{len(extra_mapping)} label "
            "không xuất hiện trong train."
        )

    print(
        "Dataset structure: OK"
    )

    # --------------------------------------------------------
    # 3. Split
    # --------------------------------------------------------

    print()
    print(
        "[3/5] Creating stratified split..."
    )

    train_items, val_items = (
        stratified_split(videos)
    )

    print(
        f"Train      : "
        f"{len(train_items)}"
    )

    print(
        f"Validation : "
        f"{len(val_items)}"
    )

    # --------------------------------------------------------
    # 4. Validate
    # --------------------------------------------------------

    print()
    print(
        "[4/5] Validating split..."
    )

    validate_split(
        videos,
        train_items,
        val_items
    )

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------

    print()
    print(
        "[5/5] Writing CSV files..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    write_csv(
        TRAIN_CSV,
        train_items
    )

    write_csv(
        VAL_CSV,
        val_items
    )

    write_summary(
        train_items,
        val_items
    )

    print_report(
        videos,
        train_items,
        val_items,
        label_mapping
    )

    print()
    print("DONE.")


if __name__ == "__main__":
    main()