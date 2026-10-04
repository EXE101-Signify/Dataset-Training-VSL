from pathlib import Path
import pickle

DATASET_ROOT = Path(r"D:\BuiGiaHuy\CN8\EXE202\dataset")
LABEL_MAPPING_FILE = DATASET_ROOT / "label_mapping.pkl"


def main():
    print("=" * 70)
    print("INSPECT LABEL MAPPING")
    print("=" * 70)

    with LABEL_MAPPING_FILE.open("rb") as f:
        mapping = pickle.load(f)

    print()
    print("TYPE:")
    print(type(mapping))

    print()
    print("LENGTH:")
    print(len(mapping))

    print()
    print("FIRST 30 ITEMS:")
    print("-" * 70)

    for index, (key, value) in enumerate(mapping.items()):
        print(
            f"{index:>3}: "
            f"key={key!r:<35} "
            f"type(key)={type(key).__name__:<10} "
            f"value={value!r:<10} "
            f"type(value)={type(value).__name__}"
        )

        if index >= 29:
            break

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()