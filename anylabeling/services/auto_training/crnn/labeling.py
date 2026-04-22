import argparse
from pathlib import Path
from typing import Set, Tuple


DEFAULT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_image_exts(csv_exts: str) -> Set[str]:
    exts: Set[str] = set()
    for item in (csv_exts or "").split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        exts.add(ext)
    return exts if exts else set(DEFAULT_IMAGE_EXTS)


def extract_label_from_filename(filename: str, split_char: str = "_") -> str:
    stem = Path(filename).stem
    if split_char not in stem:
        return ""
    return stem.split(split_char, 1)[0].strip()


def generate_labels_file(
    data_root: Path,
    labels_file: Path,
    split_char: str = "_",
    image_exts: Set[str] = None,
) -> Tuple[int, int]:
    image_exts = image_exts if image_exts is not None else DEFAULT_IMAGE_EXTS
    lines = []
    skipped = 0

    for image_file in data_root.rglob("*"):
        if not image_file.is_file():
            continue
        if image_file.suffix.lower() not in image_exts:
            continue
        label = extract_label_from_filename(image_file.name, split_char)
        if not label:
            skipped += 1
            continue
        rel_path = image_file.relative_to(data_root).as_posix()
        lines.append(f"{rel_path}\t{label}")

    lines.sort()
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with labels_file.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return len(lines), skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate CRNN labels.txt from image filename prefix."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Image root directory (supports nested subdirectories).",
    )
    parser.add_argument(
        "--labels-file",
        required=True,
        help="Output labels.txt file path.",
    )
    parser.add_argument(
        "--split-char",
        default="_",
        help="Split character in filename. Prefix before this char is label.",
    )
    parser.add_argument(
        "--image-exts",
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Comma-separated image extensions.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    labels_file = Path(args.labels_file).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    image_exts = parse_image_exts(args.image_exts)
    count, skipped = generate_labels_file(
        data_root=data_root,
        labels_file=labels_file,
        split_char=args.split_char,
        image_exts=image_exts,
    )
    print(
        f"[done] generated {count} labels -> {labels_file} (skipped={skipped})"
    )


if __name__ == "__main__":
    main()
