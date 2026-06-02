import argparse
import ast
import re
from pathlib import Path


TARGET_DOMAINS = (
    "coursera",
    "food",
    "hotel",
    "laptop",
    "phone",
    "restaurant",
    "sight",
)
TARGET_LANGS = ("en", "vi", "zh")
TARGET_SPLITS = ("train", "dev", "val")

SENTIMENT_MAP = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "conflict": "conflict",
}


def default_output_root():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "data"
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build clean_all.txt from train/dev/val for M-ABSA (en/vi/zh)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/input/datasets/tuongmacvan/m-absb-for-ppnckh/data"),
        help="Root folder of raw M-ABSA dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Root folder for generated clean_all.txt files. Defaults to /kaggle/working/data on Kaggle.",
    )
    parser.add_argument(
        "--keep-conflict",
        action="store_true",
        help="Keep conflict sentiment. Default behavior removes conflict to keep 3 labels only.",
    )
    parser.add_argument(
        "--dedupe-key",
        choices=("text", "text_and_labels"),
        default="text",
        help="How to remove duplicates (default: text only).",
    )
    args, _ = parser.parse_known_args()
    return args


def normalize_sentiment(value: str):
    s = str(value).strip().lower()
    return SENTIMENT_MAP.get(s)


def normalize_category(value: str):
    return str(value).strip().lower()


def remove_emoji_and_symbols(text: str):
    # Remove common emoji/symbol blocks but keep letters, numbers, punctuation, and CJK text.
    pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\u2600-\u26FF"
        "]+",
        flags=re.UNICODE,
    )
    return pattern.sub("", text)


def clean_text(text: str):
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = remove_emoji_and_symbols(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_line(line: str):
    if "####" not in line:
        return line.strip(), []
    text_part, label_part = line.split("####", 1)
    text = text_part.strip()
    try:
        labels = ast.literal_eval(label_part.strip())
    except Exception:
        labels = []
    if not isinstance(labels, list):
        labels = []
    return text, labels


def normalize_labels(raw_labels, keep_conflict=False):
    normalized = []
    for item in raw_labels:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        aspect, category, sentiment = item
        sentiment_norm = normalize_sentiment(sentiment)
        if sentiment_norm is None:
            continue
        if sentiment_norm == "conflict" and not keep_conflict:
            continue
        aspect_norm = str(aspect).strip()
        category_norm = normalize_category(category)
        if not category_norm:
            continue
        normalized.append([aspect_norm, category_norm, sentiment_norm])
    return normalized


def load_split_lines(file_path: Path):
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_clean_rows(lines, keep_conflict=False, dedupe_key="text"):
    clean_rows = []
    seen = set()

    for line in lines:
        text_raw, labels_raw = parse_line(line)
        text = clean_text(text_raw)
        if not text:
            continue

        labels = normalize_labels(labels_raw, keep_conflict=keep_conflict)
        if not labels:
            # Drop missing labels [] after normalization.
            continue

        key = text if dedupe_key == "text" else (text, str(labels))
        if key in seen:
            continue
        seen.add(key)

        clean_rows.append(f"{text}####{labels}")

    return clean_rows


def write_clean_all(target_dir: Path, rows):
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "clean_all.txt"
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(row + "\n")
    return out_path


def main():
    args = parse_args()
    output_root = args.output_root if args.output_root is not None else args.data_root

    total_written = 0
    generated_files = []

    for domain in TARGET_DOMAINS:
        for lang in TARGET_LANGS:
            lang_dir = args.data_root / domain / lang
            if not lang_dir.exists():
                continue
            target_dir = output_root / domain / lang

            merged_lines = []
            for split in TARGET_SPLITS:
                split_file = lang_dir / f"{split}.txt"
                merged_lines.extend(load_split_lines(split_file))

            clean_rows = build_clean_rows(
                merged_lines,
                keep_conflict=args.keep_conflict,
                dedupe_key=args.dedupe_key,
            )
            out_path = write_clean_all(target_dir, clean_rows)
            total_written += len(clean_rows)
            generated_files.append((out_path, len(clean_rows)))

    print("Preprocessing completed.")
    print(f"Generated {len(generated_files)} clean_all.txt files.")
    print(f"Total cleaned rows written: {total_written}")
    for out_path, count in generated_files:
        print(f"{out_path} -> {count} rows")


if __name__ == "__main__":
    main()
