import argparse
import ast
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


TARGET_LANGS = ("en", "vi", "zh")
TARGET_DOMAINS = (
    "coursera",
    "food",
    "hotel",
    "laptop",
    "phone",
    "restaurant",
    "sight",
)
TARGET_SPLITS = ("train", "dev", "val")
SENTIMENT_ORDER = ("conflict", "negative", "neutral", "positive")


def default_output_root() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "data_analysis" / "outputs"
    return Path("data_analysis/outputs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate M-ABSA analysis tables and figures for en/vi/zh."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/input/datasets/tuongmacvan/m-absb-for-ppnckh/data"),
        help="Path to downloaded M-ABSA root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Output folder for tables and figures. Defaults to /kaggle/working/data_analysis/outputs on Kaggle.",
    )
    args, _ = parser.parse_known_args()
    return args


def normalize_sentiment(raw_sentiment: str) -> str:
    s = str(raw_sentiment).strip().lower()
    mapping = {
        "pos": "positive",
        "positive": "positive",
        "neg": "negative",
        "negative": "negative",
        "neu": "neutral",
        "neutral": "neutral",
        "conflict": "conflict",
    }
    return mapping.get(s, s)


def normalize_category(raw_category: str) -> str:
    return str(raw_category).strip().lower()


def is_null_aspect(aspect: str) -> bool:
    val = str(aspect).strip().lower()
    return val in {"null", "none", ""}


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


def load_records(data_root: Path):
    sentence_rows = []
    triplet_rows = []

    for domain in TARGET_DOMAINS:
        for lang in TARGET_LANGS:
            for split in TARGET_SPLITS:
                file_path = data_root / domain / lang / f"{split}.txt"
                if not file_path.exists():
                    continue

                with file_path.open("r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, start=1):
                        text, labels = parse_line(line.rstrip("\n"))
                        sentence_rows.append(
                            {
                                "domain": domain,
                                "language": lang,
                                "split": split,
                                "line_no": line_idx,
                                "text": text,
                                "text_length": len(text),
                                "triplet_count": len(labels),
                                "is_missing": len(labels) == 0,
                            }
                        )

                        for triplet_idx, triplet in enumerate(labels):
                            if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                                continue
                            aspect, category, sentiment = triplet
                            triplet_rows.append(
                                {
                                    "domain": domain,
                                    "language": lang,
                                    "split": split,
                                    "line_no": line_idx,
                                    "triplet_idx": triplet_idx,
                                    "aspect": str(aspect).strip(),
                                    "category": normalize_category(category),
                                    "sentiment_raw": str(sentiment).strip(),
                                    "sentiment": normalize_sentiment(sentiment),
                                    "is_null_aspect": is_null_aspect(aspect),
                                }
                            )

    sentence_df = pd.DataFrame(sentence_rows)
    triplet_df = pd.DataFrame(triplet_rows)
    return sentence_df, triplet_df


def ensure_output_dirs(output_root: Path):
    table_dir = output_root / "tables"
    fig_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, fig_dir


def save_table(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_grouped_bar(df: pd.DataFrame, x_col: str, legend_col: str, value_col: str, title: str, ylabel: str, out_path: Path):
    x_vals = list(df[x_col].unique())
    legend_vals = list(df[legend_col].unique())

    x = np.arange(len(x_vals))
    width = 0.8 / max(len(legend_vals), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, leg in enumerate(legend_vals):
        sub = df[df[legend_col] == leg]
        y_map = {row[x_col]: row[value_col] for _, row in sub.iterrows()}
        ys = [y_map.get(xv, 0) for xv in x_vals]
        ax.bar(x + i * width - 0.4 + width / 2, ys, width=width, label=leg)

    ax.set_xticks(x)
    ax.set_xticklabels(x_vals, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=legend_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_heatmap(matrix_df: pd.DataFrame, title: str, out_path: Path):
    data = matrix_df.values
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels(matrix_df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Count")

    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            ax.text(j, i, int(matrix_df.iloc[i, j]), ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_triplets_distribution(triplet_dist: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for lang in TARGET_LANGS:
        sub = triplet_dist[triplet_dist["language"] == lang]
        ax.plot(sub["triplet_count"], sub["count"], marker="o", label=lang)
    ax.set_xlabel("Triplets per sentence")
    ax.set_ylabel("Sentence count")
    ax.set_title("Triplet Count Distribution per Sentence")
    ax.legend(title="language")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_text_length_boxplot(sentence_df: pd.DataFrame, out_path: Path):
    data = [sentence_df[sentence_df["language"] == lang]["text_length"].values for lang in TARGET_LANGS]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, tick_labels=TARGET_LANGS, showfliers=True)
    ax.set_title("Text Length Distribution by Language")
    ax.set_xlabel("language")
    ax.set_ylabel("text length (characters)")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_outputs(sentence_df: pd.DataFrame, triplet_df: pd.DataFrame, table_dir: Path, fig_dir: Path):
    # 1) Overview: sentence count by language
    lang_counts = (
        sentence_df.groupby("language", as_index=False)
        .size()
        .rename(columns={"size": "sentence_count"})
        .sort_values("language")
    )
    save_table(lang_counts, table_dir / "01_language_sentence_counts.csv")

    # 2) Domain distribution by language
    domain_lang = (
        sentence_df.groupby(["domain", "language"], as_index=False)
        .size()
        .rename(columns={"size": "sentence_count"})
    )
    save_table(domain_lang.sort_values(["domain", "language"]), table_dir / "02_domain_language_distribution.csv")
    plot_grouped_bar(
        domain_lang.sort_values(["domain", "language"]),
        x_col="domain",
        legend_col="language",
        value_col="sentence_count",
        title="Domain Distribution by Language",
        ylabel="Sentence count",
        out_path=fig_dir / "01_domain_distribution_by_language.png",
    )

    # 3) Sentiment distribution by language (triplet-level)
    sentiment_lang = (
        triplet_df.groupby(["language", "sentiment"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    sentiment_lang["sentiment"] = pd.Categorical(
        sentiment_lang["sentiment"], categories=list(SENTIMENT_ORDER), ordered=True
    )
    sentiment_lang = sentiment_lang.sort_values(["language", "sentiment"])
    save_table(sentiment_lang, table_dir / "03_sentiment_distribution_by_language.csv")
    plot_grouped_bar(
        sentiment_lang,
        x_col="sentiment",
        legend_col="language",
        value_col="count",
        title="Sentiment Distribution by Language",
        ylabel="Triplet count",
        out_path=fig_dir / "02_sentiment_distribution_by_language.png",
    )

    # 4) Aspect category analysis
    category_counts = (
        triplet_df.groupby(["language", "category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["language", "count"], ascending=[True, False])
    )
    save_table(category_counts, table_dir / "04_category_counts_by_language.csv")

    topk = 20
    top_category_rows = []
    for lang in TARGET_LANGS:
        top_lang = category_counts[category_counts["language"] == lang].head(topk).copy()
        top_category_rows.append(top_lang)
        if top_lang.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(top_lang["category"][::-1], top_lang["count"][::-1])
        ax.set_title(f"Top {topk} Aspect Categories ({lang})")
        ax.set_xlabel("Triplet count")
        ax.set_ylabel("Category")
        fig.tight_layout()
        fig.savefig(fig_dir / f"03_top_categories_{lang}.png", dpi=220)
        plt.close(fig)

    top_categories = pd.concat(top_category_rows, ignore_index=True) if top_category_rows else pd.DataFrame()
    save_table(top_categories, table_dir / "05_top20_categories_by_language.csv")

    # 5) Domain x Sentiment heatmap per language
    for lang in TARGET_LANGS:
        sub = triplet_df[triplet_df["language"] == lang]
        matrix = (
            sub.groupby(["domain", "sentiment"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .pivot(index="domain", columns="sentiment", values="count")
            .fillna(0)
        )
        matrix = matrix.reindex(index=list(TARGET_DOMAINS), fill_value=0)
        matrix = matrix.reindex(columns=list(SENTIMENT_ORDER), fill_value=0)
        matrix = matrix.astype(int)
        matrix.to_csv(table_dir / f"06_domain_sentiment_matrix_{lang}.csv", encoding="utf-8-sig")
        plot_heatmap(
            matrix,
            title=f"Domain x Sentiment ({lang})",
            out_path=fig_dir / f"04_heatmap_domain_sentiment_{lang}.png",
        )

    # 6) Missing analysis (sentence-level)
    missing = (
        sentence_df.groupby("language")
        .agg(total_sentences=("is_missing", "size"), missing_count=("is_missing", "sum"))
        .reset_index()
    )
    missing["missing_rate"] = missing["missing_count"] / missing["total_sentences"]
    save_table(missing.sort_values("language"), table_dir / "07_missing_triplet_summary.csv")

    # 7) NULL aspect analysis (triplet-level)
    null_stats = (
        triplet_df.groupby("language")
        .agg(total_triplets=("is_null_aspect", "size"), null_count=("is_null_aspect", "sum"))
        .reset_index()
    )
    null_stats["explicit_count"] = null_stats["total_triplets"] - null_stats["null_count"]
    null_stats["null_rate"] = null_stats["null_count"] / null_stats["total_triplets"]
    null_stats = null_stats[["language", "null_count", "explicit_count", "null_rate", "total_triplets"]]
    save_table(null_stats.sort_values("language"), table_dir / "08_null_aspect_summary.csv")

    # 8) Text length statistics
    text_stats = (
        sentence_df.groupby("language")["text_length"]
        .agg(min_length="min", median_length="median", mean_length="mean", max_length="max")
        .reset_index()
    )
    text_stats["mean_length"] = text_stats["mean_length"].round(2)
    text_stats["median_length"] = text_stats["median_length"].round(2)
    save_table(text_stats.sort_values("language"), table_dir / "09_text_length_stats.csv")
    plot_text_length_boxplot(sentence_df, fig_dir / "05_text_length_boxplot.png")

    # 9) Triplet count per sentence distribution
    triplet_dist = (
        sentence_df.groupby(["language", "triplet_count"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["language", "triplet_count"])
    )
    save_table(triplet_dist, table_dir / "10_triplet_count_distribution.csv")
    plot_triplets_distribution(triplet_dist, fig_dir / "06_triplet_count_distribution.png")

    # 10) Quick summary markdown for direct reading
    summary_path = table_dir / "00_quick_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# M-ABSA Analysis Summary (en/vi/zh)\n\n")
        f.write("## Total Sentences by Language\n\n")
        f.write(lang_counts.to_string(index=False))
        f.write("\n\n## Missing Triplet Summary\n\n")
        f.write(missing.to_string(index=False))
        f.write("\n\n## NULL Aspect Summary\n\n")
        f.write(null_stats.to_string(index=False))
        f.write("\n\n## Text Length Statistics\n\n")
        f.write(text_stats.to_string(index=False))
        f.write("\n")


def main():
    args = parse_args()
    table_dir, fig_dir = ensure_output_dirs(args.output_root)

    sentence_df, triplet_df = load_records(args.data_root)
    if sentence_df.empty:
        raise RuntimeError(f"No sentence data found under: {args.data_root}")
    if triplet_df.empty:
        raise RuntimeError("No triplet data parsed. Please check data format and path.")

    generate_outputs(sentence_df, triplet_df, table_dir, fig_dir)

    print("Analysis finished.")
    print(f"Tables:  {table_dir}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
