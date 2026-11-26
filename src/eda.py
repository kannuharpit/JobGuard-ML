"""
eda.py
------
Exploratory Data Analysis: generate all EDA charts and save to reports/.
Run: python src/eda.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from preprocess import load_and_preprocess, TEXT_COLS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(PROJECT_ROOT, "data", "fake_job_postings.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    df = load_and_preprocess(CSV_PATH)
    raw = pd.read_csv(CSV_PATH)

    # ── 1. Class Distribution ─────────────────────────────────────────────────
    counts = raw["fraudulent"].value_counts().sort_index()
    labels = ["Legitimate", "Fraudulent"]
    colors = ["#4C72B0", "#DD8452"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=140, pctdistance=0.8)
    axes[1].set_title("Class Imbalance Ratio")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)
    print("[EDA] Saved class_distribution.png")

    # ── 2. Missing Values Heatmap ─────────────────────────────────────────────
    missing_pct = (raw.isnull().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(missing_pct.index, missing_pct.values, color="#4C72B0")
    ax.set_xlabel("Missing Values (%)")
    ax.set_title("Missing Value Percentage by Column")
    ax.axvline(50, color="red", linestyle="--", linewidth=1, label="50% threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "missing_values.png"), dpi=150)
    plt.close(fig)
    print("[EDA] Saved missing_values.png")

    # ── 3. Categorical Feature Distributions ──────────────────────────────────
    cat_cols = ["employment_type", "required_experience", "required_education"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, cat_cols):
        vc = raw[col].fillna("Unknown").value_counts().head(8)
        vc.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "categorical_distributions.png"), dpi=150)
    plt.close(fig)
    print("[EDA] Saved categorical_distributions.png")

    # ── 4. Text Length Distribution ───────────────────────────────────────────
    df["desc_word_count"] = df["description"].fillna("").apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(8, 4))
    for label_val, color, name in [(0, "#4C72B0", "Legitimate"), (1, "#DD8452", "Fraudulent")]:
        subset = df[df["fraudulent"] == label_val]["desc_word_count"]
        ax.hist(subset.clip(upper=2000), bins=50, alpha=0.6, color=color, label=name)
    ax.set_xlabel("Description Word Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Job Description Length Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "text_length_distribution.png"), dpi=150)
    plt.close(fig)
    print("[EDA] Saved text_length_distribution.png")

    # ── 5. Word Clouds ────────────────────────────────────────────────────────
    fraud_text  = " ".join(df[df["fraudulent"] == 1]["processed_text"].fillna("").values)
    legit_text  = " ".join(df[df["fraudulent"] == 0]["processed_text"].fillna("").values)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, text, title, cmap in [
        (axes[0], fraud_text,  "Fraudulent Postings",  "Reds"),
        (axes[1], legit_text,  "Legitimate Postings",  "Blues"),
    ]:
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap=cmap, max_words=100).generate(text or "no text")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "wordclouds.png"), dpi=150)
    plt.close(fig)
    print("[EDA] Saved wordclouds.png")

    print("[EDA] All EDA charts generated in reports/")


if __name__ == "__main__":
    main()
