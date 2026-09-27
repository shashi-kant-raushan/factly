"""
Data Loader and Exploratory Data Analysis (EDA) Module.

Handles:
- Dataset acquisition (downloading and unzipping verified benchmark dataset)
- Data ingestion & column validation
- Missing value handling and deduplication
- Title and article text intelligent fusion
- Stratified Train / Validation / Test splitting (No Data Leakage)
- Comprehensive Exploratory Data Analysis & visual reporting
"""

import os
import sys
import zipfile
import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_DATASET_CSV,
    DATA_RAW_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    METADATA_JSON,
    FIGURES_DIR,
    RANDOM_SEED,
    LABEL2ID,
    ID2LABEL
)

DATASET_URL = "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv.zip"


def download_dataset(target_path: Path = RAW_DATASET_CSV) -> Path:
    """
    Downloads the benchmark Fake vs Real News dataset if not already locally cached.
    """
    if target_path.exists() and target_path.stat().st_size > 1000:
        print(f"[DataLoader] Dataset already exists at: {target_path}")
        return target_path

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_RAW_DIR / "fake_or_real_news.csv.zip"

    print(f"[DataLoader] Downloading benchmark dataset from {DATASET_URL}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(DATASET_URL, headers=headers)
    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    print(f"[DataLoader] Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_RAW_DIR)

    if zip_path.exists():
        os.remove(zip_path)

    print(f"[DataLoader] Download and extraction complete. Target: {target_path}")
    return target_path


def load_raw_data(csv_path: Path = RAW_DATASET_CSV) -> pd.DataFrame:
    """
    Loads raw CSV dataset into a pandas DataFrame.
    """
    if not csv_path.exists():
        download_dataset(csv_path)

    df = pd.read_csv(csv_path)
    print(f"[DataLoader] Loaded raw dataset with shape: {df.shape}")
    return df


def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares raw dataset:
    - Normalizes column names
    - Handles missing values in title and text
    - Intelligently merges title and article body
    - Normalizes and encodes labels (REAL: 0, FAKE: 1)
    - Removes duplicate records
    """
    print("[DataLoader] Cleaning and preparing dataset...")

    # Identify relevant columns
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    # Standardize column mapping
    title_col = next((c for c in df.columns if "title" in c or "headline" in c), None)
    text_col = next((c for c in df.columns if "text" in c or "article" in c or "content" in c), None)
    label_col = next((c for c in df.columns if "label" in c or "target" in c or "class" in c), None)

    if not text_col or not label_col:
        raise ValueError(f"Required text/label columns not found in dataset columns: {df.columns.tolist()}")

    # Fill NaNs
    if title_col:
        df[title_col] = df[title_col].fillna("").astype(str).str.strip()
    else:
        df["title"] = ""
        title_col = "title"

    df[text_col] = df[text_col].fillna("").astype(str).str.strip()

    # Intelligent combination of Title and Text
    # If title exists, prefix it nicely to provide crucial context to both models
    def merge_content(row):
        t = row[title_col]
        b = row[text_col]
        if t and b:
            return f"{t} — {b}"
        elif b:
            return b
        else:
            return t

    df["combined_text"] = df.apply(merge_content, axis=1)

    # Filter out empty texts
    df = df[df["combined_text"].str.strip().str.len() > 10].copy()

    # Normalize labels
    if df[label_col].dtype == object:
        df["label_str"] = df[label_col].astype(str).str.strip().str.upper()
        # Map common label variants
        label_map = {
            "REAL": 0, "TRUE": 0, "0": 0, 0: 0,
            "FAKE": 1, "FALSE": 1, "1": 1, 1: 1
        }
        df["label"] = df["label_str"].map(label_map)
    else:
        df["label"] = df[label_col].astype(int)

    # Drop unmapped or missing labels
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Deduplicate based on combined text
    initial_len = len(df)
    df = df.drop_duplicates(subset=["combined_text"]).reset_index(drop=True)
    dedup_dropped = initial_len - len(df)
    print(f"[DataLoader] Removed {dedup_dropped} duplicate samples. Remaining: {len(df)} samples.")

    # Calculate text length metrics for EDA
    df["char_length"] = df["combined_text"].apply(len)
    df["word_count"] = df["combined_text"].apply(lambda x: len(x.split()))

    return df[["title", text_col, "combined_text", "label", "char_length", "word_count"]]


def generate_eda_reports(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> dict:
    """
    Generates exploratory data analysis charts and summary metrics:
    - Class distribution bar chart
    - Word count distribution KDE / histogram by class
    - Summary statistical metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    class_counts = df["label"].value_counts().to_dict()
    real_count = class_counts.get(0, 0)
    fake_count = class_counts.get(1, 0)
    total_count = len(df)

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["REAL (0)", "FAKE (1)"]
    counts = [real_count, fake_count]
    colors = ["#2E7D32", "#C62828"]
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax.set_title("Class Distribution (Real vs Fake News)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Articles", fontsize=12)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height} ({height/total_count*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=300)
    plt.close()

    # 2. Text Word Count Distribution by Class
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df[df["word_count"] < 2500],
        x="word_count",
        hue="label",
        element="step",
        stat="density",
        common_norm=False,
        palette={0: "#2E7D32", 1: "#C62828"},
        bins=50,
        ax=ax
    )
    ax.set_title("Article Word Count Distribution by News Class", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Word Count (capped at 2500 for visualization)", fontsize=12)
    ax.legend(["FAKE (1)", "REAL (0)"], title="Label", frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / "word_count_distribution.png", dpi=300)
    plt.close()

    stats = {
        "total_samples": int(total_count),
        "real_count": int(real_count),
        "fake_count": int(fake_count),
        "real_percentage": round(real_count / total_count * 100, 2),
        "fake_percentage": round(fake_count / total_count * 100, 2),
        "avg_words_real": round(float(df[df['label'] == 0]['word_count'].mean()), 1),
        "avg_words_fake": round(float(df[df['label'] == 1]['word_count'].mean()), 1),
        "median_words_real": int(df[df['label'] == 0]['word_count'].median()),
        "median_words_fake": int(df[df['label'] == 1]['word_count'].median()),
    }

    with open(METADATA_JSON, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"[DataLoader] EDA summary generated: {stats}")
    return stats


def split_and_save_data(df: pd.DataFrame,
                        train_path: Path = TRAIN_CSV,
                        val_path: Path = VAL_CSV,
                        test_path: Path = TEST_CSV) -> tuple:
    """
    Performs stratified split:
    - 70% Train
    - 15% Validation
    - 15% Unseen Test Set
    Guarantees strict separation to prevent data leakage.
    """
    print("[DataLoader] Splitting dataset into Stratified Train (70%), Val (15%), Test (15%)...")

    # Step 1: Split train+val (85%) and test (15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    # Step 2: Split train (70% of total -> ~82.35% of train_val) and val (15% of total -> ~17.65% of train_val)
    val_ratio_in_train_val = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_in_train_val,
        random_state=RANDOM_SEED,
        stratify=train_val_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save to disk
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[DataLoader] Splits successfully saved:")
    print(f"  - Train set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%) -> {train_path}")
    print(f"  - Val set:   {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%) -> {val_path}")
    print(f"  - Test set:  {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%) -> {test_path}")

    return train_df, val_df, test_df


def prepare_pipeline(force_download: bool = False):
    """
    Full data preparation orchestration pipeline.
    """
    raw_path = RAW_DATASET_CSV
    if force_download or not raw_path.exists():
        download_dataset(raw_path)

    df_raw = load_raw_data(raw_path)
    df_clean = clean_and_prepare_data(df_raw)
    generate_eda_reports(df_clean)
    train_df, val_df, test_df = split_and_save_data(df_clean)
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_pipeline()
