"""
Master Orchestration Script for Training All Models and Generating Benchmarks.

Runs:
1. Dataset acquisition, cleaning, EDA, and stratified train/val/test splits.
2. TF-IDF + Logistic Regression training and evaluation.
3. TF-IDF + Random Forest training and evaluation.
4. DistilBERT fine-tuning and evaluation.
5. Multi-model comparative benchmark report & visualization generation.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import prepare_pipeline as prepare_data
from src.train_logistic import run_pipeline as run_logistic
from src.train_random_forest import run_pipeline as run_random_forest
from src.train_distilbert import run_pipeline as run_distilbert
from src.evaluate import evaluate_all_models


def main():
    print("=" * 70)
    print("      FAKE NEWS & MISINFORMATION DETECTION — TRAINING PIPELINE       ")
    print("=" * 70)
    start_total = time.time()

    # Phase 1: Data Preparation & EDA
    print("\n>>> PHASE 1: DATA PREPARATION & EXPLORATORY DATA ANALYSIS <<<")
    prepare_data()

    # Phase 2 & 3: TF-IDF + Logistic Regression
    print("\n>>> PHASE 2 & 3: TF-IDF + LOGISTIC REGRESSION TRAINING <<<")
    lr_metrics = run_logistic()

    # Phase 4: TF-IDF + Random Forest
    print("\n>>> PHASE 4: TF-IDF + RANDOM FOREST TRAINING <<<")
    rf_metrics = run_random_forest()

    # Phase 5: DistilBERT Fine-tuning
    print("\n>>> PHASE 5: DISTILBERT FINE-TUNING <<<")
    db_metrics = run_distilbert()

    # Phase 6: Final Comparison & Evaluation
    print("\n>>> PHASE 6: FINAL MODEL COMPARISON & BENCHMARKING <<<")
    all_metrics = evaluate_all_models()

    total_time = time.time() - start_total
    print(f"\n[Done] Entire Pipeline successfully completed in {total_time/60:.2f} minutes!")


if __name__ == "__main__":
    main()
