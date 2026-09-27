"""
Script to create the notebooks/EDA_and_Model_Training.ipynb notebook.
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🛡️ Fake News & Misinformation Detection Classifier
### End-to-End NLP Pipeline: Traditional ML vs. Transformer Fine-Tuning

This notebook implements an end-to-end Machine Learning and Natural Language Processing project comparing three distinct paradigms:
1. **TF-IDF + Logistic Regression** (High-speed linear baseline)
2. **TF-IDF + Random Forest** (Non-linear ensemble)
3. **Fine-Tuned DistilBERT** (State-of-the-art transformer with contextual self-attention)

---
"""))

# Cell 1: Environment and Imports
cells.append(nbf.v4.new_code_cell("""# 1. Imports and Reproducibility Setup
import os
import sys
import json
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set styles & seed
sns.set_theme(style="whitegrid", palette="muted")
np.random.seed(42)
torch.manual_seed(42)

# Ensure project root is in path
project_root = Path("..").resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print("PyTorch CUDA Available:", torch.cuda.is_available())
"""))

# Cell 2: Data Loading & EDA
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Ingestion & Exploratory Data Analysis (EDA)
We load the verified benchmark dataset containing news titles, article text, and ground truth labels (`REAL`: 0, `FAKE`: 1).
"""))

cells.append(nbf.v4.new_code_cell("""from src.data_loader import prepare_pipeline

# Execute data acquisition, cleaning, EDA, and stratified train/val/test splits
train_df, val_df, test_df = prepare_pipeline()

print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")
display(train_df.head(3))
"""))

# Cell 3: EDA Visualizations
cells.append(nbf.v4.new_markdown_cell("""### Visualizing Class & Word Count Distributions"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class Distribution
train_df['label'].value_counts().plot(kind='bar', color=['#2E7D32', '#C62828'], ax=axes[0])
axes[0].set_title("Training Set Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_xticklabels(["REAL (0)", "FAKE (1)"], rotation=0)
axes[0].set_ylabel("Count")

# Word Count Distribution
sns.histplot(data=train_df[train_df['word_count'] < 2000], x='word_count', hue='label', palette={0: '#2E7D32', 1: '#C62828'}, ax=axes[1], element="step")
axes[1].set_title("Article Word Count by Label", fontsize=13, fontweight="bold")
axes[1].legend(["FAKE", "REAL"])

plt.tight_layout()
plt.show()
"""))

# Cell 4: Text Preprocessing
cells.append(nbf.v4.new_markdown_cell("""## 3. Text Preprocessing Pipelines
We employ separate preprocessing logic:
- **Traditional Preprocessing:** Lowercasing, HTML/URL stripping, punctuation removal, stopword removal, WordNet lemmatization.
- **Transformer Preprocessing:** Preserving casing, punctuation, and contextual flow for DistilBERT self-attention.
"""))

cells.append(nbf.v4.new_code_cell("""from src.preprocessing import preprocess_traditional, preprocess_transformer, prepare_dataset_features

sample_text = "<b>BREAKING:</b> Secret miracle cure revealed at http://cure.com/now! Doctors are amazed in 2026."
print("Original Text      :", sample_text)
print("Traditional Clean  :", preprocess_traditional(sample_text))
print("Transformer Clean  :", preprocess_transformer(sample_text))
"""))

# Cell 5: TF-IDF + Logistic Regression
cells.append(nbf.v4.new_markdown_cell("""## 4. Model 1: TF-IDF + Logistic Regression"""))

cells.append(nbf.v4.new_code_cell("""from src.train_logistic import run_pipeline as run_logistic

lr_metrics = run_logistic()
"""))

# Cell 6: TF-IDF + Random Forest
cells.append(nbf.v4.new_markdown_cell("""## 5. Model 2: TF-IDF + Random Forest Classifier"""))

cells.append(nbf.v4.new_code_cell("""from src.train_random_forest import run_pipeline as run_rf

rf_metrics = run_rf()
"""))

# Cell 7: DistilBERT Fine-Tuning
cells.append(nbf.v4.new_markdown_cell("""## 6. Model 3: Fine-Tuning DistilBERT (Transformers + PyTorch)"""))

cells.append(nbf.v4.new_code_cell("""from src.train_distilbert import run_pipeline as run_distilbert

db_metrics = run_distilbert()
"""))

# Cell 8: Model Evaluation & Comparison
cells.append(nbf.v4.new_markdown_cell("""## 7. Model Benchmark Comparison & Evaluation"""))

cells.append(nbf.v4.new_code_cell("""from src.evaluate import evaluate_all_models

comparison_results = evaluate_all_models()
"""))

# Cell 9: Explainability
cells.append(nbf.v4.new_markdown_cell("""## 8. Model Explainability & Saliency Analysis
Analyzing token-level contributions for individual articles across the models.
"""))

cells.append(nbf.v4.new_code_cell("""from src.explainability import explain_prediction

test_article = (
    "SHOCKING BREAKING DISCOVERY: Doctors are furious after a whistleblower leaked a secret miracle cure that Big Pharma "
    "has been hiding from the public for decades! Mainstream media is censoring this."
)

print("--- Logistic Regression Explanation ---")
exp_lr = explain_prediction(test_article, model_choice="Logistic Regression")
print(f"Prediction: {exp_lr['prediction']} (Confidence: {exp_lr['confidence']}%)")
print("Top Contributory Tokens:", exp_lr['top_contributing_words'][:6])

print("\\n--- DistilBERT Explanation ---")
exp_db = explain_prediction(test_article, model_choice="DistilBERT")
print(f"Prediction: {exp_db['prediction']} (Confidence: {exp_db['confidence']}%)")
print("Top Contributory Tokens:", exp_db['top_contributing_words'][:6])
"""))

nb.cells = cells

notebook_dir = Path(r"c:\Users\amitk\OneDrive\Desktop\Fake\notebooks")
notebook_dir.mkdir(parents=True, exist_ok=True)
notebook_path = notebook_dir / "EDA_and_Model_Training.ipynb"

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created notebook at {notebook_path}")
