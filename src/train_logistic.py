"""
Training and Evaluation Module for TF-IDF + Logistic Regression Baseline.

Features:
- Fit TfidfVectorizer (ngram_range=(1,2), sublinear_tf, max_features)
- Train Logistic Regression classifier with L2 regularization
- Evaluate on unseen test set (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Serialize trained vectorizer and model artifacts to disk
"""

import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TFIDF_VECTORIZER_PATH,
    LOGISTIC_MODEL_PATH,
    RANDOM_SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    TFIDF_SUBLINEAR_TF,
    LOGISTIC_C,
    LOGISTIC_MAX_ITER
)
from src.preprocessing import prepare_dataset_features


def load_processed_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, val, and test splits.
    """
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, val_df, test_df


def train_logistic_regression(
    train_texts: list,
    train_labels: list,
    val_texts: list = None,
    val_labels: list = None
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Fits TF-IDF Vectorizer and trains Logistic Regression model.
    """
    print(f"[LogisticRegression] Fitting TfidfVectorizer with max_features={TFIDF_MAX_FEATURES}, ngrams={TFIDF_NGRAM_RANGE}, min_df={TFIDF_MIN_DF}, max_df={TFIDF_MAX_DF}...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        strip_accents="unicode",
        stop_words=None
    )

    X_train = vectorizer.fit_transform(train_texts)
    y_train = np.array(train_labels)

    print(f"[LogisticRegression] Training LogisticRegression (C={LOGISTIC_C}, max_iter={LOGISTIC_MAX_ITER})...")
    model = LogisticRegression(
        C=LOGISTIC_C,
        max_iter=LOGISTIC_MAX_ITER,
        random_state=RANDOM_SEED,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)

    if val_texts is not None and val_labels is not None:
        X_val = vectorizer.transform(val_texts)
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"[LogisticRegression] Validation Accuracy: {val_acc:.4f}")

    return vectorizer, model


def evaluate_model(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    test_texts: list,
    test_labels: list
) -> Dict[str, Any]:
    """
    Evaluates the model on test data and returns a comprehensive metrics dictionary.
    """
    X_test = vectorizer.transform(test_texts)
    y_test = np.array(test_labels)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["REAL (0)", "FAKE (1)"], output_dict=True)

    metrics = {
        "model_name": "TF-IDF + Logistic Regression",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "test_size": len(y_test)
    }

    print("\n" + "=" * 55)
    print("      LOGISTIC REGRESSION TEST EVALUATION REPORT      ")
    print("=" * 55)
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
    print("=" * 55 + "\n")

    return metrics


def save_artifacts(vectorizer: TfidfVectorizer, model: LogisticRegression):
    """
    Persists trained vectorizer and model to disk.
    """
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(model, LOGISTIC_MODEL_PATH)
    print(f"[LogisticRegression] Saved vectorizer to {TFIDF_VECTORIZER_PATH}")
    print(f"[LogisticRegression] Saved model to {LOGISTIC_MODEL_PATH}")


def run_pipeline() -> Dict[str, Any]:
    """
    Full pipeline execution for TF-IDF + Logistic Regression.
    """
    train_df, val_df, test_df = load_processed_splits()

    print("[LogisticRegression] Preprocessing texts for traditional ML...")
    train_texts = prepare_dataset_features(train_df, for_transformer=False)
    val_texts = prepare_dataset_features(val_df, for_transformer=False)
    test_texts = prepare_dataset_features(test_df, for_transformer=False)

    train_labels = train_df["label"].tolist()
    val_labels = val_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    vectorizer, model = train_logistic_regression(train_texts, train_labels, val_texts, val_labels)
    metrics = evaluate_model(model, vectorizer, test_texts, test_labels)
    save_artifacts(vectorizer, model)
    return metrics


if __name__ == "__main__":
    run_pipeline()
