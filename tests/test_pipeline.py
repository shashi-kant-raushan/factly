"""
Unit and Integration Tests for Fake News Detection Pipeline.

Verifies:
- Preprocessing pipelines (Traditional vs Transformer)
- Model artifact existence and loading
- Inference and explainability outputs
- SHAP attribution sanity checks
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TFIDF_VECTORIZER_PATH,
    LOGISTIC_MODEL_PATH,
    RANDOM_FOREST_PATH,
    DISTILBERT_DIR,
    TRAIN_CSV,
    TEST_CSV
)
from src.preprocessing import preprocess_traditional, preprocess_transformer
from src.explainability import (
    explain_traditional_prediction,
    explain_distilbert_prediction,
    explain_all_models
)


def test_traditional_preprocessing():
    sample = "<b>BREAKING:</b> Check http://news.com/cure! Doctors are shocking in 2026."
    cleaned = preprocess_traditional(sample)
    assert "<b>" not in cleaned
    assert "http" not in cleaned
    assert "2026" not in cleaned
    assert cleaned.islower()
    assert len(cleaned) > 0


def test_transformer_preprocessing():
    sample = "<b>BREAKING:</b> NASA announced findings on Mars."
    cleaned = preprocess_transformer(sample)
    assert "<b>" not in cleaned
    assert "NASA" in cleaned  # Preserves case
    assert "Mars" in cleaned


def test_processed_splits_exist():
    assert TRAIN_CSV.exists(), "train.csv does not exist"
    assert TEST_CSV.exists(), "test.csv does not exist"


def test_saved_model_artifacts_exist():
    assert TFIDF_VECTORIZER_PATH.exists(), "tfidf_vectorizer.pkl missing"
    assert LOGISTIC_MODEL_PATH.exists(), "logistic_regression.pkl missing"
    assert RANDOM_FOREST_PATH.exists(), "random_forest.pkl missing"
    assert DISTILBERT_DIR.exists(), "models/distilbert missing"


def test_logistic_shap_explainability():
    sample = "NASA officials confirmed satellite data reveals water molecules on the distant exoplanet."
    result = explain_traditional_prediction(sample, model_type="logistic")
    assert "prediction" in result
    assert result["prediction"] in ["REAL", "FAKE"]
    assert "confidence" in result
    assert "probabilities" in result
    assert "top_contributing_words" in result
    assert len(result["top_contributing_words"]) > 0


def test_distilbert_gradient_saliency():
    sample = "SHOCKING: Secret whistleblower leaks mind control memo hidden by government."
    result = explain_distilbert_prediction(sample)
    assert "prediction" in result
    assert result["prediction"] in ["REAL", "FAKE"]
    assert "confidence" in result
    assert "probabilities" in result
    assert "top_contributing_words" in result


def test_all_models_ensemble_comparison():
    sample = "The Federal Reserve kept interest rates steady amid economic growth."
    result = explain_all_models(sample)
    assert "Logistic Regression" in result
    assert "Random Forest" in result
    assert "DistilBERT" in result
    for m in ["Logistic Regression", "Random Forest", "DistilBERT"]:
        assert result[m]["prediction"] in ["REAL", "FAKE"]
        assert result[m]["confidence"] >= 50.0
