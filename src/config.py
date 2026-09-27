"""
Configuration module for Fake News / Misinformation Detection System.
Centralizes all file paths, random seeds, hyperparameter settings, and model configurations.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DIR = BASE_DIR / "app"

# Ensure essential directories exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset paths
RAW_DATASET_CSV = DATA_RAW_DIR / "fake_or_real_news.csv"
TRAIN_CSV = DATA_PROCESSED_DIR / "train.csv"
VAL_CSV = DATA_PROCESSED_DIR / "val.csv"
TEST_CSV = DATA_PROCESSED_DIR / "test.csv"
METADATA_JSON = DATA_PROCESSED_DIR / "dataset_metadata.json"

# Saved Model Paths
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.pkl"
DISTILBERT_DIR = MODELS_DIR / "distilbert"
EVALUATION_METRICS_PATH = MODELS_DIR / "model_comparison_metrics.json"

# Reproducibility
RANDOM_SEED = 42

# Label Mapping
# 0: REAL, 1: FAKE
LABEL2ID = {"REAL": 0, "FAKE": 1}
ID2LABEL = {0: "REAL", 1: "FAKE"}

# Preprocessing Hyperparameters
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True

# Traditional ML Hyperparameters
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 35
RF_MIN_SAMPLES_SPLIT = 5

# DistilBERT Fine-tuning Parameters
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
DISTILBERT_MAX_LENGTH = 96
DISTILBERT_BATCH_SIZE = 32
DISTILBERT_LEARNING_RATE = 3e-5
DISTILBERT_EPOCHS = 2
DISTILBERT_WEIGHT_DECAY = 0.01
DISTILBERT_WARMUP_RATIO = 0.1
