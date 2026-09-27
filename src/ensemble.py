"""
Hybrid Super-Ensemble Classifier for Fake News Detection.

Combines all three model paradigms:
1. TF-IDF + Logistic Regression (Linear)
2. TF-IDF + Random Forest (Non-linear Tree Ensemble)
3. Fine-tuned DistilBERT (Contextual Deep Transformer)

Uses Weighted Soft-Voting / Stacking Meta-Learner to maximize test accuracy and reduce classification variance.
"""

import sys
import json
import torch
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
from typing import Dict, Any, Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    VAL_CSV,
    TEST_CSV,
    TFIDF_VECTORIZER_PATH,
    LOGISTIC_MODEL_PATH,
    RANDOM_FOREST_PATH,
    DISTILBERT_DIR,
    MODELS_DIR,
    ID2LABEL
)
from src.preprocessing import prepare_dataset_features, preprocess_traditional, preprocess_transformer

ENSEMBLE_WEIGHTS_PATH = MODELS_DIR / "ensemble_weights.json"


def get_model_probabilities(texts_trad: List[str], texts_trans: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes probabilities for LR, RF, and DistilBERT across document lists.
    """
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    lr_model = joblib.load(LOGISTIC_MODEL_PATH)
    rf_model = joblib.load(RANDOM_FOREST_PATH)

    # 1. TF-IDF Transform
    X_tfidf = vectorizer.transform(texts_trad)

    # 2. LR Probabilities
    lr_probs = lr_model.predict_proba(X_tfidf)[:, 1]

    # 3. RF Probabilities
    rf_probs = rf_model.predict_proba(X_tfidf)[:, 1]

    # 4. DistilBERT Probabilities
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    db_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    db_model.to(device)
    db_model.eval()

    db_probs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts_trans), batch_size):
            batch = texts_trans[i:i + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
            out = db_model(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            db_probs.extend(probs.tolist())

    db_probs = np.array(db_probs)

    return lr_probs, rf_probs, db_probs


def find_optimal_ensemble_weights() -> Dict[str, float]:
    """
    Optimizes ensemble weights using the validation split to maximize F1 / Accuracy.
    """
    print("[Ensemble] Finding optimal combination weights on validation set...")
    val_df = pd.read_csv(VAL_CSV)
    y_val = val_df["label"].values

    texts_trad = prepare_dataset_features(val_df, for_transformer=False)
    texts_trans = prepare_dataset_features(val_df, for_transformer=True)

    lr_val, rf_val, db_val = get_model_probabilities(texts_trad, texts_trans)

    # Objective: Minimize log-loss / negative F1
    def loss_func(weights):
        w1, w2, w3 = weights
        w_sum = w1 + w2 + w3
        if w_sum <= 0:
            return 1e5
        p_ens = (w1 * lr_val + w2 * rf_val + w3 * db_val) / w_sum
        p_clipped = np.clip(p_ens, 1e-6, 1.0 - 1e-6)
        log_loss = -np.mean(y_val * np.log(p_clipped) + (1 - y_val) * np.log(1 - p_clipped))
        return log_loss

    initial_weights = [0.33, 0.33, 0.34]
    bounds = [(0.05, 0.8), (0.05, 0.8), (0.1, 0.85)]
    res = minimize(loss_func, initial_weights, bounds=bounds, method="L-BFGS-B")

    raw_w = res.x
    norm_w = raw_w / np.sum(raw_w)

    weights_dict = {
        "Logistic_Regression": round(float(norm_w[0]), 4),
        "Random_Forest": round(float(norm_w[1]), 4),
        "DistilBERT": round(float(norm_w[2]), 4)
    }

    with open(ENSEMBLE_WEIGHTS_PATH, "w") as f:
        json.dump(weights_dict, f, indent=4)

    print(f"[Ensemble] Optimal weights found: {weights_dict}")
    return weights_dict


def evaluate_ensemble() -> Dict[str, Any]:
    """
    Evaluates the Hybrid Ensemble on the unseen test set.
    """
    if not ENSEMBLE_WEIGHTS_PATH.exists():
        find_optimal_ensemble_weights()

    with open(ENSEMBLE_WEIGHTS_PATH, "r") as f:
        weights = json.load(f)

    w_lr = weights["Logistic_Regression"]
    w_rf = weights["Random_Forest"]
    w_db = weights["DistilBERT"]

    test_df = pd.read_csv(TEST_CSV)
    y_test = test_df["label"].values

    texts_trad = prepare_dataset_features(test_df, for_transformer=False)
    texts_trans = prepare_dataset_features(test_df, for_transformer=True)

    print("[Ensemble] Evaluating Hybrid Ensemble on Unseen Test Set...")
    lr_test, rf_test, db_test = get_model_probabilities(texts_trad, texts_trans)

    # Compute weighted soft voting probabilities
    y_proba = (w_lr * lr_test + w_rf * rf_test + w_db * db_test)
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["REAL (0)", "FAKE (1)"], output_dict=True)

    metrics = {
        "model_name": "Hybrid Super-Ensemble (LR + RF + DistilBERT)",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "weights": weights,
        "test_size": len(y_test)
    }

    print("\n" + "=" * 65)
    print("      HYBRID SUPER-ENSEMBLE TEST EVALUATION REPORT      ")
    print("=" * 65)
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Weights   : LR={w_lr:.2f}, RF={w_rf:.2f}, DistilBERT={w_db:.2f}")
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
    print("=" * 65 + "\n")

    return metrics


def predict_ensemble(text: str) -> Dict[str, Any]:
    """
    Real-time ensemble prediction combining all three models with token-level fusion.
    """
    from src.explainability import (
        explain_traditional_prediction,
        explain_distilbert_prediction
    )

    if not ENSEMBLE_WEIGHTS_PATH.exists():
        find_optimal_ensemble_weights()

    with open(ENSEMBLE_WEIGHTS_PATH, "r") as f:
        weights = json.load(f)

    w_lr = weights.get("Logistic_Regression", 0.3)
    w_rf = weights.get("Random_Forest", 0.2)
    w_db = weights.get("DistilBERT", 0.5)

    res_lr = explain_traditional_prediction(text, model_type="logistic")
    res_rf = explain_traditional_prediction(text, model_type="random_forest")
    res_db = explain_distilbert_prediction(text)

    # Extract individual probabilities defensively
    def extract_prob_and_meta(res, default_weight):
        if not res or "error" in res:
            return 0.5, "UNKNOWN", 50.0, default_weight
        p_fake = res.get("probabilities", {}).get("FAKE", 50.0) / 100.0
        pred = res.get("prediction", "UNKNOWN")
        conf = res.get("confidence", 50.0)
        return p_fake, pred, conf, default_weight

    p_fake_lr, pred_lr, conf_lr, w_lr = extract_prob_and_meta(res_lr, w_lr)
    p_fake_rf, pred_rf, conf_rf, w_rf = extract_prob_and_meta(res_rf, w_rf)
    p_fake_db, pred_db, conf_db, w_db = extract_prob_and_meta(res_db, w_db)

    # Weighted Ensemble Probability
    total_w = w_lr + w_rf + w_db
    if total_w > 0:
        p_fake_ens = (w_lr * p_fake_lr + w_rf * p_fake_rf + w_db * p_fake_db) / total_w
    else:
        p_fake_ens = 0.5
    p_real_ens = 1.0 - p_fake_ens

    pred_label_id = 1 if p_fake_ens >= 0.5 else 0
    confidence = p_fake_ens if pred_label_id == 1 else p_real_ens
    label_str = ID2LABEL[pred_label_id]

    # Combine top tokens from both linear SHAP and transformer gradient saliency
    combined_tokens = {}
    for tok in res_lr.get("top_contributing_words", []):
        w = tok.get("word")
        if w:
            combined_tokens[w] = combined_tokens.get(w, 0.0) + w_lr * tok.get("score", 0.0)

    for tok in res_db.get("top_contributing_words", []):
        w = tok.get("clean_token") or tok.get("word")
        if w:
            combined_tokens[w] = combined_tokens.get(w, 0.0) + w_db * tok.get("score", 0.0)

    fused_tokens = [
        {
            "word": k,
            "score": round(float(v), 4),
            "favors": "FAKE" if v > 0 else "REAL",
            "magnitude": round(abs(float(v)), 4)
        }
        for k, v in combined_tokens.items()
    ]
    fused_tokens.sort(key=lambda x: x["magnitude"], reverse=True)

    return {
        "model_type": "🔥 Hybrid Super-Ensemble (LR + RF + DistilBERT)",
        "prediction": label_str,
        "prediction_id": pred_label_id,
        "confidence": round(float(confidence * 100), 2),
        "probabilities": {
            "REAL": round(float(p_real_ens * 100), 2),
            "FAKE": round(float(p_fake_ens * 100), 2)
        },
        "individual_models": {
            "Logistic Regression": {"pred": pred_lr, "conf": conf_lr, "weight": w_lr},
            "Random Forest": {"pred": pred_rf, "conf": conf_rf, "weight": w_rf},
            "DistilBERT": {"pred": pred_db, "conf": conf_db, "weight": w_db}
        },
        "top_contributing_words": fused_tokens[:12],
        "method": "Multi-Model Soft-Voting Ensemble (Weighted Probability Fusion & Cross-Saliency)"
    }


if __name__ == "__main__":
    evaluate_ensemble()
