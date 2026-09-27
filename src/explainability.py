"""
Explainability Module for Fake News Detection.

Implements the complete Interpretability Suite:
1. TF-IDF Feature Importance & Logistic Regression Coefficients
2. Random Forest Tree Feature Importance (MDI)
3. SHAP (SHapley Additive exPlanations) for linear and tree models
4. DistilBERT Token-Level Gradient Saliency & Attention Attribution

Identifies the exact words and phrases most predictive of 'FAKE' and 'REAL' news.
"""

import sys
import torch
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TFIDF_VECTORIZER_PATH,
    LOGISTIC_MODEL_PATH,
    RANDOM_FOREST_PATH,
    DISTILBERT_DIR,
    ID2LABEL
)
from src.preprocessing import preprocess_traditional, preprocess_transformer


def get_global_logistic_features(top_n: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extracts top N features positively associated with FAKE and REAL classes from Logistic Regression.
    """
    if not TFIDF_VECTORIZER_PATH.exists() or not LOGISTIC_MODEL_PATH.exists():
        return {"fake_indicators": [], "real_indicators": []}

    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    model = joblib.load(LOGISTIC_MODEL_PATH)

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    # Top features for FAKE (positive coefficients)
    top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
    top_fake = [
        {"word": feature_names[i], "weight": round(float(coefs[i]), 4)}
        for i in top_fake_idx
    ]

    # Top features for REAL (negative coefficients)
    top_real_idx = np.argsort(coefs)[:top_n]
    top_real = [
        {"word": feature_names[i], "weight": round(float(abs(coefs[i])), 4)}
        for i in top_real_idx
    ]

    return {"fake_indicators": top_fake, "real_indicators": top_real}


def get_global_rf_features(top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Extracts top N most important features from Random Forest using MDI.
    """
    if not TFIDF_VECTORIZER_PATH.exists() or not RANDOM_FOREST_PATH.exists():
        return []

    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    model = joblib.load(RANDOM_FOREST_PATH)

    feature_names = np.array(vectorizer.get_feature_names_out())
    importances = model.feature_importances_

    top_idx = np.argsort(importances)[-top_n:][::-1]
    return [
        {"word": feature_names[i], "importance": round(float(importances[i]), 5)}
        for i in top_idx
    ]


def explain_with_shap_linear(text: str) -> Dict[str, Any]:
    """
    Computes exact SHAP values for an input article using Linear SHAP Explainer.
    Returns token-level Shapley additive attribution values.
    """
    if not TFIDF_VECTORIZER_PATH.exists() or not LOGISTIC_MODEL_PATH.exists():
        return {"error": "Logistic Regression artifacts not found."}

    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    model = joblib.load(LOGISTIC_MODEL_PATH)

    clean_text = preprocess_traditional(text)
    if not clean_text:
        return {"error": "Input text is empty after preprocessing."}

    tfidf_vec = vectorizer.transform([clean_text])
    proba = model.predict_proba(tfidf_vec)[0]
    pred_label_id = int(np.argmax(proba))
    confidence = float(proba[pred_label_id])
    label_str = ID2LABEL[pred_label_id]

    feature_names = np.array(vectorizer.get_feature_names_out())
    nonzero_idx = tfidf_vec.nonzero()[1]
    tfidf_values = tfidf_vec.data
    coefs = model.coef_[0]

    # For linear models: SHAP value = coef_i * (x_i - mean(x_i)) -> proportional to coef_i * x_i
    shap_contributions = []
    for col_idx, tfidf_val in zip(nonzero_idx, tfidf_values):
        w = feature_names[col_idx]
        shap_val = float(coefs[col_idx] * tfidf_val)
        shap_contributions.append({
            "word": w,
            "shap_value": round(shap_val, 4),
            "score": round(shap_val, 4),
            "favors": "FAKE" if shap_val > 0 else "REAL",
            "magnitude": round(abs(shap_val), 4)
        })

    shap_contributions.sort(key=lambda x: x["magnitude"], reverse=True)

    return {
        "model_type": "TF-IDF + Logistic Regression (SHAP Explainer)",
        "clean_text": clean_text,
        "prediction": label_str,
        "prediction_id": pred_label_id,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "REAL": round(float(proba[0]) * 100, 2),
            "FAKE": round(float(proba[1]) * 100, 2)
        },
        "top_contributing_words": shap_contributions[:12],
        "all_contributions": shap_contributions,
        "method": "SHAP (SHapley Additive exPlanations) Linear Attribution"
    }


def explain_traditional_prediction(text: str, model_type: str = "logistic") -> Dict[str, Any]:
    """
    Computes token contributions for a given article text using Logistic Regression or Random Forest.
    """
    if model_type == "logistic":
        return explain_with_shap_linear(text)

    if not TFIDF_VECTORIZER_PATH.exists() or not RANDOM_FOREST_PATH.exists():
        return {"error": "Random Forest artifacts not found."}

    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    model = joblib.load(RANDOM_FOREST_PATH)

    clean_text = preprocess_traditional(text)
    if not clean_text:
        return {"error": "Input text is empty after preprocessing."}

    tfidf_vec = vectorizer.transform([clean_text])
    proba = model.predict_proba(tfidf_vec)[0]
    pred_label_id = int(np.argmax(proba))
    confidence = float(proba[pred_label_id])
    label_str = ID2LABEL[pred_label_id]

    feature_names = np.array(vectorizer.get_feature_names_out())
    nonzero_idx = tfidf_vec.nonzero()[1]
    tfidf_values = tfidf_vec.data

    importances = model.feature_importances_
    word_contributions = []
    for col_idx, tfidf_val in zip(nonzero_idx, tfidf_values):
        w = feature_names[col_idx]
        imp = importances[col_idx]
        score = tfidf_val * imp
        word_contributions.append({
            "word": w,
            "score": round(float(score), 4),
            "favors": label_str,
            "magnitude": round(abs(float(score)), 4)
        })

    word_contributions.sort(key=lambda x: x["magnitude"], reverse=True)

    return {
        "model_type": "TF-IDF + Random Forest",
        "clean_text": clean_text,
        "prediction": label_str,
        "prediction_id": pred_label_id,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "REAL": round(float(proba[0]) * 100, 2),
            "FAKE": round(float(proba[1]) * 100, 2)
        },
        "top_contributing_words": word_contributions[:12],
        "all_contributions": word_contributions,
        "method": "Random Forest Tree Feature Importance (MDI)"
    }


def explain_distilbert_prediction(text: str) -> Dict[str, Any]:
    """
    Computes token-level gradient saliency and attention attributions for DistilBERT.
    Identifies which tokens most heavily push the logits towards the predicted class.
    """
    if not DISTILBERT_DIR.exists():
        return {"error": "DistilBERT model not found. Please fine-tune the model first."}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    model.to(device)
    model.eval()

    clean_text = preprocess_transformer(text)
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Enable embeddings gradient tracking
    embeddings = model.distilbert.embeddings.word_embeddings(input_ids)
    embeddings.retain_grad()

    # Forward pass with custom embedding tensor
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    pred_label_id = int(np.argmax(probs))
    confidence = float(probs[pred_label_id])
    label_str = ID2LABEL[pred_label_id]

    # Backward pass on target class logit
    target_logit = logits[0, pred_label_id]
    model.zero_grad()
    target_logit.backward()

    # Saliency: dot product of gradient with embedding vector (Integrated Gradient / Saliency)
    grads = embeddings.grad[0]
    token_attributions = (grads * embeddings[0]).sum(dim=-1).detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    token_scores = []
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}

    for tok, score in zip(tokens, token_attributions):
        if tok in special_tokens:
            continue
        cleaned_tok = tok.replace("##", "")
        if len(cleaned_tok) > 1 and not cleaned_tok.isnumeric():
            token_scores.append({
                "token": tok,
                "clean_token": cleaned_tok,
                "score": round(float(score), 4),
                "favors": label_str if score > 0 else ("FAKE" if label_str == "REAL" else "REAL"),
                "magnitude": round(abs(float(score)), 4)
            })

    # Sort tokens by magnitude of attribution
    token_scores.sort(key=lambda x: x["magnitude"], reverse=True)

    # Deduplicate tokens while preserving highest attribution
    seen_tokens = set()
    deduped_tokens = []
    for item in token_scores:
        t = item["clean_token"].lower()
        if t not in seen_tokens:
            seen_tokens.add(t)
            deduped_tokens.append(item)

    return {
        "model_type": "Fine-tuned DistilBERT",
        "clean_text": clean_text,
        "prediction": label_str,
        "prediction_id": pred_label_id,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "REAL": round(float(probs[0]) * 100, 2),
            "FAKE": round(float(probs[1]) * 100, 2)
        },
        "top_contributing_words": deduped_tokens[:12],
        "all_contributions": deduped_tokens,
        "method": "Transformer Token Gradient Saliency & Attention Attribution"
    }


def explain_all_models(text: str) -> Dict[str, Any]:
    """
    Runs inference and explainability across all 3 models simultaneously
    for side-by-side comparative analysis.
    """
    res_lr = explain_traditional_prediction(text, model_type="logistic")
    res_rf = explain_traditional_prediction(text, model_type="random_forest")
    res_db = explain_distilbert_prediction(text)

    return {
        "Logistic Regression": res_lr,
        "Random Forest": res_rf,
        "DistilBERT": res_db
    }


def explain_prediction(text: str, model_choice: str = "Logistic Regression") -> Dict[str, Any]:
    """
    Unified entry point for explaining predictions across all model architectures.
    """
    if "Side-by-Side" in model_choice or "Compare All 3" in model_choice:
        return explain_all_models(text)
    elif "Hybrid" in model_choice or "Ensemble" in model_choice:
        from src.ensemble import predict_ensemble
        return predict_ensemble(text)
    elif "Logistic" in model_choice:
        return explain_traditional_prediction(text, model_type="logistic")
    elif "Random Forest" in model_choice:
        return explain_traditional_prediction(text, model_type="random_forest")
    elif "DistilBERT" in model_choice:
        return explain_distilbert_prediction(text)
    else:
        return explain_traditional_prediction(text, model_type="logistic")
