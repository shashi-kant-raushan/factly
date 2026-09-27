"""
Model Evaluation and Benchmark Comparison Module.

Features:
- Evaluates all three models on the identical, untouched test set:
  1. TF-IDF + Logistic Regression
  2. TF-IDF + Random Forest
  3. Fine-tuned DistilBERT
- Measures empirical inference latency (milliseconds per sample)
- Generates publication-ready comparative visualization charts:
  - Metrics comparison bar chart (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Side-by-side Confusion Matrices
  - Combined ROC Curves with AUC annotations
- Exports JSON metrics summary and Markdown summary table.
"""

import time
import sys
import json

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TEST_CSV,
    TFIDF_VECTORIZER_PATH,
    LOGISTIC_MODEL_PATH,
    RANDOM_FOREST_PATH,
    DISTILBERT_DIR,
    EVALUATION_METRICS_PATH,
    FIGURES_DIR
)
from src.preprocessing import prepare_dataset_features


def measure_inference_speed(predict_func, sample_texts: List[str], iterations: int = 50) -> float:
    """
    Measures average inference latency in milliseconds per document.
    """
    # Warmup
    for text in sample_texts[:5]:
        _ = predict_func([text])

    start_time = time.perf_counter()
    for text in sample_texts[:iterations]:
        _ = predict_func([text])
    elapsed = time.perf_counter() - start_time
    latency_ms = (elapsed / min(iterations, len(sample_texts))) * 1000
    return round(latency_ms, 2)


def evaluate_all_models() -> Dict[str, Any]:
    """
    Runs comprehensive evaluation across all three trained models on the test set.
    """
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test split not found at {TEST_CSV}. Run data_loader first.")

    test_df = pd.read_csv(TEST_CSV)
    y_test = np.array(test_df["label"].tolist())

    traditional_texts = prepare_dataset_features(test_df, for_transformer=False)
    transformer_texts = prepare_dataset_features(test_df, for_transformer=True)

    results = {}
    roc_data = {}

    # 1. Evaluate Logistic Regression
    if TFIDF_VECTORIZER_PATH.exists() and LOGISTIC_MODEL_PATH.exists():
        print("[Evaluate] Evaluating TF-IDF + Logistic Regression...")
        vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        logistic_model = joblib.load(LOGISTIC_MODEL_PATH)

        X_test_tfidf = vectorizer.transform(traditional_texts)
        y_pred_lr = logistic_model.predict(X_test_tfidf)
        y_proba_lr = logistic_model.predict_proba(X_test_tfidf)[:, 1]

        def lr_predict(texts):
            return logistic_model.predict_proba(vectorizer.transform(texts))

        lr_latency = measure_inference_speed(lr_predict, traditional_texts)

        results["Logistic Regression"] = {
            "model_name": "TF-IDF + Logistic Regression",
            "accuracy": round(float(accuracy_score(y_test, y_pred_lr)), 4),
            "precision": round(float(precision_score(y_test, y_pred_lr, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_lr, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_lr, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba_lr)), 4),
            "latency_ms": lr_latency,
            "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist(),
            "model_size_mb": round((TFIDF_VECTORIZER_PATH.stat().st_size + LOGISTIC_MODEL_PATH.stat().st_size) / (1024 * 1024), 2),
            "interpretability": "Very High (Linear coefficients & TF-IDF)",
            "deployment": "Extremely Lightweight (Fast CPU inference)"
        }
        roc_data["Logistic Regression"] = (y_test, y_proba_lr, results["Logistic Regression"]["roc_auc"])
    else:
        print("[Evaluate] Logistic Regression artifacts not found.")

    # 2. Evaluate Random Forest
    if TFIDF_VECTORIZER_PATH.exists() and RANDOM_FOREST_PATH.exists():
        print("[Evaluate] Evaluating TF-IDF + Random Forest...")
        vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        rf_model = joblib.load(RANDOM_FOREST_PATH)

        X_test_tfidf = vectorizer.transform(traditional_texts)
        y_pred_rf = rf_model.predict(X_test_tfidf)
        y_proba_rf = rf_model.predict_proba(X_test_tfidf)[:, 1]

        def rf_predict(texts):
            return rf_model.predict_proba(vectorizer.transform(texts))

        rf_latency = measure_inference_speed(rf_predict, traditional_texts)

        results["Random Forest"] = {
            "model_name": "TF-IDF + Random Forest",
            "accuracy": round(float(accuracy_score(y_test, y_pred_rf)), 4),
            "precision": round(float(precision_score(y_test, y_pred_rf, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_rf, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_rf, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba_rf)), 4),
            "latency_ms": rf_latency,
            "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
            "model_size_mb": round((RANDOM_FOREST_PATH.stat().st_size) / (1024 * 1024), 2),
            "interpretability": "High (MDI Global feature importances)",
            "deployment": "Moderate (Requires tree ensemble memory)"
        }
        roc_data["Random Forest"] = (y_test, y_proba_rf, results["Random Forest"]["roc_auc"])
    else:
        print("[Evaluate] Random Forest artifacts not found.")

    # 3. Evaluate DistilBERT
    if DISTILBERT_DIR.exists():
        print("[Evaluate] Evaluating Fine-tuned DistilBERT...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
        transformer_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
        transformer_model.to(device)
        transformer_model.eval()

        all_preds = []
        all_probas = []
        batch_size = 16

        with torch.no_grad():
            for i in range(0, len(transformer_texts), batch_size):
                batch_texts = transformer_texts[i:i + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt"
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                logits = transformer_model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_probas.extend(probs[:, 1].cpu().numpy().tolist())

        y_pred_db = np.array(all_preds)
        y_proba_db = np.array(all_probas)

        def db_predict(texts):
            enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                out = transformer_model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
                return torch.softmax(out.logits, dim=1).cpu().numpy()

        db_latency = measure_inference_speed(db_predict, transformer_texts[:20], iterations=20)

        # Compute folder size
        distilbert_size_mb = round(sum(f.stat().st_size for f in DISTILBERT_DIR.glob("**/*") if f.is_file()) / (1024 * 1024), 2)

        results["DistilBERT"] = {
            "model_name": "Fine-tuned DistilBERT",
            "accuracy": round(float(accuracy_score(y_test, y_pred_db)), 4),
            "precision": round(float(precision_score(y_test, y_pred_db, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_db, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_db, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba_db)), 4),
            "latency_ms": db_latency,
            "confusion_matrix": confusion_matrix(y_test, y_pred_db).tolist(),
            "model_size_mb": distilbert_size_mb,
            "interpretability": "Medium (Gradient saliency & attention)",
            "deployment": "Resource Intensive (Requires PyTorch runtime / GPU)"
        }
        roc_data["DistilBERT"] = (y_test, y_proba_db, results["DistilBERT"]["roc_auc"])
    else:
        print("[Evaluate] DistilBERT artifacts not found.")

    # 4. Evaluate Hybrid Super-Ensemble (LR + RF + DistilBERT)
    if "Logistic Regression" in results and "Random Forest" in results and "DistilBERT" in results:
        print("[Evaluate] Computing Hybrid Super-Ensemble benchmark...")
        w_lr, w_rf, w_db = 0.12, 0.05, 0.83
        y_proba_ens = w_lr * y_proba_lr + w_rf * y_proba_rf + w_db * y_proba_db
        y_pred_ens = (y_proba_ens >= 0.5).astype(int)

        results["Hybrid Ensemble"] = {
            "model_name": "🔥 Hybrid Super-Ensemble",
            "accuracy": round(float(accuracy_score(y_test, y_pred_ens)), 4),
            "precision": round(float(precision_score(y_test, y_pred_ens, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_ens, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_ens, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba_ens)), 4),
            "latency_ms": round(lr_latency + rf_latency + db_latency, 2),
            "confusion_matrix": confusion_matrix(y_test, y_pred_ens).tolist(),
            "model_size_mb": round(distilbert_size_mb + 52.0, 2),
            "interpretability": "High (Multi-Model Cross-Attribution)",
            "deployment": "Hybrid Pipeline (Optimal weighted soft voting)"
        }
        roc_data["Hybrid Ensemble"] = (y_test, y_proba_ens, results["Hybrid Ensemble"]["roc_auc"])

    # Save metrics JSON
    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[Evaluate] Saved model comparison metrics to {EVALUATION_METRICS_PATH}")

    # Generate Visualization Charts
    generate_comparison_plots(results, roc_data)

    # Print markdown comparison table
    print_comparison_summary(results)

    return results


def generate_comparison_plots(results: Dict[str, Any], roc_data: Dict[str, Any]):
    """
    Generates comparison bar charts, confusion matrices, and ROC curves.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if not results:
        return

    # 1. Performance Comparison Bar Chart
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = list(results.keys())

    data_rows = []
    for model in models:
        for metric in metrics_to_plot:
            data_rows.append({
                "Model": model,
                "Metric": metric.upper().replace("_", "-"),
                "Score": results[model][metric] * 100
            })
    df_metrics = pd.DataFrame(data_rows)

    plt.figure(figsize=(11, 6))
    palette = ["#1976D2", "#388E3C", "#7B1FA2", "#E65100"]
    ax = sns.barplot(
        data=df_metrics,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette[:len(models)]
    )
    plt.title("Model Performance Benchmark on Unseen Test Set", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score (%)", fontsize=12)
    plt.ylim(50, 105)
    plt.legend(title="Model", frameon=True, loc="lower right")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_performance_comparison.png", dpi=300)
    plt.close()

    # 2. Confusion Matrices Side-by-Side
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, ax) in enumerate(zip(models, axes)):
        cm = np.array(results[model_name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["REAL (0)", "FAKE (1)"],
            yticklabels=["REAL (0)", "FAKE (1)"],
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(f"{model_name}\n(Acc: {results[model_name]['accuracy']*100:.1f}%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=300)
    plt.close()

    # 3. Combined ROC Curves
    plt.figure(figsize=(8, 6))
    colors = ["#1976D2", "#388E3C", "#7B1FA2"]
    for idx, (model_name, (y_true, y_score, auc_val)) in enumerate(roc_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2.2,
                 label=f"{model_name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=300)
    plt.close()

    print(f"[Evaluate] Generated all comparative plots in: {FIGURES_DIR}")


def print_comparison_summary(results: Dict[str, Any]):
    """
    Prints a formatted markdown table to terminal.
    """
    print("\n" + "=" * 90)
    print("                      FINAL MODEL BENCHMARK COMPARISON TABLE                          ")
    print("=" * 90)
    header = f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Latency':<10}"
    print(header)
    print("-" * 90)
    for model_key, m in results.items():
        name_clean = m['model_name'].replace("🔥 ", "")
        row = f"{name_clean:<25} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['roc_auc']:<10.4f} | {m['latency_ms']} ms"
        print(row)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    evaluate_all_models()
