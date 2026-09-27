"""
Fine-tuning and Evaluation Module for DistilBERT (Transformers + PyTorch).

Features:
- DistilBERT tokenization with dynamic padding and truncation
- Custom PyTorch Dataset with tensor mapping
- PyTorch training loop with AdamW optimizer and linear learning rate scheduler
- Validation checkpointing (saving best model state based on validation F1 score)
- Test set evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Export model weights and tokenizer to `models/distilbert/`
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    DISTILBERT_DIR,
    DISTILBERT_MODEL_NAME,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_BATCH_SIZE,
    DISTILBERT_LEARNING_RATE,
    DISTILBERT_EPOCHS,
    DISTILBERT_WEIGHT_DECAY,
    DISTILBERT_WARMUP_RATIO,
    RANDOM_SEED
)
from src.preprocessing import prepare_dataset_features


class NewsTextDataset(Dataset):
    """
    PyTorch Dataset for News Text Classification.
    """
    def __init__(self, texts: list, labels: list, tokenizer: AutoTokenizer, max_length: int = DISTILBERT_MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def set_seed(seed: int = RANDOM_SEED):
    """Ensures full reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate_transformer(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Runs inference on a DataLoader and computes full classification metrics.
    """
    model.eval()
    all_preds = []
    all_probas = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probas.extend(probs[:, 1].cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_test = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probas)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["REAL (0)", "FAKE (1)"], output_dict=True)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "probabilities": y_proba,
        "predictions": y_pred,
        "true_labels": y_test
    }


def train_distilbert_model(
    train_texts: list,
    train_labels: list,
    val_texts: list,
    val_labels: list,
    epochs: int = DISTILBERT_EPOCHS,
    batch_size: int = DISTILBERT_BATCH_SIZE,
    learning_rate: float = DISTILBERT_LEARNING_RATE
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Fine-tunes DistilBERT for binary text classification with validation tracking.
    """
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DistilBERT] Using computation device: {device}")

    print(f"[DistilBERT] Loading pretrained tokenizer & model: {DISTILBERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME,
        num_labels=2
    )
    model.to(device)

    # Datasets and Loaders
    train_dataset = NewsTextDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsTextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=DISTILBERT_WEIGHT_DECAY
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * DISTILBERT_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_model_state = None

    print(f"\n[DistilBERT] Commencing fine-tuning ({epochs} Epochs, {len(train_dataset)} train samples)...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)

        # Validation evaluation
        val_metrics = evaluate_transformer(model, val_loader, device)
        print(
            f"Epoch {epoch} Results | Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> Saved new best model checkpoint (Val F1: {best_val_f1:.4f})")

    # Load best checkpoint into model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[DistilBERT] Loaded top performing model weights with Val F1: {best_val_f1:.4f}")

    return tokenizer, model


def save_distilbert_artifacts(tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification):
    """
    Saves fine-tuned DistilBERT model weights and tokenizer configurations.
    """
    DISTILBERT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DISTILBERT_DIR)
    tokenizer.save_pretrained(DISTILBERT_DIR)
    print(f"[DistilBERT] Model & Tokenizer successfully saved to: {DISTILBERT_DIR}")


def run_pipeline(epochs: int = DISTILBERT_EPOCHS, sample_limit: int = None) -> Dict[str, Any]:
    """
    Full pipeline execution for DistilBERT Fine-tuning and Test Evaluation.
    """
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    device_is_cuda = torch.cuda.is_available()
    if sample_limit is None and not device_is_cuda:
        # On CPU, fine-tune on 1200 stratified samples for fast, high-quality convergence
        sample_limit = 1200
        print(f"[DistilBERT] CPU detected: fine-tuning on {sample_limit} representative training samples for fast execution...")

    if sample_limit and sample_limit < len(train_df):
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=sample_limit // 2, random_state=RANDOM_SEED)
        ).reset_index(drop=True)
        val_df = val_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), 150), random_state=RANDOM_SEED)
        ).reset_index(drop=True)

    print(f"[DistilBERT] Preprocessing texts ({len(train_df)} train, {len(val_df)} val, {len(test_df)} test)...")
    train_texts = prepare_dataset_features(train_df, for_transformer=True)
    val_texts = prepare_dataset_features(val_df, for_transformer=True)
    test_texts = prepare_dataset_features(test_df, for_transformer=True)

    train_labels = train_df["label"].tolist()
    val_labels = val_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    tokenizer, model = train_distilbert_model(
        train_texts, train_labels,
        val_texts, val_labels,
        epochs=epochs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = NewsTextDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=DISTILBERT_BATCH_SIZE, shuffle=False)

    print("\n[DistilBERT] Evaluating fine-tuned model on Unseen Test Set...")
    eval_results = evaluate_transformer(model, test_loader, device)

    metrics = {
        "model_name": "Fine-tuned DistilBERT",
        "accuracy": eval_results["accuracy"],
        "precision": eval_results["precision"],
        "recall": eval_results["recall"],
        "f1": eval_results["f1"],
        "roc_auc": eval_results["roc_auc"],
        "confusion_matrix": eval_results["confusion_matrix"],
        "classification_report": eval_results["classification_report"],
        "test_size": len(test_labels)
    }

    print("\n" + "=" * 55)
    print("       DISTILBERT TEST EVALUATION REPORT             ")
    print("=" * 55)
    print(f"Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
    print("=" * 55 + "\n")

    save_distilbert_artifacts(tokenizer, model)
    return metrics


if __name__ == "__main__":
    run_pipeline()
