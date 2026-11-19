"""Evaluation and diagnostic testing script for ReviewGuard-AI."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.dataset_loader import load_dataset
from src.xai.explainer import ReviewExplainer


def evaluate_dataset(csv_path: str = None):
    print("=" * 65)
    print("📊 ReviewGuard-AI: Evaluation and Diagnostic Suite")
    print("=" * 65)

    explainer = ReviewExplainer(artifacts_dir="models_dir")
    if not explainer.is_ready:
        print("❌ Model artifacts not found in `models_dir/`. Please run `python train.py` first.")
        return

    df = load_dataset(csv_path) if csv_path else load_dataset()
    print(f"Loaded {len(df)} review samples for evaluation.")

    texts = df["review_text"].tolist()
    ratings = df["rating"].tolist() if "rating" in df.columns else [None] * len(df)
    true_labels = df["label"].values if "label" in df.columns else None

    predictions = []
    fake_probs = []

    for i, (text, rating) in enumerate(zip(texts, ratings)):
        res = explainer.explain(text, rating=rating)
        prob = res["prediction"]["fake_probability"]
        pred_label = 1 if prob >= 0.5 else 0
        predictions.append(pred_label)
        fake_probs.append(prob)

    if true_labels is not None:
        print("\n📈 Classification Report (Ensemble Decision):")
        print(classification_report(true_labels, predictions, target_names=["Genuine", "Fake"]))

        cm = confusion_matrix(true_labels, predictions)
        print("Confusion Matrix:")
        print(f"  TN: {cm[0, 0]:<5} | FP: {cm[0, 1]:<5}")
        print(f"  FN: {cm[1, 0]:<5} | TP: {cm[1, 1]:<5}")

        auc = roc_auc_score(true_labels, fake_probs)
        print(f"\nROC-AUC Score: {auc:.4f}")

    print("\n🔍 Qualitative Analysis on First 3 Samples:")
    for i in range(min(3, len(texts))):
        res = explainer.explain(texts[i], rating=ratings[i])
        print(f"\nSample #{i + 1}: \"{texts[i][:80]}...\"")
        print(f"  Fake Probability : {res['prediction']['fake_probability'] * 100:.1f}%")
        print(f"  Risk Category    : {res['prediction']['risk_level']}")
        print(f"  Summary          : {res['diagnostics']['summary']}")
        print(f"  Red Flags Count  : {len(res['diagnostics']['red_flags'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReviewGuard-AI models on review datasets.")
    parser.add_argument("--csv", type=str, default=None, help="Path to evaluation CSV file.")
    args = parser.parse_args()
    evaluate_dataset(args.csv)
