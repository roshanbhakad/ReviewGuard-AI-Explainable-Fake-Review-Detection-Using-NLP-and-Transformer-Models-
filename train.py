"""Training pipeline for ReviewGuard-AI.
Trains classical ML models (Logistic Regression, Naive Bayes, Random Forest, XGBoost, LightGBM)
and the Deep Transformer Encoder with Multi-Head Self-Attention.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.dataset_loader import load_dataset
from src.models.ml_models import ReviewGuardMLSuite
from src.models.transformer_model import ReviewTransformerPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def main():
    print("=" * 70)
    print("🚀 ReviewGuard-AI: Model Training & Evaluation Pipeline")
    print("=" * 70)

    # 1. Load Dataset
    print("\n[1/4] Loading and preparing benchmark review dataset...")
    df = load_dataset()
    print(f"  ✓ Loaded dataset with {len(df)} samples.")
    print(f"  ✓ Class distribution:\n{df['label_name'].value_counts().to_string()}")

    # Stratified Train/Test Split (80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
    print(f"  ✓ Training split: {len(train_df)} samples | Testing split: {len(test_df)} samples")

    # 2. Train Classical and Boosted ML Suite
    print("\n[2/4] Training Classical and Boosted ML Models...")
    print("  - Logistic Regression (L2 penalty, balanced weights)")
    print("  - Multinomial Naive Bayes (alpha=0.5)")
    print("  - Random Forest Classifier (100 estimators, max depth 16)")
    print("  - XGBoost Classifier (tree boosting on hybrid features)")
    print("  - LightGBM Classifier (fast gradient boosting)")

    t0 = time.time()
    ml_suite = ReviewGuardMLSuite(artifacts_dir="models_dir")
    ml_metrics = ml_suite.train_all(train_df, test_df)
    ml_time = time.time() - t0
    print(f"  ✓ ML Suite training complete in {ml_time:.2f}s!")

    # 3. Train Keras Transformer with Multi-Head Attention
    print("\n[3/4] Training Transformer Encoder with Multi-Head Self-Attention...")
    print("  - Architecture: Token + Positional Embedding -> MultiHeadAttention -> FeedForward -> Classification")
    t0 = time.time()
    transformer = ReviewTransformerPipeline(
        max_vocab_size=5000,
        sequence_length=100,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        artifacts_dir="models_dir",
    )

    train_texts = train_df["review_text"].tolist()
    y_train = train_df["label"].values
    test_texts = test_df["review_text"].tolist()
    y_test = test_df["label"].values

    history = transformer.fit(train_texts, y_train, epochs=4, batch_size=32, validation_split=0.15)
    tf_time = time.time() - t0
    print(f"  ✓ Transformer training complete in {tf_time:.2f}s!")

    # Evaluate Transformer on test set
    tf_probs = transformer.predict_proba(test_texts)
    tf_preds = (tf_probs >= 0.5).astype(int)
    tf_metrics = {
        "accuracy": round(float(accuracy_score(y_test, tf_preds)), 4),
        "precision": round(float(precision_score(y_test, tf_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, tf_preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, tf_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, tf_probs)), 4),
    }
    ml_metrics["transformer_mha"] = tf_metrics

    # Save combined benchmark metrics
    import joblib
    joblib.dump(ml_metrics, os.path.join("models_dir", "all_benchmark_metrics.joblib"))

    # 4. Display Benchmark Leaderboard
    print("\n[4/4] 🏆 Model Performance Benchmark Leaderboard (Held-out Test Set):")
    print("-" * 75)
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 75)
    for model_name, m in ml_metrics.items():
        display_name = model_name.replace("_", " ").title()
        print(f"{display_name:<25} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['roc_auc']:<10.4f}")
    print("-" * 75)
    print("\n✅ All models and artifacts successfully persisted to `models_dir/`!")


if __name__ == "__main__":
    main()
