"""Traditional and ensemble Machine Learning models for fake review detection.
Includes Logistic Regression, Naive Bayes, Random Forest, XGBoost, and LightGBM
operating on hybrid TF-IDF + Stylometric feature representations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import xgboost as xgb
import lightgbm as lgb

from src.nlp.text_preprocessor import TextPreprocessor
from src.nlp.stylometry_extractor import StylometryExtractor
from src.nlp.sentiment_analyzer import SentimentAnalyzer


class HybridFeatureExtractor:
    """Combines TF-IDF n-grams with normalized stylometric and sentiment features."""

    def __init__(self, max_tfidf_features: int = 3000):
        self.preprocessor = TextPreprocessor(lowercase=True)
        self.stylometry_extractor = StylometryExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_tfidf_features,
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _extract_dense_features(self, texts: List[str], ratings: Optional[List[float]] = None) -> np.ndarray:
        """Extracts stylometric, readability, and sentiment dissonance features."""
        dense_rows = []
        for i, text in enumerate(texts):
            rating = ratings[i] if ratings is not None and i < len(ratings) else None
            # Stylometric features (20 features)
            sty_feats = list(self.stylometry_extractor.extract_features(text).values())
            # Sentiment features (polarity, subjectivity, dissonance)
            sent_info = self.sentiment_analyzer.analyze(text, rating=rating)
            sent_feats = [
                sent_info["polarity"],
                sent_info["subjectivity"],
                sent_info["dissonance_score"],
                1.0 if sent_info["is_dissonant"] else 0.0,
            ]
            dense_rows.append(sty_feats + sent_feats)
        return np.array(dense_rows, dtype=np.float32)

    def fit(self, texts: List[str], ratings: Optional[List[float]] = None) -> "HybridFeatureExtractor":
        """Fits TF-IDF and dense feature scaler."""
        clean_texts = [self.preprocessor.clean_text(t) for t in texts]
        self.tfidf.fit(clean_texts)
        dense_feats = self._extract_dense_features(texts, ratings)
        self.scaler.fit(dense_feats)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str], ratings: Optional[List[float]] = None) -> csr_matrix:
        """Transforms raw texts and optional ratings into sparse feature matrix."""
        if not self.is_fitted:
            raise ValueError("HybridFeatureExtractor must be fitted before transforming data.")
        clean_texts = [self.preprocessor.clean_text(t) for t in texts]
        sparse_tfidf = self.tfidf.transform(clean_texts)
        dense_feats = self._extract_dense_features(texts, ratings)
        scaled_dense = self.scaler.transform(dense_feats)
        return hstack([sparse_tfidf, csr_matrix(scaled_dense)]).tocsr()

    def fit_transform(self, texts: List[str], ratings: Optional[List[float]] = None) -> csr_matrix:
        """Fits extractor and transforms data in one call."""
        return self.fit(texts, ratings).transform(texts, ratings)


class ReviewGuardMLSuite:
    """Manages training, evaluation, persistence, and inference for classical & boosted ML models."""

    def __init__(self, artifacts_dir: str = "models_dir"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.feature_extractor = HybridFeatureExtractor()
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}

    def train_all(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Trains Logistic Regression, Naive Bayes, Random Forest, XGBoost, and LightGBM."""
        train_texts = train_df["review_text"].tolist()
        train_ratings = train_df["rating"].tolist() if "rating" in train_df.columns else None
        y_train = train_df["label"].values

        test_texts = test_df["review_text"].tolist()
        test_ratings = test_df["rating"].tolist() if "rating" in test_df.columns else None
        y_test = test_df["label"].values

        # Fit feature extractor
        X_train = self.feature_extractor.fit_transform(train_texts, train_ratings)
        X_test = self.feature_extractor.transform(test_texts, test_ratings)

        # 1. Logistic Regression
        lr = LogisticRegression(max_iter=1000, C=1.5, random_state=42)
        lr.fit(X_train, y_train)
        self.models["logistic_regression"] = lr

        # 2. Multinomial Naive Bayes (runs purely on TF-IDF part due to non-negative constraint)
        X_train_tfidf = self.feature_extractor.tfidf.transform([self.feature_extractor.preprocessor.clean_text(t) for t in train_texts])
        X_test_tfidf = self.feature_extractor.tfidf.transform([self.feature_extractor.preprocessor.clean_text(t) for t in test_texts])
        nb = MultinomialNB(alpha=0.5)
        nb.fit(X_train_tfidf, y_train)
        self.models["naive_bayes"] = nb

        # 3. Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models["random_forest"] = rf

        # 4. XGBoost Classifier
        xgb_clf = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_clf.fit(X_train, y_train)
        self.models["xgboost"] = xgb_clf

        # 5. LightGBM Classifier
        try:
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            X_train_dense = np.ascontiguousarray(X_train.toarray(), dtype=np.float32)
            y_train_arr = np.ascontiguousarray(y_train, dtype=np.int32)
            lgb_clf.fit(X_train_dense, y_train_arr)
            self.models["lightgbm"] = lgb_clf
        except Exception as e:
            print(f"  Note: LightGBM skipped ({e}), other ensemble models active.")

        # Evaluate all models on test set
        for name, model in self.models.items():
            if name == "naive_bayes":
                preds = model.predict(X_test_tfidf)
                probs = model.predict_proba(X_test_tfidf)[:, 1]
            elif name == "lightgbm":
                X_test_dense = np.ascontiguousarray(X_test.toarray(), dtype=np.float32)
                preds = model.predict(X_test_dense)
                probs = model.predict_proba(X_test_dense)[:, 1]
            else:
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, 1]

            self.metrics[name] = {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            }

        self.save_artifacts()
        return self.metrics

    def predict(self, text: str, rating: Optional[float] = None, model_name: str = "xgboost") -> Dict[str, Any]:
        """Predicts fake probability for a single review."""
        if model_name not in self.models:
            model_name = "xgboost" if "xgboost" in self.models else list(self.models.keys())[0]

        model = self.models[model_name]
        if model_name == "naive_bayes":
            clean_text = self.feature_extractor.preprocessor.clean_text(text)
            X = self.feature_extractor.tfidf.transform([clean_text])
            prob = float(model.predict_proba(X)[0, 1])
        elif model_name == "lightgbm":
            X = self.feature_extractor.transform([text], [rating] if rating else None)
            prob = float(model.predict_proba(X.toarray())[0, 1])
        else:
            X = self.feature_extractor.transform([text], [rating] if rating else None)
            prob = float(model.predict_proba(X)[0, 1])

        label = "Fake" if prob >= 0.5 else "Genuine"
        return {
            "model_used": model_name,
            "fake_probability": round(prob, 4),
            "genuine_probability": round(1.0 - prob, 4),
            "predicted_label": label,
        }

    def save_artifacts(self) -> None:
        """Saves models, feature extractors, and metrics to disk."""
        joblib.dump(self.feature_extractor, os.path.join(self.artifacts_dir, "feature_extractor.joblib"))
        joblib.dump(self.models, os.path.join(self.artifacts_dir, "ml_models.joblib"))
        joblib.dump(self.metrics, os.path.join(self.artifacts_dir, "ml_metrics.joblib"))

    def load_artifacts(self) -> bool:
        """Loads pre-trained models from disk if available."""
        feat_path = os.path.join(self.artifacts_dir, "feature_extractor.joblib")
        models_path = os.path.join(self.artifacts_dir, "ml_models.joblib")
        metrics_path = os.path.join(self.artifacts_dir, "ml_metrics.joblib")

        if os.path.exists(feat_path) and os.path.exists(models_path):
            self.feature_extractor = joblib.load(feat_path)
            self.models = joblib.load(models_path)
            if os.path.exists(metrics_path):
                self.metrics = joblib.load(metrics_path)
            return True
        return False
