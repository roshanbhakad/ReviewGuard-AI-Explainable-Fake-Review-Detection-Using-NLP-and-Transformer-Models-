"""Meta-Ensemble combining Transformer Attention predictions, XGBoost, and Stylometric heuristics."""

from typing import Dict, Any, List, Optional
import numpy as np

from src.models.ml_models import ReviewGuardMLSuite
from src.models.transformer_model import ReviewTransformerPipeline
from src.nlp.stylometry_extractor import StylometryExtractor
from src.nlp.sentiment_analyzer import SentimentAnalyzer


class ReviewGuardEnsemble:
    """Combines Transformer, XGBoost, Random Forest, and Stylometric heuristics into unified decision."""

    def __init__(self, artifacts_dir: str = "models_dir"):
        self.artifacts_dir = artifacts_dir
        self.ml_suite = ReviewGuardMLSuite(artifacts_dir=artifacts_dir)
        self.transformer = ReviewTransformerPipeline(artifacts_dir=artifacts_dir)
        self.stylometry_extractor = StylometryExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.is_ready = False

    def load(self) -> bool:
        """Loads both ML suite and Transformer artifacts."""
        ml_loaded = self.ml_suite.load_artifacts()
        tf_loaded = self.transformer.load()
        self.is_ready = ml_loaded or tf_loaded
        return self.is_ready

    def predict(self, text: str, rating: Optional[float] = None) -> Dict[str, Any]:
        """Calculates multi-model consensus prediction and risk categorization."""
        model_scores = {}

        # 1. Classical ML predictions
        if self.ml_suite.models:
            for model_name in ["xgboost", "random_forest", "logistic_regression", "lightgbm"]:
                if model_name in self.ml_suite.models:
                    res = self.ml_suite.predict(text, rating=rating, model_name=model_name)
                    model_scores[model_name] = res["fake_probability"]

        # 2. Transformer prediction
        tf_score = None
        if self.transformer.model is not None:
            try:
                tf_score = float(self.transformer.predict_proba([text])[0])
                model_scores["transformer_mha"] = round(tf_score, 4)
            except Exception as e:
                pass

        # 3. Stylometric & Sentiment Anomaly penalty
        sty_feats = self.stylometry_extractor.extract_features(text)
        sent_info = self.sentiment_analyzer.analyze(text, rating=rating)

        # Heuristic anomaly score (0.0 to 1.0)
        anomaly_points = 0.0
        if sty_feats["exclamation_density"] > 3.0:
            anomaly_points += 0.25
        if sty_feats["all_caps_word_ratio"] > 0.25:
            anomaly_points += 0.25
        if sty_feats["promotional_marker_count"] >= 1:
            anomaly_points += min(0.35, sty_feats["promotional_marker_count"] * 0.15)
        if sent_info["is_dissonant"]:
            anomaly_points += 0.35
        if sty_feats["hyperbole_ratio"] > 0.08:
            anomaly_points += 0.20
        rule_anomaly_score = min(1.0, anomaly_points)

        # 4. Weighted Ensemble Probability
        # Balanced consensus across all available models
        model_weights_map = {
            "xgboost": 0.30,
            "random_forest": 0.20,
            "logistic_regression": 0.20,
            "transformer_mha": 0.20,
            "lightgbm": 0.10,
        }
        active_weights = {m: model_weights_map.get(m, 0.15) for m in model_scores}
        total_w = sum(active_weights.values())

        if total_w > 0:
            blended_ml = sum(model_scores[m] * (active_weights[m] / total_w) for m in active_weights)
            # Blend 85% ML model consensus + 15% rule anomaly
            final_prob = 0.85 * blended_ml + 0.15 * rule_anomaly_score
        else:
            final_prob = rule_anomaly_score

        final_prob = max(0.01, min(0.99, final_prob))

        # Determine Risk Level and Badge
        if final_prob < 0.25:
            risk_level = "Verified Authentic"
            risk_badge = "success"  # green
        elif final_prob < 0.50:
            risk_level = "Likely Authentic"
            risk_badge = "info"     # teal / blue
        elif final_prob < 0.75:
            risk_level = "Suspicious"
            risk_badge = "warning"  # yellow / amber
        else:
            risk_level = "High Risk / Deceptive"
            risk_badge = "danger"   # red

        # Model agreement
        votes_fake = sum(1 for p in model_scores.values() if p >= 0.5)
        total_models = len(model_scores)
        consensus_text = f"{votes_fake}/{total_models} models voted Fake" if total_models > 0 else "Heuristic evaluation"

        return {
            "fake_probability": round(float(final_prob), 4),
            "genuine_probability": round(float(1.0 - final_prob), 4),
            "risk_level": risk_level,
            "risk_badge": risk_badge,
            "consensus": consensus_text,
            "model_scores": model_scores,
            "rule_anomaly_score": round(rule_anomaly_score, 3),
            "stylometric_summary": {
                "exclamation_density": sty_feats["exclamation_density"],
                "all_caps_ratio": sty_feats["all_caps_word_ratio"],
                "hyperbole_ratio": sty_feats["hyperbole_ratio"],
                "promotional_markers": int(sty_feats["promotional_marker_count"]),
                "first_person_pronouns": sty_feats["first_person_singular_ratio"],
                "flesch_reading_ease": sty_feats["flesch_reading_ease"],
            },
            "sentiment_summary": {
                "polarity": sent_info["polarity"],
                "label": sent_info["sentiment_label"],
                "dissonance_score": sent_info["dissonance_score"],
                "is_dissonant": sent_info["is_dissonant"],
            },
        }
