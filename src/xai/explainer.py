"""Master Explainability engine for ReviewGuard-AI.
Orchestrates model inference, token saliency, attention rollout, stylometric comparison, and diagnostics.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from src.models.ensemble import ReviewGuardEnsemble
from src.nlp.stylometry_extractor import StylometryExtractor
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.xai.diagnostic_rules import DiagnosticReasoner
from src.xai.attention_visualizer import SaliencyVisualizer


class ReviewExplainer:
    """End-to-end Explainable Fake Review Detection Engine."""

    def __init__(self, artifacts_dir: str = "models_dir"):
        self.artifacts_dir = artifacts_dir
        self.ensemble = ReviewGuardEnsemble(artifacts_dir=artifacts_dir)
        self.stylometry_extractor = StylometryExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.diagnostic_reasoner = DiagnosticReasoner()
        self.visualizer = SaliencyVisualizer()

        # Attempt to load model artifacts
        self.is_ready = self.ensemble.load()

    def explain(self, text: str, rating: Optional[float] = None) -> Dict[str, Any]:
        """Performs full end-to-end prediction and multi-layered explainability analysis."""
        if not text or not text.strip():
            return {
                "error": "Empty review text provided.",
                "fake_probability": 0.0,
                "risk_level": "Unknown",
            }

        # 1. Prediction via Meta-Ensemble
        pred = self.ensemble.predict(text, rating=rating)
        fake_prob = pred["fake_probability"]

        # 2. Extract Stylometrics and Sentiment
        stylometrics = self.stylometry_extractor.extract_features(text)
        sentiment_info = self.sentiment_analyzer.analyze(text, rating=rating)

        # 3. Token-Level Attribution
        tokens_attribution = []
        # Try Transformer gradient saliency first
        if self.ensemble.transformer.model is not None:
            try:
                tf_saliency = self.ensemble.transformer.get_token_saliency(text)
                tokens_attribution = tf_saliency.get("tokens", [])
            except Exception as e:
                pass

        # Fallback to TF-IDF / Perturbation attribution if Transformer is unavailable or returned empty
        if not tokens_attribution:
            tokens_attribution = self._compute_surrogate_token_attribution(text, fake_prob)

        # 4. Generate highlighted HTML and top word contributions
        highlighted_html = self.visualizer.render_highlighted_html(tokens_attribution)
        top_words = self.visualizer.get_top_salient_words(tokens_attribution, top_k=5)

        # 5. Natural Language Diagnostics
        diagnostics = self.diagnostic_reasoner.generate_diagnostics(
            text=text,
            stylometrics=stylometrics,
            sentiment_info=sentiment_info,
            fake_prob=fake_prob,
            rating=rating,
        )

        # 6. Stylometric Radar Metrics vs Benchmark
        benchmarks = diagnostics["benchmarks"]
        radar_data = {
            "labels": [
                "Exclamation Intensity",
                "Capitalization Ratio",
                "Hyperbolic Density",
                "First-Person Pronouns",
                "Lexical Richness (TTR)",
                "Readability Index",
            ],
            "review_values": [
                min(100.0, stylometrics["exclamation_density"] * 15.0),
                min(100.0, stylometrics["all_caps_word_ratio"] * 250.0),
                min(100.0, stylometrics["hyperbole_ratio"] * 400.0),
                min(100.0, stylometrics["first_person_singular_ratio"] * 400.0),
                min(100.0, stylometrics["type_token_ratio"] * 100.0),
                min(100.0, stylometrics["flesch_reading_ease"]),
            ],
            "benchmark_values": [
                min(100.0, benchmarks["exclamation_density"] * 15.0),
                min(100.0, benchmarks["all_caps_word_ratio"] * 250.0),
                min(100.0, benchmarks["hyperbole_ratio"] * 400.0),
                min(100.0, benchmarks["first_person_singular_ratio"] * 400.0),
                min(100.0, benchmarks["type_token_ratio"] * 100.0),
                min(100.0, benchmarks["flesch_reading_ease"]),
            ],
        }

        return {
            "prediction": pred,
            "tokens": tokens_attribution,
            "highlighted_html": highlighted_html,
            "top_words": top_words,
            "stylometrics": stylometrics,
            "sentiment": sentiment_info,
            "diagnostics": diagnostics,
            "radar_data": radar_data,
        }

    def _compute_surrogate_token_attribution(self, text: str, base_prob: float) -> List[Dict[str, Any]]:
        """Computes surrogate attribution via leave-one-out perturbation or TF-IDF weights."""
        words = text.split()
        if not words:
            return []

        tokens_attribution = []
        for i, word in enumerate(words):
            cleaned_word = word.lower().strip(".,!?:;\"'()[]{}")
            if not cleaned_word:
                continue

            # Check if model has TF-IDF + Logistic Regression
            sal = 0.0
            if (
                self.ensemble.ml_suite.models
                and "logistic_regression" in self.ensemble.ml_suite.models
                and hasattr(self.ensemble.ml_suite.feature_extractor, "tfidf")
            ):
                tfidf = self.ensemble.ml_suite.feature_extractor.tfidf
                vocab = tfidf.vocabulary_
                if cleaned_word in vocab:
                    idx = vocab[cleaned_word]
                    coef = self.ensemble.ml_suite.models["logistic_regression"].coef_[0]
                    if idx < len(coef):
                        sal = float(coef[idx])

            # If no linear weights, use heuristic polarity/spam markers
            if sal == 0.0:
                from src.nlp.lexicons import HYPERBOLIC_WORDS, SPAM_PROMOTIONAL_MARKERS, BOOSTER_WORDS
                if cleaned_word in HYPERBOLIC_WORDS or word.isupper() and len(word) > 1:
                    sal = 0.45
                elif "!" in word:
                    sal = 0.35
                elif cleaned_word in BOOSTER_WORDS:
                    sal = 0.25
                else:
                    sal = -0.15  # standard neutral/content words lean authentic

            tokens_attribution.append({
                "token": word,
                "position": i,
                "saliency": round(sal, 4),
                "attention_weight": round(abs(sal), 4),
                "direction": "fake" if sal > 0 else "genuine",
                "normalized_score": round(max(-1.0, min(1.0, sal / 1.5)), 3),
            })

        return tokens_attribution
