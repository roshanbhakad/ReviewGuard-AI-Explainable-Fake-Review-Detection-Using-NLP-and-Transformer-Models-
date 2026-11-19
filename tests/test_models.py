"""Unit tests for ML models, Transformer pipeline, and meta-ensemble."""

import unittest
import numpy as np
from src.models.ml_models import ReviewGuardMLSuite
from src.models.transformer_model import ReviewTransformerPipeline
from src.models.ensemble import ReviewGuardEnsemble


class TestModelPipelines(unittest.TestCase):

    def setUp(self):
        self.ml_suite = ReviewGuardMLSuite(artifacts_dir="models_dir")
        self.transformer = ReviewTransformerPipeline(artifacts_dir="models_dir")
        self.ensemble = ReviewGuardEnsemble(artifacts_dir="models_dir")

    def test_ml_suite_artifacts_loaded(self):
        loaded = self.ml_suite.load_artifacts()
        self.assertTrue(loaded, "ML models should be successfully loaded from models_dir/")
        self.assertIn("xgboost", self.ml_suite.models)
        self.assertIn("logistic_regression", self.ml_suite.models)

    def test_transformer_inference_and_attention(self):
        loaded = self.transformer.load()
        self.assertTrue(loaded, "Transformer weights should be loaded from models_dir/")
        text = "Awesome quality, best purchase ever!"
        probs = self.transformer.predict_proba([text])
        self.assertEqual(len(probs), 1)
        self.assertTrue(0.0 <= probs[0] <= 1.0)

        saliency = self.transformer.get_token_saliency(text)
        self.assertIn("tokens", saliency)
        self.assertIn("attention_matrix", saliency)
        self.assertGreater(len(saliency["tokens"]), 0)

    def test_ensemble_prediction(self):
        loaded = self.ensemble.load()
        self.assertTrue(loaded, "Ensemble should be successfully loaded.")
        fake_text = "BEST PRODUCT EVERRRR!!!! Changed my life forever!! MUST BUY 100%!"
        res = self.ensemble.predict(fake_text, rating=5.0)
        self.assertIn("fake_probability", res)
        self.assertIn("risk_level", res)
        self.assertGreater(res["fake_probability"], 0.70)
        self.assertIn("HIGH RISK", res["risk_level"].upper())


if __name__ == "__main__":
    unittest.main()
