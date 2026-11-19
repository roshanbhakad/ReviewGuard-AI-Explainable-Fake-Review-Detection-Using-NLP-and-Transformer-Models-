"""Unit tests for Explainable AI (XAI) engine, saliency visualizer, and diagnostics."""

import unittest
from src.xai.diagnostic_rules import DiagnosticReasoner
from src.xai.attention_visualizer import SaliencyVisualizer
from src.xai.explainer import ReviewExplainer


class TestXAI(unittest.TestCase):

    def setUp(self):
        self.reasoner = DiagnosticReasoner()
        self.visualizer = SaliencyVisualizer()
        self.explainer = ReviewExplainer(artifacts_dir="models_dir")

    def test_diagnostic_reasoner_flags(self):
        text = "BUY NOW!! BEST EVERRR!!"
        stylometrics = {
            "exclamation_density": 8.5,
            "all_caps_word_ratio": 0.5,
            "promotional_marker_count": 2,
            "hyperbole_ratio": 0.1,
            "type_token_ratio": 0.4,
            "digits_ratio": 0.0,
            "first_person_singular_ratio": 0.0,
        }
        sent = {"is_dissonant": False, "polarity": 0.8}
        diag = self.reasoner.generate_diagnostics(text, stylometrics, sent, fake_prob=0.92)
        self.assertGreaterEqual(len(diag["red_flags"]), 3)
        self.assertIn("summary", diag)

    def test_saliency_visualizer_rendering(self):
        tokens = [
            {"token": "BEST", "saliency": 0.8, "attention_weight": 0.4, "direction": "fake", "normalized_score": 0.9},
            {"token": "earbuds", "saliency": -0.5, "attention_weight": 0.2, "direction": "genuine", "normalized_score": -0.6},
        ]
        html_out = self.visualizer.render_highlighted_html(tokens)
        self.assertIn("<span", html_out)
        self.assertIn("BEST", html_out)
        self.assertIn("earbuds", html_out)

    def test_explainer_end_to_end(self):
        text = "Decent headphones for everyday use. Battery lasts about 20 hours. Sound is clear."
        res = self.explainer.explain(text, rating=4.0)
        self.assertIn("prediction", res)
        self.assertIn("tokens", res)
        self.assertIn("highlighted_html", res)
        self.assertIn("radar_data", res)
        self.assertIn("diagnostics", res)
        self.assertLess(res["prediction"]["fake_probability"], 0.40)


if __name__ == "__main__":
    unittest.main()
