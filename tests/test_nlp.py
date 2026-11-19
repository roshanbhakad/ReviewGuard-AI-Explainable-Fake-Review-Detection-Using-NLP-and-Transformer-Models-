"""Unit tests for NLP preprocessing, stylometrics, and sentiment analysis."""

import unittest
from src.nlp.text_preprocessor import TextPreprocessor
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.nlp.stylometry_extractor import StylometryExtractor


class TestNLPModules(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.stylometry_extractor = StylometryExtractor()

    def test_text_preprocessor(self):
        raw = "I can't believe this! Visit http://spam.com for 50% off <br> now!"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("http://spam.com", cleaned)
        self.assertNotIn("<br>", cleaned)
        self.assertIn("can not", cleaned)

    def test_sentiment_analyzer_positive(self):
        text = "This product is fantastic, exceptional quality and works wonderfully!"
        res = self.sentiment_analyzer.analyze(text, rating=5.0)
        self.assertGreater(res["polarity"], 0.2)
        self.assertEqual(res["sentiment_label"], "positive")
        self.assertFalse(res["is_dissonant"])

    def test_sentiment_analyzer_dissonance(self):
        # 5-star rating with strongly negative complaint
        text = "Worst purchase ever, completely broke and horrible customer service."
        res = self.sentiment_analyzer.analyze(text, rating=5.0)
        self.assertLess(res["polarity"], -0.2)
        self.assertTrue(res["is_dissonant"])
        self.assertGreater(res["dissonance_score"], 0.7)

    def test_stylometry_extractor_features(self):
        spam_text = "BUY IT NOW!!! BEST PRODUCT EVERRRR!! 100% MUST BUY!!"
        feats = self.stylometry_extractor.extract_features(spam_text)
        self.assertIn("exclamation_density", feats)
        self.assertIn("all_caps_word_ratio", feats)
        self.assertIn("promotional_marker_count", feats)
        self.assertGreater(feats["exclamation_density"], 3.0)
        self.assertGreater(feats["all_caps_word_ratio"], 0.3)
        self.assertGreaterEqual(feats["promotional_marker_count"], 1)


if __name__ == "__main__":
    unittest.main()
