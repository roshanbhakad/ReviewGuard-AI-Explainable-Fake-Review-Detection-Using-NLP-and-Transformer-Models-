"""Stylometric and linguistic feature extractor for deceptive review analysis.
Extracts lexical richness, syntactic cues, readability metrics, and psychological indicators.
"""

import re
import math
from typing import Dict, Any, List
import numpy as np

from src.nlp.lexicons import (
    PRONOUNS_FIRST_SINGULAR,
    PRONOUNS_FIRST_PLURAL,
    PRONOUNS_SECOND,
    PRONOUNS_THIRD,
    HYPERBOLIC_WORDS,
    SPAM_PROMOTIONAL_MARKERS,
    STOP_WORDS,
)
from src.nlp.text_preprocessor import TextPreprocessor


class StylometryExtractor:
    """Extracts interpretable stylometric and linguistic features from review text."""

    def __init__(self):
        self.preprocessor = TextPreprocessor(lowercase=False, expand_contractions=False)

    def count_syllables(self, word: str) -> int:
        """Estimates syllable count using vowel heuristic."""
        word = word.lower().strip()
        if not word:
            return 1
        word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
        word = re.sub(r"^y", "", word)
        syllables = len(re.findall(r"[aeiouy]{1,2}", word))
        return max(1, syllables)

    def calculate_readability(self, text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
        """Calculates Flesch Reading Ease and Automated Readability Index (ARI)."""
        num_words = max(1, len(words))
        num_sentences = max(1, len(sentences))
        num_chars = sum(len(w) for w in words)
        total_syllables = sum(self.count_syllables(w) for w in words)

        # Flesch Reading Ease: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        flesch = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (total_syllables / num_words)
        flesch = max(0.0, min(100.0, flesch))

        # ARI = 4.71*(chars/words) + 0.5*(words/sentences) - 21.43
        ari = 4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
        ari = max(1.0, min(20.0, ari))

        return {
            "flesch_reading_ease": round(flesch, 2),
            "automated_readability_index": round(ari, 2),
        }

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extracts complete stylometric and linguistic profile as numeric dictionary."""
        if not text or not isinstance(text, str):
            text = ""

        raw_char_count = len(text)
        words_cased = re.findall(r"\b[a-zA-Z0-9'-]+\b", text)
        words_lower = [w.lower() for w in words_cased]
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences and text.strip():
            sentences = [text.strip()]

        num_words = len(words_lower)
        num_unique_words = len(set(words_lower))
        num_sentences = len(sentences)

        # 1. Length features
        avg_sentence_len = (num_words / num_sentences) if num_sentences > 0 else 0.0
        avg_word_len = (sum(len(w) for w in words_lower) / num_words) if num_words > 0 else 0.0

        # 2. Lexical richness
        ttr = (num_unique_words / num_words) if num_words > 0 else 0.0
        word_freqs = {}
        for w in words_lower:
            word_freqs[w] = word_freqs.get(w, 0) + 1
        hapax_count = sum(1 for w, cnt in word_freqs.items() if cnt == 1)
        hapax_ratio = (hapax_count / num_words) if num_words > 0 else 0.0

        # 3. Capitalization features
        cap_chars = sum(1 for c in text if c.isupper())
        capital_ratio = (cap_chars / raw_char_count) if raw_char_count > 0 else 0.0
        all_caps_words = sum(1 for w in words_cased if len(w) > 1 and w.isupper())
        all_caps_ratio = (all_caps_words / num_words) if num_words > 0 else 0.0

        # 4. Punctuation intensity
        exclamation_count = text.count("!")
        question_count = text.count("?")
        excl_density = (exclamation_count / max(1, raw_char_count)) * 100.0
        quest_density = (question_count / max(1, raw_char_count)) * 100.0

        # 5. Pronoun distribution (Ott et al. deception markers)
        num_first_sing = sum(1 for w in words_lower if w in PRONOUNS_FIRST_SINGULAR)
        num_first_plur = sum(1 for w in words_lower if w in PRONOUNS_FIRST_PLURAL)
        num_second = sum(1 for w in words_lower if w in PRONOUNS_SECOND)
        num_third = sum(1 for w in words_lower if w in PRONOUNS_THIRD)

        first_sing_ratio = (num_first_sing / num_words) if num_words > 0 else 0.0
        first_plur_ratio = (num_first_plur / num_words) if num_words > 0 else 0.0
        second_ratio = (num_second / num_words) if num_words > 0 else 0.0
        third_ratio = (num_third / num_words) if num_words > 0 else 0.0

        # 6. Hyperbole and promotional triggers
        text_lower = text.lower()
        hyperbole_count = sum(1 for w in words_lower if w in HYPERBOLIC_WORDS)
        hyperbole_ratio = (hyperbole_count / num_words) if num_words > 0 else 0.0

        promotional_matches = sum(1 for marker in SPAM_PROMOTIONAL_MARKERS if marker in text_lower)

        # 7. Specificity cues (digits, dollar signs, measurements)
        digits_count = sum(1 for c in text if c.isdigit())
        digits_ratio = (digits_count / max(1, raw_char_count)) * 100.0

        # 8. Readability
        readability = self.calculate_readability(text, words_lower, sentences)

        return {
            "char_count": float(raw_char_count),
            "word_count": float(num_words),
            "sentence_count": float(num_sentences),
            "avg_sentence_length": round(avg_sentence_len, 2),
            "avg_word_length": round(avg_word_len, 2),
            "type_token_ratio": round(ttr, 4),
            "hapax_legomena_ratio": round(hapax_ratio, 4),
            "capital_char_ratio": round(capital_ratio, 4),
            "all_caps_word_ratio": round(all_caps_ratio, 4),
            "exclamation_density": round(excl_density, 3),
            "question_density": round(quest_density, 3),
            "first_person_singular_ratio": round(first_sing_ratio, 4),
            "first_person_plural_ratio": round(first_plur_ratio, 4),
            "second_person_ratio": round(second_ratio, 4),
            "third_person_ratio": round(third_ratio, 4),
            "hyperbole_ratio": round(hyperbole_ratio, 4),
            "promotional_marker_count": float(promotional_matches),
            "digits_ratio": round(digits_ratio, 3),
            "flesch_reading_ease": readability["flesch_reading_ease"],
            "automated_readability_index": readability["automated_readability_index"],
        }

    def feature_vector(self, text: str) -> np.ndarray:
        """Returns ordered feature values as NumPy array for ML models."""
        feats = self.extract_features(text)
        return np.array(list(feats.values()), dtype=np.float32)

    def feature_names(self) -> List[str]:
        """Returns ordered list of feature names."""
        dummy = self.extract_features("Test review text.")
        return list(dummy.keys())
