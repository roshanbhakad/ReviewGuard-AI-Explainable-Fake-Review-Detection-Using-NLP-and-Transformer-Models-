"""Rule-based sentiment and sentiment dissonance analyzer for fake review detection."""

import re
from typing import Dict, Any, List
from src.nlp.lexicons import SENTIMENT_LEXICON, NEGATION_WORDS, BOOSTER_WORDS
from src.nlp.text_preprocessor import TextPreprocessor


class SentimentAnalyzer:
    """Analyzes sentiment polarity, subjectivity, and star rating alignment."""

    def __init__(self):
        self.preprocessor = TextPreprocessor(lowercase=True)

    def analyze(self, text: str, rating: float = None) -> Dict[str, Any]:
        """Analyzes sentiment polarity (-1.0 to 1.0), subjectivity (0 to 1), and dissonance.

        Args:
            text: Review text content.
            rating: Optional user star rating (1 to 5).
        """
        if not text or not isinstance(text, str):
            return {
                "polarity": 0.0,
                "sentiment_label": "neutral",
                "subjectivity": 0.0,
                "dissonance_score": 0.0,
                "is_dissonant": False,
                "positive_score": 0.0,
                "negative_score": 0.0,
            }

        words = re.findall(r"\b[a-zA-Z'-]+\b", text.lower())
        total_words = len(words)
        if total_words == 0:
            return {
                "polarity": 0.0,
                "sentiment_label": "neutral",
                "subjectivity": 0.0,
                "dissonance_score": 0.0,
                "is_dissonant": False,
                "positive_score": 0.0,
                "negative_score": 0.0,
            }

        pos_sum = 0.0
        neg_sum = 0.0
        sentiment_word_count = 0

        for i, word in enumerate(words):
            if word in SENTIMENT_LEXICON:
                valence = SENTIMENT_LEXICON[word]
                sentiment_word_count += 1

                # Check window of 3 preceding words for negations and boosters
                window = words[max(0, i - 3) : i]
                negated = any(w in NEGATION_WORDS for w in window)
                booster_multiplier = 1.0
                for w in window:
                    if w in BOOSTER_WORDS:
                        booster_multiplier *= BOOSTER_WORDS[w]

                effective_valence = valence * booster_multiplier
                if negated:
                    effective_valence = -0.74 * effective_valence

                if effective_valence > 0:
                    pos_sum += effective_valence
                else:
                    neg_sum += abs(effective_valence)

        total_sentiment = pos_sum + neg_sum
        if total_sentiment > 0:
            # Normalized polarity between -1.0 and +1.0
            polarity = (pos_sum - neg_sum) / (total_sentiment + 2.0)
            polarity = max(-1.0, min(1.0, polarity))
            # Subjectivity: proportion of sentiment words vs total text length
            subjectivity = min(1.0, (sentiment_word_count / total_words) * 3.5)
        else:
            polarity = 0.0
            subjectivity = 0.0

        if polarity > 0.15:
            sentiment_label = "positive"
        elif polarity < -0.15:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        # Calculate dissonance if rating provided (1.0 to 5.0 scale)
        dissonance_score = 0.0
        is_dissonant = False
        if rating is not None and 1.0 <= rating <= 5.0:
            # Map rating to [-1.0, 1.0] scale: 1 -> -1, 3 -> 0, 5 -> +1
            normalized_rating = (rating - 3.0) / 2.0
            # Dissonance is difference between textual sentiment and stated rating
            diff = abs(polarity - normalized_rating)
            # Normalize diff to [0, 1] range (max theoretical diff is 2.0)
            dissonance_score = round(min(1.0, diff / 1.5), 3)
            # Flag if significant dissonance: e.g. 5-star with negative text or 1-star with positive text
            if (rating >= 4.0 and polarity < -0.2) or (rating <= 2.0 and polarity > 0.2):
                is_dissonant = True

        return {
            "polarity": round(polarity, 3),
            "sentiment_label": sentiment_label,
            "subjectivity": round(subjectivity, 3),
            "dissonance_score": dissonance_score,
            "is_dissonant": is_dissonant,
            "positive_score": round(pos_sum, 2),
            "negative_score": round(neg_sum, 2),
        }
