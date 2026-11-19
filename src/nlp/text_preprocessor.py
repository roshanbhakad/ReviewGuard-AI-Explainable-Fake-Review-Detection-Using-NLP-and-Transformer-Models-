"""Text preprocessing, normalization, and tokenization utilities."""

import re
from typing import List, Tuple, Dict, Any

# Common contraction mapping
CONTRACTIONS = {
    "won't": "will not",
    "can't": "can not",
    "cannot": "can not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
}


class TextPreprocessor:
    """Preprocesses raw review text for machine learning and transformer pipelines."""

    def __init__(self, lowercase: bool = True, expand_contractions: bool = True):
        self.lowercase = lowercase
        self.expand_contractions = expand_contractions

    def expand_text_contractions(self, text: str) -> str:
        """Expands common English contractions."""
        text_lower = text.lower()
        for contraction, replacement in CONTRACTIONS.items():
            if contraction in text_lower:
                pattern = re.compile(re.escape(contraction), re.IGNORECASE)
                text = pattern.sub(replacement, text)
        return text

    def clean_text(self, text: str) -> str:
        """Cleans URLs, HTML tags, excess whitespace, keeping standard punctuation."""
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML tags
        cleaned = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
        # Expand contractions if enabled
        if self.expand_contractions:
            cleaned = self.expand_text_contractions(cleaned)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def tokenize_words(self, text: str, keep_punctuation: bool = False) -> List[str]:
        """Splits text into words/tokens."""
        if not text:
            return []
        if keep_punctuation:
            return re.findall(r"\b\w+\b|[^\w\s]", text)
        return re.findall(r"\b\w+\b", text)

    def tokenize_sentences(self, text: str) -> List[str]:
        """Splits text into sentences based on punctuation boundaries."""
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def transform(self, text: str) -> str:
        """Full pipeline transformation for vectorizer/model input."""
        cleaned = self.clean_text(text)
        if self.lowercase:
            cleaned = cleaned.lower()
        return cleaned


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generates n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
