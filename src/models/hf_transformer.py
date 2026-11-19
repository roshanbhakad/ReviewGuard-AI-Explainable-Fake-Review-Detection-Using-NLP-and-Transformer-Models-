"""Hugging Face Transformers integration (RoBERTa, DeBERTa, DistilBERT).
Provides optional deep transformer fine-tuning and inference when torch and transformers are installed.
"""

from typing import Dict, Any, List, Optional
import os


class HFTransformerDetector:
    """Detects and provides interface to Hugging Face models if installed."""

    def __init__(self, model_name: str = "roberta-base"):
        self.model_name = model_name
        self.is_available = False
        self.tokenizer = None
        self.model = None

        try:
            import torch
            import transformers
            self.is_available = True
            self.torch = torch
            self.transformers = transformers
        except ImportError:
            self.is_available = False

    def get_status(self) -> Dict[str, Any]:
        """Returns Hugging Face library and model status."""
        return {
            "hf_available": self.is_available,
            "supported_models": ["roberta-base", "microsoft/deberta-v3-base", "distilbert-base-uncased"],
            "current_model": self.model_name,
            "installation_guide": "Run `pip install transformers torch` to enable Hugging Face pretrained RoBERTa/DeBERTa models.",
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """Performs inference if HF is available, else returns advisory."""
        if not self.is_available:
            return {
                "available": False,
                "message": "Hugging Face transformers not installed. The native TensorFlow Multi-Head Attention Transformer is active and serving predictions.",
            }

        # If torch and transformers are present, provide HF pipeline execution
        try:
            from transformers import pipeline
            clf = pipeline("text-classification", model=self.model_name)
            res = clf(text)
            return {
                "available": True,
                "result": res,
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            }
