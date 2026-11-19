"""Attention and saliency visualizer for token-level explainability."""

from typing import List, Dict, Any
import html


class SaliencyVisualizer:
    """Formats word-level saliency scores into annotated visual HTML and structured payloads."""

    @staticmethod
    def render_highlighted_html(tokens: List[Dict[str, Any]]) -> str:
        """Generates an HTML snippet with color-coded token spans and interactive tooltips.

        Red highlights correspond to tokens contributing toward Fake/Deceptive.
        Green highlights correspond to tokens contributing toward Genuine/Authentic.
        """
        if not tokens:
            return "<p class='text-gray-400 italic'>No token attribution available.</p>"

        spans = []
        for t in tokens:
            tok_text = html.escape(t.get("token", ""))
            norm_score = t.get("normalized_score", 0.0)
            sal = t.get("saliency", 0.0)
            att = t.get("attention_weight", 0.0)
            direction = t.get("direction", "fake")

            alpha = min(0.85, max(0.12, abs(norm_score)))
            if direction == "fake":
                bg_color = f"rgba(239, 68, 68, {alpha:.2f})"
                text_color = "#ffffff" if alpha > 0.55 else "#fca5a5"
                border_color = "rgba(220, 38, 38, 0.5)"
            else:
                bg_color = f"rgba(16, 185, 129, {alpha:.2f})"
                text_color = "#ffffff" if alpha > 0.55 else "#6ee7b7"
                border_color = "rgba(5, 150, 105, 0.5)"

            tooltip = f"Word: '{tok_text}' | Importance: {sal:+.3f} | Saliency: {norm_score:+.2f} | Attention: {att:.3f} | Leans: {direction.upper()}"

            span = (
                f"<span class='inline-block px-1.5 py-0.5 m-0.5 rounded text-sm transition-all duration-200 cursor-pointer hover:scale-105' "
                f"style='background-color: {bg_color}; color: {text_color}; border: 1px solid {border_color};' "
                f"title=\"{tooltip}\" data-score=\"{norm_score}\" data-direction=\"{direction}\">"
                f"{tok_text}"
                f"</span>"
            )
            spans.append(span)

        return " ".join(spans)

    @staticmethod
    def get_top_salient_words(tokens: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Filters top tokens driving the classification toward Fake and Genuine."""
        fake_tokens = [t for t in tokens if t.get("direction") == "fake"]
        genuine_tokens = [t for t in tokens if t.get("direction") == "genuine"]

        fake_sorted = sorted(fake_tokens, key=lambda x: abs(x.get("saliency", 0)), reverse=True)[:top_k]
        genuine_sorted = sorted(genuine_tokens, key=lambda x: abs(x.get("saliency", 0)), reverse=True)[:top_k]

        return {
            "top_fake_triggers": fake_sorted,
            "top_genuine_signals": genuine_sorted,
        }
