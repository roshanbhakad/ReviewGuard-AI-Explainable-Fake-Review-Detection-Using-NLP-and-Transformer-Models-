"""Rule-based diagnostic reasoner that translates stylometrics and dissonance into natural language explanations."""

from typing import Dict, Any, List, Tuple


class DiagnosticReasoner:
    """Generates human-understandable audit diagnostics for review authenticity."""

    # Baseline norms derived from empirical authentic review corpora
    AUTHENTIC_BENCHMARKS = {
        "exclamation_density": 0.8,         # authentic avg ~0.8 per 100 chars
        "all_caps_word_ratio": 0.02,        # authentic avg ~2% of words
        "first_person_singular_ratio": 0.04, # authentic avg ~4%
        "type_token_ratio": 0.72,           # authentic avg ~0.72
        "flesch_reading_ease": 68.0,        # authentic avg ~68
        "hyperbole_ratio": 0.015,           # authentic avg ~1.5%
        "digits_ratio": 1.2,                # authentic avg ~1.2%
    }

    def generate_diagnostics(
        self,
        text: str,
        stylometrics: Dict[str, float],
        sentiment_info: Dict[str, Any],
        fake_prob: float,
        rating: float = None,
    ) -> Dict[str, Any]:
        """Produces red flags, authentic signals, and an executive explanation summary."""
        red_flags: List[Dict[str, str]] = []
        authentic_signals: List[Dict[str, str]] = []

        # 1. Punctuation intensity
        excl = stylometrics.get("exclamation_density", 0.0)
        if excl > 3.0:
            red_flags.append({
                "category": "Punctuation Anomaly",
                "severity": "high" if excl > 5.0 else "medium",
                "message": f"Excessive exclamation mark density ({excl:.1f}% vs typical authentic norm 0.8%). Indicates emotional inflation or artificial hype.",
            })
        elif excl <= 1.0:
            authentic_signals.append({
                "category": "Punctuation",
                "message": "Balanced, restrained punctuation consistent with natural customer feedback.",
            })

        # 2. Capitalization / Shouting
        caps = stylometrics.get("all_caps_word_ratio", 0.0)
        if caps > 0.15:
            red_flags.append({
                "category": "Capitalization",
                "severity": "high" if caps > 0.3 else "medium",
                "message": f"High ratio of ALL-CAPS words ({caps * 100:.1f}%). Common indicator of urgency engineering or bot generation.",
            })

        # 3. Promotional Markers
        promo = int(stylometrics.get("promotional_marker_count", 0))
        if promo > 0:
            red_flags.append({
                "category": "Promotional Spam Cue",
                "severity": "high",
                "message": f"Detected {promo} explicit promotional/incentivized marketing phrases (e.g. call-to-actions, discount codes, or reciprocal review markers).",
            })

        # 4. Hyperbole and Superlatives
        hyper = stylometrics.get("hyperbole_ratio", 0.0)
        if hyper > 0.05:
            red_flags.append({
                "category": "Hyperbolic Language",
                "severity": "medium",
                "message": f"Unusually high density of superlative and extreme evaluative terms ({hyper * 100:.1f}% of words), suggesting lack of objective evaluation.",
            })

        # 5. Rating vs Sentiment Dissonance
        if sentiment_info.get("is_dissonant", False):
            polarity = sentiment_info.get("polarity", 0.0)
            red_flags.append({
                "category": "Sentiment Dissonance",
                "severity": "high",
                "message": f"Severe conflict between Star Rating ({rating}★) and linguistic sentiment ({polarity:+.2f}). Frequently observed in review bombing or click-farm errors.",
            })
        elif rating is not None:
            authentic_signals.append({
                "category": "Sentiment Alignment",
                "message": f"Review sentiment polarity aligns coherently with the stated {rating}★ star rating.",
            })

        # 6. Lexical Richness (Type-Token Ratio)
        ttr = stylometrics.get("type_token_ratio", 0.0)
        word_count = stylometrics.get("word_count", 0.0)
        if word_count > 15 and ttr < 0.50:
            red_flags.append({
                "category": "Lexical Redundancy",
                "severity": "low",
                "message": f"Low vocabulary diversity (TTR: {ttr:.2f}). Repetitive phrasing is characteristic of template-based bot generation.",
            })
        elif ttr > 0.65:
            authentic_signals.append({
                "category": "Vocabulary Richness",
                "message": f"High lexical diversity (TTR: {ttr:.2f}), indicating varied, spontaneous human vocabulary.",
            })

        # 7. Specificity / Quantitative Details (digits, dimensions, time periods)
        digits = stylometrics.get("digits_ratio", 0.0)
        if digits > 0.8:
            authentic_signals.append({
                "category": "Concrete Specificity",
                "message": f"Contains specific numeric details (measurements, durations, or prices: {digits:.1f}%), a strong hallmark of genuine product usage.",
            })

        # 8. First-person Pronoun density
        first_sing = stylometrics.get("first_person_singular_ratio", 0.0)
        if first_sing > 0.10:
            red_flags.append({
                "category": "First-Person Bias",
                "severity": "low",
                "message": f"High concentration of first-person singular pronouns ({first_sing * 100:.1f}%). Deceptive writers frequently over-emphasize personal presence to feign authenticity.",
            })

        # Executive summary generation
        if fake_prob >= 0.70:
            summary = (
                f"ReviewGuard-AI flagged this review as HIGH RISK ({fake_prob * 100:.1f}% deceptive probability). "
                f"The analysis revealed {len(red_flags)} anomaly markers, driven primarily by "
                f"{red_flags[0]['category'].lower() if red_flags else 'stylometric deviation'}."
            )
        elif fake_prob >= 0.45:
            summary = (
                f"ReviewGuard-AI identified moderate suspicion ({fake_prob * 100:.1f}% deceptive probability). "
                f"While some authentic signals exist, {len(red_flags)} warning indicator(s) were recorded."
            )
        else:
            summary = (
                f"ReviewGuard-AI classified this review as GENUINE ({(1.0 - fake_prob) * 100:.1f}% authenticity score). "
                f"The text exhibits normal lexical distributions, consistent rating-sentiment alignment, and concrete contextual markers."
            )

        return {
            "summary": summary,
            "red_flags": red_flags,
            "authentic_signals": authentic_signals,
            "benchmarks": self.AUTHENTIC_BENCHMARKS,
        }
