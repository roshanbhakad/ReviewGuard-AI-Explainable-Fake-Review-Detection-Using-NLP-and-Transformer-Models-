"""Command-Line Interface (CLI) for ReviewGuard-AI.
Inspect reviews, view token attributions, and examine stylometric red flags directly in the terminal.
"""

import sys
import os
import argparse
from typing import Optional

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.xai.explainer import ReviewExplainer


def run_cli():
    parser = argparse.ArgumentParser(
        description="ReviewGuard-AI: Explainable Fake Review Detection CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="Review text to analyze",
    )
    parser.add_argument(
        "--rating", "-r",
        type=float,
        default=None,
        help="Optional star rating (1.0 to 5.0) for dissonance analysis",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="ensemble",
        choices=["ensemble", "transformer", "xgboost", "random_forest", "logistic_regression"],
        help="Model architecture to use for decision (default: ensemble)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed stylometrics and token-level saliency scores",
    )

    args = parser.parse_args()

    explainer = ReviewExplainer(artifacts_dir="models_dir")
    if not explainer.is_ready:
        print("\n⚠️ Warning: Pre-trained models not found in `models_dir/`. Running `python train.py` is recommended.")

    res = explainer.explain(args.text, rating=args.rating)
    pred = res["prediction"]
    diag = res["diagnostics"]
    sty = res["stylometrics"]
    sent = res["sentiment"]

    print("\n" + "=" * 65)
    print("🛡️  ReviewGuard-AI Audit Report")
    print("=" * 65)
    print(f"Review Text       : \"{args.text}\"")
    if args.rating is not None:
        print(f"Star Rating       : {args.rating} ★")
    print("-" * 65)
    print(f"Prediction        : {pred['risk_level'].upper()}")
    print(f"Fake Probability  : {pred['fake_probability'] * 100:.1f}%")
    print(f"Model Consensus   : {pred['consensus']}")
    label_str = sent.get("sentiment_label", sent.get("label", "neutral")).capitalize()
    print(f"Sentiment Valence : {sent['polarity']:+.2f} ({label_str})")
    if sent["is_dissonant"]:
        print(f"⚠️ Dissonance Flag : YES (Dissonance Score: {sent['dissonance_score']:.2f})")

    print("\n📋 Executive Summary:")
    print(f"  {diag['summary']}")

    if diag["red_flags"]:
        print(f"\n🚩 Red Flags Detected ({len(diag['red_flags'])}):")
        for rf in diag["red_flags"]:
            print(f"  • [{rf['category']}] ({rf.get('severity', 'info').upper()}): {rf['message']}")

    if diag["authentic_signals"]:
        print(f"\n✅ Authenticity Signals ({len(diag['authentic_signals'])}):")
        for sig in diag["authentic_signals"]:
            print(f"  • [{sig['category']}]: {sig['message']}")

    if args.verbose:
        print("\n🔬 Detailed Stylometric Features:")
        print(f"  • Exclamation Density    : {sty['exclamation_density']:.2f}%")
        print(f"  • All-Caps Words Ratio   : {sty['all_caps_word_ratio'] * 100:.1f}%")
        print(f"  • Lexical Diversity (TTR): {sty['type_token_ratio']:.3f}")
        print(f"  • Flesch Reading Ease    : {sty['flesch_reading_ease']:.1f}")
        print(f"  • Hyperbole Words Ratio  : {sty['hyperbole_ratio'] * 100:.2f}%")
        print(f"  • Promotional Cues Count : {int(sty['promotional_marker_count'])}")

        print("\n🔤 Top Salient Tokens:")
        top_words = res.get("top_words", {})
        if top_words.get("top_fake_triggers"):
            print("  Fake Drivers:")
            for t in top_words["top_fake_triggers"]:
                print(f"    - '{t['token']}' (Score: {t['saliency']:+.3f})")
        if top_words.get("top_genuine_signals"):
            print("  Genuine Drivers:")
            for t in top_words["top_genuine_signals"]:
                print(f"    - '{t['token']}' (Score: {t['saliency']:+.3f})")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_cli()
