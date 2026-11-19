"""REST API endpoints for ReviewGuard-AI."""

import os
import io
import pandas as pd
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify, current_app

from src.xai.explainer import ReviewExplainer
from src.models.hf_transformer import HFTransformerDetector

api_bp = Blueprint("api_bp", __name__)
explainer = ReviewExplainer(artifacts_dir="models_dir")
hf_detector = HFTransformerDetector()


@api_bp.route("/v1/predict", methods=["POST"])
def predict_endpoint():
    """Predicts fake review probability and risk classification."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    rating = data.get("rating")
    if rating is not None:
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            rating = None

    if not text:
        return jsonify({"error": "Missing or empty 'text' field."}), 400

    res = explainer.explain(text, rating=rating)
    return jsonify({
        "status": "success",
        "data": res["prediction"],
    })


@api_bp.route("/v1/explain", methods=["POST"])
def explain_endpoint():
    """Provides full explainability analysis (token saliency, stylometrics, diagnostics, radar data)."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    rating = data.get("rating")
    if rating is not None:
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            rating = None

    if not text:
        return jsonify({"error": "Missing or empty 'text' field."}), 400

    res = explainer.explain(text, rating=rating)
    return jsonify({
        "status": "success",
        "data": res,
    })


@api_bp.route("/v1/batch", methods=["POST"])
def batch_endpoint():
    """Processes batch of reviews via JSON payload or CSV file upload."""
    reviews_data: List[Dict[str, Any]] = []

    # Check if multipart form with CSV file
    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            try:
                stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
                df = pd.read_csv(stream)
                text_col = None
                for col in ["review_text", "text", "review", "content"]:
                    if col in df.columns:
                        text_col = col
                        break
                if not text_col:
                    return jsonify({"error": "CSV must contain a 'review_text' or 'text' column."}), 400

                rating_col = None
                for col in ["rating", "stars", "star_rating"]:
                    if col in df.columns:
                        rating_col = col
                        break

                for idx, row in df.iterrows():
                    reviews_data.append({
                        "id": idx + 1,
                        "text": str(row[text_col]),
                        "rating": float(row[rating_col]) if rating_col and pd.notna(row[rating_col]) else None,
                    })
            except Exception as e:
                return jsonify({"error": f"Failed to parse CSV file: {str(e)}"}), 400

    # Else parse JSON array
    if not reviews_data:
        data = request.get_json(silent=True) or {}
        reviews_data = data.get("reviews", [])

    if not reviews_data:
        return jsonify({"error": "No reviews provided. Send JSON array under 'reviews' or upload CSV."}), 400

    results = []
    fake_count = 0
    total_fake_prob = 0.0

    for idx, item in enumerate(reviews_data[:200]):  # Cap at 200 items per request for responsiveness
        text = str(item.get("text", "")).strip()
        rating = item.get("rating")
        if not text:
            continue

        res = explainer.explain(text, rating=rating)
        prob = res["prediction"]["fake_probability"]
        is_fake = prob >= 0.5
        if is_fake:
            fake_count += 1
        total_fake_prob += prob

        results.append({
            "id": item.get("id", idx + 1),
            "text": text,
            "rating": rating,
            "fake_probability": prob,
            "risk_level": res["prediction"]["risk_level"],
            "risk_badge": res["prediction"]["risk_badge"],
            "red_flags_count": len(res["diagnostics"]["red_flags"]),
            "is_dissonant": res["sentiment"]["is_dissonant"],
        })

    total_valid = len(results)
    avg_fake_prob = round(total_fake_prob / total_valid, 4) if total_valid > 0 else 0.0
    spam_rate = round((fake_count / total_valid) * 100.0, 1) if total_valid > 0 else 0.0

    return jsonify({
        "status": "success",
        "summary": {
            "total_reviews": total_valid,
            "flagged_deceptive": fake_count,
            "authentic_count": total_valid - fake_count,
            "deceptive_percentage": spam_rate,
            "average_fake_probability": avg_fake_prob,
        },
        "results": results,
    })


@api_bp.route("/v1/models", methods=["GET"])
def models_endpoint():
    """Returns benchmark leaderboard metrics and status of all models."""
    import joblib
    metrics_path = os.path.join("models_dir", "all_benchmark_metrics.joblib")
    metrics = {}
    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)

    hf_status = hf_detector.get_status()

    return jsonify({
        "status": "success",
        "benchmark_metrics": metrics,
        "huggingface_status": hf_status,
        "active_models": list(metrics.keys()),
    })


@api_bp.route("/v1/samples", methods=["GET"])
def samples_endpoint():
    """Returns curated sample reviews for quick demo testing."""
    samples = [
        {
            "category": "Obvious 5-Star Bot Spam",
            "text": "BEST PRODUCT EVERRRR!!!! Absolutely blown away by the quality!! Changed my life forever!! MUST BUY 100% recommended to everyone!! A++++++ service!!!",
            "rating": 5.0,
            "expected": "Fake",
        },
        {
            "category": "Genuine Verified Tech Review",
            "text": "I've been using this wireless earbuds for about 6 months. Sound stage is solid and ANC noise cancellation works as expected. The companion mobile app could be improved, but overall decent value for $79.",
            "rating": 4.0,
            "expected": "Genuine",
        },
        {
            "category": "Smear 1-Star Competitor Attack",
            "text": "DO NOT BUY!!! TOTAL SCAM!! Broke in 5 seconds!! Garbage trash worthless junk!! Worst company on planet earth!! RUN AWAY SCAM SCAM SCAM!!",
            "rating": 1.0,
            "expected": "Fake",
        },
        {
            "category": "Rating-Sentiment Dissonance",
            "text": "Completely broken on arrival, total disaster and useless waste of money. Worst thing ever.",
            "rating": 5.0,
            "expected": "Fake (Dissonant 5-Star)",
        },
        {
            "category": "Genuine Kitchen Appliance Review",
            "text": "We purchased this air fryer for daily meal prep. Cuts prep time in half and dishwasher cleaning is straightforward. The lid latch requires a firm press to seal properly.",
            "rating": 4.0,
            "expected": "Genuine",
        },
    ]
    return jsonify({"status": "success", "samples": samples})
