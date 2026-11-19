"""Main launcher for ReviewGuard-AI application.
Usage:
    python run_app.py [--port 5000] [--train]
"""

import os
import sys
import argparse

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.web_server import create_app


def main():
    parser = argparse.ArgumentParser(description="ReviewGuard-AI Web Dashboard Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--train", action="store_true", help="Force retrain models prior to launch")
    args = parser.parse_args()

    # Check if models exist
    models_file = os.path.join("models_dir", "ml_models.joblib")
    if args.train or not os.path.exists(models_file):
        print("⚡ Pre-trained models not found or --train specified. Running training pipeline...")
        from train import main as train_main
        train_main()

    app = create_app()

    print("\n" + "=" * 70)
    print("🛡️  ReviewGuard-AI: Explainable Fake Review Detection System")
    print("=" * 70)
    print(f"  • Web Dashboard URL : http://{args.host}:{args.port}")
    print(f"  • Single Inspector  : http://{args.host}:{args.port}/")
    print(f"  • Batch Auditor     : http://{args.host}:{args.port}/batch")
    print(f"  • Model Arena       : http://{args.host}:{args.port}/benchmark")
    print(f"  • XAI Deep Dive Lab : http://{args.host}:{args.port}/xai")
    print(f"  • REST API Console  : http://{args.host}:{args.port}/api-docs")
    print("=" * 70)
    print("Press Ctrl+C to stop the server.\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
