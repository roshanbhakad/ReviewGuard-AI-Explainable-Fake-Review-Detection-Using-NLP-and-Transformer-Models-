"""Flask Web Application for ReviewGuard-AI Dashboard."""

import os
import sys
from flask import Flask, render_template

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api import api_bp


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SECRET_KEY"] = os.urandom(24)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

    # Register API blueprint
    app.register_blueprint(api_bp, url_prefix="/api")

    # Frontend Page Routes
    @app.route("/")
    def index():
        return render_template("index.html", active_page="inspector")

    @app.route("/batch")
    def batch():
        return render_template("batch.html", active_page="batch")

    @app.route("/benchmark")
    def benchmark():
        return render_template("benchmark.html", active_page="benchmark")

    @app.route("/xai")
    def xai():
        return render_template("xai.html", active_page="xai")

    @app.route("/api-docs")
    def api_docs():
        return render_template("api_docs.html", active_page="api_docs")

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting ReviewGuard-AI server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
