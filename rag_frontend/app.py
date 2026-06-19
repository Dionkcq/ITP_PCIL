"""
RAG frontend - minimal Flask UI that calls the PCIL orchestrator.
All retrieval and generation logic lives in the orchestrator.
This file only serves static assets and proxies a run request.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
DEFAULT_CONFIG = os.environ.get(
    "PCIL_CONFIG_PATH", "systems/inkjet_printer/config.yaml"
)
FRONTEND_DIR = Path(__file__).resolve().parent


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/app.js", methods=["GET"])
def app_js():
    return send_from_directory(FRONTEND_DIR, "app.js")


@app.route("/styles.css", methods=["GET"])
def styles_css():
    return send_from_directory(FRONTEND_DIR, "styles.css")


@app.route("/run", methods=["POST"])
def run_pipeline():
    """Proxy request to the orchestrator to avoid browser CORS issues."""
    try:
        resp = requests.post(
            f"{ORCHESTRATOR_URL}/pipeline/run",
            json={"config_path": DEFAULT_CONFIG, "persist": False},
            timeout=60,
        )
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            return jsonify({"error": detail}), resp.status_code
        return jsonify(resp.json())
    except requests.ConnectionError:
        return jsonify({
            "error": f"Cannot reach orchestrator at {ORCHESTRATOR_URL}. Is it running?",
        }), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
