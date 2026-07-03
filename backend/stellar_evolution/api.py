"""Flask application factory and route definitions.

Routes
------
GET  /health       Liveness probe.
GET  /             Serve the frontend index.html.
GET  /<path>       Serve static frontend assets (JS, CSS).
POST /simulate     Run a single static stellar structure solve.
POST /evolution    Generate a full precomputed evolution timeline.
POST /color        Convert a blackbody temperature to sRGB colour.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from .color import blackbody_to_srgb
from .evolution import generate_evolution_timeline
from .models import SimulationConfig
from .solver import simulate_structure

# Resolve the frontend directory relative to the project root so Flask can
# serve the static files regardless of the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__, static_folder=None)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> Response:
        """Return a simple liveness response."""
        return jsonify({"status": "ok"})

    # ------------------------------------------------------------------
    # Static frontend serving
    # ------------------------------------------------------------------

    @app.get("/")
    def index() -> Response:
        """Serve the main single-page application entry point."""
        return send_from_directory(_FRONTEND_DIR, "index.html")

    @app.get("/<path:filename>")
    def static_files(filename: str) -> Response:
        """Serve CSS, JS, and any other frontend assets."""
        return send_from_directory(_FRONTEND_DIR, filename)

    # ------------------------------------------------------------------
    # /simulate — single static stellar structure solve
    # ------------------------------------------------------------------

    @app.post("/simulate")
    def simulate() -> Response:
        """Run one radial stellar-structure integration.

        Request body (JSON)
        -------------------
        {
            "mass_solar": 1.0,
            "radius_solar": 1.0,
            "central_temperature": 1.55e7,
            "central_pressure": 2.45e17,
            "composition": {"hydrogen": 0.70, "helium": 0.28, "metals": 0.02},
            "radial_steps": 1200
        }

        Response (JSON)
        ---------------
        {
            "summary": { ... },
            "profiles": { "radius_cm": [...], "temperature": [...], ... }
        }
        """
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        try:
            config = SimulationConfig.from_dict(body)
            config.validate()
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            result = simulate_structure(config)
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            return jsonify({"error": f"Simulation failed: {exc}"}), 500

        return jsonify(result)

    # ------------------------------------------------------------------
    # /evolution — precomputed timeline for the frontend slider
    # ------------------------------------------------------------------

    @app.post("/evolution")
    def evolution() -> Response:
        """Generate a precomputed stellar evolution timeline.

        Uses the same fields as /simulate plus an optional ``frames``
        integer (24–360, default 180) that controls the number of
        slider positions returned.

        Each frame in the response ``timeline`` list contains:
        - index, age_fraction, age_years, phase
        - radius_solar, luminosity_solar, effective_temperature
        - color: {hex, srgb, linear_srgb, xyz, label}

        Response (JSON)
        ---------------
        {
            "structure_summary": { ... },
            "timeline": [ { "index": 0, "color": { "hex": "#..." }, ... }, ... ]
        }
        """
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        try:
            config = SimulationConfig.from_dict(body)
            config.validate()
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

        frames = int(body.get("frames", 180))

        try:
            result = generate_evolution_timeline(config, frames=frames)
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            return jsonify({"error": f"Evolution simulation failed: {exc}"}), 500

        return jsonify(result)

    # ------------------------------------------------------------------
    # /color — standalone colour endpoint
    # ------------------------------------------------------------------

    @app.post("/color")
    def color() -> Response:
        """Convert a blackbody temperature to a perceived sRGB colour.

        This endpoint is useful for testing the colour pipeline directly
        and for any future UI component that needs a live colour preview
        without running the full structure solver.

        Request body (JSON)
        -------------------
        { "temperature": 5772 }

        Response (JSON)
        ---------------
        {
            "temperature": 5772,
            "hex": "#fff5e0",
            "srgb": [1.0, 0.9648, 0.8784],
            "linear_srgb": [1.0, 0.9220, 0.7533],
            "xyz": [0.9505, 1.0, 1.0886],
            "label": "yellow-white"
        }
        """
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        try:
            temperature = float(body["temperature"])
        except KeyError:
            return jsonify({"error": "'temperature' (float, Kelvin) is required."}), 400
        except (TypeError, ValueError):
            return jsonify({"error": "'temperature' must be a number."}), 400

        if temperature <= 0:
            return jsonify({"error": "'temperature' must be positive."}), 400

        color_result = blackbody_to_srgb(temperature)

        return jsonify(
            {
                "temperature": temperature,
                "hex": color_result.hex,
                "srgb": color_result.srgb,
                "linear_srgb": color_result.linear_srgb,
                "xyz": color_result.xyz,
                "label": color_result.label,
            }
        )

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(exc) -> Response:
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc) -> Response:
        return jsonify({"error": "Method not allowed."}), 405

    return app
