from flask import Flask, request, jsonify
import logging
import secrets
import warnings

# 🛡️ Sentinel: Deprecation Warning
logger = logging.getLogger(__name__)
logger.warning("SECURITY WARNING: This module (core/api.py) is DEPRECATED and unsafe.")
logger.warning("Please use services/webapp/api.py or core/api/main.py instead.")
warnings.warn("core/api.py is deprecated and will be removed.", DeprecationWarning, stacklevel=2)
from core.system.agent_orchestrator import AgentOrchestrator
from core.system.echo import Echo
from core.utils.api_utils import (
    get_knowledge_graph_data,
    update_knowledge_graph_node,
)
from core.settings import settings

app = Flask(__name__)

# Initialize Agent Orchestrator and Echo System
agent_orchestrator = AgentOrchestrator()
# Fix: Echo requires a config argument to initialize, preventing startup crash.
echo_system = Echo(config={})

# 🛡️ Sentinel: Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

@app.route("/", methods=["POST"])
def api_endpoint():
    """
    Adam v17.0 API Endpoint

    This is the unified API endpoint for interacting with all of Adam's functionalities.
    """
    # 🛡️ Sentinel: Enforce API Key Authentication
    api_key = request.headers.get("X-API-Key")
    # Use constant-time comparison to prevent timing attacks
    if not api_key or not secrets.compare_digest(api_key, settings.adam_api_key):
        logging.warning(f"Unauthorized access attempt from {request.remote_addr}")
        return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 401

    try:
        data = request.get_json()
        module = data.get("module")
        action = data.get("action")
        parameters = data.get("parameters")

        if module == "knowledge_graph":
            if action == "get_data":
                results = get_knowledge_graph_data(**parameters)
            elif action == "update_node":
                results = update_knowledge_graph_node(**parameters)
            else:
                return jsonify({"error": "Invalid action for knowledge_graph module."}), 400
        elif module == "agent_orchestrator":
            if action == "run_analysis":
                results = agent_orchestrator.run_analysis(**parameters)
            else:
                return jsonify({"error": "Invalid action for agent_orchestrator module."}), 400
        elif module == "echo_system":
            if action == "get_insights":
                results = echo_system.get_insights(**parameters)
            else:
                return jsonify({"error": "Invalid action for echo_system module."}), 400
        else:
            return jsonify({"error": "Invalid module."}), 400

        return jsonify({"results": results}), 200

    except Exception as e:
        # 🛡️ Sentinel: Log the full error securely, but return a generic message to the user
        # to prevent leaking sensitive system details (stack traces, paths, etc.)
        logging.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500


if __name__ == "__main__":
    import os
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=debug_mode)
