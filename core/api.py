import logging

from flask import Flask, jsonify, request

from core.system.agent_orchestrator import AgentOrchestrator
from core.system.echo import Echo
from core.utils.api_utils import (
    get_knowledge_graph_data,
    update_knowledge_graph_node,
)

app = Flask(__name__)

# Initialize Agent Orchestrator and Echo System
agent_orchestrator = AgentOrchestrator()
echo_system = Echo()

@app.route("/", methods=["POST"])
def api_endpoint():
    """
    Adam v17.0 API Endpoint

    This is the unified API endpoint for interacting with all of Adam's functionalities.
    """
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
    app.run(debug=False)
