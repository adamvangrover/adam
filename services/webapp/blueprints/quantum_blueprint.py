from flask import Blueprint, jsonify, request
from core.agents.specialized.quantum_retrieval_agent import QuantumRetrievalAgent
import logging

quantum_bp = Blueprint('quantum', __name__)
logger = logging.getLogger(__name__)

# Initialize single instance of the agent for the API
agent = QuantumRetrievalAgent()

@quantum_bp.route('/optimize', methods=['POST'])
def optimize_schedule():
    """
    Triggers the Adam optimizer to find the best annealing schedule.
    """
    try:
        data = request.get_json() or {}
        haystack_size = float(data.get("haystack_size", 1e15))
        max_steps = int(data.get("max_steps", 50))

        result = agent.find_needle(haystack_size=haystack_size, max_steps=max_steps)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Quantum optimization failed: {e}")
        return jsonify({"error": str(e)}), 500

@quantum_bp.route('/simulate', methods=['POST'])
def simulate_search():
    """
    Runs a single simulation shot with provided parameters.
    """
    # For now, just alias to optimize for the demo,
    # but theoretically this could run a specific schedule without optimization.
    return optimize_schedule()
