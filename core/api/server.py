import os
import sys
import logging
import asyncio
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.utils.logging_utils import setup_logging
from core.prompting.scanner import PromptScanner


# Safe Import for Orchestrators
try:
    from core.engine.meta_orchestrator import MetaOrchestrator
    from core.system.agent_orchestrator import AgentOrchestrator
except ImportError as e:
    print(f"Warning: AI Engine dependencies missing ({e}). Starting in Lightweight Mode.")
    MetaOrchestrator = None
    AgentOrchestrator = None

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# SHOWCASE_DIR = os.path.join(BASE_DIR, 'showcase')
SHOWCASE_DIR = os.path.join(BASE_DIR, 'services/webapp/client/build')
STATIC_DIR = os.path.join(SHOWCASE_DIR, 'static')

# Initialize Flask with explicit static folder to avoid default behavior
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')

# Security: Configure CORS
# Default to localhost for development. In production, set ALLOWED_ORIGINS env var.
# Example: ALLOWED_ORIGINS="https://mydomain.com,https://api.mydomain.com"
allowed_origins_str = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5000,"
    "http://127.0.0.1:3000,http://127.0.0.1:5000"
)
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

CORS(app, resources={r"/*": {"origins": allowed_origins}})

# Global State
meta_orchestrator = None
simulation_engine = None
# LOG_BUFFER removed to prevent sensitive data exposure
# LOG_BUFFER = deque(maxlen=50)


def setup_log_capture():
    # Logging to memory buffer removed to prevent sensitive data exposure via API
    # Standard stdout/file logging is sufficient and safer
    pass


def init_orchestrator():
    global meta_orchestrator
    logger = logging.getLogger(__name__)
    if meta_orchestrator is not None:
        return

    if MetaOrchestrator is None:
        logger.warning("MetaOrchestrator class not found. Skipping initialization.")
        return

    logger.info("Initializing Orchestration Layer...")
    try:
        # Initialize Legacy Orchestrator (v21/v22)
        try:
            legacy_orchestrator = AgentOrchestrator()
        except Exception as e:
            logger.error(f"Failed to initialize AgentOrchestrator (Legacy): {e}")
            legacy_orchestrator = None

        # Initialize Meta Orchestrator (v23 Brain)
        meta_orchestrator = MetaOrchestrator(legacy_orchestrator=legacy_orchestrator)
        logger.info("MetaOrchestrator Initialized Successfully.")
    except Exception as e:
        logger.fatal(f"Failed to initialize MetaOrchestrator: {e}")

    # Initialize Crisis Adjudicator Engine
    global simulation_engine
    try:
        from core.engine.adjudicator_engine import AdjudicatorEngine
        simulation_engine = AdjudicatorEngine()
        logger.info("AdjudicatorEngine Initialized Successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize AdjudicatorEngine: {e}")


@app.route('/')
def serve_index():
    return send_from_directory(SHOWCASE_DIR, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    # Fallback catch-all for SPA client-side routing
    # Check if file exists in root (e.g. manifest.json, favicon.ico)
    if os.path.exists(os.path.join(SHOWCASE_DIR, path)):
        return send_from_directory(SHOWCASE_DIR, path)
    return send_from_directory(SHOWCASE_DIR, 'index.html')


@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """
    Returns the list of prompts with ROI scores.
    Optional query param: ?context=keyword1,keyword2
    """
    try:
        context_str = request.args.get('context', '')
        context = [k.strip() for k in context_str.split(',')] if context_str else None

        prompts = PromptScanner.scan(context=context)
        return jsonify(prompts)
    except Exception as e:
        logging.error(f"Error scanning prompts: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/state', methods=['GET'])
def get_state():
    # Return basic system stats and logs
    return jsonify({
        "status": "online",
        "version": "v23.5-Adaptive",
        "mode": "Live Runtime" if meta_orchestrator else "Lightweight Mode (No AI)",
        "orchestrator": "MetaOrchestrator (Active)" if meta_orchestrator else "Offline",
        # "logs": [] # Logs removed to prevent information leakage
    })


@app.route('/api/chat', methods=['POST'])
async def chat():
    if not meta_orchestrator:
        return jsonify({
            "error": "Orchestrator is offline (Lightweight Mode)",
            "human_readable_status": "**System Notice:** The AI Engine is currently offline (dependencies missing). Please install full requirements to enable Deep Dive analysis."
        }), 200  # Return 200 so UI handles it gracefully as a message

    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    logging.info(f"Received API Query: {query}")

    try:
        # The Orchestrator is async, Flask supports awaiting async functions
        result = await meta_orchestrator.route_request(query)

        # Ensure result is JSON serializable
        if hasattr(result, 'dict'):
            result = result.dict()
        elif hasattr(result, 'model_dump'):
            result = result.model_dump()

        return jsonify(result)
    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    if not simulation_engine:
        return jsonify({"error": "Simulation Engine not available"}), 503
    simulation_engine.reset()
    return jsonify({"status": "Simulation Started", "state": simulation_engine.get_state()})


@app.route('/api/simulation/state', methods=['GET'])
def get_simulation_state():
    if not simulation_engine:
         return jsonify({"error": "Simulation Engine not available"}), 503
    return jsonify(simulation_engine.get_state())


@app.route('/api/simulation/action', methods=['POST'])
def submit_action():
    if not simulation_engine:
         return jsonify({"error": "Simulation Engine not available"}), 503

    data = request.json
    inject_id = data.get('inject_id')
    option_id = data.get('option_id')

    if not inject_id or not option_id:
        return jsonify({"error": "Missing inject_id or option_id"}), 400

    simulation_engine.resolve_action(inject_id, option_id)
    return jsonify({"status": "Action Resolved", "state": simulation_engine.get_state()})

if __name__ == '__main__':
    setup_logging()
    setup_log_capture()
    init_orchestrator()
    print(f"Serving Adam v23.5 from {SHOWCASE_DIR}")
    print(f"Static Dir: {STATIC_DIR}")
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    print(f"Access the Live Interface at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
