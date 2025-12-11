import os
import sys
import logging
import asyncio
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from collections import deque

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.utils.logging_utils import setup_logging

# Safe Import for Orchestrators
try:
    from core.engine.meta_orchestrator import MetaOrchestrator
    from core.system.agent_orchestrator import AgentOrchestrator
except ImportError as e:
    print(f"Warning: AI Engine dependencies missing ({e}). Starting in Lightweight Mode.")
    MetaOrchestrator = None
    AgentOrchestrator = None

app = Flask(__name__)

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

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHOWCASE_DIR = os.path.join(BASE_DIR, 'showcase')

# Global State
meta_orchestrator = None
LOG_BUFFER = deque(maxlen=50)

class ListHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = self.format(record)
            LOG_BUFFER.append(log_entry)
        except Exception:
            self.handleError(record)

def setup_log_capture():
    handler = ListHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s') # Simplified format for UI
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    # Ensure we capture 'core' logs
    logging.getLogger('core').setLevel(logging.INFO)

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

@app.route('/')
def serve_index():
    return send_from_directory(SHOWCASE_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(SHOWCASE_DIR, path)

@app.route('/api/state', methods=['GET'])
def get_state():
    # Return basic system stats and logs
    return jsonify({
        "status": "online",
        "version": "v23.5-Adaptive",
        "mode": "Live Runtime" if meta_orchestrator else "Lightweight Mode (No AI)",
        "orchestrator": "MetaOrchestrator (Active)" if meta_orchestrator else "Offline",
        "logs": list(LOG_BUFFER)
    })

@app.route('/api/chat', methods=['POST'])
async def chat():
    if not meta_orchestrator:
        return jsonify({
            "error": "Orchestrator is offline (Lightweight Mode)",
            "human_readable_status": "**System Notice:** The AI Engine is currently offline (dependencies missing). Please install full requirements to enable Deep Dive analysis."
        }), 200 # Return 200 so UI handles it gracefully as a message

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

if __name__ == '__main__':
    setup_logging()
    setup_log_capture()
    init_orchestrator()
    print(f"Serving Adam v23.5 from {SHOWCASE_DIR}")
    print(f"Access the Live Interface at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
