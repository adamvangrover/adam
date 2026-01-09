import os
import json
import time
import random
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

# Define the showcase directory relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOWCASE_DIR = os.path.join(BASE_DIR, 'showcase')
DATA_FILE = os.path.join(SHOWCASE_DIR, 'data', 'ui_data.json')

app = Flask(__name__, static_folder=SHOWCASE_DIR)
CORS(app)


@app.route('/')
def serve_index():
    """Serves the main index.html file."""
    return send_from_directory(SHOWCASE_DIR, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serves static files from the showcase directory."""
    return send_from_directory(SHOWCASE_DIR, path)


@app.route('/api/state')
def get_state():
    """
    Retrieves the current system state.

    Simulates real-time updates by reading generated data and adding noise
    to system statistics.
    """
    # Simulate real-time updates by reading the generated data and adding some noise
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)

        # Add dynamic elements
        data['system_stats']['cpu_usage'] = round(random.uniform(10, 60), 1)
        data['system_stats']['memory_usage'] = round(random.uniform(30, 80), 1)
        data['system_stats']['active_tasks'] = random.randint(1, 5)
        data['system_stats']['uptime'] = time.time() - data['generated_at']

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/files')
def get_files():
    """Retrieves the file structure of the repository."""
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data['files'])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/agents')
def get_agents():
    """Retrieves the list of active agents."""
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data['agents'])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print(f"Serving UI from {SHOWCASE_DIR}")
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    app.run(host=host, port=port, debug=False)
