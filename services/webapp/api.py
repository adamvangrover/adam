from __future__ import annotations
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, JWTManager, get_jwt_identity, get_jwt
from werkzeug.exceptions import HTTPException
from werkzeug.security import generate_password_hash, check_password_hash
import sys
import os
import logging
import json
import functools
import asyncio
import re
import threading
from datetime import datetime, timezone, timedelta
from .config import config
from .celery_app import celery
from .governance import GovernanceMiddleware

# Import the Live Mock Engine for dynamic simulation data
try:
    from core.engine.live_mock_engine import live_engine
    from core.engine.forecasting_engine import forecasting_engine
    from core.engine.conviction_manager import conviction_manager
except ImportError as e:
    # Fallback if core is not in path or other import issues
    logging.warning(f"Could not import core engines: {e}. Simulation endpoints will be static.")
    live_engine = None
    forecasting_engine = None
    conviction_manager = None

# ---------------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------------- #

IP_BLOCK_THRESHOLD = 5
BLOCK_DURATION_MINUTES = 15

# 🛡️ Sentinel: Pre-calculate dummy hash for timing attack mitigation
# This ensures that invalid username lookups take roughly the same amount of time
# as valid ones (by performing a hash check in both cases).
DUMMY_PASSWORD_HASH = generate_password_hash('dummy_password_for_timing_mitigation')

# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _get_allowed_simulations():
    simulations_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'core', 'simulations')
    if not os.path.exists(simulations_dir):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(simulations_dir) if f.endswith('.py') and not f.startswith('__')]


def _is_password_strong(password: str) -> bool:
    """
    🛡️ Sentinel: Validate password strength.
    Requires:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    """
    if not password:
        return False
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    # Check for at least one special character (anything not alphanumeric)
    if not re.search(r'[^a-zA-Z0-9]', password):
        return False
    return True


def _validate_username(username: str) -> bool:
    """
    🛡️ Sentinel: Validate username format.
    Requires:
    - Length between 4 and 30 characters
    - Alphanumeric characters, underscores, and hyphens only
    - Cannot start or end with special characters
    """
    if not username:
        return False

    # Check length
    if len(username) < 4 or len(username) > 30:
        return False

    # Check allowed characters (alphanumeric, underscore, hyphen)
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        return False

    return True

def _validate_portfolio_name(name: str) -> bool:
    """
    🛡️ Sentinel: Validate portfolio name.
    Requires:
    - Length between 1 and 120 characters
    - No HTML tags or dangerous characters
    """
    if not name or not isinstance(name, str):
        return False
    if len(name) < 1 or len(name) > 120:
        return False
    # Basic anti-XSS: Don't allow < or >
    if '<' in name or '>' in name:
        return False
    return True

def _validate_asset_symbol(symbol: str) -> bool:
    """
    🛡️ Sentinel: Validate asset symbol.
    Requires:
    - Length between 1 and 20 characters
    - Alphanumeric (uppercase preferred, but we'll case-insensitive check and convert later if needed)
    """
    if not symbol or not isinstance(symbol, str):
        return False
    if len(symbol) < 1 or len(symbol) > 20:
        return False
    # Should be alphanumeric
    if not re.match(r'^[a-zA-Z0-9\.]+$', symbol):
        return False
    return True

# ---------------------------------------------------------------------------- #
# Initialize Extensions
# ---------------------------------------------------------------------------- #


db = SQLAlchemy()
socketio = SocketIO()
jwt = JWTManager()
agent_orchestrator = None
meta_orchestrator = None
_neo4j_driver = None
_neo4j_lock = threading.Lock()


def get_neo4j_driver():
    """
    ⚡ Bolt Optimization: Singleton pattern for Neo4j driver.
    Prevents creating a new connection pool on every request.
    Uses double-checked locking for thread safety.
    """
    global _neo4j_driver

    if _neo4j_driver:
        return _neo4j_driver

    with _neo4j_lock:
        if _neo4j_driver:
            return _neo4j_driver

        try:
            from neo4j import GraphDatabase
            uri = os.environ.get('NEO4J_URI', 'bolt://neo4j:7687')
            user = os.environ.get('NEO4J_USER', 'neo4j')
            password = os.environ.get('NEO4J_PASSWORD')  # 🛡️ Sentinel: No hardcoded default password
            auth = (user, password) if password else None

            _neo4j_driver = GraphDatabase.driver(uri, auth=auth)
            return _neo4j_driver
        except ImportError:
            logging.getLogger(__name__).warning("Neo4j driver could not be imported.")
            return None
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to initialize Neo4j driver: {e}")
            return None


# ---------------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------------- #

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(80), nullable=False, default='user')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('portfolios', lazy=True))
    assets = db.relationship('PortfolioAsset', backref='portfolio', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Portfolio {self.name}>'


class PortfolioAsset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    purchase_price = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<PortfolioAsset {self.symbol}>'


class SimulationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(36), unique=True, nullable=False)
    simulation_name = db.Column(db.String(120), nullable=False)
    result = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('simulation_results', lazy=True))

    def __repr__(self):
        return f'<SimulationResult {self.simulation_name}>'


class TokenBlocklist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(36), nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False)


class LoginAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(50), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    successful = db.Column(db.Boolean, default=False)


# ---------------------------------------------------------------------------- #
# Application Factory
# ---------------------------------------------------------------------------- #


def create_app(config_name='default'):
    """
    Application factory.
    """
    global agent_orchestrator
    global meta_orchestrator

    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Initialize extensions
    db.init_app(app)
    # 🛡️ Sentinel: Configured CORS from settings to avoid wildcard access.
    socketio.init_app(app, cors_allowed_origins=app.config.get('CORS_ALLOWED_ORIGINS', []))
    jwt.init_app(app)

    # 🛡️ Governance: Initialize middleware
    # Protocol: ADAM-V-NEXT
    GovernanceMiddleware(app)

    # Configure Celery
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask

    # Initialize the core components
    global meta_orchestrator
    if app.config['CORE_INTEGRATION']:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from core.utils.config_utils import load_app_config
        from core.system.agent_orchestrator import AgentOrchestrator
        from core.system.knowledge_base import KnowledgeBase
        from core.system.data_manager import DataManager
        from core.engine.meta_orchestrator import MetaOrchestrator

        core_config = load_app_config()
        # knowledge_base and data_manager are not used by AgentOrchestrator v2
        # knowledge_base = KnowledgeBase(core_config)
        # data_manager = DataManager(core_config)
        agent_orchestrator = AgentOrchestrator()
        meta_orchestrator = MetaOrchestrator(legacy_orchestrator=agent_orchestrator)

    # ---------------------------------------------------------------------------- #
    # Security Headers
    # ---------------------------------------------------------------------------- #

    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        return response

    # ---------------------------------------------------------------------------- #
    # API Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/hello')
    def hello_world():
        return 'Hello, World!'

    # ---------------------------------------------------------------------------- #
    # Mission Control / Synthesizer Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/synthesizer/confidence', methods=['GET'])
    @jwt_required()
    def get_synthesizer_confidence():
        """
        Returns the system's aggregated confidence score and market pulse signals.
        Powered by the LiveMockEngine to simulate real-time volatility.
        """
        if live_engine:
            pulse = live_engine.get_market_pulse()
            consensus_result = live_engine.get_synthesizer_score()

            # Protocol: ADAM-V-NEXT - Enterprise Payload
            # Extract normalized score for backward compatibility, pass full object for new UI
            score_val = consensus_result.get('normalized_score', 50)

            return jsonify({
                "score": score_val,
                "consensus": consensus_result,
                "pulse": pulse,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                "score": 50,
                "pulse": {},
                "status": "Simulation Engine Offline"
            })

    @app.route('/api/intercom/stream', methods=['GET'])
    @jwt_required()
    def get_intercom_stream():
        """
        Returns a stream of 'thoughts' from the agent swarm.
        """
        if live_engine:
            thoughts = live_engine.get_agent_stream(limit=10)
            return jsonify(thoughts)
        else:
            return jsonify(["System offline.", "Waiting for agent connection..."])

    @app.route('/api/synthesizer/forecast/<symbol>', methods=['GET'])
    @jwt_required()
    def get_forecast(symbol):
        """
        Returns historical data and a 30-day probabilistic forecast for the given symbol.
        """
        if not forecasting_engine:
             return jsonify({'error': 'Forecasting Engine Unavailable'}), 503

        # Load history from the generated JSON
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'core', 'data', 'generated_history.json')
            with open(data_path, 'r') as f:
                history_db = json.load(f)

            symbol = symbol.upper()
            if symbol not in history_db:
                return jsonify({'error': 'Symbol not found'}), 404

            history = history_db[symbol]
            forecast = forecasting_engine.generate_forecast(symbol, history, days=30)

            return jsonify({
                "history": history[-90:], # Return last 90 days for context
                "forecast": forecast
            })
        except Exception as e:
            app.logger.error(f"Forecasting error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/synthesizer/conviction', methods=['GET'])
    @jwt_required()
    def get_conviction():
        """
        Returns the current conviction heatmap of the agent swarm.
        """
        if conviction_manager:
            return jsonify(conviction_manager.get_conviction_map())
        else:
            return jsonify({'error': 'Conviction Manager Unavailable'}), 503

    @app.route('/api/v23/analyze', methods=['POST'])
    @jwt_required()
    def run_v23_analysis():
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # 🛡️ Sentinel: Validate query length and type
        if not isinstance(query, str):
            return jsonify({'error': 'Query must be a string'}), 400
        if len(query) > 5000:
            return jsonify({'error': 'Query too long (max 5000 chars)'}), 400

        if app.config['CORE_INTEGRATION'] and meta_orchestrator:
            import asyncio
            try:
                # Ensure we have an event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    # If we are already in a loop (e.g. uvicorn/hypercorn), handle differently
                    # But Flask dev server is threaded.
                    future = asyncio.run_coroutine_threadsafe(meta_orchestrator.route_request(query), loop)
                    result = future.result()
                else:
                    result = loop.run_until_complete(meta_orchestrator.route_request(query))

                return jsonify(result)
            except Exception as e:
                app.logger.error(f"Error in v23 analysis: {e}", exc_info=True)
                return jsonify({'error': 'An internal error occurred during analysis.'}), 500
        else:
            return jsonify({'status': 'Mock Result', 'analysis': 'Core not integrated or MetaOrchestrator not ready.'})

    # ---------------------------------------------------------------------------- #
    # Agent Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/agents')
    def get_agents():
        """
        Returns a list of available agents from the configuration.
        """
        if app.config['CORE_INTEGRATION']:
            from core.utils.config_utils import load_app_config
            core_config = load_app_config()
            agents = core_config.get('agents', {})
            agent_names = list(agents.keys())
            return jsonify(agent_names)
        else:
            return jsonify([])

    @app.route('/api/agents/<agent_name>/invoke', methods=['POST'])
    @jwt_required()
    def invoke_agent(agent_name):
        """
        Invokes a specific agent with the given arguments.
        """
        data = request.get_json()
        if agent_orchestrator:
            result = agent_orchestrator.execute_agent(agent_name, data)
            return jsonify(result)
        return jsonify({'error': 'Agent Orchestrator not initialized'}), 503

    @app.route('/api/agents/<agent_name>/schema')
    def get_agent_schema(agent_name):
        """
        Returns the input schema for a specified agent.
        """
        if app.config['CORE_INTEGRATION']:
            from core.utils.config_utils import load_app_config
            core_config = load_app_config()
            agents = core_config.get('agents', {})
            agent_config = agents.get(agent_name)
            if agent_config and 'input_schema' in agent_config:
                return jsonify(agent_config['input_schema'])
            else:
                return jsonify({})  # Return empty schema if not found
        else:
            return jsonify({})

    # ---------------------------------------------------------------------------- #
    # Auth Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/register', methods=['POST'])
    def register():
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        # 🛡️ Sentinel: Validate username format
        if not _validate_username(username):
            return jsonify({
                'error': 'Invalid username format. Must be 4-30 characters, alphanumeric, underscores, or hyphens.'
            }), 400

        # 🛡️ Sentinel: Enforce strong password policy
        if not _is_password_strong(password):
            return jsonify({
                'error': 'Password is too weak. It must be at least 8 characters and contain '
                         'uppercase, lowercase, number, and special character.'
            }), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'User already exists'}), 400

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'User created successfully'}), 201

    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        jti = jwt_payload["jti"]
        token = db.session.query(TokenBlocklist.id).filter_by(jti=jti).scalar()
        return token is not None

    @app.route('/api/login', methods=['POST'])
    def login():
        """
        Login endpoint.
        """
        # 🛡️ Sentinel: Rate limiting check
        ip_address = request.remote_addr
        cutoff_time = datetime.utcnow() - timedelta(minutes=BLOCK_DURATION_MINUTES)

        # Count failed attempts in the last window
        failed_attempts = LoginAttempt.query.filter(
            LoginAttempt.ip_address == ip_address,
            LoginAttempt.successful == False,
            LoginAttempt.timestamp > cutoff_time
        ).count()

        if failed_attempts >= IP_BLOCK_THRESHOLD:
            # Check if there is a recent successful login to reset counter?
            # Standard practice is strict block.
            return jsonify({'error': 'Too many failed login attempts. Please try again later.'}), 429

        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        user = User.query.filter_by(username=username).first()

        # 🛡️ Sentinel: Mitigate Timing Attacks (User Enumeration)
        # We must perform a hash check even if the user is not found to ensure
        # that the response time is consistent regardless of user existence.
        if user:
            authorized = user.check_password(password)
        else:
            # Perform a dummy check against a constant hash
            check_password_hash(DUMMY_PASSWORD_HASH, password if password else '')
            authorized = False

        if authorized:
            # Log successful attempt
            db.session.add(LoginAttempt(ip_address=ip_address, successful=True))
            db.session.commit()

            access_token = create_access_token(identity=str(user.id))
            refresh_token = create_refresh_token(identity=str(user.id))
            return jsonify(access_token=access_token, refresh_token=refresh_token)
        else:
            # Log failed attempt
            db.session.add(LoginAttempt(ip_address=ip_address, successful=False))
            db.session.commit()
            return jsonify({'error': 'Invalid credentials'}), 401

    @app.route('/api/logout', methods=['POST'])
    @jwt_required()
    def logout():
        jti = get_jwt()['jti']
        now = datetime.now(timezone.utc)
        db.session.add(TokenBlocklist(jti=jti, created_at=now))
        db.session.commit()
        return jsonify(msg="JWT revoked")

    @app.route('/api/refresh', methods=['POST'])
    @jwt_required(refresh=True)
    def refresh():
        identity = get_jwt_identity()
        access_token = create_access_token(identity=identity)
        return jsonify(access_token=access_token)

    # ---------------------------------------------------------------------------- #
    # Data Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/data/<filename>')
    def get_data(filename):
        """
        Returns the contents of a JSON file from the data directory.
        """
        allowed_files = ['company_data.json', 'knowledge_graph.json']
        if filename not in allowed_files:
            return jsonify({'error': 'File not found'}), 404

        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return jsonify(data)
        except FileNotFoundError:
            return jsonify({'error': 'File not found'}), 404
        except Exception as e:
            app.logger.error(f"Error serving data file {filename}: {e}", exc_info=True)
            return jsonify({'error': 'An internal error occurred.'}), 500

    # ---------------------------------------------------------------------------- #
    # WebSocket Endpoints
    # ---------------------------------------------------------------------------- #

    @socketio.on('connect')
    def test_connect():
        emit('my response', {'data': 'Connected'})

    @socketio.on('test event')
    def handle_test_event(json):
        print('received json: ' + str(json))
        emit('test response', json)

    # ---------------------------------------------------------------------------- #
    # Celery Tasks
    # ---------------------------------------------------------------------------- #

    @celery.task(bind=True)
    def run_simulation_task(self, simulation_name, user_id):
        """
        Celery task to run a simulation.
        """
        import importlib
        try:
            module = importlib.import_module(f"core.simulations.{simulation_name}")
            simulation_class = getattr(module, simulation_name)
            simulation = simulation_class()
            result = simulation.run()

            # Store the result in the database
            sim_result = SimulationResult(
                task_id=self.request.id,
                simulation_name=simulation_name,
                result=json.dumps(result),
                user_id=user_id
            )
            db.session.add(sim_result)
            db.session.commit()

            socketio.emit('simulation_complete', {'task_id': self.request.id, 'simulation_name': simulation_name})
            return {'status': 'success', 'message': f'Simulation {simulation_name} completed.'}
        except (ImportError, AttributeError):
            return {'status': 'failure', 'message': f'Simulation {simulation_name} not found.'}

    # ---------------------------------------------------------------------------- #
    # Simulation Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/simulations', methods=['GET'])
    @jwt_required()
    def get_simulations():
        """
        Returns a list of available simulations.
        """
        return jsonify(_get_allowed_simulations())

    @app.route('/api/simulations/history', methods=['GET'])
    @jwt_required()
    def get_simulation_history():
        """
        Returns the history of simulations for the current user.
        """
        current_user_id = get_jwt_identity()
        results = SimulationResult.query.filter_by(user_id=current_user_id).order_by(SimulationResult.id.desc()).all()
        return jsonify([{
            'task_id': r.task_id,
            'simulation_name': r.simulation_name,
            'status': run_simulation_task.AsyncResult(r.task_id).state
        } for r in results])

    @app.route('/api/simulations/<simulation_name>', methods=['POST'])
    @jwt_required()
    def run_simulation(simulation_name):
        """
        Runs a specific simulation.
        """
        current_user_id = get_jwt_identity()

        # 🛡️ Sentinel: Validate simulation name format
        if not re.match(r'^[a-zA-Z0-9_]+$', simulation_name):
            return jsonify({'error': 'Invalid simulation name format'}), 400

        # 🛡️ Sentinel: Validate simulation name against allowlist
        if simulation_name not in _get_allowed_simulations():
            return jsonify({'error': 'Invalid simulation name'}), 400

        task = run_simulation_task.delay(simulation_name, current_user_id)
        return jsonify({'task_id': task.id})

    # ---------------------------------------------------------------------------- #
    # Knowledge Graph Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/knowledge_graph')
    def get_knowledge_graph():
        """
        Returns the knowledge graph data from Neo4j.
        Accepts a 'query' parameter to search for a starting node.
        """
        driver = get_neo4j_driver()
        if not driver:
            return jsonify({'error': 'Graph database connection unavailable'}), 503

        query = request.args.get('query')
        cypher_query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25"
        params = {}

        if query:
            cypher_query = "MATCH (n)-[r]->(m) WHERE n.name CONTAINS $query RETURN n, r, m LIMIT 100"
            params = {'query': query}

        try:
            with driver.session() as session:
                result = session.run(cypher_query, params)
                nodes = {}
                links = []
                for record in result:
                    source_node = record["n"]
                    target_node = record["m"]

                    source_id = source_node.id
                    target_id = target_node.id

                    if source_id not in nodes:
                        nodes[source_id] = {"id": source_node.get('name'), "labels": list(
                            source_node.labels), "properties": dict(source_node)}
                    if target_id not in nodes:
                        nodes[target_id] = {"id": target_node.get('name'), "labels": list(
                            target_node.labels), "properties": dict(target_node)}

                    links.append({
                        "source": source_node.get('name'),
                        "target": target_node.get('name'),
                        "type": type(record["r"]).__name__,
                        "properties": dict(record["r"])
                    })

                return jsonify({"nodes": list(nodes.values()), "links": links})
        except Exception as e:
            app.logger.error(f"Error querying Neo4j: {e}")
            return jsonify({'error': 'Error querying knowledge graph'}), 500

    # ---------------------------------------------------------------------------- #
    # Task Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/tasks/<task_id>')
    def get_task_status(task_id):
        """
        Returns the status of a Celery task.
        """
        task = run_simulation_task.AsyncResult(task_id)
        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'status': 'Pending...'
            }
        elif task.state == 'SUCCESS':
            sim_result = SimulationResult.query.filter_by(task_id=task_id).first()
            if sim_result:
                response = {
                    'state': task.state,
                    'status': 'Completed',
                    'result': json.loads(sim_result.result)
                }
            else:
                response = {
                    'state': 'FAILURE',
                    'status': 'Result not found in database.'
                }
        elif task.state != 'FAILURE':
            response = {
                'state': task.state,
                'status': task.info.get('status', ''),
                'result': task.info.get('result', '')
            }
        else:
            # something went wrong in the background
            response = {
                'state': task.state,
                'status': str(task.info),  # this is the exception raised
            }
        return jsonify(response)

    # ---------------------------------------------------------------------------- #
    # Portfolio Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/portfolios', methods=['POST'])
    @jwt_required()
    def create_portfolio():
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'Missing name in request body'}), 400

        # 🛡️ Sentinel: Validate portfolio name
        if not _validate_portfolio_name(data['name']):
             return jsonify({'error': 'Invalid portfolio name. Must be 1-120 characters and contain no dangerous characters.'}), 400

        current_user_id = get_jwt_identity()
        new_portfolio = Portfolio(name=data['name'], user_id=current_user_id)
        db.session.add(new_portfolio)
        db.session.commit()
        return jsonify({'id': new_portfolio.id, 'name': new_portfolio.name})

    @app.route('/api/portfolios', methods=['GET'])
    @jwt_required()
    def get_portfolios():
        current_user_id = get_jwt_identity()
        portfolios = Portfolio.query.filter_by(user_id=current_user_id).all()
        return jsonify([{'id': p.id, 'name': p.name} for p in portfolios])

    @app.route('/api/portfolios/<int:id>', methods=['GET'])
    @jwt_required()
    def get_portfolio(id):
        current_user_id = get_jwt_identity()
        portfolio = Portfolio.query.filter_by(id=id, user_id=current_user_id).first_or_404()
        assets = [{'id': asset.id, 'symbol': asset.symbol, 'quantity': asset.quantity,
                   'purchase_price': asset.purchase_price} for asset in portfolio.assets]
        return jsonify({'id': portfolio.id, 'name': portfolio.name, 'assets': assets})

    @app.route('/api/portfolios/<int:id>', methods=['PUT'])
    @jwt_required()
    def update_portfolio(id):
        data = request.get_json()
        current_user_id = get_jwt_identity()
        portfolio = Portfolio.query.filter_by(id=id, user_id=current_user_id).first_or_404()

        # 🛡️ Sentinel: Validate portfolio name
        if 'name' in data and not _validate_portfolio_name(data['name']):
             return jsonify({'error': 'Invalid portfolio name.'}), 400

        portfolio.name = data['name']
        db.session.commit()
        return jsonify({'id': portfolio.id, 'name': portfolio.name})

    @app.route('/api/portfolios/<int:id>', methods=['DELETE'])
    @jwt_required()
    def delete_portfolio(id):
        current_user_id = get_jwt_identity()
        portfolio = Portfolio.query.filter_by(id=id, user_id=current_user_id).first_or_404()
        db.session.delete(portfolio)
        db.session.commit()
        return jsonify({'result': True})

    @app.route('/api/portfolios/<int:portfolio_id>/assets', methods=['POST'])
    @jwt_required()
    def add_portfolio_asset(portfolio_id):
        data = request.get_json()
        current_user_id = get_jwt_identity()
        portfolio = Portfolio.query.filter_by(id=portfolio_id, user_id=current_user_id).first_or_404()

        # 🛡️ Sentinel: Validate input
        if not _validate_asset_symbol(data.get('symbol')):
             return jsonify({'error': 'Invalid symbol.'}), 400

        try:
            quantity = float(data.get('quantity', 0))
            price = float(data.get('purchase_price', 0))
        except (ValueError, TypeError):
             return jsonify({'error': 'Invalid quantity or price.'}), 400

        if quantity <= 0:
             return jsonify({'error': 'Quantity must be positive.'}), 400
        if price < 0:
             return jsonify({'error': 'Price cannot be negative.'}), 400

        new_asset = PortfolioAsset(
            portfolio_id=portfolio.id,
            symbol=data['symbol'],
            quantity=quantity,
            purchase_price=price
        )
        db.session.add(new_asset)
        db.session.commit()
        return jsonify({'id': new_asset.id, 'symbol': new_asset.symbol, 'quantity': new_asset.quantity, 'purchase_price': new_asset.purchase_price})

    @app.route('/api/portfolios/<int:portfolio_id>/assets/<int:asset_id>', methods=['PUT'])
    @jwt_required()
    def update_portfolio_asset(portfolio_id, asset_id):
        data = request.get_json()
        current_user_id = get_jwt_identity()
        portfolio = Portfolio.query.filter_by(id=portfolio_id, user_id=current_user_id).first_or_404()
        asset = PortfolioAsset.query.filter_by(id=asset_id, portfolio_id=portfolio.id).first_or_404()

        # 🛡️ Sentinel: Validate input
        if 'symbol' in data:
            if not _validate_asset_symbol(data['symbol']):
                return jsonify({'error': 'Invalid symbol.'}), 400
            asset.symbol = data['symbol']

        if 'quantity' in data:
            try:
                quantity = float(data['quantity'])
                if quantity <= 0:
                     return jsonify({'error': 'Quantity must be positive.'}), 400
                asset.quantity = quantity
            except (ValueError, TypeError):
                 return jsonify({'error': 'Invalid quantity.'}), 400

        if 'purchase_price' in data:
            try:
                price = float(data['purchase_price'])
                if price < 0:
                     return jsonify({'error': 'Price cannot be negative.'}), 400
                asset.purchase_price = price
            except (ValueError, TypeError):
                 return jsonify({'error': 'Invalid price.'}), 400

        db.session.commit()
        return jsonify({'id': asset.id, 'symbol': asset.symbol, 'quantity': asset.quantity, 'purchase_price': asset.purchase_price})

    @app.route('/api/portfolios/<int:portfolio_id>/assets/<int:asset_id>', methods=['DELETE'])
    @jwt_required()
    def delete_portfolio_asset(portfolio_id, asset_id):
        current_user_id = get_jwt_identity()
        portfolio = Portfolio.query.filter_by(id=portfolio_id, user_id=current_user_id).first_or_404()
        asset = PortfolioAsset.query.filter_by(id=asset_id, portfolio_id=portfolio.id).first_or_404()
        db.session.delete(asset)
        db.session.commit()
        return jsonify({'result': True})

    # ---------------------------------------------------------------------------- #
    # Error Handlers
    # ---------------------------------------------------------------------------- #

    @app.errorhandler(Exception)
    def handle_exception(e):
        # pass through HTTP errors
        if isinstance(e, HTTPException):
            return e

        # 🛡️ Sentinel: Log the full error but return a generic message to prevent leaking sensitive info
        app.logger.error(f"Unhandled Exception: {e}", exc_info=True)

        # Return generic error message
        return jsonify(error="An unexpected error occurred."), 500

    return app


if __name__ == '__main__':
    app = create_app(os.getenv('FLASK_CONFIG') or 'default')
    with app.app_context():
        db.create_all()
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    socketio.run(app, debug=debug_mode, port=5001)
