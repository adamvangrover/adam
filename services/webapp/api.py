from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import create_access_token, jwt_required, JWTManager, get_jwt_identity
from werkzeug.exceptions import HTTPException
from werkzeug.security import generate_password_hash, check_password_hash
import sys
import os
import json
from .config import config
from .celery import celery

# ---------------------------------------------------------------------------- #
# Initialize Extensions
# ---------------------------------------------------------------------------- #

db = SQLAlchemy()
socketio = SocketIO()
jwt = JWTManager()
agent_orchestrator = None


# ---------------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------------- #

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class SimulationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(36), unique=True, nullable=False)
    simulation_name = db.Column(db.String(120), nullable=False)
    result = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<SimulationResult {self.simulation_name}>'

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('portfolios', lazy=True))

    def __repr__(self):
        return f'<Portfolio {self.name}>'

# ---------------------------------------------------------------------------- #
# Application Factory
# ---------------------------------------------------------------------------- #

def create_app(config_name='default'):
    """
    Application factory.
    """
    global agent_orchestrator
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Initialize extensions
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")
    jwt.init_app(app)

    # Configure Celery
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask


    # Initialize the core components
    if app.config['CORE_INTEGRATION']:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from core.utils.config_utils import load_app_config
        from core.system.agent_orchestrator import AgentOrchestrator
        from core.system.knowledge_base import KnowledgeBase
        from core.system.data_manager import DataManager
        core_config = load_app_config()
        knowledge_base = KnowledgeBase(core_config)
        data_manager = DataManager(core_config)
        agent_orchestrator = AgentOrchestrator(core_config, knowledge_base, data_manager)

    # ---------------------------------------------------------------------------- #
    # API Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/hello')
    def hello_world():
        return 'Hello, World!'

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
    def invoke_agent(agent_name):
        """
        Invokes a specific agent with the given arguments.
        """
        data = request.get_json()
        result = agent_orchestrator.run_agent(agent_name, data)
        return jsonify(result)

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
                return jsonify({}) # Return empty schema if not found
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
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'User already exists'}), 400

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'User created successfully'}), 201


    @app.route('/api/login', methods=['POST'])
    def login():
        """
        Login endpoint.
        """
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            access_token = create_access_token(identity=username)
            return jsonify(access_token=access_token)
        else:
            return jsonify({'error': 'Invalid credentials'}), 401

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
            return jsonify({'error': str(e)}), 500

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
    def run_simulation_task(self, simulation_name):
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
                result=json.dumps(result)
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

    @app.route('/api/simulations/<simulation_name>', methods=['POST'])
    def run_simulation(simulation_name):
        """
        Runs a specific simulation.
        """
        task = run_simulation_task.delay(simulation_name)
        return jsonify({'task_id': task.id})

    # ---------------------------------------------------------------------------- #
    # Knowledge Graph Endpoints
    # ---------------------------------------------------------------------------- #

    @app.route('/api/knowledge_graph')
    def get_knowledge_graph():
        """
        Returns the knowledge graph data from Neo4j.
        """
        from neo4j import GraphDatabase
        uri = os.environ.get('NEO4J_URI', 'bolt://neo4j:7687')
        user = os.environ.get('NEO4J_USER', 'neo4j')
        password = os.environ.get('NEO4J_PASSWORD', 'password')
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")
            nodes = {}
            links = []
            for record in result:
                source_name = record["n"]["name"]
                target_name = record["m"]["name"]
                if source_name not in nodes:
                    nodes[source_name] = {"id": source_name, "labels": list(record["n"].labels)}
                if target_name not in nodes:
                    nodes[target_name] = {"id": target_name, "labels": list(record["m"].labels)}

                links.append({
                    "source": source_name,
                    "target": target_name,
                    "type": type(record["r"]).__name__
                })

            return jsonify({"nodes": list(nodes.values()), "links": links})

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
        current_user = get_jwt_identity()
        user = User.query.filter_by(username=current_user).first()
        new_portfolio = Portfolio(name=data['name'], user_id=user.id)
        db.session.add(new_portfolio)
        db.session.commit()
        return jsonify({'id': new_portfolio.id, 'name': new_portfolio.name})

    @app.route('/api/portfolios', methods=['GET'])
    @jwt_required()
    def get_portfolios():
        current_user = get_jwt_identity()
        user = User.query.filter_by(username=current_user).first()
        portfolios = Portfolio.query.filter_by(user_id=user.id).all()
        return jsonify([{'id': p.id, 'name': p.name} for p in portfolios])

    @app.route('/api/portfolios/<int:id>', methods=['GET'])
    @jwt_required()
    def get_portfolio(id):
        current_user = get_jwt_identity()
        user = User.query.filter_by(username=current_user).first()
        portfolio = Portfolio.query.filter_by(id=id, user_id=user.id).first_or_404()
        return jsonify({'id': portfolio.id, 'name': portfolio.name})

    @app.route('/api/portfolios/<int:id>', methods=['PUT'])
    @jwt_required()
    def update_portfolio(id):
        data = request.get_json()
        current_user = get_jwt_identity()
        user = User.query.filter_by(username=current_user).first()
        portfolio = Portfolio.query.filter_by(id=id, user_id=user.id).first_or_404()
        portfolio.name = data['name']
        db.session.commit()
        return jsonify({'id': portfolio.id, 'name': portfolio.name})

    @app.route('/api/portfolios/<int:id>', methods=['DELETE'])
    @jwt_required()
    def delete_portfolio(id):
        current_user = get_jwt_identity()
        user = User.query.filter_by(username=current_user).first()
        portfolio = Portfolio.query.filter_by(id=id, user_id=user.id).first_or_404()
        db.session.delete(portfolio)
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
        # now you're handling non-HTTP exceptions only
        return jsonify(error=str(e)), 500

    return app

if __name__ == '__main__':
    app = create_app(os.getenv('FLASK_CONFIG') or 'default')
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, port=5001)
