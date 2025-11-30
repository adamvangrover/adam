````markdown
---
project: "Adam Web Application"
version: "1.0.0"
spec_type: "Product & Design Specification"
status: "Draft"
author: "ADAM"
last_updated: "2025-09-15"
---
````
# Adam Web Application: Product & Design Specification

This document provides a comprehensive, implementable specification for building the Adam Web Application. It is intended for both human developers and automated software agents.

***

## 1. Vision & High-Level Goals

### Vision
To transform the Adam repository from a collection of powerful command-line scripts and agents into a cohesive, user-friendly, and production-ready web application. The application will serve as an interactive platform for financial analysis, leveraging the full capabilities of the underlying AI agent system.

### High-Level Goals
* **Full Integration**: Incorporate the entire suite of agents, simulations, and data sources from the `core` repository into the web application.
* **Intuitive UI/UX**: Create a responsive, well-designed, and intuitive user interface that makes the complex capabilities of the Adam system accessible to financial analysts, researchers, and investors.
* **Robust & Scalable Architecture**: Build a future-proof application with a clear separation of concerns, robust error handling, and a scalable architecture that can accommodate future growth.
* **Production-Ready**: Ensure the application is secure, well-documented, and thoroughly tested, making it suitable for deployment.

***

## 2. System Architecture

The application will follow a modern, containerized, microservices-oriented architecture to ensure separation of concerns, scalability, and maintainability.

### 2.1. Component Overview

| Service               | Technology Stack               | Role & Responsibilities                                                                                             |
| --------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **Frontend** | React, Vite, Axios, Redux      | Single-Page Application (SPA) for all UI rendering, state management, and user interaction. Communicates with the Backend API. |
| **Backend (API)** | Flask, Flask-RESTx, SQLAlchemy | Central RESTful API. Handles business logic, authentication, and orchestrates communication between all other services. |
| **Core Logic** | Python (Existing `core` dir)   | The heart of the system. Contains all AI agents and simulation logic. Invoked by the Backend API via the Task Queue. |
| **Database** | PostgreSQL                     | Primary relational database for storing user data, portfolios, simulation results, API keys, and other structured data.   |
| **Task Queue** | Celery, Redis                  | Manages and executes long-running, asynchronous tasks (e.g., simulations, complex analyses) to keep the API responsive.     |
| **Knowledge Graph** | Neo4j                          | Stores and queries highly interconnected graph data used by agents. Provides insights into entity relationships.     |
| **Containerization** | Docker, Docker Compose         | Containerizes each service for consistent, isolated, and reproducible environments across development, testing, and production. |

### 2.2. Architecture Diagram (Mermaid)

```mermaid
graph TD
    subgraph "User's Browser"
        A[React Frontend SPA]
    end

    subgraph "Docker Environment"
        B[Backend API - Flask]
        C[Task Queue - Celery Workers]
        D[Database - PostgreSQL]
        E[Cache / Message Broker - Redis]
        F[Knowledge Graph - Neo4j]
        G[Core Logic - Python Agents/Sims]
    end

    A -- "HTTP/S (REST API)" --> B
    B -- "Adds Task" --> E
    B -- "Reads/Writes Data" --> D
    B -- "Queries Graph" --> F
    C -- "Listens for Tasks" --> E
    C -- "Executes" --> G
    G -- "Writes Results" --> D
````

-----

## 3\. Detailed Feature Specifications

### 3.1. User Authentication & Authorization

#### Models

  * **File Location**: `backend/models/user.py`
  * **`User` Model Schema**:
    ```python
    class User(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(128), nullable=False)
        role = db.Column(db.String(20), nullable=False, default='user') # 'user', 'analyst', 'admin'
        created_at = db.Column(db.DateTime, server_default=db.func.now())
        updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())
    ```

#### API Endpoints

  * **Blueprint**: `/api/auth`
  * **Endpoints**:
      * `POST /register`:
          * **Request Body**: `{ "username": "...", "email": "...", "password": "..." }`
          * **Response (201)**: `{ "message": "User created successfully" }`
      * `POST /login`:
          * **Request Body**: `{ "email": "...", "password": "..." }`
          * **Response (200)**: `{ "access_token": "...", "refresh_token": "..." }`
      * `POST /logout`:
          * **Headers**: `Authorization: Bearer <access_token>`
          * **Logic**: Revokes both access and refresh tokens by adding their JTI (JWT ID) to a blocklist in Redis.
          * **Response (200)**: `{ "message": "Successfully logged out" }`
      * `POST /refresh`:
          * **Headers**: `Authorization: Bearer <refresh_token>`
          * **Response (200)**: `{ "access_token": "..." }`

#### Developer Implementation Notes

  * **Backend**: Use `Flask-JWT-Extended` for managing JWTs. Use a Redis set for the token blocklist. Implement `@jwt_required` and custom role-based decorators (e.g., `@admin_required`) for endpoint protection.
  * **Frontend**: Store tokens in `localStorage`. Use an Axios interceptor to automatically attach the `Authorization` header to outgoing requests and to handle 401 Unauthorized errors by attempting to use the refresh token or redirecting to the login page.

-----

### 3.2. Portfolio Management (Full CRUD)

#### Models

  * **File Location**: `backend/models/portfolio.py`
  * **`Portfolio` Model Schema**:
    ```python
    class Portfolio(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(100), nullable=False)
        description = db.Column(db.Text, nullable=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
        user = db.relationship('User', backref=db.backref('portfolios', lazy=True))
        assets = db.relationship('PortfolioAsset', backref='portfolio', lazy='dynamic', cascade="all, delete-orphan")
    ```
  * **`PortfolioAsset` Model Schema**:
    ```python
    class PortfolioAsset(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
        asset_ticker = db.Column(db.String(20), nullable=False) # e.g., 'AAPL', 'BTC-USD'
        quantity = db.Column(db.Float, nullable=False)
        purchase_price = db.Column(db.Float, nullable=False)
        purchase_date = db.Column(db.Date, nullable=False)
    ```

#### API Endpoints

  * **Blueprint**: `/api/portfolios`
  * **Endpoints** (all require authentication):
      * `GET /`: List all portfolios for the authenticated user.
      * `POST /`: Create a new portfolio.
      * `GET /<int:portfolio_id>`: Get details for a single portfolio, including its assets.
      * `PUT /<int:portfolio_id>`: Update a portfolio's details (name, description).
      * `DELETE /<int:portfolio_id>`: Delete a portfolio and all its associated assets.
      * `POST /<int:portfolio_id>/assets`: Add a new asset to a portfolio.
      * `PUT /assets/<int:asset_id>`: Update an asset within a portfolio.
      * `DELETE /assets/<int:asset_id>`: Remove an asset from a portfolio.

#### Developer Implementation Notes

  * **Backend**: Use Flask-SQLAlchemy for database interactions. Ensure all queries are scoped to the `current_user`'s ID to prevent data leakage.
  * **Frontend**: Create a dedicated React context or Redux slice for managing portfolio state. Use a library like `react-router` for navigation between the portfolio list and detail views. For charts, consider using `Chart.js` or `Recharts`. Implement reusable form components for creating/editing portfolios and assets.

-----

### 3.3. Analysis Tools (Custom UIs)

#### Goal

Move beyond the generic `AgentRunner` to create a tailored experience for each key analysis agent.

#### Dynamic Form Generation

  * **Backend**: Create a new endpoint `GET /api/agents/schemas` that reads `config/agents.yaml` and returns a JSON representation of the `input_schema` for all available agents.
      * **Example `agents.yaml` entry**:
        ```yaml
        fundamental_analyst_agent:
          name: "Fundamental Analyst"
          description: "Performs fundamental analysis on a stock."
          input_schema:
            ticker: { type: "string", required: true, label: "Stock Ticker" }
            time_horizon: { type: "integer", required: false, label: "Time Horizon (Years)", default: 5 }
        ```
  * **Frontend**: On the analysis page, fetch the agent schemas. When a user selects an agent from a dropdown, use the corresponding schema to dynamically render a form with appropriate input fields (`<input type="text">`, `<input type="number">`, etc.) and labels.

#### Custom Result Visualization

  * **Backend**: The agent execution endpoint (`POST /api/agents/run/<agent_name>`) will return a structured JSON object. The structure should be predictable for each agent type.
      * **`fundamental_analyst_agent` Output**:
        ```json
        {
          "type": "fundamental_analysis",
          "data": {
            "ratios": { "P/E": 15.2, "P/B": 2.1, ... },
            "dcf": { "intrinsic_value": 150.75, "assumptions": "..." }
          }
        }
        ```
  * **Frontend**: Create specific React components for rendering each result `type`.
      * **`FundamentalAnalysisResult.js`**: Renders a table for `ratios` and a summary card for the `dcf` results.
      * **`TechnicalAnalysisResult.js`**: Uses a charting library (e.g., `TradingView Lightweight Charts`) to display price data and overlays for indicators like moving averages and RSI.
      * **`RiskAssessmentResult.js`**: Uses a library like `react-gauge-chart` to visualize risk scores.

-----

### 3.4. Real Simulation Integration

#### Backend

  * **Celery Task**: `tasks.py` will contain `run_simulation_task`.
    ```python
    from core.simulations import get_simulation_class

    @celery.task(bind=True)
    def run_simulation_task(self, simulation_name, params, user_id):
        SimulationClass = get_simulation_class(simulation_name)
        simulation = SimulationClass(**params)
        results = simulation.run()
        # Store results in the SimulationResult table linked to user_id
        # ...
        return {"status": "Complete", "result_id": new_result.id}
    ```
  * **API Endpoint**: `POST /api/simulations` will trigger the Celery task and immediately return a task ID.
      * **Response (202)**: `{ "task_id": "...", "status_url": "/api/tasks/..." }`
  * **Task Status Endpoint**: `GET /api/tasks/<task_id>` will check Celery's backend for the task's status (`PENDING`, `PROGRESS`, `SUCCESS`, `FAILURE`).

#### Frontend

  * **UI Flow**:
    1.  User fills out the simulation form and clicks "Run".
    2.  The UI makes a `POST` request to `/api/simulations`.
    3.  The UI receives the `task_id` and begins polling the `/api/tasks/<task_id>` endpoint every 2-3 seconds. A progress bar or spinner is displayed.
    4.  A WebSocket connection (using `Socket.IO`) is established. The backend will push a "simulation\_complete" event when the task finishes. This is more efficient than polling.
    5.  Upon receiving the WebSocket event or a "SUCCESS" status from polling, the UI notifies the user (e.g., with a toast notification) and provides a link to the results page: `/simulations/results/<result_id>`.
  * **Results Page**: Fetches the detailed simulation results and presents them using interactive charts and data tables.

-----

### 3.5. Knowledge Graph Visualization

#### Backend

  * **Endpoint**: `/api/knowledge_graph`
  * **Query Parameters**:
      * `q` (search term): Find nodes matching a specific name or property.
      * `node_id` (expand): Return a node and its immediate neighbors (1-hop relationships).
      * `depth` (integer): Used with `node_id` to specify the number of hops to expand (e.g., `depth=2`).
  * **Logic**: The Flask endpoint will connect to the Neo4j instance, execute the appropriate Cypher query, and format the results into a standard graph JSON format (nodes and links).
      * **Nodes**: `[ { "id": "AAPL", "label": "Company", "properties": { ... } } ]`
      * **Links**: `[ { "source": "AAPL", "target": "Tim Cook", "type": "CEO_OF" } ]`

#### Frontend

  * **Component**: `KnowledgeGraph.js`
  * **Library**: Use `react-force-graph`, `d3-force`, or a similar library for rendering the graph.
  * **Features**:
      * **Interactivity**: When a user clicks a node, an API call is made to expand from that node (`/api/knowledge_graph?node_id=...`). New nodes and links are added to the graph visualization.
      * **Information Panel**: A side panel will display the `properties` of the currently selected node or link.
      * **Search**: A search bar will hit the `q` parameter on the API to find and center the graph on a specific entity.
      * **Controls**: Add checkboxes or dropdowns to filter the view by node `label` or link `type`.

-----

## 4\. Non-Functional Requirements

  * **Security**:
      * Use `HTTPS` in production.
      * Implement CORS policies on the Flask backend.
      * Use `Bcrypt` for password hashing.
      * Sanitize all user inputs to prevent XSS and SQL Injection (SQLAlchemy helps with the latter).
      * Perform regular dependency scans with tools like `pip-audit` and `npm audit`.
  * **Performance**:
      * Implement pagination for all list-based API endpoints (portfolios, simulations).
      * Use database indexes on foreign keys and frequently queried columns (e.g., `user.email`).
      * Optimize frontend asset delivery using code splitting and lazy loading in React.
  * **Scalability**:
      * The architecture allows for horizontal scaling. Use `docker-compose up --scale celery-worker=4` to increase the number of Celery workers based on load.
      * Use a production-grade WSGI server like `Gunicorn` or `uWSGI` behind an `Nginx` reverse proxy.
  * **Maintainability**:
      * Adhere to a consistent code style (`Black` for Python, `Prettier` for JS/React).
      * Use Flask Blueprints to organize the backend by feature (e.g., `auth`, `portfolios`).
      * Write comprehensive docstrings for Python functions and JSDoc for React components.
  * **Testability**:
      * **Backend**: Use `pytest`. Write unit tests for business logic and integration tests for API endpoints (mocking external services).
      * **Frontend**: Use `Jest` and `React Testing Library` for component unit tests. Use `Cypress` or `Playwright` for end-to-end tests that simulate user workflows.
      * **CI/CD**: Set up a GitHub Actions workflow to run all tests on every push/pull request.

-----

## 5\. Future-Proofing & Upgrades

  * **Dependency Management**:
      * Pin dependency versions in `requirements.txt` and `package.json` for reproducibility. Use a tool like Dependabot to automatically create pull requests for updates.
      * **`psycopg2-binary` Resolution**: The production `Dockerfile` for the backend service will be based on a Python image that includes build essentials.
        ```dockerfile
        FROM python:3.11-slim

        # Install build tools needed for packages like psycopg2
        RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential \
            libpq-dev \
            && rm -rf /var/lib/apt/lists/*

        # ... rest of Dockerfile
        ```
  * **Architectural Patterns**: The decoupled microservices architecture allows individual components to be updated, replaced, or scaled independently without affecting the rest of the system.
  * **Documentation**:
      * Maintain an up-to-date `README.md` with setup and deployment instructions.
      * Use a tool like Swagger or OpenAPI (integrated with Flask-RESTx) to automatically generate and host interactive API documentation.
      * Document architectural decisions in a designated `/docs` directory within the repository.

<!-- end list -->

```
```
