# Adam Web Application

This is the web application for the Adam project. It provides a user-friendly interface for interacting with the Adam AI system.

## 🚀 Quick Start (Docker)

The easiest way to run the full stack is with Docker Compose.

1.  **Configure Environment:**
    ```bash
    cp ../../.env.example .env
    # Ensure OPENAI_API_KEY is set
    ```

2.  **Run Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    The app will be available at `http://localhost:80`.

---

## 💻 Local Development (No Docker)

For faster development cycles (Hot Reloading), run the Frontend and Backend separately.

### 1. Backend (Flask)
Start the API server on port 5000.

```bash
# From the repository root
source .venv/bin/activate
export FLASK_APP=app.py
export FLASK_ENV=development
python app.py
```

### 2. Frontend (React)
Start the React development server on port 3000.

```bash
cd services/webapp/client

# Install dependencies
pnpm install  # or npm install

# Start the dev server
pnpm start    # or npm start
```

**Note:** The React app is configured to proxy API requests to `http://localhost:5000` via `package.json` proxy settings (or CORS setup).

---

## 🏗️ Architecture

The web application consists of the following components:

*   **Flask Backend:** A Python web server built with Flask. It provides a REST API for the frontend to interact with the Adam AI system.
*   **React Frontend:** A single-page application built with React. It provides the user interface for the web application.
*   **PostgreSQL Database:** A relational database for storing user data, portfolios, and analysis results.
*   **Redis:** An in-memory data store used as a message broker for Celery.
*   **Celery:** A task queue for running long-running asynchronous tasks, such as simulations.
*   **Neo4j:** A graph database for storing and querying the knowledge graph.
*   **Nginx:** A web server that serves the React frontend and acts as a reverse proxy for the Flask backend.

## 🔌 API Endpoints

The Flask backend provides the following API endpoints:

*   `GET /api/hello`: A simple test endpoint.
*   `GET /api/agents`: Returns a list of all available agents.
*   `GET /api/agents/<agent_name>/schema`: Returns the input schema for a specified agent.
*   `POST /api/agents/<agent_name>/invoke`: Invokes a specific agent with the given arguments.
*   `POST /api/login`: A mock login endpoint.
*   `GET /api/data/<filename>`: Returns the contents of a JSON file from the `data` directory.
*   `POST /api/simulations/<simulation_name>`: Starts a simulation.
*   `GET /api/tasks/<task_id>`: Returns the status of a Celery task.
*   `GET /api/knowledge_graph`: Returns the knowledge graph data from Neo4j.

## 📡 Agent Intercom (Live Streams)

The frontend visualizes the "Agent's Mind" via the **Intercom System**.

### Architecture
1.  **Backend:** Agents emit "Thought" events (e.g., `fetching_data`, `reasoning`, `tool_call`) to a Redis Pub/Sub channel.
2.  **API:** The Flask server exposes a Server-Sent Events (SSE) endpoint at `/api/intercom/stream`.
3.  **Frontend:** `AgentIntercom.tsx` subscribes to this stream and renders thoughts as a scrolling terminal log.

### Event Format
```json
{
  "agent": "RiskAnalyst",
  "action": "THOUGHT",
  "content": "Calculating Debt/EBITDA ratio...",
  "timestamp": "2023-10-27T10:00:00Z"
}
```
