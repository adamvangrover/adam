# Adam Web Application

This is the web application for the Adam project. It provides a user-friendly interface for interacting with the Adam AI system.

## Running the Application

The easiest way to run the application is with Docker Compose.

1.  **Create a `.env` file:**

    Create a `.env` file in the root of the project with the following content:

    ```
    POSTGRES_USER=user
    POSTGRES_PASSWORD=password
    POSTGRES_DB=adam
    ```

2.  **Run Docker Compose:**

    From the root of the project, run the following command:

    ```
    docker-compose up --build
    ```

    This will build the Docker images and start all the services. The web application will be available at `http://localhost:80`.

## Architecture

The web application consists of the following components:

*   **Flask Backend:** A Python web server built with Flask. It provides a REST API for the frontend to interact with the Adam AI system.
*   **React Frontend:** A single-page application built with React. It provides the user interface for the web application.
*   **PostgreSQL Database:** A relational database for storing user data, portfolios, and analysis results.
*   **Redis:** An in-memory data store used as a message broker for Celery.
*   **Celery:** A task queue for running long-running asynchronous tasks, such as simulations.
*   **Neo4j:** A graph database for storing and querying the knowledge graph.
*   **Nginx:** A web server that serves the React frontend and acts as a reverse proxy for the Flask backend.

## API Endpoints

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
