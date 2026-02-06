# Adam v26.0 API Reference

The Adam API allows external applications to interact with the cognitive engine. It is built on **Flask** and exposes RESTful endpoints.

## 🔑 Authentication

Most endpoints require an API Key (if configured in `.env`).
```http
Authorization: Bearer <YOUR_API_KEY>
```

## 📡 Base URL
`http://localhost:5000` (Default)

## Endpoints

### 1. System State
Get the current health and status of the engine.

*   **GET** `/api/state`
*   **Response:**
    ```json
    {
        "status": "online",
        "version": "v26.0",
        "active_agents": 5,
        "system_health": "nominal"
    }
    ```

### 2. Chat / Query
Send a natural language query to the Meta Orchestrator.

*   **POST** `/api/chat`
*   **Body:**
    ```json
    {
        "message": "Analyze the credit risk of AAPL.",
        "user_id": "user_123"
    }
    ```
*   **Response:**
    ```json
    {
        "response": "Based on the 2023 10-K...",
        "thought_process": ["Fetching data...", "Calculating ratios..."]
    }
    ```

### 3. Crisis Simulation
Trigger a macro-economic scenario.

*   **POST** `/api/simulation/start`
*   **Body:**
    ```json
    {
        "scenario_id": "SCENARIO_JAN30_LIQUIDITY_SHOCK",
        "parameters": {
            "vix_spike": 45,
            "repo_haircut": 0.15
        }
    }
    ```

### 4. Agent Economy
Interact with the internal agent marketplace.

*   **GET** `/api/economy/state`
*   **POST** `/api/economy/step` (Advance simulation by one tick)

## ⚠️ Rate Limiting
The API enforces a limit of 100 requests per minute per IP.

## 📄 OpenAPI Spec
A full OpenAPI (Swagger) specification is available at `/api/docs` when running the server in debug mode.
