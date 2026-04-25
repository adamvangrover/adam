# Adam Architecture Extensions & Integrations

This file scaffolds the connection points and environment runtimes needed to interface Adam with third-party systems, commercial software, and custom AI microservices.

## 1. Microservices & API Endpoints
Adam's core API is distributed across:
- `backend/api.py`: FastAPI server for core operations.
- `server/server.py`: Model Context Protocol (MCP) Server for tool access.
- `core/api/routers/`: Specific domain endpoints (agents, risk, etc).

**To Add a New Microservice API:**
```python
# scaffold inside core/api/routers/new_service.py
from fastapi import APIRouter
router = APIRouter()

@router.post("/plugin/execute")
async def execute_plugin(payload: dict):
    # Connect to custom commercial software here
    return {"status": "placeholder"}
```

## 2. App Plugins
Plugins are integrated via the `MCP` framework. Add tools to `server/server.py` and register them in `mcp.json`.

**Example Prompt for Advanced AI Generation:**
> "Generate a Python MCP Tool for Adam v26 that interfaces with the Bloomberg Terminal API via bbglink. Ensure the input parameters follow the AgentInput schema."

## 3. Environment Runtimes
Adam uses Docker for isolated runtimes.
- `Dockerfile.core`: Standard Execution
- `Dockerfile.swarm`: Multi-Agent Asynchronous (Requires high concurrency setup)
- `Dockerfile.rust`: Rust Pricing Engine compilation

**To connect a new kernel (e.g., Jupyter, proprietary runtime):**
Ensure your kernel supports passing standard primitive types to Python, or interact over the FastAPI HTTP layer.

## 4. Third-Party Vendor Initialization
*   For data providers (FactSet, CapitalIQ): Implement an interface in `core/data_access/` and inject it into the `LakehouseConnector` or specific domain agent.
*   For custom LLMs: Register the provider in `core/llm_plugin.py` adapting to the `BaseLLM` standard.
