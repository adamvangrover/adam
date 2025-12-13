# Async Coding Agents: Development Guide (v23.5)

This guide provides instructions for interacting with and developing using the **Autonomous Async Coding Agents**, primarily the **Code Alchemist**. These agents are designed to operate within the Adam v23.5 architecture, leveraging asynchronous workflows and graph-based reasoning.

## **1. Overview**

The **Code Alchemist** (`core/agents/code_alchemist.py`) is the primary development agent. It is capable of:
*   **Code Generation:** Creating high-quality, typed, and documented Python code.
*   **Validation:** Checking syntax and performing static analysis (via LLM).
*   **Optimization:** Applying performance strategies (e.g., vectorization, caching).
*   **Deployment:** Saving code to files or pushing to API endpoints.

It operates asynchronously, making it suitable for high-throughput pipelines and "DevOps" loops where multiple agents (e.g., Red Team, Test Runner) interact.

## **2. Environment Configuration**

To fully utilize the coding agents, ensure your environment is configured with the following variables. These are loaded via `core/settings.py`.

### **Core AI & API Keys**
Required for the agent's intelligence engine.

| Key | Description |
| :--- | :--- |
| `OPENAI_API_KEY` | **Required.** The primary brain of the Code Alchemist. |
| `ANTHROPIC_API_KEY`| Optional. Fallback for complex reasoning tasks. |
| `COHERE_API_KEY` | Optional. Used for embedding generation if needed. |
| `TINKER_API_KEY` | Required for accessing the Tinker Lab experiments. |

### **Infrastructure**
The agents may need to connect to these services to deploy code or verify database interactions.

| Key | Default Value (Local) | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/adam` | PostgreSQL connection string. |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for caching and async task queues. |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Knowledge Graph URI. |
| `NEO4J_USER` | `neo4j` | Neo4j username. |

*Note: In the Tinker Lab environment, refer to `.env.example` for the latest defaults.*

## **3. Interacting with the Code Alchemist**

The `CodeAlchemist` class inherits from `AgentBase` and uses an `async def execute` method.

### **Initialization**

```python
from core.agents.code_alchemist import CodeAlchemist
from core.settings import settings

# Initialize with system settings
config = {
    "optimization_strategies": ["vectorization", "caching"],
    "validation_tool_url": "http://localhost:8000/validate", # Optional external tool
    # API keys are loaded automatically from settings if not passed here
}

agent = CodeAlchemist(config)
```

### **Task: Generate Code**

```python
import asyncio

async def main():
    intent = "Create a Pydantic model for a 'CorporateBond' with fields for issuer, coupon, and maturity."
    context = {"library": "pydantic"}
    constraints = {"validation": "strict"}

    code = await agent.execute(
        action="generate_code",
        intent=intent,
        context=context,
        constraints=constraints
    )
    print(code)

if __name__ == "__main__":
    asyncio.run(main())
```

### **Task: Validate & Optimize**

```python
code_snippet = """
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)
"""

# Validate
validation = await agent.execute(action="validate_code", code=code_snippet)
print("Validation:", validation)

# Optimize
optimized = await agent.execute(
    action="optimize_code",
    code=code_snippet,
    optimization_strategies=["memoization"]
)
print("Optimized Code:\n", optimized)
```

## **4. Prompting Strategy**

The Code Alchemist uses the **`LIB-META-008`** system prompt (located in `prompt_library/AOPL-v1.0/system_architecture/`). This prompt defines its persona as a "Senior Principal Software Engineer."

When interacting with the agent, be specific about:
1.  **Architecture:** Mention if the code needs to fit into v21 (sync), v22 (async), or v23 (graph) architectures.
2.  **Dependencies:** List available libraries (e.g., "Use `pandas` and `networkx`").
3.  **Error Handling:** Request specific error handling strategies (e.g., "Log errors to `core.utils.logging_utils`").

## **5. Best Practices**

*   **Async All The Way:** The agent is designed to run in an `asyncio` loop. Do not call its methods synchronously.
*   **Review Generated Code:** While the agent validates syntax, human review is still recommended for logic and business requirements.
*   **Sandboxing:** For automated execution of generated code, ensure you are running in a containerized or sandboxed environment (e.g., Docker) to prevent accidental system modification.
*   **Secrets:** Never ask the agent to hardcode secrets. Direct it to use `os.getenv` or `core.settings`.

---
**Author:** Adam v23.5 System Architect
