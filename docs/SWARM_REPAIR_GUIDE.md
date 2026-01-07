# Swarm Repair Guide

This guide is for the "Async Coding Swarm" (future agents or developers) to maintain and repair the Adam repository efficiently.

## 1. Environment Setup

**Crucial:** The repository structure requires the root directory to be in your Python path.
```bash
export PYTHONPATH=.
```
Always prepend this to your python commands:
```bash
PYTHONPATH=. python tests/verify_v23_full.py
```

## 2. Dependency Management
The system has heavy dependencies. If you encounter `ModuleNotFoundError`, check `requirements.txt` first, but be aware that the sandbox environment might need manual installation for verification scripts.

**Common Missing Packages:**
*   `langgraph`, `langchain`, `semantic-kernel` (Core Logic)
*   `pydantic`, `networkx` (Data Structures)
*   `tiktoken`, `transformers` (NLP)
*   `pandas`, `scikit-learn` (Data Science)

## 3. Logging & Telemetry
Do not use `print()`. Use the standardized logging infrastructure.

**Usage:**
```python
from core.utils.logging_utils import get_logger, SwarmLogger

logger = get_logger(__name__)
swarm = SwarmLogger()

logger.info("Starting task...")
swarm.log_event("TASK_START", "MyAgent", {"target": "Analysis"})
```

**Logs Location:**
*   `logs/system.log` (if configured)
*   `logs/swarm_telemetry.jsonl` (Structured events)

## 4. Known Error Patterns & Fixes

### A. `AgentBase` Instantiation Error
**Error:** `TypeError: Can't instantiate abstract class ... without an implementation for abstract method 'execute'`
**Cause:** A legacy agent inherits from `AgentBase` but hasn't been updated to the v23 interface.
**Fix:**
1.  Open the agent file.
2.  Implement `async def execute(self, *args, **kwargs):`.
3.  Ensure it calls `super().__init__(config)`.

### B. `ModuleNotFoundError: core`
**Cause:** Running a script from `tests/` or `scripts/` without `PYTHONPATH`.
**Fix:** See Section 1.

### C. Import Errors in `core/schemas`
**Cause:** `core/schemas/__init__.py` references files that haven't been created yet (scaffolding).
**Fix:** Keep them commented out until the files `cognitive_state.py`, `observability.py`, etc., are implemented.

## 5. Verification Protocols
Before submitting *any* change, run these verifiers:

1.  **Full System Check:**
    ```bash
    PYTHONPATH=. python tests/verify_v23_full.py
    ```
    *Must print "Execution Complete" and "Graph converged".*

2.  **Update Check:**
    ```bash
    PYTHONPATH=. python tests/verify_v23_updates.py
    ```
    *Must verify Reflector and Crisis agents.*

## 6. Async Communication
Use the `core.system.v22_async` package for inter-agent messaging.
*   **Broker:** `core.system.v22_async.message_broker.MessageBroker`
*   **Base:** `core.system.v22_async.async_agent_base.AsyncAgentBase`

Build additively. Do not break the message bus.
