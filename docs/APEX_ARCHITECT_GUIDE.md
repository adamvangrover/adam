# Apex Architect: System Evolution Guide (v23.5)

**"We build towers, we do not dig graves."**

## Overview
This document outlines the architectural evolution introduced by the "Apex Architect" protocol. It focuses on the transition to a non-blocking, asynchronous "Temporal Pulse" for system maintenance and the hardening of security protocols for Enterprise deployment.

## 1. The Temporal Engine ("The Pulse")
### Philosophy
The legacy `task_scheduler.py` relied on blocking `time.sleep()` calls, incompatible with high-frequency agent operations. The new **Temporal Engine** (`core/system/temporal_engine.py`) operates within the `asyncio` event loop, providing a non-blocking "heartbeat" for the system.

### Architecture
- **PulseTask**: Wraps a coroutine with schedule logic.
- **TemporalEngine**: Manages the schedule and executes tasks without blocking the main loop.
- **Launcher**: `scripts/launch_system_pulse.py` serves as the new entry point for the "Always On" runtime.

### Usage
To start the system pulse:
```bash
make pulse
```

## 2. Configuration & Security
### LLM Plugin Configuration
The `config/llm_plugin.yaml` has been standardized.
- **Provider Selection**: Use the `provider` key to switch between `openai`, `anthropic`, `cohere`, `gemini`, `huggingface`, or `mock`.
- **Security Policies**: New `security_policies` section enforces machine-level checks (e.g., HTTPS).
- **Human Oversight**: New `human_oversight` section defines approval gates for high-stakes actions.

### Automated Auditing
A new security auditor (`ops/security/audit_config.py`) scans configuration files for hardcoded secrets (API keys). This runs automatically as part of:
```bash
make security
```

## 3. Developer Instructions
### Adding a New Recurring Task
1. Define an `async` method in `core/procedures/`.
2. Register it in `scripts/launch_system_pulse.py`:
   ```python
   temporal_engine.register_task(
       name="My New Task",
       coro_func=my_procedure_instance.run_task,
       interval_seconds=3600
   )
   ```

### Switching LLM Providers
Edit `config/llm_plugin.yaml`:
```yaml
provider: openai # Switch from 'mock' to 'openai'
```
Ensure `OPENAI_API_KEY` is set in your environment (`.env`).

## 4. DevOps
- **Docker**: The system is container-ready. Use `docker-compose up` to launch.
- **Make**: Use `make verify-pulse` to test the startup sequence in CI/CD pipelines.

---
*Signed,*
**The Apex Architect**
