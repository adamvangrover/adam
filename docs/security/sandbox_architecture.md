# Secure Sandbox Architecture

## Overview
The Adam v23.5 system includes a `SecureSandbox` module designed to execute untrusted Python code safely. This module replaces the legacy `execute_python_sandbox` tool which was vulnerable to Remote Code Execution (RCE).

## Defense-in-Depth Strategy

The sandbox employs a 4-layer security model to ensure that executed code cannot compromise the host system, access sensitive data, or cause denial-of-service.

### Layer 1: Static Analysis (AST Validation)
Before execution, the code is parsed into an Abstract Syntax Tree (AST). The `SecureSandbox` validates the tree against a strict whitelist of allowed nodes.
*   **Allowed:** Basic data structures, control flow (if/for/while), arithmetic, function definitions.
*   **Rejected:** `import` statements, `async`/`await`, private attribute access (e.g., `__subclasses__`).
*   **Banned Functions:** Explicitly bans calls to `exec`, `eval`, `open`, `compile`, etc.

### Layer 2: Restricted Globals
The code runs with a custom `globals()` dictionary.
*   `__builtins__` is set to an empty dictionary `{}` to prevent access to standard built-in functions like `__import__` or `open`.
*   A whitelist of safe functions (e.g., `print`, `math.*`, `json.*`) is manually injected.
*   This prevents "escape" via standard library functions.

### Layer 3: Process Isolation
Code execution happens in a separate `multiprocessing.Process`.
*   **Memory Isolation:** The worker process has its own memory space. Crashes (segfaults) do not bring down the main application.
*   **Resource Containment:** In future versions, this allows for OS-level limits (cgroups/rlimit) to be applied to the worker PID.

### Layer 4: Execution Timeouts
The main process monitors the worker process.
*   If the worker does not return a result within the `timeout` window (default 5.0s), the process is forcefully terminated (`SIGTERM`/`SIGKILL`).
*   This prevents infinite loops or resource exhaustion attacks.

## Usage

```python
from core.security.sandbox import SecureSandbox

code = """
import math
x = math.sqrt(16)
print(x)
"""

# Execute with 2-second timeout
result = SecureSandbox.execute(code, timeout=2.0)

if result["status"] == "success":
    print(result["output"])
else:
    print(f"Error: {result['error']}")
```

## Limitations & Future Work
*   **Memory Limits:** Currently relies on the host OS. Future work should implement `resource.setrlimit` in the worker process.
*   **Network Access:** The sandbox does not currently block network calls at the OS level (e.g., `socket`), but the AST validator blocks `import socket`, making this difficult to exploit.
*   **Complex Libraries:** Only basic libraries (`math`, `json`, `datetime`) are available. Support for `pandas` or `numpy` is experimental and depends on availability in the environment.
