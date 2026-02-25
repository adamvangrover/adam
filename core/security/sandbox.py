# core/security/sandbox.py

import ast
import multiprocessing
import queue
import time
import json
import logging
from typing import Dict, Any, Optional
from core.security.governance import GovernanceEnforcer, ApprovalRequired

logger = logging.getLogger(__name__)

class SecurityViolation(Exception):
    """Raised when code violates security policies."""
    pass

class ExecutionTimeout(Exception):
    """Raised when code execution exceeds time limit."""
    pass

class SecureSandbox:
    """
    A robust, multi-layer sandbox for executing untrusted Python code.

    Philosophy: "Defense in Depth"
    1. Static Analysis (AST): Rejects dangerous constructs (imports, exec, eval, private access).
    2. Restricted Globals: Only allows a strict whitelist of safe functions/modules.
    3. Process Isolation: Runs code in a separate process to contain memory/crashes.
    4. Timeouts: Prevents infinite loops.
    """

    # --------------------------------------------------------------------------
    # Layer 1: Static Analysis Configuration
    # --------------------------------------------------------------------------

    # Allowed AST nodes. Anything else is rejected.
    ALLOWED_NODES = {
        'Module', 'Expr', 'Assign', 'AugAssign', 'AnnAssign',
        'Num', 'Str', 'Bytes', 'List', 'Tuple', 'Set', 'Dict', 'Constant',
        'Name', 'Load', 'Store', 'BinOp', 'UnaryOp', 'BoolOp', 'Compare',
        'If', 'IfExp', 'For', 'While', 'Break', 'Continue', 'Pass',
        'Call', 'Attribute', 'Subscript', 'Index', 'Slice', 'ExtSlice',
        'ListComp', 'SetComp', 'DictComp', 'GeneratorExp',
        'FunctionDef', 'Return', 'Lambda', 'arg', 'arguments', 'keyword',
        'JoinedStr', 'FormattedValue',
        # Operators
        'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Pow', 'LShift', 'RShift', 'BitOr', 'BitXor', 'BitAnd', 'MatMult',
        'UAdd', 'USub', 'Not', 'Invert',
        'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'Is', 'IsNot', 'In', 'NotIn',
        'And', 'Or'
    }

    # Explicitly banned functions (even if they exist in allowed modules)
    BANNED_FUNCTIONS = {'open', 'compile', 'exec', 'eval', 'globals', 'locals', 'input', 'breakpoint', 'help', 'exit', 'quit'}

    # --------------------------------------------------------------------------
    # Layer 2: Restricted Globals Configuration
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_safe_globals() -> Dict[str, Any]:
        """Returns the dictionary of globals allowed in the sandbox."""
        safe_globals = {
            "__builtins__": {},  # CRITICAL: Disable builtins
            "print": print,      # We might want to capture this instead of allowing direct stdout
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "bool": bool,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
            "math": __import__("math"),
            "json": __import__("json"),
            "datetime": __import__("datetime"),
        }

        # Optional: Add pandas/numpy if available, but be careful
        try:
            import pandas as pd
            safe_globals["pd"] = pd
        except ImportError:
            pass

        try:
            import numpy as np
            safe_globals["np"] = np
        except ImportError:
            pass

        return safe_globals

    # --------------------------------------------------------------------------
    # Implementation
    # --------------------------------------------------------------------------

    @classmethod
    def execute(cls, code: str, timeout: float = 5.0, security_level: str = "standard", memory_limit_mb: int = 512) -> Dict[str, Any]:
        """
        Executes the provided code in a secure sandbox.

        Args:
            code (str): The Python code to execute.
            timeout (float): Maximum execution time in seconds.
            security_level (str): "standard", "governed", or "air_gapped".
            memory_limit_mb (int): Maximum memory usage in MB. Defaults to 512MB.

        Returns:
            Dict containing 'status' ('success'/'error'), 'output', and 'result' (last expression).
        """
        try:
            # Step 0: Governance Check (for higher security tiers)
            if security_level != "standard":
                try:
                    GovernanceEnforcer.validate(code, context=f"sandbox_{security_level}")
                except ApprovalRequired as e:
                     return {"status": "error", "error": f"Governance Approval Required: {e}"}
                except Exception as e:
                     return {"status": "error", "error": f"Governance Check Failed: {e}"}

            # Step 1: Static Analysis
            cls._validate_ast(code)

            # Step 2 & 3: Process Isolation & Execution
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=cls._worker,
                args=(code, result_queue, memory_limit_mb)
            )
            process.start()
            process.join(timeout)

            if process.is_alive():
                process.terminate()
                process.join()
                raise ExecutionTimeout(f"Code execution exceeded {timeout}s limit.")

            if result_queue.empty():
                # Process died without returning (e.g., segfault or OOM)
                return {"status": "error", "error": "Process crashed unexpectedly."}

            result = result_queue.get()
            if result["status"] == "error":
                return result

            return result

        except SecurityViolation as e:
            logger.warning(f"Sandbox Security Violation: {e}")
            return {"status": "error", "error": f"Security Violation: {e}"}
        except ExecutionTimeout as e:
            logger.warning(f"Sandbox Timeout: {e}")
            return {"status": "error", "error": str(e)}
        except SyntaxError as e:
            return {"status": "error", "error": f"Syntax Error: {e}"}
        except Exception as e:
            logger.error(f"Sandbox Unexpected Error: {e}", exc_info=True)
            return {"status": "error", "error": f"System Error: {e}"}

    @classmethod
    def _validate_ast(cls, code: str):
        """Parses code into AST and checks for banned nodes/attributes."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise e

        for node in ast.walk(tree):
            # Check Node Type
            if type(node).__name__ not in cls.ALLOWED_NODES:
                raise SecurityViolation(f"Operation '{type(node).__name__}' is not allowed.")

            # Check for Import
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SecurityViolation("Imports are strictly forbidden.")

            # Check for Attributes starting with __ (Double Underscore)
            # This prevents access to internal properties like __class__, __subclasses__, etc.
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise SecurityViolation(f"Access to private attribute '{node.attr}' is forbidden.")

            # Check for banned function calls
            if isinstance(node, ast.Name) and node.id in cls.BANNED_FUNCTIONS:
                raise SecurityViolation(f"Function '{node.id}' is banned.")

            # Check for banned method calls (format/format_map)
            # These are banned because they can be used to bypass AST checks for private attributes
            if isinstance(node, ast.Attribute) and node.attr in {'format', 'format_map'}:
                raise SecurityViolation(f"Method '{node.attr}' is banned for security reasons.")

    @classmethod
    def _worker(cls, code: str, result_queue: multiprocessing.Queue, memory_limit_mb: int):
        """
        The function running inside the isolated process.
        """
        import io
        from contextlib import redirect_stdout, redirect_stderr
        try:
            import resource
            # Limit memory usage (RLIMIT_AS is virtual memory size)
            # Convert MB to Bytes
            limit = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ImportError, ValueError, OSError):
            # Ignore if resource module not present (Windows) or other error
            pass

        output_buffer = io.StringIO()
        safe_globals = cls._get_safe_globals()

        # Explicitly define an empty local scope to prevent leaking implementation details
        # unless necessary for specific execution contexts
        safe_locals = {}

        try:
            # We use compile to get code object
            # exec() is used here, but ONLY on the code that passed AST validation
            # and ONLY with the restricted globals dictionary.
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                # Usage of exec is guarded by strict AST validation and restricted globals
                exec(code, safe_globals, safe_locals)  # nosec B102

            # Attempt to capture the last expression if possible (like a REPL)
            # but simple exec doesn't return it. For now, rely on print output.

            result_queue.put({
                "status": "success",
                "output": output_buffer.getvalue()
            })
        except Exception as e:
            result_queue.put({
                "status": "error",
                "error": f"{type(e).__name__}: {str(e)}",
                "output": output_buffer.getvalue()
            })
