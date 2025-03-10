# core/system/error_handler.py

from core.utils.config_utils import load_error_codes # Import the function

class AdamError(Exception):
    """
    Base class for Adam-specific errors.  All custom exceptions should inherit from this.

    Attributes:
        code (int):  An integer error code (see config/errors.yaml).
        message (str):  A human-readable error message.
    """
    def __init__(self, code: int, message: str):
        super().__init__(message)  # Initialize the base Exception class
        self.code = code
        self.message = message

    def __str__(self):
        return f"Error Code {self.code}: {self.message}"


class DataNotFoundError(AdamError):
    """
    Raised when requested data is not found (e.g., in a file, knowledge base, etc.).
    """
    def __init__(self, data_identifier: str = "", source: str = ""):
        code = 101  # Or get from config/errors.yaml if available
        message = f"Data not found: '{data_identifier}'"
        if source:
            message += f" in source: '{source}'"
        super().__init__(code, message)


class AgentNotFoundError(AdamError):
    """
    Raised when a requested agent is not found or cannot be loaded.
    """
    def __init__(self, agent_name: str = ""):
        code = 102
        message = f"Agent not found: '{agent_name}'"
        super().__init__(code, message)


class InvalidInputError(AdamError):
    """
    Raised when user input is invalid, incomplete, or cannot be parsed.
    """
    def __init__(self, input_string: str = "", reason: str = ""):
        code = 103
        message = f"Invalid input: '{input_string}'"
        if reason:
            message += f". Reason: {reason}"
        super().__init__(code, message)


class ConfigurationError(AdamError):
    """
    Raised when there is an error in a configuration file (e.g., YAML parsing error,
    missing required settings).
    """
    def __init__(self, config_file: str = "", message: str = "Configuration error"):
        code = 104
        message = f"Configuration error in '{config_file}': {message}" if config_file else message
        super().__init__(code, message)

class FileReadError(AdamError):
    """
    Raised when there's an error reading a file (e.g., FileNotFoundError, PermissionError).
    """
    def __init__(self, file_path: str, message: str = "File read error"):
        code = 105
        message = f"Error reading file '{file_path}': {message}"
        super().__init__(code, message)

class WorkflowExecutionError(AdamError):
    """
    Raised when an error occurs during the execution of a workflow.
    """
    def __init__(self, workflow_name: str = "", step: int = -1, message: str = "Workflow execution error"):
        code = 106
        message = f"Error in workflow '{workflow_name}'"
        if step >= 0:
            message += f" at step {step}"
        message += f": {message}"
        super().__init__(code, message)
class AgentExecutionError(AdamError):
     """
     Raised by an agent if it has a problem with it's execute method.
     """
     def __init__(self, agent_name: str = "", message: str = "Workflow execution error"):
        code = 107
        message = f"Error in agent '{agent_name}'"
        message += f": {message}"
        super().__init__(code, message)
class LLMPluginError(AdamError):
     """
     Raised if there is an issue when attempting to use the LLM.
     """
     def __init__(self, message: str = "Workflow execution error"):
        code = 108
        message = f"Error in LLM Plugin"
        message += f": {message}"
        super().__init__(code, message)

# --- Utility Function (Optional, but Recommended) ---

def get_error_message(code: int, default_message: str = "An unknown error occurred.") -> str:
    """
    Retrieves an error message based on its code (from config/errors.yaml).

    Args:
        code: The error code.
        default_message:  The message to return if the code is not found.

    Returns:
        The error message.
    """
    try:
        # In a real application, you'd load this from config/errors.yaml
        # For this example, we'll use a hardcoded dictionary:
        error_messages = {
            101: "The requested data could not be found.",
            102: "The requested agent is not registered.",
            103: "The input provided is invalid or could not be processed.",
            104: "A configuration error occurred.",
            105: "An error occurred while reading a file.",
            106: "An error occurred while running the workflow.",
            107: "An error occurred while running the agent.",
            108: "An error occurred with the LLM Plugin."
            # ... add other error codes ...
        }
        return error_messages.get(code, default_message)
    except Exception:
        return default_message
