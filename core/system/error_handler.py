#core/system/error_handler.py

class AdamError(Exception):
    """Base class for Adam-specific errors."""
    pass

class DataNotFoundError(AdamError):
    """Raised when requested data is not found."""
    pass

class AgentNotFoundError(AdamError):
    """Raised when a requested agent is not found."""
    pass

class InvalidInputError(AdamError):
    """Raised when user input is invalid."""
    pass

# Add more specific error types as needed
