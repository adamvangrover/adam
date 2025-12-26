import warnings
import functools
import logging

logger = logging.getLogger(__name__)

def deprecated(version: str, replacement: str = None):
    """
    Decorator to mark functions or classes as deprecated.

    Args:
        version (str): The version in which the feature was deprecated.
        replacement (str, optional): The name of the replacement feature.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated since version {version}."
            if replacement:
                message += f" Use {replacement} instead."

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            logger.warning(message)
            return func(*args, **kwargs)
        return wrapper
    return decorator
