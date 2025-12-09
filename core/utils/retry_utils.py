import time
import logging
from functools import wraps
import random

logger = logging.getLogger(__name__)

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"Function {func.__name__} failed after {retries} retries: {e}")
                        raise
                    else:
                        sleep = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                        logger.warning(f"Function {func.__name__} failed: {e}. Retrying in {sleep:.2f}s...")
                        time.sleep(sleep)
                        x += 1
        return wrapper
    return decorator
