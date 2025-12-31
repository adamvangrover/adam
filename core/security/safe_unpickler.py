# core/security/safe_unpickler.py

import pickle
import io
import logging
from typing import Any, List, Set

logger = logging.getLogger(__name__)

class SafeUnpickler(pickle.Unpickler):
    """
    A safer unpickler that restricts loading to a specific whitelist of modules.
    This prevents Remote Code Execution (RCE) attacks via insecure deserialization
    by blocking dangerous modules (like os, subprocess, builtins.exec).
    """

    # Allowed root modules
    ALLOWED_MODULE_PREFIXES = {
        'numpy',
        'sklearn',
        'pandas',
        'scipy',
        'joblib',
        'pytz', # Often used in pandas
        'datetime', # often used in pandas
        'torch', # Allow torch tensors
    }

    # Specific allowed classes in builtins or other sensitive modules
    # (Only add if absolutely necessary for the model)
    ALLOWED_GLOBALS = {
        # ('builtins', 'set'), # Example if needed, though usually handled by opcodes
    }

    def find_class(self, module: str, name: str) -> Any:
        # Check explicit whitelist for globals
        if (module, name) in self.ALLOWED_GLOBALS:
            return super().find_class(module, name)

        # Check module prefixes
        if any(module.startswith(prefix) or module == prefix for prefix in self.ALLOWED_MODULE_PREFIXES):
             # Additional safety: blacklist commonly used dangerous functions even in allowed modules?
             # (Unlikely to exist in numpy/sklearn, but good practice if extending)
             return super().find_class(module, name)

        # Violation
        error_msg = f"SafeUnpickler: Blocked attempt to load '{module}.{name}'"
        logger.error(error_msg)
        raise pickle.UnpicklingError(error_msg)

def safe_load(file_obj) -> Any:
    """
    Safely load a pickle file.

    Args:
        file_obj: File-like object opened in binary mode.

    Returns:
        Unpickled object.

    Raises:
        pickle.UnpicklingError: If a disallowed class is encountered.
    """
    return SafeUnpickler(file_obj).load()

def safe_loads(data: bytes) -> Any:
    """
    Safely load pickle data from bytes.

    Args:
        data: Pickled bytes.

    Returns:
        Unpickled object.
    """
    return SafeUnpickler(io.BytesIO(data)).load()
