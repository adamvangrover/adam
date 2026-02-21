import re
import os
import html
import logging

logger = logging.getLogger(__name__)

class InputShield:
    """
    🛡️ Sentinel Shield: Centralized Input Validation & Sanitization
    """

    ALLOWED_EXTENSIONS = {'.json', '.csv', '.txt', '.md', '.pdf', '.xlsx', '.xls', '.docx'}

    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """
        Validates a financial ticker symbol.
        - Length: 1-12 chars
        - Allowed: Uppercase letters, numbers, dots (e.g. BRK.B), hyphens.
        """
        if not ticker or not isinstance(ticker, str):
            return False
        if len(ticker) > 12 or len(ticker) < 1:
            return False
        # Allow dots and hyphens for things like BRK.B or BTC-USD
        return bool(re.match(r'^[A-Z0-9\.-]+$', ticker.upper()))

    @staticmethod
    def validate_filename(filename: str) -> bool:
        """
        Validates a filename to prevent path traversal and enforce allowed extensions.
        """
        if not filename or not isinstance(filename, str):
            return False

        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            return False

        # Check extension
        _, ext = os.path.splitext(filename.lower())
        if ext not in InputShield.ALLOWED_EXTENSIONS:
            return False

        # Check for dangerous characters in the name part
        # Allow alphanumeric, underscore, hyphen, space, dot
        if not re.match(r'^[a-zA-Z0-9_\-\. ]+$', filename):
            return False

        return True

    @staticmethod
    def sanitize_text(text: str, max_length: int = 5000) -> str:
        """
        Sanitizes text input for display (basic HTML escaping) and enforces length limits.
        """
        if not text:
            return ""

        # Truncate
        if len(text) > max_length:
            logger.warning(f"InputShield: Truncated text from {len(text)} to {max_length}")
            text = text[:max_length]

        # Basic HTML Escape (prevents Stored XSS if raw output is used)
        # Note: Frontend frameworks (React) usually auto-escape, but this is defense-in-depth for API responses.
        return html.escape(text)

    @staticmethod
    def validate_username(username: str) -> bool:
        """
        Validates a username.
        - 4-30 chars
        - Alphanumeric, underscore, hyphen
        """
        if not username or not isinstance(username, str):
            return False
        if len(username) < 4 or len(username) > 30:
            return False
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', username))

    @staticmethod
    def validate_portfolio_name(name: str) -> bool:
        """
        Validates portfolio names.
        - 1-64 chars
        - Safe characters
        """
        if not name or not isinstance(name, str):
            return False
        if len(name) < 1 or len(name) > 64:
            return False
        # Allow alphanumeric, spaces, hyphens, underscores
        return bool(re.match(r'^[a-zA-Z0-9 _-]+$', name))
