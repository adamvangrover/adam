import re
from typing import List, Tuple

class SQLValidator:
    """
    A robust, zero-dependency SQL validator for ensuring read-only access.
    It parses the SQL string into tokens to safely handle comments, string literals,
    and whitespace, preventing common bypasses of regex-based checks.
    """

    @staticmethod
    def validate_read_only(sql: str) -> bool:
        """
        Validates that the SQL query is a strict read-only SELECT statement.

        Rules:
        1. Must start with SELECT.
        2. Must contain only one statement (no semicolons outside quotes/comments).
        3. Comments are stripped before validation to prevent hiding commands.
        """
        tokens = SQLValidator._tokenize(sql)

        # Filter out comments and whitespace for logic checks
        meaningful_tokens = [t for t in tokens if t[0] not in ('COMMENT', 'WHITESPACE')]

        if not meaningful_tokens:
            return False

        # Rule 1: Must start with SELECT
        first_token_type, first_token_value = meaningful_tokens[0]
        if first_token_type != 'KEYWORD' or first_token_value.upper() != 'SELECT':
            return False

        # Rule 2: No multiple statements
        # Check for semicolons in meaningful tokens (operators)
        for token_type, token_value in meaningful_tokens:
            if token_type == 'OPERATOR' and ';' in token_value:
                return False

        return True

    @staticmethod
    def _tokenize(sql: str) -> List[Tuple[str, str]]:
        """
        A simple SQL tokenizer.
        Returns a list of (type, value) tuples.
        Types: KEYWORD, STRING, COMMENT, WHITESPACE, OPERATOR, IDENTIFIER
        """
        token_specs = [
            ('STRING',     r"'([^']|'')*'"),       # Single quoted string
            ('STRING_DQ',  r'"([^"]|""|\\")*"'),   # Double quoted string (or identifier in some dialects)
            ('COMMENT_ML', r'/\*[\s\S]*?\*/'),     # Multi-line comment
            ('COMMENT_SL', r'--.*'),               # Single-line comment
            ('WHITESPACE', r'\s+'),                # Whitespace
            ('KEYWORD',    r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|WITH)\b'), # Keywords we care about
            ('OPERATOR',   r'[;(),=]'),            # Operators
            ('IDENTIFIER', r'[a-zA-Z0-9_]+'),      # Identifiers
            ('OTHER',      r'.'),                  # Any other character
        ]

        # Compile regex
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specs)
        get_token = re.compile(tok_regex, re.IGNORECASE).match

        pos = 0
        mo = get_token(sql)
        tokens = []

        while mo is not None:
            kind = mo.lastgroup
            value = mo.group(kind)

            # Normalize token types
            if kind in ('STRING_DQ', 'STRING'):
                kind = 'STRING'
            elif kind in ('COMMENT_ML', 'COMMENT_SL'):
                kind = 'COMMENT'
            elif kind == 'KEYWORD':
                # Ensure we capture keywords case-insensitively for the type,
                # but keep original value
                pass

            tokens.append((kind, value))
            pos = mo.end()
            mo = get_token(sql, pos)

        return tokens
