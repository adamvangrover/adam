import re
from typing import List, Tuple

class SQLValidator:
    """
    A robust, zero-dependency SQL validator for ensuring read-only access.
    It parses the SQL string into tokens to safely handle comments, string literals,
    and whitespace, preventing common bypasses of regex-based checks.
    """

    # 🛡️ Sentinel: Expanded list of dangerous keywords and identifiers
    DANGEROUS_KEYWORDS = {
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE',
        'GRANT', 'REVOKE', 'CREATE', 'REPLACE', 'MERGE', 'CALL',
        'EXEC', 'EXECUTE', 'PREPARE', 'DEALLOCATE', 'DESCRIBE', 'SHOW',
        'USE', 'SET', 'DECLARE', 'GO', 'INTO', 'PRAGMA', 'EXPLAIN'
    }

    # 🛡️ Sentinel: Dangerous identifiers (converted to uppercase for comparison)
    DANGEROUS_IDENTIFIERS = {
        'XP_CMDSHELL', 'SP_CONFIGURE', 'XP_REGREAD', 'XP_REGWRITE',
        'XP_DIRTREE', 'XP_FILEEXIST', 'WAITFOR', 'LOAD_FILE', 'BENCHMARK',
        'PG_SLEEP', 'SLEEP', 'SYS_EVAL', 'SYS_EXEC'
    }

    @staticmethod
    def validate_read_only(sql: str) -> bool:
        """
        Validates that the SQL query is a strict read-only SELECT statement.

        Rules:
        1. Must start with SELECT.
        2. Must contain only one statement (no semicolons outside quotes/comments).
        3. No dangerous keywords (EXEC, DROP, etc.) allowed anywhere in the query.
        4. No dangerous identifiers (xp_cmdshell, etc.) allowed anywhere in the query.
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
        for token_type, token_value in meaningful_tokens:
            if token_type == 'OPERATOR' and ';' in token_value:
                return False

        # Rule 3 & 4: No dangerous keywords or identifiers ANYWHERE
        for token_type, token_value in meaningful_tokens:
            upper_val = token_value.upper()

            if token_type == 'KEYWORD' and upper_val in SQLValidator.DANGEROUS_KEYWORDS:
                return False

            if token_type == 'IDENTIFIER' and upper_val in SQLValidator.DANGEROUS_IDENTIFIERS:
                return False

        return True

    @staticmethod
    def _tokenize(sql: str) -> List[Tuple[str, str]]:
        """
        A simple SQL tokenizer.
        Returns a list of (type, value) tuples.
        Types: KEYWORD, STRING, COMMENT, WHITESPACE, OPERATOR, IDENTIFIER
        """
        # 🛡️ Sentinel: Expanded keyword regex to catch more DDL/DML commands
        # Note: We must be careful not to include substrings of common words unless bounded by \b which is handled below.
        keywords = (
            'SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|WITH|'
            'EXEC|EXECUTE|MERGE|REPLACE|CREATE|PRAGMA|EXPLAIN|SHOW|DESCRIBE|USE|'
            'SET|DECLARE|GO|INTO|CALL|PREPARE|DEALLOCATE'
        )

        token_specs = [
            ('STRING',     r"'([^']|'')*'"),       # Single quoted string
            ('STRING_DQ',  r'"([^"]|""|\\")*"'),   # Double quoted string (or identifier in some dialects)
            ('COMMENT_ML', r'/\*[\s\S]*?\*/'),     # Multi-line comment
            ('COMMENT_SL', r'--.*'),               # Single-line comment
            ('WHITESPACE', r'\s+'),                # Whitespace
            ('KEYWORD',    r'\b({})\b'.format(keywords)), # Keywords we care about
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
                pass

            tokens.append((kind, value))
            pos = mo.end()
            mo = get_token(sql, pos)

        return tokens
