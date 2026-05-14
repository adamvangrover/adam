# tests/test_token_utils.py

import unittest
from unittest.mock import patch, MagicMock
from core.utils.token_utils import (
    count_tokens,
    check_token_limit,
    truncate_text,
    _get_encoding,
    _get_encoding_for_model,
    TokenManager,
)


class TestTokenUtils(unittest.TestCase):
    def setUp(self):
        # Clear lru_cache for isolated tests
        # Access __func__ to call cache_clear on the underlying wrapped function
        # to ensure compatibility across different Python minor versions
        TokenManager.get_encoding.__func__.cache_clear()
        TokenManager.get_encoding_for_model.__func__.cache_clear()

    # --- Tests for existing functionality (Backward Compatibility) ---

    def test_count_tokens_empty_string(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_tokens(None), 0)

    def test_count_tokens_simple_string(self):
        self.assertEqual(count_tokens("hello world"), 2)

    def test_count_tokens_with_punctuation(self):
        self.assertEqual(count_tokens("hello, world!"), 4)

    def test_check_token_limit_within_limit(self):
        config = {"token_limit": 10}
        self.assertTrue(check_token_limit("short text", config=config, margin=2))  # 2 <= 10-2=8

    def test_check_token_limit_exceeds_limit(self):
        config = {"token_limit": 5}
        self.assertFalse(check_token_limit("This is a longer text", config=config, margin=1))  # 5 <= 5-1=4 is false

    def test_check_token_limit_near_limit_with_margin_pass(self):
        config = {"token_limit": 5}
        self.assertTrue(check_token_limit("This is text", config=config, margin=2))  # 3 <= 5-2=3 is true

    def test_check_token_limit_near_limit_with_margin_fail(self):
        config = {"token_limit": 5}
        self.assertFalse(check_token_limit("This is text", config=config, margin=3))  # 3 <= 5-3=2 is false

    def test_check_token_limit_at_limit(self):
        config = {"token_limit": 3}
        self.assertTrue(check_token_limit("This is text", config=config, margin=0))  # 3 <= 3 is true

    def test_check_token_limit_no_config(self):
        self.assertTrue(check_token_limit("This is text", config=None))

    def test_count_tokens_with_model_name(self):
        # gpt-4 model uses cl100k_base
        self.assertEqual(count_tokens("hello world", model_name="gpt-4"), 2)

    def test_count_tokens_invalid_model_fallback(self):
        # Should gracefully fallback to cl100k_base for unknown models
        self.assertEqual(count_tokens("hello world", model_name="unknown-model-xyz"), 2)

    def test_count_tokens_invalid_encoding_fallback(self):
        # Should gracefully fallback to cl100k_base for unknown encodings
        self.assertEqual(count_tokens("hello world", encoding_name="invalid_encoding_name"), 2)

    @patch("core.utils.token_utils.tiktoken.get_encoding")
    def test_lru_cache_behavior(self, mock_get_encoding):
        # Setup mock to track calls
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2]
        mock_get_encoding.return_value = mock_encoding

        # First call should hit tiktoken.get_encoding
        count_tokens("hello world", encoding_name="cl100k_base")
        mock_get_encoding.assert_called_once_with("cl100k_base")

        # Second call should use cache
        count_tokens("hello world again", encoding_name="cl100k_base")
        mock_get_encoding.assert_called_once()  # Call count shouldn't increase

    # --- Tests for new Innovator functionality ---

    def test_truncate_text_within_limit(self):
        text = "hello world"
        # 2 tokens. Truncating to 5 should return the original string
        self.assertEqual(truncate_text(text, 5), text)

    def test_truncate_text_exceeds_limit(self):
        text = "This is a longer piece of text."
        # The tokens should be ["This", " is", " a", " longer", " piece", " of", " text", "."]
        truncated = truncate_text(text, 4)
        self.assertTrue(len(truncated) < len(text))
        self.assertTrue(count_tokens(truncated) <= 4)

    def test_truncate_text_invalid_inputs(self):
        self.assertEqual(truncate_text("", 10), "")
        self.assertEqual(truncate_text(None, 10), "")
        self.assertEqual(truncate_text("hello", 0), "")
        self.assertEqual(truncate_text("hello", -5), "")

    @patch("core.utils.token_utils.tiktoken.get_encoding")
    def test_truncate_text_fallback_on_exception(self, mock_get_encoding):
        mock_get_encoding.side_effect = Exception("Simulated crash")
        text = "This is some test text that is fairly long"
        # Exception thrown, should fallback to text[:max_tokens*4]
        truncated = truncate_text(text, max_tokens=5)
        self.assertEqual(truncated, text[:20])

if __name__ == "__main__":
    unittest.main()
