# tests/test_token_utils.py

import unittest
from unittest.mock import patch, MagicMock
from core.utils.token_utils import count_tokens, check_token_limit, _get_encoding, _get_encoding_for_model


class TestTokenUtils(unittest.TestCase):

    def setUp(self):
        # Clear lru_cache for isolated tests
        _get_encoding.cache_clear()
        _get_encoding_for_model.cache_clear()

    def test_count_tokens_empty_string(self):
        self.assertEqual(count_tokens(""), 0)

    def test_count_tokens_simple_string(self):
        self.assertEqual(count_tokens("hello world"), 2)

    def test_count_tokens_with_punctuation(self):
        self.assertEqual(count_tokens("hello, world!"), 4)

    def test_check_token_limit_within_limit(self):
        config = {'token_limit': 10}
        self.assertTrue(check_token_limit("short text", config=config, margin=2))  # 2 <= 10-2=8

    def test_check_token_limit_exceeds_limit(self):
        config = {'token_limit': 5}
        self.assertFalse(check_token_limit("This is a longer text", config=config, margin=1))  # 5 <= 5-1=4 is false

    def test_check_token_limit_near_limit_with_margin_pass(self):
        config = {'token_limit': 5}
        self.assertTrue(check_token_limit("This is text", config=config, margin=2))  # 3 <= 5-2=3 is true

    def test_check_token_limit_near_limit_with_margin_fail(self):
        config = {'token_limit': 5}
        self.assertFalse(check_token_limit("This is text", config=config, margin=3))  # 3 <= 5-3=2 is false

    def test_check_token_limit_at_limit(self):
        config = {'token_limit': 3}
        self.assertTrue(check_token_limit("This is text", config=config, margin=0))  # 3 <= 3 is true

    def test_count_tokens_with_model_name(self):
        # gpt-4 model uses cl100k_base
        self.assertEqual(count_tokens("hello world", model_name="gpt-4"), 2)

    def test_count_tokens_invalid_model_fallback(self):
        # Should gracefully fallback to cl100k_base for unknown models
        self.assertEqual(count_tokens("hello world", model_name="unknown-model-xyz"), 2)

    def test_count_tokens_invalid_encoding_fallback(self):
        # Should gracefully fallback to cl100k_base for unknown encodings
        self.assertEqual(count_tokens("hello world", encoding_name="invalid_encoding_name"), 2)

    @patch('core.utils.token_utils.tiktoken.get_encoding')
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


if __name__ == '__main__':
    unittest.main()
