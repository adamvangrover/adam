# tests/test_token_utils.py

import unittest
from core.utils.token_utils import count_tokens, check_token_limit


class TestTokenUtils(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
