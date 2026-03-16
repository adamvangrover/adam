from unittest.mock import MagicMock, patch

import tiktoken

from core.utils.token_utils import _get_encoding, check_token_limit, count_tokens, get_token_limit, trim_to_token_limit_with_ai


def test_count_tokens_standard():
    text = "This is a standard text string for testing token counts."
    # Using cl100k_base
    encoding = tiktoken.get_encoding("cl100k_base")
    expected_tokens = len(encoding.encode(text))

    assert count_tokens(text) == expected_tokens

def test_count_tokens_invalid_encoding():
    text = "Test fallback encoding."
    # Fallback to cl100k_base
    encoding = tiktoken.get_encoding("cl100k_base")
    expected_tokens = len(encoding.encode(text))

    assert count_tokens(text, encoding_name="invalid_encoding") == expected_tokens

def test_count_tokens_error_handling():
    with patch("core.utils.token_utils._get_encoding", side_effect=Exception("Mocked error")):
        assert count_tokens("Error text") == 0

def test_get_token_limit():
    config = {"token_limit": 500}
    assert get_token_limit(config) == 500
    assert get_token_limit({}) == 4096

def test_check_token_limit():
    text = "Short text"
    config = {"token_limit": 10}
    # Short text should be well under 10 tokens
    assert check_token_limit(text, config) is True

    config_low = {"token_limit": 1}
    assert check_token_limit(text, config_low) is False

def test_check_token_limit_with_margin():
    text = "Short text"
    config = {"token_limit": 10}
    # Margin of 5 means target limit is 5. 'Short text' is 2 tokens, so <= 5 is True
    assert check_token_limit(text, config, margin=5) is True

    # Margin of 9 means target limit is 1. 'Short text' is 2 tokens, so <= 1 is False
    assert check_token_limit(text, config, margin=9) is False

@patch("os.environ.get")
@patch.dict("sys.modules", {"litellm": MagicMock()})
def test_trim_to_token_limit_with_ai_within_limit(mock_env):
    text = "Short text"
    config = {"token_limit": 100}

    result = trim_to_token_limit_with_ai(text, config)

    assert result == text

@patch("os.environ.get")
@patch.dict("sys.modules", {"litellm": MagicMock()})
def test_trim_to_token_limit_with_ai_exceeds_limit(mock_env):
    mock_env.return_value = "fake_api_key"
    import sys
    mock_litellm = sys.modules["litellm"]

    # Create a long text
    long_text = "This is a very long text. " * 50
    config = {"token_limit": 10}

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Summarized text"
    mock_litellm.completion.return_value = mock_response

    result = trim_to_token_limit_with_ai(long_text, config)

    assert result == "Summarized text"
    mock_litellm.completion.assert_called_once()

@patch("os.environ.get")
@patch.dict("sys.modules", {"litellm": MagicMock()})
def test_trim_to_token_limit_with_ai_fallback_compaction(mock_env):
    # Simulate missing litellm API key or error to trigger fallback
    mock_env.return_value = None

    # Test whitespace compaction
    whitespace_text = "Word    Word    Word"
    config_compaction = {"token_limit": 3}
    result_compaction = trim_to_token_limit_with_ai(whitespace_text, config_compaction)
    assert result_compaction == "Word Word Word"

    # Test middle-out graceful fallback
    long_text = "Start " + ("Middle " * 50) + "End"
    config_fallback = {"token_limit": 10}

    result_fallback = trim_to_token_limit_with_ai(long_text, config_fallback)

    encoding = tiktoken.get_encoding("cl100k_base")
    ellipsis = encoding.encode(" ... ")
    half_limit = (10 - len(ellipsis)) // 2
    encoded_compacted = encoding.encode(long_text.strip())
    expected_truncated = encoding.decode(encoded_compacted[:half_limit] + ellipsis + encoded_compacted[-half_limit:])

    assert result_fallback == expected_truncated

    # Test hard truncate fallback (when limit is very small)
    long_text_small_limit = "A B C D E F G H I J K L M N O P"
    config_small_limit = {"token_limit": 3}
    result_small_limit = trim_to_token_limit_with_ai(long_text_small_limit, config_small_limit)
    encoded_small = encoding.encode(long_text_small_limit)
    assert result_small_limit == encoding.decode(encoded_small[:3])

def test_lru_cache_behavior():
    _get_encoding.cache_clear()

    encoding1 = _get_encoding("cl100k_base")
    encoding2 = _get_encoding("cl100k_base")

    assert encoding1 is encoding2
    assert _get_encoding.cache_info().hits == 1
    assert _get_encoding.cache_info().misses == 1