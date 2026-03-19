import asyncio
import time
from unittest.mock import patch, MagicMock
import pytest
from core.utils.retry_utils import retry_with_backoff


def test_sync_success_first_try():
    mock_func = MagicMock(return_value="success")

    @retry_with_backoff(retries=3, backoff_in_seconds=0.1)
    def test_func():
        return mock_func()

    result = test_func()
    assert result == "success"
    assert mock_func.call_count == 1


def test_sync_retry_then_success():
    mock_func = MagicMock(side_effect=[ValueError("fail 1"), ValueError("fail 2"), "success"])

    @retry_with_backoff(retries=3, backoff_in_seconds=0.1, max_backoff=0.2, exceptions=(ValueError,))
    def test_func():
        return mock_func()

    result = test_func()
    assert result == "success"
    assert mock_func.call_count == 3


def test_sync_max_retries_exceeded():
    mock_func = MagicMock(side_effect=ValueError("persistent failure"))

    @retry_with_backoff(retries=2, backoff_in_seconds=0.1, exceptions=(ValueError,))
    def test_func():
        return mock_func()

    with pytest.raises(ValueError, match="persistent failure"):
        test_func()

    assert mock_func.call_count == 3  # Initial + 2 retries


def test_sync_unhandled_exception():
    mock_func = MagicMock(side_effect=TypeError("unhandled"))

    @retry_with_backoff(retries=3, exceptions=(ValueError,))
    def test_func():
        return mock_func()

    with pytest.raises(TypeError, match="unhandled"):
        test_func()

    assert mock_func.call_count == 1  # Should fail immediately without retries


@pytest.mark.asyncio
async def test_async_success_first_try():
    mock_func = MagicMock(return_value="success")

    @retry_with_backoff(retries=3, backoff_in_seconds=0.1)
    async def test_func():
        return mock_func()

    result = await test_func()
    assert result == "success"
    assert mock_func.call_count == 1


@pytest.mark.asyncio
async def test_async_retry_then_success():
    mock_func = MagicMock(side_effect=[ValueError("fail 1"), ValueError("fail 2"), "success"])

    @retry_with_backoff(retries=3, backoff_in_seconds=0.1, max_backoff=0.2, exceptions=(ValueError,))
    async def test_func():
        return mock_func()

    result = await test_func()
    assert result == "success"
    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_async_max_retries_exceeded():
    mock_func = MagicMock(side_effect=ValueError("persistent failure"))

    @retry_with_backoff(retries=2, backoff_in_seconds=0.1, exceptions=(ValueError,))
    async def test_func():
        return mock_func()

    with pytest.raises(ValueError, match="persistent failure"):
        await test_func()

    assert mock_func.call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_async_unhandled_exception():
    mock_func = MagicMock(side_effect=TypeError("unhandled"))

    @retry_with_backoff(retries=3, exceptions=(ValueError,))
    async def test_func():
        return mock_func()

    with pytest.raises(TypeError, match="unhandled"):
        await test_func()

    assert mock_func.call_count == 1  # Should fail immediately without retries
