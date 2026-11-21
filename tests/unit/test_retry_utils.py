import asyncio
import unittest
from unittest.mock import patch

from core.utils.retry_utils import retry_with_backoff, _calculate_jitter


class TestRetryUtils(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.call_count = 0

    def test_calculate_jitter(self):
        """Test jitter calculation to ensure it respects max_backoff and bounds."""
        jitter = _calculate_jitter(attempt=0, backoff_in_seconds=1.0, max_backoff=5.0)
        self.assertTrue(0 <= jitter <= 1.0)

        jitter_max = _calculate_jitter(attempt=10, backoff_in_seconds=1.0, max_backoff=5.0)
        self.assertTrue(0 <= jitter_max <= 5.0)

    @patch("time.sleep", return_value=None)
    def test_sync_retry_success(self, mock_sleep):
        """Test synchronous retry succeeding after failures."""
        @retry_with_backoff(retries=3, backoff_in_seconds=0.1)
        def flaky_func():
            self.call_count += 1
            if self.call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = flaky_func()
        self.assertEqual(result, "success")
        self.assertEqual(self.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("time.sleep", return_value=None)
    def test_sync_retry_exhausted(self, mock_sleep):
        """Test synchronous retry raising exception when exhausted."""
        @retry_with_backoff(retries=2, backoff_in_seconds=0.1)
        def failing_func():
            self.call_count += 1
            raise ValueError("Permanent failure")

        with self.assertRaises(ValueError):
            failing_func()

        self.assertEqual(self.call_count, 3) # 1 initial + 2 retries
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("asyncio.sleep", return_value=None)
    async def test_async_retry_success(self, mock_sleep):
        """Test asynchronous retry succeeding after failures."""
        @retry_with_backoff(retries=2, backoff_in_seconds=0.1)
        async def flaky_async_func():
            self.call_count += 1
            if self.call_count < 2:
                raise KeyError("Temporary missing key")
            return "async_success"

        result = await flaky_async_func()
        self.assertEqual(result, "async_success")
        self.assertEqual(self.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("asyncio.sleep", return_value=None)
    async def test_async_retry_exhausted(self, mock_sleep):
        """Test asynchronous retry exhausting retries."""
        @retry_with_backoff(retries=1, backoff_in_seconds=0.1)
        async def failing_async_func():
            self.call_count += 1
            raise ConnectionError("Network down")

        with self.assertRaises(ConnectionError):
            await failing_async_func()

        self.assertEqual(self.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

if __name__ == "__main__":
    unittest.main()
