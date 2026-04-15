import sys
from unittest.mock import MagicMock

class MockLimiter:
    def __init__(self, *args, **kwargs):
        pass

mock_ratelimiter = MagicMock()
mock_ratelimiter.create_rate_limiter.return_value = MockLimiter()
mock_httpxthrottlecache = MagicMock()
mock_httpxthrottlecache.ratelimiter = mock_ratelimiter

sys.modules['httpxthrottlecache.ratelimiter'] = mock_ratelimiter
sys.modules['pyrate_limiter'] = MagicMock()

import src.market_mayhem.scanners
print("Patched edgar successfully")
