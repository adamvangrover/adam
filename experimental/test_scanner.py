import sys
from unittest.mock import MagicMock

# Patch httpxthrottlecache to avoid the pyrate_limiter v3 issue
mock_ratelimiter = MagicMock()
class MockLimiter:
    def __init__(self, *args, **kwargs):
        pass
mock_ratelimiter.create_rate_limiter.return_value = MockLimiter()
sys.modules['httpxthrottlecache.ratelimiter'] = mock_ratelimiter
sys.modules['pyrate_limiter'] = MagicMock()

import pandas as pd
from src.market_mayhem.scanners import WhaleScanner

def test_scanner_vectorization():
    scanner = WhaleScanner("Test <test@test.com>")

    # Mock filings
    q0_df = pd.DataFrame({
        'cusip': ['111', '222', '333'],
        'ticker': ['AAPL', 'MSFT', 'TSLA'],
        'value': [1000, 2000, 3000],
        'shares': [100, 200, 300],
        'share_type': ['SH', 'SH', 'PRN'],
        'issuer': ['Apple', 'Microsoft', 'Tesla']
    })

    q1_df = pd.DataFrame({
        'cusip': ['111', '444'],
        'ticker': ['AAPL', 'AMZN'],
        'value': [500, 4000],
        'shares': [50, 400],
        'share_type': ['SH', 'SH'],
        'issuer': ['Apple', 'Amazon']
    })

    # Mock methods
    scanner._fetch_filings = MagicMock(return_value=["mock_filing1", "mock_filing2"])
    scanner._parse_13f_xml = MagicMock(side_effect=[q0_df, q1_df])

    # Run
    signals = scanner.calculate_fund_sentiment("OAKTREE", lookback=2)

    print("Detected signals:")
    for sig in signals:
        print(f"- {sig.ticker}: {sig.signal_type} ({sig.change_pct}%)")

    # Validations
    assert len(signals) == 3

    # AAPL should be an ACCUMULATION
    aapl = next((s for s in signals if s.ticker == 'AAPL'), None)
    assert aapl is not None
    assert aapl.signal_type == 'ACCUMULATION'
    assert aapl.change_pct == 100.0  # 50 -> 100

    # MSFT should be a NEW_ENTRY
    msft = next((s for s in signals if s.ticker == 'MSFT'), None)
    assert msft is not None
    assert msft.signal_type == 'VULTURE_ENTRY'
    assert msft.change_pct == 100.0

    # TSLA should be a NEW_ENTRY (PRN)
    tsla = next((s for s in signals if s.ticker == 'TSLA'), None)
    assert tsla is not None
    assert tsla.signal_type == 'VULTURE_ENTRY'
    assert 'DEBT/CONVERTIBLE' in tsla.description

test_scanner_vectorization()
print("All tests passed.")
