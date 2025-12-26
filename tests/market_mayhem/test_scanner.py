"""
Market Mayhem - Unit Tests
Author: Principal Software Architect
Description: Quality Assurance for WhaleScanner logic.
             Mocks SEC EDGAR responses to test parsing and signal generation.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.market_mayhem.scanners import WhaleScanner, WhaleSignal

# Mock Data for 13F Information Table
# Represents Quarter 0 (Current)
MOCK_INFOTABLE_Q0 = pd.DataFrame({
    'issuer': ['CONVERTIBLE CO', 'DISTRESSED INC'],
    'cusip': ['123456789', '987654321'],
    'ticker': ['CNVT', 'DSTR'],
    'value': [5000, 1000],
    'shares': [1000, 1000],
    'share_type': ['PRN', 'SH'],
    'discretion': ['SOLE', 'SOLE']
})

# Represents Quarter 1 (Previous)
MOCK_INFOTABLE_Q1 = pd.DataFrame({
    'issuer': ['DISTRESSED INC'],
    'cusip': ['987654321'],
    'ticker': ['DSTR'],
    'value': [800],
    'shares': [800],
    'share_type': ['SH'],
    'discretion': ['SOLE']
})

@pytest.fixture
def mock_scanner():
    """Returns a WhaleScanner instance with a mock identity."""
    return WhaleScanner(user_agent="Test Agent <test@example.com>")

@patch("src.market_mayhem.scanners.WhaleScanner._fetch_filings")
@patch("src.market_mayhem.scanners.WhaleScanner._parse_13f_xml")
def test_vulture_entry_signal(mock_parse, mock_fetch, mock_scanner):
    """
    Verifies that a new position (present in Q0, absent in Q1)
    triggers a VULTURE_ENTRY signal.
    """
    # Setup Mocks
    mock_filing_q0 = MagicMock()
    mock_filing_q1 = MagicMock()
    mock_fetch.return_value = [mock_filing_q0, mock_filing_q1]

    # Mock parse to return our DataFrames
    # First call returns Q0, second call returns Q1
    mock_parse.side_effect = [MOCK_INFOTABLE_Q0, MOCK_INFOTABLE_Q1]

    # Execute
    signals = mock_scanner.calculate_fund_sentiment("OAKTREE")

    # Verify
    assert len(signals) >= 1

    # Check New Entry Signal
    entry_signal = next((s for s in signals if s.ticker == "CNVT"), None)
    assert entry_signal is not None
    assert entry_signal.signal_type == "VULTURE_ENTRY"
    assert entry_signal.share_type == "PRN"
    assert "New Position" in entry_signal.description

@patch("src.market_mayhem.scanners.WhaleScanner._fetch_filings")
@patch("src.market_mayhem.scanners.WhaleScanner._parse_13f_xml")
def test_accumulation_signal(mock_parse, mock_fetch, mock_scanner):
    """
    Verifies that an increased position (>20%) triggers an ACCUMULATION signal.
    """
    # Setup Mocks
    mock_fetch.return_value = [MagicMock(), MagicMock()]
    mock_parse.side_effect = [MOCK_INFOTABLE_Q0, MOCK_INFOTABLE_Q1]

    # Execute
    signals = mock_scanner.calculate_fund_sentiment("OAKTREE")

    # Check Accumulation Signal
    # DSTR shares went from 800 (Q1) to 1000 (Q0) -> +25%
    acc_signal = next((s for s in signals if s.ticker == "DSTR"), None)
    assert acc_signal is not None
    assert acc_signal.signal_type == "ACCUMULATION"
    assert acc_signal.change_pct == 25.0
    assert acc_signal.share_type == "SH"

def test_invalid_fund_key(mock_scanner):
    """Verifies error handling for unknown funds."""
    with pytest.raises(ValueError):
        mock_scanner.calculate_fund_sentiment("UNKNOWN_FUND")
