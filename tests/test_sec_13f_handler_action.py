import pytest
import pandas as pd
from core.vertical_risk_agent.ingestion.sec_13f_handler import Sec13FHandler

def test_determine_action():
    handler = Sec13FHandler()

    # Create test data spanning all conditions
    current_df = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
        'shares': [100, 0, 150, 50, 0]
    })

    previous_df = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NFLX'],
        'shares': [0, 100, 100, 100, 0]
    })

    # Calculate delta
    result = handler.calculate_delta(current_df, previous_df)

    # Assertions
    # AAPL: prev 0, curr 100 -> NEW
    assert result.loc[result['ticker'] == 'AAPL', 'action_calculated'].iloc[0] == 'NEW'

    # GOOGL: prev 100, curr 0 -> EXIT
    assert result.loc[result['ticker'] == 'GOOGL', 'action_calculated'].iloc[0] == 'EXIT'

    # MSFT: prev 100, curr 150 -> ADD
    assert result.loc[result['ticker'] == 'MSFT', 'action_calculated'].iloc[0] == 'ADD'

    # AMZN: prev 100, curr 50 -> REDUCE
    assert result.loc[result['ticker'] == 'AMZN', 'action_calculated'].iloc[0] == 'REDUCE'

    # META: prev 0 (NaN->0), curr 0 -> HOLD
    assert result.loc[result['ticker'] == 'META', 'action_calculated'].iloc[0] == 'HOLD'

    # NFLX: prev 0, curr 0 (NaN->0) -> HOLD
    assert result.loc[result['ticker'] == 'NFLX', 'action_calculated'].iloc[0] == 'HOLD'
