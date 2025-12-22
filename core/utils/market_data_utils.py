
import numpy as np
import pandas as pd
import json
from datetime import datetime


def convert_to_python_types(data):
    """
    Recursively converts numpy types and pandas timestamps to standard Python types.
    """
    if isinstance(data, dict):
        return {k: convert_to_python_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(v) for v in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return convert_to_python_types(data.tolist())
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        return data


def format_market_data_gold_standard(symbol: str, snapshot: dict, intraday: list, intra_year: list, long_term: list) -> dict:
    """
    Formats the market data into the Adam v23.5 Gold Standard structure.
    """

    # Clean data (remove NaNs, convert types)
    snapshot = convert_to_python_types(snapshot)
    intraday = convert_to_python_types(intraday)
    intra_year = convert_to_python_types(intra_year)
    long_term = convert_to_python_types(long_term)

    return {
        "meta": {
            "target": symbol,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model_version": "Adam-v23.5"
        },
        "nodes": {
            "market_data": {
                "snapshot": snapshot,
                "intraday": intraday,
                "intra_year": intra_year,
                "long_term": long_term
            }
        }
    }
