# tests/test_json_file_source_upgraded.py

import json
from pathlib import Path
import pytest

from core.data_access.json_file_source import JsonFileSource
from core.system.error_handler import FileReadError, InvalidInputError

@pytest.fixture
def temp_data_dir(tmp_path):
    """Fixture providing a temporary directory with mocked financial JSON files."""
    base = tmp_path / "data"
    base.mkdir()

    # Mock valid financial statements
    fin_path = base / "TESTCO_financials.json"
    fin_path.write_text(json.dumps({"revenue": 1000, "profit": 200}))

    # Mock valid historical prices
    prices_path = base / "TESTCO_prices.json"
    prices_path.write_text(json.dumps({
        "prices": [
            {"date": "2023-01-01", "price": 100},
            {"date": "2023-01-05", "price": 110},
            {"date": "2023-01-10", "price": 105}
        ]
    }))

    # Mock invalid list-based JSON
    invalid_path = base / "INVALID_financials.json"
    invalid_path.write_text(json.dumps([1, 2, 3]))

    # Mock malformed JSON
    malformed_path = base / "MALFORMED_financials.json"
    malformed_path.write_text("{malformed: true")

    return str(base)

def test_load_valid_financials(temp_data_dir):
    source = JsonFileSource(base_path=temp_data_dir)
    data = source.get_financial_statements("TESTCO")
    assert data == {"revenue": 1000, "profit": 200}

def test_load_missing_file(temp_data_dir):
    source = JsonFileSource(base_path=temp_data_dir)
    with pytest.raises(FileReadError):
        source.get_financial_statements("MISSING")

def test_load_invalid_json_structure(temp_data_dir):
    source = JsonFileSource(base_path=temp_data_dir)
    with pytest.raises(InvalidInputError):
        source.get_financial_statements("INVALID")

def test_load_malformed_json(temp_data_dir):
    source = JsonFileSource(base_path=temp_data_dir)
    with pytest.raises(FileReadError):
        source.get_financial_statements("MALFORMED")

def test_historical_prices_filtering(temp_data_dir):
    source = JsonFileSource(base_path=temp_data_dir)
    data = source.get_historical_prices("TESTCO", "2023-01-02", "2023-01-10")

    assert data is not None
    assert "2023-01-01" not in data
    assert "2023-01-05" in data
    assert data["2023-01-05"] == 110
    assert data["2023-01-10"] == 105
    assert len(data) == 2
