import json
import csv
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from core.utils.data_utils import DataLoader, load_data
from core.system.error_handler import FileReadError, InvalidInputError

@pytest.fixture
def temp_json_file(tmp_path):
    file_path = tmp_path / "data.json"
    data = {"key": "value"}
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def temp_csv_file(tmp_path):
    file_path = tmp_path / "data.csv"
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["col1", "col2"])
        writer.writeheader()
        writer.writerow({"col1": "val1", "col2": "val2"})
    return file_path

@pytest.fixture
def temp_yaml_file(tmp_path):
    file_path = tmp_path / "data.yaml"
    data = {"yaml_key": "yaml_value"}
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path

@pytest.fixture
def temp_smart_text_file(tmp_path):
    file_path = tmp_path / "data.txt"
    text = "MockCompany Inc. CEO John Doe reported Q3 Revenue of $50M."
    with open(file_path, "w") as f:
        f.write(text)
    return file_path

def test_load_json_success(temp_json_file):
    loader = DataLoader(use_cache=False)
    result = loader.load({"type": "json", "path": str(temp_json_file)})
    assert result == {"key": "value"}

def test_load_csv_success(temp_csv_file):
    loader = DataLoader(use_cache=False)
    result = loader.load({"type": "csv", "path": str(temp_csv_file)})
    assert result == [{"col1": "val1", "col2": "val2"}]

def test_load_yaml_success(temp_yaml_file):
    loader = DataLoader(use_cache=False)
    result = loader.load({"type": "yaml", "path": str(temp_yaml_file)})
    assert result == {"yaml_key": "yaml_value"}

def test_load_smart_text_success(temp_smart_text_file):
    loader = DataLoader(use_cache=False)
    result = loader.load({"type": "smart_text", "path": str(temp_smart_text_file)})

    assert "source_file" in result
    assert result["source_file"] == str(temp_smart_text_file)
    assert result["text_length"] == 58
    assert result["extracted_entities"] == ["MockCompany Inc.", "CEO John Doe", "Q3 Revenue $50M"]

def test_load_api_financial_data():
    loader = DataLoader()
    result = loader.load({"type": "api", "provider": "example_financial_data_api"})

    assert "income_statement" in result
    assert result["income_statement"]["revenue"] == [1000, 1100, 1250]

def test_load_api_market_data():
    loader = DataLoader()
    result = loader.load({"type": "api", "provider": "example_market_data_api"})

    assert "market_trends" in result
    assert result["market_trends"][0]["sector"] == "healthcare"

def test_load_api_unknown_provider():
    loader = DataLoader()
    result = loader.load({"type": "api", "provider": "unknown_provider"})
    assert result is None

def test_load_missing_path():
    loader = DataLoader()
    with pytest.raises(InvalidInputError, match="Missing 'path' in source_config"):
        loader.load({"type": "json"})

def test_load_unsupported_type(temp_json_file):
    loader = DataLoader()
    with pytest.raises(InvalidInputError, match="Unsupported data type: xml"):
        loader.load({"type": "xml", "path": str(temp_json_file)})

def test_load_file_not_found(tmp_path):
    loader = DataLoader()
    file_path = tmp_path / "nonexistent.json"
    with pytest.raises(FileReadError, match="File not found"):
        loader.load({"type": "json", "path": str(file_path)})

def test_load_file_size_limit_exceeded(tmp_path):
    loader = DataLoader()
    file_path = tmp_path / "large.json"

    with open(file_path, "wb") as f:
        f.truncate(DataLoader.MAX_FILE_SIZE_BYTES + 1)

    with pytest.raises(FileReadError, match="File exceeds maximum size limit"):
        loader.load({"type": "json", "path": str(file_path)})

def test_caching_behavior(temp_json_file):
    loader = DataLoader(use_cache=True)

    result1 = loader.load({"type": "json", "path": str(temp_json_file)})
    assert result1 == {"key": "value"}

    with open(temp_json_file, "w") as f:
        json.dump({"new_key": "new_value"}, f)

    result2 = loader.load({"type": "json", "path": str(temp_json_file)})
    assert result2 == {"key": "value"}

def test_caching_disabled(temp_json_file):
    loader = DataLoader(use_cache=False)

    result1 = loader.load({"type": "json", "path": str(temp_json_file)})
    assert result1 == {"key": "value"}

    with open(temp_json_file, "w") as f:
        json.dump({"new_key": "new_value"}, f)

    result2 = loader.load({"type": "json", "path": str(temp_json_file)})
    assert result2 == {"new_key": "new_value"}

def test_load_data_wrapper(temp_json_file):
    result = load_data({"type": "json", "path": str(temp_json_file)}, cache=False)
    assert result == {"key": "value"}
