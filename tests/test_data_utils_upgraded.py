import asyncio
import pytest
import json
import csv
import yaml
from pathlib import Path
from unittest.mock import patch
from core.utils.data_utils import DataLoader, load_data
from core.system.error_handler import FileReadError, InvalidInputError

@pytest.fixture
def temp_files(tmp_path):
    # Setup temp files
    json_path = tmp_path / "data.json"
    json_path.write_text('{"key": "value"}')

    csv_path = tmp_path / "data.csv"
    csv_path.write_text('col1,col2\nval1,val2')

    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text('key: value')

    txt_path = tmp_path / "data.txt"
    txt_path.write_text('Unstructured text data.')

    large_path = tmp_path / "large.json"
    large_path.write_text('{"key": "value"}')

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "yaml": str(yaml_path),
        "smart_text": str(txt_path),
        "large": str(large_path)
    }

def test_load_json(temp_files):
    loader = DataLoader(use_cache=False)
    data = loader.load({"type": "json", "path": temp_files["json"]})
    assert data == {"key": "value"}

def test_load_csv(temp_files):
    loader = DataLoader(use_cache=False)
    data = loader.load({"type": "csv", "path": temp_files["csv"]})
    assert data == [{"col1": "val1", "col2": "val2"}]

def test_load_yaml(temp_files):
    loader = DataLoader(use_cache=False)
    data = loader.load({"type": "yaml", "path": temp_files["yaml"]})
    assert data == {"key": "value"}

@patch("core.llm_plugin.LLMPlugin.generate_text")
def test_load_smart_text_success(mock_generate_text, temp_files):
    # Mock successful LLM response containing JSON
    mock_generate_text.return_value = '```json\n{"extracted_entities": ["AI", "Parsing"], "summary": "Real summary"}\n```'
    loader = DataLoader(use_cache=False)
    data = loader.load({"type": "smart_text", "path": temp_files["smart_text"]})
    assert "source_file" in data
    assert data["summary"] == "Real summary"
    assert "AI" in data["extracted_entities"]
    assert data["text_length"] == 23

@patch("core.llm_plugin.LLMPlugin.generate_text")
def test_load_smart_text_fallback(mock_generate_text, temp_files):
    # Mock LLM failing with exception
    mock_generate_text.side_effect = Exception("API timeout")
    loader = DataLoader(use_cache=False)
    data = loader.load({"type": "smart_text", "path": temp_files["smart_text"]})

    # Assert fallback to mock behavior
    assert "source_file" in data
    assert "summary" in data
    assert data["summary"] == "This is a mock AI-generated summary of the unstructured text."
    assert data["text_length"] == 23

def test_caching(temp_files):
    loader = DataLoader(use_cache=True)
    data1 = loader.load({"type": "json", "path": temp_files["json"]})

    # Modify file directly, shouldn't affect cache
    Path(temp_files["json"]).write_text('{"key": "new_value"}')

    data2 = loader.load({"type": "json", "path": temp_files["json"]})
    assert data1 == data2
    assert data2 == {"key": "value"}

def test_invalid_input(temp_files):
    loader = DataLoader(use_cache=False)
    with pytest.raises(InvalidInputError):
        loader.load({"type": "unknown", "path": temp_files["json"]})
    with pytest.raises(InvalidInputError):
        loader.load({"type": "json"})

def test_file_not_found():
    loader = DataLoader(use_cache=False)
    with pytest.raises(FileReadError):
        loader.load({"type": "json", "path": "nonexistent.json"})

def test_file_size_limit(temp_files, monkeypatch):
    loader = DataLoader(use_cache=False)
    monkeypatch.setattr(DataLoader, "MAX_FILE_SIZE_BYTES", 5)
    with pytest.raises(FileReadError, match="maximum size limit"):
        loader.load({"type": "json", "path": temp_files["large"]})

def test_load_data_wrapper(temp_files):
    data = load_data({"type": "json", "path": temp_files["json"]}, cache=False)
    assert data == {"key": "value"}

@pytest.mark.asyncio
async def test_async_load(temp_files):
    loader = DataLoader(use_cache=True)
    data = await loader.async_load({"type": "json", "path": temp_files["json"]})
    assert data == {"key": "value"}
    # Verify caching in async mode
    assert loader._data_cache[temp_files["json"]] == {"key": "value"}

@patch("core.utils.data_utils.load_config")
def test_api_whitelisting_valid(mock_load_config):
    mock_load_config.return_value = {
        "data_sources": {
            "whitelisted_api_feeds": ["example_market_data_api"]
        }
    }
    loader = DataLoader(use_cache=True)
    data = loader.load({"type": "api", "provider": "example_market_data_api"})
    assert "market_trends" in data
    # Verify API result was cached
    assert "api:example_market_data_api" in loader._data_cache

    # Second call should fetch from cache, load_config should not be called again
    loader.load({"type": "api", "provider": "example_market_data_api"})
    mock_load_config.assert_called_once()


@patch("core.utils.data_utils.load_config")
def test_api_whitelisting_invalid(mock_load_config):
    mock_load_config.return_value = {
        "data_sources": {
            "whitelisted_api_feeds": ["some_other_api"]
        }
    }
    loader = DataLoader()
    with pytest.raises(InvalidInputError, match="is not whitelisted"):
        loader.load({"type": "api", "provider": "example_market_data_api"})


@patch("core.utils.data_utils.load_config")
def test_api_whitelisting_missing_config(mock_load_config):
    mock_load_config.return_value = None
    loader = DataLoader()
    with pytest.raises(InvalidInputError, match="Configuration missing"):
        loader.load({"type": "api", "provider": "example_market_data_api"})
