import csv
import json

import pytest
import yaml

from core.system.error_handler import FileReadError, InvalidInputError
from core.utils.data_utils import DataLoader, load_data


@pytest.fixture
def temp_json_file(tmp_path):
    file_path = tmp_path / "test.json"
    with file_path.open("w") as f:
        json.dump({"key": "value"}, f)
    return file_path

@pytest.fixture
def temp_csv_file(tmp_path):
    file_path = tmp_path / "test.csv"
    with file_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["col1", "col2"])
        writer.writeheader()
        writer.writerow({"col1": "val1", "col2": "val2"})
    return file_path

@pytest.fixture
def temp_yaml_file(tmp_path):
    file_path = tmp_path / "test.yaml"
    with file_path.open("w") as f:
        yaml.dump({"key": "value"}, f)
    return file_path

@pytest.fixture
def temp_text_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is unstructured text.")
    return file_path

def test_load_json(temp_json_file):
    loader = DataLoader()
    data = loader.load({"type": "json", "path": str(temp_json_file)})
    assert data == {"key": "value"}

def test_load_json_invalid(tmp_path):
    file_path = tmp_path / "invalid.json"
    file_path.write_text("{invalid json")
    loader = DataLoader()
    with pytest.raises(FileReadError):
        loader.load({"type": "json", "path": str(file_path)})

def test_load_csv(temp_csv_file):
    loader = DataLoader()
    data = loader.load({"type": "csv", "path": str(temp_csv_file)})
    assert data == [{"col1": "val1", "col2": "val2"}]

def test_load_csv_empty(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")
    loader = DataLoader()
    data = loader.load({"type": "csv", "path": str(file_path)})
    assert data == []

def test_load_yaml(temp_yaml_file):
    loader = DataLoader()
    data = loader.load({"type": "yaml", "path": str(temp_yaml_file)})
    assert data == {"key": "value"}

def test_load_smart_text(temp_text_file):
    loader = DataLoader()
    data = loader.load({"type": "smart_text", "path": str(temp_text_file)})
    assert "source_file" in data
    assert data["text_length"] > 0
    assert "extracted_entities" in data
    assert "summary" in data
    assert "rag_context" in data
    assert data["rag_context"]["vector_db_matches"] == 3
    assert data["rag_context"]["confidence_score"] == 0.92
    assert "finance" in data["rag_context"]["keywords"]

def test_load_yaml_invalid(tmp_path):
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text("invalid: [yaml: string")
    loader = DataLoader()
    with pytest.raises(FileReadError):
        loader.load({"type": "yaml", "path": str(file_path)})

def test_load_api_financial():
    loader = DataLoader()
    data = loader.load({"type": "api", "provider": "example_financial_data_api"})
    assert "income_statement" in data
    assert "revenue" in data["income_statement"]

def test_load_api_market():
    loader = DataLoader()
    data = loader.load({"type": "api", "provider": "example_market_data_api"})
    assert "market_trends" in data
    assert len(data["market_trends"]) == 2

def test_load_api_unknown():
    loader = DataLoader()
    data = loader.load({"type": "api", "provider": "unknown_provider"})
    assert data is None

def test_missing_path():
    loader = DataLoader()
    with pytest.raises(InvalidInputError, match="Missing 'path'"):
        loader.load({"type": "json"})

def test_unsupported_type(temp_json_file):
    loader = DataLoader()
    with pytest.raises(InvalidInputError, match="Unsupported data type"):
        loader.load({"type": "unsupported", "path": str(temp_json_file)})

def test_file_not_found():
    loader = DataLoader()
    with pytest.raises(FileReadError, match="File not found"):
        loader.load({"type": "json", "path": "nonexistent_file.json"})

def test_max_file_size(tmp_path):
    loader = DataLoader()
    file_path = tmp_path / "large_file.json"
    # Write a file that is larger than the 50MB limit (mock this by patching the size)
    with file_path.open("w") as f:
        f.write("{}")

    class MockStat:
        @property
        def st_size(self):
            return 51 * 1024 * 1024

    # Mocking the stat() return object to trick the size check
    with pytest.MonkeyPatch.context() as mp:
        import pathlib
        mp.setattr(pathlib.Path, "stat", lambda self, **kwargs: MockStat())
        with pytest.raises(FileReadError, match="File exceeds maximum size limit"):
            loader.load({"type": "json", "path": str(file_path)})

def test_caching(temp_json_file):
    loader = DataLoader(use_cache=True)
    # First load reads from file
    data1 = loader.load({"type": "json", "path": str(temp_json_file)})
    assert data1 == {"key": "value"}

    # Modify the file
    with temp_json_file.open("w") as f:
        json.dump({"key": "new_value"}, f)

    # Second load should use cache, returning original value
    data2 = loader.load({"type": "json", "path": str(temp_json_file)})
    assert data2 == {"key": "value"}

def test_no_caching(temp_json_file):
    loader = DataLoader(use_cache=False)
    # First load reads from file
    data1 = loader.load({"type": "json", "path": str(temp_json_file)})
    assert data1 == {"key": "value"}

    # Modify the file
    with temp_json_file.open("w") as f:
        json.dump({"key": "new_value"}, f)

    # Second load should read from file
    data2 = loader.load({"type": "json", "path": str(temp_json_file)})
    assert data2 == {"key": "new_value"}

def test_load_data_wrapper(temp_json_file):
    data = load_data({"type": "json", "path": str(temp_json_file)})
    assert data == {"key": "value"}
