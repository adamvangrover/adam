import pytest
import json
import csv
import yaml
from pathlib import Path
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

def test_load_smart_text(temp_files):
    loader = DataLoader(use_cache=False)
    data = loader.load({"type": "smart_text", "path": temp_files["smart_text"]})
    assert "source_file" in data
    assert "summary" in data
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
