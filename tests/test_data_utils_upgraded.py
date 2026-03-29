import pytest
import json
import csv
import yaml
from pathlib import Path
from collections import OrderedDict
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

def test_lru_cache_eviction(tmp_path, monkeypatch):
    """
    Tests the new O(1) LRU caching behavior using OrderedDict.
    Verifies that the oldest item is evicted and that recently
    accessed items are bumped to the end of the eviction queue.
    """
    monkeypatch.setattr(DataLoader, "MAX_CACHE_ENTRIES", 3)
    loader = DataLoader(use_cache=True)

    # Create 4 dummy json files
    files = []
    for i in range(4):
        p = tmp_path / f"test_{i}.json"
        p.write_text(f'{{"id": {i}}}')
        files.append(str(p))

    # Load first 3
    loader.load({"type": "json", "path": files[0]})
    loader.load({"type": "json", "path": files[1]})
    loader.load({"type": "json", "path": files[2]})

    # Cache should be exactly 3 items long
    assert len(loader._data_cache) == 3

    # Re-access the oldest item (files[0]), making it the NEWEST accessed
    loader.load({"type": "json", "path": files[0]})

    # Load the 4th item, which should evict the OLDEST (which is now files[1], because files[0] was bumped)
    loader.load({"type": "json", "path": files[3]})

    # Verify files[1] was evicted and the cache size remains 3
    assert len(loader._data_cache) == 3
    resolved_files = [str(Path(f).resolve()) for f in files]

    assert resolved_files[1] not in loader._data_cache
    assert resolved_files[0] in loader._data_cache
    assert resolved_files[2] in loader._data_cache
    assert resolved_files[3] in loader._data_cache

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
