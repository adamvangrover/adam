import json
from pathlib import Path

import pytest

from core.data_processing.universal_ingestor import ArtifactType, UniversalIngestor


@pytest.fixture
def temp_workspace(tmp_path):
    # Setup mock files
    json_file = tmp_path / "data.json"
    json_file.write_text('{"title": "Test Report", "data": [1, 2, 3]}', encoding="utf-8")

    jsonl_file = tmp_path / "records.jsonl"
    jsonl_file.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    md_file = tmp_path / "prompt.md"
    md_file.write_text("# Main Header\nThis is a test prompt.\n## Subheader\nMore text.", encoding="utf-8")

    py_file = tmp_path / "script.py"
    py_file.write_text('"""Docstring here"""\ndef my_func():\n    pass', encoding="utf-8")

    txt_file = tmp_path / "log.txt"
    txt_file.write_text("Some random text logs.", encoding="utf-8")

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json format", encoding="utf-8")

    return tmp_path

def test_ingestor_scan_directory(temp_workspace):
    ingestor = UniversalIngestor()
    ingestor.scan_directory(str(temp_workspace))

    # Expect 5 valid artifacts (bad.json is skipped)
    assert len(ingestor.artifacts) == 5

    # Check types
    types = [a.type for a in ingestor.artifacts]
    assert ArtifactType.DATA.value in types
    assert ArtifactType.PROMPT.value in types
    assert ArtifactType.CODE_DOC.value in types
    assert ArtifactType.UNKNOWN.value in types

def test_markdown_chunking(temp_workspace):
    ingestor = UniversalIngestor()
    ingestor.process_file(str(temp_workspace / "prompt.md"))

    assert len(ingestor.artifacts) == 1
    artifact = ingestor.artifacts[0]

    # Verify semantic chunking feature
    assert "semantic_chunks" in artifact.metadata
    assert artifact.metadata["semantic_chunks"] == 2

    assert "chunks" in artifact.content
    assert len(artifact.content["chunks"]) == 2
    assert "Main Header" in artifact.content["chunks"][0]
    assert "Subheader" in artifact.content["chunks"][1]

def test_save_to_jsonl(temp_workspace):
    ingestor = UniversalIngestor()
    ingestor.process_file(str(temp_workspace / "data.json"))
    ingestor.process_file(str(temp_workspace / "records.jsonl"))

    output_file = temp_workspace / "output" / "out.jsonl"
    ingestor.save_to_jsonl(str(output_file))

    assert output_file.exists()

    with output_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 2
        data1 = json.loads(lines[0])
        assert "id" in data1
        assert "source_path" in data1

def test_large_file_skip(temp_workspace, monkeypatch):
    ingestor = UniversalIngestor()
    file_path = temp_workspace / "huge.txt"
    file_path.write_text("a", encoding="utf-8")

    # Mock stat to return a size > 10MB
    class MockStat:
        @property
        def st_size(self):
            return 11 * 1024 * 1024

    original_stat = Path.stat
    def mock_stat(self):
        if self.name == "huge.txt":
            return MockStat()
        return original_stat(self)

    monkeypatch.setattr(Path, "stat", mock_stat)

    ingestor.process_file(str(file_path))
    assert len(ingestor.artifacts) == 0  # Should be skipped due to size limits
