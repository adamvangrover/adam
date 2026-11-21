import pytest
from core.utils.repo_compiler.scanner import RepoScanner
from core.utils.repo_compiler.chunker import RepoChunker
from core.utils.repo_compiler.formatter import PromptFormatter
from core.utils.repo_compiler.models import FileDocument

def test_chunker_directory():
    docs = [
        FileDocument(path="core/utils/test.py", content="print('test')", size=14, lines=1, extension=".py"),
        FileDocument(path="core/models/model.py", content="class Model: pass", size=17, lines=1, extension=".py"),
        FileDocument(path="scripts/run.py", content="pass", size=4, lines=1, extension=".py"),
        FileDocument(path="README.md", content="Read me", size=7, lines=1, extension=".md")
    ]

    chunker = RepoChunker()
    chunks = chunker.chunk_by_directory(docs)

    assert len(chunks) == 3
    dir_names = {c.chunk_id for c in chunks}
    assert dir_names == {"core", "scripts", "root"}

def test_formatter_markdown():
    doc = FileDocument(path="src/main.py", content="def main():\n    pass", size=20, lines=2, extension=".py")
    fmt = PromptFormatter()
    output = fmt.format_document_markdown(doc)

    assert "### File: `src/main.py`" in output
    assert "```py" in output
    assert "def main():" in output

def test_formatter_xml():
    doc = FileDocument(path="src/main.py", content="def main():\n    pass", size=20, lines=2, extension=".py")
    fmt = PromptFormatter()
    output = fmt.format_document_xml(doc)

    assert '<file path="src/main.py">' in output
    assert "def main():" in output
    assert '</file>' in output
