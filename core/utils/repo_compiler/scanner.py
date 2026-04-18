import os
import pathspec
from typing import List, Set
from core.utils.repo_compiler.models import FileDocument

# Default ignores
DEFAULT_IGNORES = [
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode",
    "dist", "build", ".pytest_cache", "site-packages", ".egg-info", "adam_project.egg-info",
    "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.exe", ".DS_Store"
]

# Standard binary extensions to ignore
BINARY_EXTENSIONS: Set[str] = {
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.webp',
    '.pdf', '.zip', '.tar', '.gz', '.7z', '.rar',
    '.parquet', '.h5', '.pkl', '.pt', '.pth', '.onnx', '.bin',
    '.db', '.sqlite', '.sqlite3',
    '.eot', '.ttf', '.woff', '.woff2',
    '.mp4', '.webm', '.mp3', '.wav'
}

class RepoScanner:
    """
    Scans a repository to collect file documents, honoring gitignore and filtering binaries.
    """
    def __init__(self, root_dir: str = ".", max_file_size_mb: float = 1.0):
        self.root_dir = os.path.abspath(root_dir)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.spec = self._load_gitignore()

    def _load_gitignore(self) -> pathspec.PathSpec:
        """Loads .gitignore rules and combines them with defaults."""
        patterns = list(DEFAULT_IGNORES)
        gitignore_path = os.path.join(self.root_dir, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                patterns.extend(f.readlines())
        return pathspec.PathSpec.from_lines('gitwildmatch', patterns)

    def _is_binary(self, filepath: str) -> bool:
        """Checks if a file is binary based on extension."""
        ext = os.path.splitext(filepath)[1].lower()
        return ext in BINARY_EXTENSIONS

    def scan(self) -> List[FileDocument]:
        """
        Scans the repository and returns a list of FileDocument objects.
        """
        documents = []
        for root, dirs, files in os.walk(self.root_dir):
            rel_root = os.path.relpath(root, self.root_dir)
            if rel_root == ".":
                rel_root = ""

            # Filter directories based on pathspec
            dirs[:] = [d for d in dirs if not self.spec.match_file(os.path.join(rel_root, d))]

            for file in files:
                rel_path = os.path.join(rel_root, file)
                full_path = os.path.join(root, file)

                # Skip ignored files
                if self.spec.match_file(rel_path):
                    continue

                # Skip binary files
                if self._is_binary(full_path):
                    continue

                try:
                    size = os.path.getsize(full_path)
                    if size > self.max_file_size_bytes:
                        # Skip files that are too large
                        continue

                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    lines = len(content.splitlines())
                    extension = os.path.splitext(file)[1].lower()

                    doc = FileDocument(
                        path=rel_path,
                        content=content,
                        size=size,
                        lines=lines,
                        extension=extension
                    )
                    documents.append(doc)

                except Exception as e:
                    # Ignore unreadable files (e.g. permission issues, symlinks to nowhere)
                    pass

        return documents
