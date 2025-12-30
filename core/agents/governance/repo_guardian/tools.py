import os
import subprocess
from typing import List, Dict, Any, Optional
import logging

class GitTools:
    """
    A wrapper for git operations to inspect the repo state.
    """

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path

    def _run_command(self, args: List[str]) -> str:
        try:
            result = subprocess.run(
                args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Git command failed: {e.cmd}, stderr: {e.stderr}")
            raise

    def get_current_branch(self) -> str:
        return self._run_command(["git", "branch", "--show-current"])

    def get_diff(self, target_branch: str = "main") -> str:
        """Gets the diff between current HEAD and target branch."""
        return self._run_command(["git", "diff", target_branch])

    def get_file_content(self, filepath: str, revision: str = "HEAD") -> str:
        """Gets file content at a specific revision."""
        return self._run_command(["git", "show", f"{revision}:{filepath}"])

    def list_files(self, path: str = ".") -> List[str]:
        output = self._run_command(["git", "ls-files", path])
        return output.splitlines()

class StaticAnalyzer:
    """
    Simple static analysis tools.
    """

    @staticmethod
    def count_loc(content: str) -> int:
        return len(content.splitlines())

    @staticmethod
    def check_docstrings(content: str) -> bool:
        """Naive check for docstrings in python code."""
        # This is very basic; a real implementation would use AST.
        lines = content.splitlines()
        for line in lines:
            if '"""' in line or "'''" in line:
                return True
        return False

    @staticmethod
    def check_imports(content: str) -> List[str]:
        """Extracts imports."""
        imports = []
        for line in content.splitlines():
            if line.startswith("import ") or line.startswith("from "):
                imports.append(line)
        return imports
