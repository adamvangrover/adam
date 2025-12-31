import os
import subprocess
import ast
import re
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

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
            logger.error(f"Git command failed: {e.cmd}, stderr: {e.stderr}")
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

class SecurityScanner:
    """
    Scans content for security vulnerabilities and secrets.
    """

    SECRET_PATTERNS = {
        "aws_access_key": r"AKIA[0-9A-Z]{16}",
        "google_api_key": r"AIza[0-9A-Za-z\\-_]{35}",
        "private_key": r"-----BEGIN [A-Z]+ PRIVATE KEY-----",
        "generic_secret": r"(?i)(password|secret|token|key)\s*=\s*['\"][^\s'\"]+['\"]",
        "slack_token": r"xox[baprs]-([0-9a-zA-Z]{10,48})?",
        "github_token": r"gh[pousr]_[A-Za-z0-9_]{36,255}",
    }

    @staticmethod
    def scan_content(content: str) -> List[Dict[str, str]]:
        """
        Scans a string for secrets.
        Returns a list of findings: {"type": ..., "match": ...}
        """
        findings = []
        for name, pattern in SecurityScanner.SECRET_PATTERNS.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                # Redact the match for reporting
                matched_text = match.group(0)
                redacted = matched_text[:4] + "***" + matched_text[-4:] if len(matched_text) > 8 else "***"
                findings.append({
                    "type": name,
                    "snippet": redacted
                })
        return findings

class StaticAnalyzer:
    """
    Advanced static analysis tools using AST.
    """

    DANGEROUS_FUNCTIONS = {"eval", "exec", "os.system", "subprocess.call", "subprocess.Popen", "input"}

    @staticmethod
    def count_loc(content: str) -> int:
        return len(content.splitlines())

    @staticmethod
    def analyze_python_code(content: str, filepath: str = "unknown") -> Dict[str, Any]:
        """
        Performs AST analysis on Python code.
        Returns a report of issues.
        """
        report = {
            "missing_docstrings": [],
            "missing_type_hints": [],
            "dangerous_functions": [],
            "imports": [],
            "classes": [],
            "functions": []
        }

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"SyntaxError parsing {filepath}: {e}")
            return report

        for node in ast.walk(tree):
            # 1. Imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        report["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                     module = node.module or ""
                     for alias in node.names:
                         report["imports"].append(f"{module}.{alias.name}")

            # 2. Function Definitions
            if isinstance(node, ast.FunctionDef):
                report["functions"].append(node.name)

                # Check Docstring
                if not ast.get_docstring(node):
                    report["missing_docstrings"].append(f"Function: {node.name}")

                # Check Return Type Hint (skip __init__)
                if node.name != "__init__" and node.returns is None:
                    report["missing_type_hints"].append(f"Function Return: {node.name}")

                # Check Argument Type Hints
                for arg in node.args.args:
                    if arg.arg != "self" and arg.arg != "cls" and arg.annotation is None:
                         report["missing_type_hints"].append(f"Function Arg: {node.name}.{arg.arg}")

            # 3. Class Definitions
            if isinstance(node, ast.ClassDef):
                report["classes"].append(node.name)
                if not ast.get_docstring(node):
                    report["missing_docstrings"].append(f"Class: {node.name}")

            # 4. Dangerous Functions
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in StaticAnalyzer.DANGEROUS_FUNCTIONS:
                        report["dangerous_functions"].append(f"Line {node.lineno}: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    # Handle module.func calls like os.system
                    try:
                        full_name = f"{node.func.value.id}.{node.func.attr}" # type: ignore
                        if full_name in StaticAnalyzer.DANGEROUS_FUNCTIONS:
                            report["dangerous_functions"].append(f"Line {node.lineno}: {full_name}")
                    except AttributeError:
                        pass

        return report
