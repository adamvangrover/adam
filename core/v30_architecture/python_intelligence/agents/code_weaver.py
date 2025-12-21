import ast
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger("CodeWeaver")

@dataclass
class TechnicalDebtIssue:
    """Standardized representation of a detected technical debt item."""
    file_path: str
    line_number: int
    issue_type: str
    description: str
    severity: str = "medium"
    context_snippet: Optional[str] = None

class CodeWeaverAgent:
    """
    Appendix C: "Code Weaver" Self-Maintenance Agent v2.0.
    
    A specialized agent that scans the Monorepo for technical debt.
    It uses Abstract Syntax Trees (AST) for deep Python analysis and 
    Regex patterns for polyglot support (Rust, JS, Markdown).
    """
    
    def __init__(self, repo_path="./"):
        self.repo_path = repo_path
        # Standard noise filters
        self.ignore_dirs = {'.git', '__pycache__', '.venv', 'node_modules', 'dist', 'build'}

    def scan_for_debt(self) -> List[Dict[str, Any]]:
        """
        Orchestrates the scanning process.
        Routes files to specific analyzers based on extension.
        """
        logger.info("Code Weaver: Scanning for technical debt (AST + Pattern Match)...")
        issues: List[TechnicalDebtIssue] = []

        for root, dirs, files in os.walk(self.repo_path):
            # 1. Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            # 2. Scope Check (Preserved from v30 architecture requirements)
            # In a real run, you might remove this check to scan the whole repo.
            if "v30_architecture" in root or self.repo_path == "./": 
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Route to Python AST Analyzer
                    if file.endswith(".py"):
                        issues.extend(self._analyze_python_ast(file_path))
                    
                    # Route to Polyglot Text Analyzer
                    elif file.endswith((".rs", ".js", ".ts", ".md", ".txt")):
                        issues.extend(self._analyze_text_patterns(file_path))

        logger.info(f"Code Weaver: Scan complete. Found {len(issues)} issues.")
        # Return serializable dicts for downstream agents/JSON consumption
        return [asdict(issue) for issue in issues]

    def _analyze_python_ast(self, filepath: str) -> List[TechnicalDebtIssue]:
        """
        Performs deep structural analysis using Python's AST.
        Detects deprecated libraries and dangerous coding patterns.
        """
        issues = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=filepath)
            
            for node in ast.walk(tree):
                # Check 1: Deprecated Dependencies (e.g., TensorFlow -> PyTorch migration)
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module_name = node.module if isinstance(node, ast.ImportFrom) else None
                    
                    # Check 'from tensorflow import ...'
                    if module_name == 'tensorflow':
                        issues.append(self._create_issue(filepath, node.lineno, "DEPRECATED_DEPENDENCY", 
                            "TensorFlow is deprecated in v30; migrate to PyTorch/JAX.", "high"))
                    
                    # Check 'import tensorflow as tf'
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == 'tensorflow':
                                issues.append(self._create_issue(filepath, node.lineno, "DEPRECATED_DEPENDENCY", 
                                    "TensorFlow is deprecated in v30.", "high"))

                # Check 2: Broad Exception Handling (Pokemon Exception Handling)
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None or (isinstance(node.type, ast.Name) and node.type.id == 'Exception'):
                        issues.append(self._create_issue(filepath, node.lineno, "BROAD_EXCEPTION", 
                            "Avoid catching generic Exception; specify error types.", "low"))

            # Check 3: Fallback to text analysis for comments (AST strips comments)
            issues.extend(self._analyze_text_patterns(filepath))

        except SyntaxError as e:
            logger.error(f"Syntax error parsing {filepath}: {e}")
        except Exception as e:
            logger.error(f"Failed to analyze AST for {filepath}: {e}")
        
        return issues

    def _analyze_text_patterns(self, filepath: str) -> List[TechnicalDebtIssue]:
        """
        Regex-based scanner for comments and non-Python files.
        """
        issues = []
        # Compile patterns once for efficiency
        todo_pattern = re.compile(r'(#|//)\s*TODO', re.IGNORECASE)
        hack_pattern = re.compile(r'(#|//)\s*HACK', re.IGNORECASE)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    
                    if todo_pattern.search(line):
                        issues.append(self._create_issue(filepath, i + 1, "TODO_COMMENT", 
                            line.strip(), "low", line.strip()))
                            
                    if hack_pattern.search(line):
                        issues.append(self._create_issue(filepath, i + 1, "HACK_PATTERN", 
                            "Temporary hack detected; requires refactoring.", "medium", line.strip()))
                            
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            
        return issues

    def _create_issue(self, path, line, type, desc, severity, context=None):
        """Helper to construct the data class."""
        return TechnicalDebtIssue(
            file_path=path,
            line_number=line,
            issue_type=type,
            description=desc,
            severity=severity,
            context_snippet=context
        )

    def generate_fix_pr(self, issue: Dict[str, Any]) -> str:
        """
        Generates a Pull Request structure to fix the identified debt.
        (Simulates an LLM call).
        """
        file_path = issue.get('file_path', 'unknown')
        line_num = issue.get('line_number', '?')
        issue_type = issue.get('issue_type', 'issue')
        
        logger.info(f"Generating PR Plan for {issue_type} in {file_path}")
        
        pr_content = (
            f"PR-AUTOGEN: fix(tech-debt): Resolve {issue_type} in {os.path.basename(file_path)}\n"
            f"--------------------------------------------------------------------------\n"
            f"**Target:** `{file_path}:{line_num}`\n"
            f"**Description:** {issue.get('description')}\n"
            f"\n"
            f"**Proposed Action Plan:**\n"
            f"1.  Navigate to line {line_num}.\n"
            f"2.  Refactor code to remove '{issue_type}'.\n"
            f"3.  Run `pytest {file_path}` to ensure no regression.\n"
        )
        return pr_content

# --- Integration Test ---
if __name__ == "__main__":
    # Create a dummy file to test the agent logic immediately
    test_file = "v30_test_debt.py"
    with open(test_file, "w") as f:
        f.write("import tensorflow as tf\n# TODO: Refactor this logic\ntry:\n    x = 1/0\nexcept Exception:\n    pass")
    
    try:
        agent = CodeWeaverAgent()
        found_issues = agent.scan_for_debt()
        
        print("\n=== Scan Results ===")
        for iss in found_issues:
            print(f"[{iss['issue_type']}] {iss['file_path']}:{iss['line_number']} -> {iss['description']}")
        
        if found_issues:
            print("\n=== Automated Fix Proposal ===")
            print(agent.generate_fix_pr(found_issues[0]))
            
    finally:
        # Clean up artifact
        if os.path.exists(test_file):
            os.remove(test_file)
