import ast
import logging
from typing import List, Dict, Any
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class EvolutionaryOptimizer(AgentBase):
    """
    A Meta-Agent that analyzes the codebase (using AST) to suggest optimizations.
    It represents the 'Self-Improving' capability of the swarm.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "EvolutionaryOptimizer"

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """
        Statically analyzes a Python file for complexity and improvement opportunities.
        """
        try:
            with open(filepath, "r") as f:
                source = f.read()

            tree = ast.parse(source)

            analysis = {
                "filepath": filepath,
                "classes": [],
                "functions": [],
                "imports": [],
                "suggestions": []
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(node.name)
                    # Heuristic: Check for large functions
                    if len(node.body) > 50:
                        analysis["suggestions"].append(f"Function '{node.name}' is too long ({len(node.body)} lines). Consider refactoring.")
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    # tracking imports
                    pass

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze {filepath}: {e}")
            return {"error": str(e)}

    async def execute(self, target_directory: str = "core/agents") -> Dict[str, Any]:
        """
        Runs the optimizer across a directory.
        """
        import os

        report = {
            "scanned_files": 0,
            "suggestions": []
        }

        for root, _, files in os.walk(target_directory):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    analysis = self.analyze_file(full_path)
                    report["scanned_files"] += 1
                    if "suggestions" in analysis and analysis["suggestions"]:
                        report["suggestions"].extend(analysis["suggestions"])

        return report
