import ast
import os
import networkx as nx
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class RepoGraphBuilder:
    """
    Parses the repository codebase to build a semantic graph of Agents, Classes, and dependencies.
    Provides 'Self-Awareness' to the system.
    """
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.graph = nx.DiGraph()

    def build(self) -> nx.DiGraph:
        """
        Scans the codebase and populates the graph.
        """
        logger.info(f"Scanning repository from {self.root_dir}...")
        for root, _, files in os.walk(self.root_dir):
            if "venv" in root or ".git" in root or "__pycache__" in root:
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    self._process_file(file_path)

        self._analyze_relationships()
        return self.graph

    def _process_file(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            rel_path = os.path.relpath(file_path, self.root_dir)

            # Add File Node
            self.graph.add_node(rel_path, type="File", path=rel_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._process_class(node, rel_path)
                elif isinstance(node, ast.FunctionDef):
                    self._process_function(node, rel_path)

        except Exception as e:
            # logger.warning(f"Failed to process {file_path}: {e}")
            pass

    def _process_class(self, node: ast.ClassDef, file_path: str):
        class_name = node.name
        node_id = f"{file_path}::{class_name}"

        # Determine if it's an Agent
        is_agent = any(
            (isinstance(b, ast.Name) and 'Agent' in b.id) or
            (isinstance(b, ast.Attribute) and 'Agent' in b.attr)
            for b in node.bases
        )
        node_type = "Agent" if is_agent else "Class"

        docstring = ast.get_docstring(node)

        self.graph.add_node(node_id, type=node_type, name=class_name, doc=docstring, file=file_path)
        self.graph.add_edge(file_path, node_id, relation="defines")

        # Check base classes (inheritance)
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.graph.add_edge(node_id, base.id, relation="inherits_from")

    def _process_function(self, node: ast.FunctionDef, file_path: str):
        func_name = node.name
        node_id = f"{file_path}::{func_name}"
        self.graph.add_node(node_id, type="Function", name=func_name, file=file_path)
        self.graph.add_edge(file_path, node_id, relation="defines")

    def _analyze_relationships(self):
        # Placeholder for more complex dependency analysis (imports)
        pass

    def export_to_json(self) -> Dict[str, Any]:
        return nx.node_link_data(self.graph)
