#!/usr/bin/env python3
import json
import logging
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OmniGraphLoader:
    def __init__(self, base_path: str = "data/omni_graph"):
        """
        Initialize the OmniGraphLoader.
        
        Args:
            base_path (str): Relative or absolute path to the graph data directory.
        """
        self.base_path = Path(base_path)
        self.graph = nx.DiGraph()
        
        # Verify path exists
        if not self.base_path.exists():
            logger.warning(f"Base path {self.base_path} does not exist. Graph will be empty.")

    def _load_json_safe(self, file_path: Path) -> Optional[Union[Dict, List]]:
        """Helper to safely load JSON files with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error in {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading {file_path.name}: {e}")
        return None

    def _get_node_id(self, data: Dict, filename: str) -> Optional[str]:
        """
        Determines the canonical ID for a Hero node. 
        Prioritizes Ticker/Target from metadata, falls back to Legal Name.
        """
        # 1. Try Ticker/Target (Forward aligned for Financial Data)
        if 'meta' in data and 'target' in data['meta']:
            return data['meta']['target']
        
        # 2. Fallback to Entity Ecosystem Name
        try:
            return data['nodes']['entity_ecosystem']['legal_entity']['name']
        except KeyError:
            logger.warning(f"Could not resolve Node ID for {filename}. Skipping.")
            return None

    def load_universe(self):
        """
        Step 1: Load Hero Dossiers (Rich Nodes).
        These are the primary entities in the knowledge graph.
        """
        dossier_path = self.base_path / "dossiers"
        
        if not dossier_path.exists():
            logger.info("No dossiers directory found.")
            return

        count = 0
        for file_path in dossier_path.glob("*.json"):
            data = self._load_json_safe(file_path)
            if not data:
                continue

            node_id = self._get_node_id(data, file_path.name)
            
            if node_id:
                # Store the full rich data object
                self.graph.add_node(node_id, type="HERO", data=data, label=node_id)
                logger.debug(f"Loaded Hero Node: {node_id}")
                count += 1
        
        logger.info(f"Summary: Loaded {count} Hero dossiers.")

    def load_constellations(self):
        """
        Step 2: Load Satellite Nodes (Constellations).
        These provide context around Heroes (suppliers, competitors, etc.).
        """
        const_path = self.base_path / "constellations"
        
        if not const_path.exists():
            logger.info("No constellations directory found.")
            return

        count = 0
        for file_path in const_path.glob("*.json"):
            nodes = self._load_json_safe(file_path)
            
            if not isinstance(nodes, list):
                logger.warning(f"Skipping {file_path.name}: Expected list of nodes.")
                continue

            for n in nodes:
                node_id = n.get('id')
                if not node_id:
                    continue

                # MERGE LOGIC: Don't overwrite a HERO with a SATELLITE
                if self.graph.has_node(node_id):
                    if self.graph.nodes[node_id].get('type') == 'HERO':
                        # Update metadata tags if present, but keep HERO status
                        continue 
                
                # Add as new Satellite node
                self.graph.add_node(node_id, type="SATELLITE", **n)
                count += 1
                
                # Forward Alignment: Auto-create edge if implicit relationship exists
                if 'relationship_to_hero' in n:
                    # Logic to link back to specific heroes could go here
                    pass

        logger.info(f"Summary: Loaded {count} Satellite nodes.")

    def load_relationships(self):
        """
        Step 3: Load Explicit Edges (Relationships).
        Connects nodes to form the graph.
        """
        rel_path = self.base_path / "relationships"
        
        if not rel_path.exists():
            logger.info("No relationships directory found.")
            return

        count = 0
        for file_path in rel_path.glob("*.json"):
            edges = self._load_json_safe(file_path)
            
            if not isinstance(edges, list):
                logger.warning(f"Skipping {file_path.name}: Expected list of edges.")
                continue

            for e in edges:
                source = e.get('source')
                target = e.get('target')

                if source and target:
                    # Ensure nodes exist (create skeletons if they don't)
                    if not self.graph.has_node(source):
                        self.graph.add_node(source, type="GHOST", label=source)
                    if not self.graph.has_node(target):
                        self.graph.add_node(target, type="GHOST", label=target)

                    self.graph.add_edge(source, target, **e)
                    count += 1

        logger.info(f"Summary: Loaded {count} relationships.")

    def export_for_ui(self) -> Dict:
        """
        Exports the graph to a D3.js / React-Force-Graph compatible JSON format.
        """
        data = nx.node_link_data(self.graph)
        logger.info("Graph exported for UI rendering.")
        return data

    def run_pipeline(self):
        """Executes the full loading pipeline."""
        logger.info("Starting OmniGraph Loading Pipeline...")
        self.load_universe()
        self.load_constellations()
        self.load_relationships()
        logger.info(f"Pipeline Complete. Graph contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

if __name__ == "__main__":
    # Example usage
    loader = OmniGraphLoader()
    loader.run_pipeline()
    
    # Optional: serialization example
    # with open("omni_graph_output.json", "w") as f:
    #     json.dump(loader.export_for_ui(), f, indent=2)