#!/usr/bin/env python3
import json
import os
import networkx as nx

class OmniGraphLoader:
    def __init__(self, base_path="data/omni_graph"):
        self.base_path = base_path
        self.graph = nx.DiGraph()

    def load_universe(self):
        # 1. Load Hero Dossiers (Rich Nodes)
        dossier_path = os.path.join(self.base_path, "dossiers")
        if os.path.exists(dossier_path):
            for filename in os.listdir(dossier_path):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(dossier_path, filename)) as f:
                            data = json.load(f)
                            # Handle different structures if necessary, but assuming v23.5 schema
                            if 'meta' in data and 'target' in data['meta']:
                                node_id = data['meta']['target']
                            elif 'nodes' in data and 'entity_ecosystem' in data['nodes']:
                                node_id = data['nodes']['entity_ecosystem']['legal_entity']['name']
                            else:
                                node_id = "UNKNOWN_" + filename

                            # Store the FULL schema object in the node
                            self.graph.add_node(node_id, type="HERO", data=data)
                            print(f"Loaded Hero: {node_id}")
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")

    def load_constellations(self):
        # 2. Load Satellite Nodes
        const_path = os.path.join(self.base_path, "constellations")
        if os.path.exists(const_path):
            for filename in os.listdir(const_path):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(const_path, filename)) as f:
                            nodes = json.load(f)
                            if isinstance(nodes, list):
                                for n in nodes:
                                    if 'id' in n:
                                        self.graph.add_node(n['id'], type="SATELLITE", **n)
                                        # Add simple edge if relationship exists
                                        if 'relationship_to_hero' in n:
                                            # In a real app, you'd find the hero node ID dynamically
                                            # For now, we just note it
                                            pass
                                    else:
                                        print(f"Skipping node in {filename}: Missing 'id'")
                            else:
                                print(f"Skipping {filename}: Not a list of nodes")
                    except Exception as e:
                         print(f"Error loading {filename}: {e}")

    def load_relationships(self):
        # 3. Load Relationship Edges
        rel_path = os.path.join(self.base_path, "relationships")
        if os.path.exists(rel_path):
            for filename in os.listdir(rel_path):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(rel_path, filename)) as f:
                            data = json.load(f)
                            if 'edges' in data:
                                for edge in data['edges']:
                                    source = edge.get('source')
                                    target = edge.get('target')
                                    if source and target:
                                        self.graph.add_edge(source, target, **edge)
                    except Exception as e:
                        print(f"Error loading relationships from {filename}: {e}")

    def export_for_ui(self):
        # Convert to D3.js compatible JSON
        return nx.node_link_data(self.graph)

if __name__ == "__main__":
    loader = OmniGraphLoader()
    loader.load_universe()
    loader.load_constellations()
    loader.load_relationships()
    print(f"Graph constructed with {loader.graph.number_of_nodes()} nodes and {loader.graph.number_of_edges()} edges.")
