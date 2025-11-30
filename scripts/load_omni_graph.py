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
                    with open(os.path.join(dossier_path, filename)) as f:
                        data = json.load(f)
                        # Depending on structure, assuming root object or data['nodes']
                        # The snippet says: node_id = data['nodes']['entity_ecosystem']['legal_entity']['name']
                        try:
                            node_id = data['nodes']['entity_ecosystem']['legal_entity']['name']
                            self.graph.add_node(node_id, type="HERO", data=data)
                            print(f"Loaded Hero: {node_id}")
                        except KeyError as e:
                            print(f"Skipping {filename}: Missing key {e}")

    def load_constellations(self):
        # 2. Load Satellite Nodes
        const_path = os.path.join(self.base_path, "constellations")
        if os.path.exists(const_path):
            for filename in os.listdir(const_path):
                if filename.endswith(".json"):
                    with open(os.path.join(const_path, filename)) as f:
                        nodes = json.load(f)
                        if isinstance(nodes, list):
                            for n in nodes:
                                self.graph.add_node(n['id'], type="SATELLITE", **n)
                                # Add simple edge if relationship exists
                                if 'relationship_to_hero' in n:
                                    # In a real app, you'd find the hero node ID dynamically
                                    pass

    def load_relationships(self):
        # 3. Load Relationships
        rel_path = os.path.join(self.base_path, "relationships")
        if os.path.exists(rel_path):
            for filename in os.listdir(rel_path):
                if filename.endswith(".json"):
                    with open(os.path.join(rel_path, filename)) as f:
                        rels = json.load(f)
                        if isinstance(rels, list):
                            for r in rels:
                                source = r.get('source')
                                target = r.get('target')
                                if source and target:
                                    # Add nodes if they don't exist (optional, but good for robust graph)
                                    if source not in self.graph:
                                        self.graph.add_node(source, type="UNKNOWN")
                                    if target not in self.graph:
                                        self.graph.add_node(target, type="UNKNOWN")
                                    self.graph.add_edge(source, target, **r)
                                    print(f"Loaded Edge: {source} -> {target}")

    def export_for_ui(self):
        # Convert to D3.js compatible JSON
        return nx.node_link_data(self.graph)

if __name__ == "__main__":
    loader = OmniGraphLoader()
    loader.load_universe()
    loader.load_constellations()
    loader.load_relationships()
    print(f"Graph constructed with {loader.graph.number_of_nodes()} nodes and {loader.graph.number_of_edges()} edges.")
