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
                        # Extract the node ID safely, handling potential missing keys
                        try:
                            node_id = data['nodes']['entity_ecosystem']['legal_entity']['name']
                            # Store the FULL schema object in the node
                            self.graph.add_node(node_id, type="HERO", data=data)
                            print(f"Loaded Hero: {node_id}")
                        except KeyError as e:
                            print(f"Error loading {filename}: Missing key {e}")

    def load_constellations(self):
        # 2. Load Satellite Nodes
        const_path = os.path.join(self.base_path, "constellations")
        if os.path.exists(const_path):
            for filename in os.listdir(const_path):
                if filename.endswith(".json"):
                    with open(os.path.join(const_path, filename)) as f:
                        nodes = json.load(f)
                        for n in nodes:
                            self.graph.add_node(n['id'], type="SATELLITE", **n)
                            # Add simple edge if relationship exists
                            if 'relationship_to_hero' in n:
                                # In a real app, you'd find the hero node ID dynamically
                                pass
                            print(f"Loaded Satellite: {n.get('id')}")

    def export_for_ui(self):
        # Convert to D3.js compatible JSON
        return nx.node_link_data(self.graph)

if __name__ == "__main__":
    loader = OmniGraphLoader()
    loader.load_universe()
    loader.load_constellations()
    print(f"Graph constructed with {loader.graph.number_of_nodes()} nodes.")
