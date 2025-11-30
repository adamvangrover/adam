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

                        # Use Ticker from meta as ID if available to match other datasets
                        if 'meta' in data and 'target' in data['meta']:
                            node_id = data['meta']['target']
                        else:
                            # Fallback to legal name
                            try:
                                node_id = data['nodes']['entity_ecosystem']['legal_entity']['name']
                            except KeyError:
                                print(f"Skipping malformed dossier: {filename}")
                                continue

                        # Store the FULL schema object in the node
                        self.graph.add_node(node_id, type="HERO", data=data)
                        print(f"Loaded Hero: {node_id}")

    def load_constellations(self):
        # 2. Load Satellite Nodes
        const_path = os.path.join(self.base_path, "constellations")
        if os.path.exists(const_path):
            for filename in os.listdir(const_path):
                if filename.endswith(".json"):
                    with open(os.path.join(const_path, filename)) as f:
                        try:
                            nodes = json.load(f)
                            for n in nodes:
                                # Ensure we don't overwrite a HERO node with a SATELLITE node if it already exists
                                if not self.graph.has_node(n['id']):
                                    self.graph.add_node(n['id'], type="SATELLITE", **n)
                                else:
                                    # Update existing node with constellation data if needed, or just skip
                                    pass

                                # Add simple edge if relationship exists
                                if 'relationship_to_hero' in n:
                                    # In a real app, you'd find the hero node ID dynamically
                                    pass
                            print(f"Loaded Constellation: {filename}")
                        except json.JSONDecodeError:
                             print(f"Error decoding constellation file: {filename}")

    def load_relationships(self):
        # 3. Load Edges (Relationships)
        rel_path = os.path.join(self.base_path, "relationships")
        if os.path.exists(rel_path):
             for filename in os.listdir(rel_path):
                if filename.endswith(".json"):
                    with open(os.path.join(rel_path, filename)) as f:
                        try:
                            edges = json.load(f)
                            for e in edges:
                                source = e.get('source')
                                target = e.get('target')
                                if source and target:
                                    self.graph.add_edge(source, target, **e)
                            print(f"Loaded Relationships: {filename}")
                        except json.JSONDecodeError:
                            print(f"Error decoding relationship file: {filename}")


    def export_for_ui(self):
        # Convert to D3.js compatible JSON
        return nx.node_link_data(self.graph)

if __name__ == "__main__":
    loader = OmniGraphLoader()
    loader.load_universe()
    loader.load_constellations()
    loader.load_relationships()
    print(f"Graph constructed with {loader.graph.number_of_nodes()} nodes and {loader.graph.number_of_edges()} edges.")
