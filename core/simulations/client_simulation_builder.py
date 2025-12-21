# core/simulations/client_simulation_builder.py

import json
import os


class ClientSimulationBuilder:
    def __init__(self, seed_path: str = "data/v23_ukg_seed.json", output_path: str = "services/webapp/client/public/data/client_state.json"):
        self.seed_path = seed_path
        self.output_path = output_path

    def build(self):
        print(f"Building client simulation from {self.seed_path}...")

        if not os.path.exists(self.seed_path):
             print(f"Seed file {self.seed_path} not found. Skipping.")
             return

        try:
            with open(self.seed_path, 'r') as f:
                seed_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding seed file: {e}")
            return

        kg = seed_data.get("v23_unified_knowledge_graph", {})

        # Transform for Client UI
        # We flatten the structure slightly for the frontend dashboard
        client_state = {
            "meta": kg.get("meta"),
            "dashboard_config": kg.get("system_config"),
            "entities": kg.get("nodes", {}).get("legal_entities", []),
            "scenarios": kg.get("simulation_parameters", {}).get("crisis_scenarios", []),
            "macro_indicators": kg.get("nodes", {}).get("macro_indicators", []),
            "supply_chain": kg.get("nodes", {}).get("supply_chain_relations", []),
            "esg_profiles": kg.get("nodes", {}).get("esg_profiles", [])
        }

        # Create directory if needed
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, 'w') as f:
            json.dump(client_state, f, indent=2)

        print(f"Client simulation state saved to {self.output_path}")

if __name__ == "__main__":
    ClientSimulationBuilder().build()
