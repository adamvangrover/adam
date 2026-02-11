import json
import os
import sys
import subprocess
from pathlib import Path
import re

# Add repo root to path for imports if needed
sys.path.append(os.getcwd())

GRAPH_FILE = "showcase/data/system_knowledge_graph.json"

class SystemGuideAgent:
    def __init__(self):
        self.graph_data = self.load_graph()
        self.index = self.build_index()
        self.simulations = {
            "market_mayhem": "python3 scripts/generate_market_mayhem_archive.py",
            "market_coding_bridge": "python3 scripts/bridge_market_coding.py",
            "system_graph": "python3 scripts/generate_system_graph.py"
        }

    def load_graph(self):
        if not os.path.exists(GRAPH_FILE):
            print(f"Graph file not found at {GRAPH_FILE}. Generating...")
            subprocess.run(["python3", "scripts/generate_system_graph.py"], check=True)

        with open(GRAPH_FILE, "r") as f:
            return json.load(f)

    def build_index(self):
        """Builds a simple keyword index from nodes."""
        index = {}
        for node in self.graph_data.get("nodes", []):
            text = f"{node.get('label', '')} {node.get('preview', '')} {node.get('docstring', '')}".lower()
            keywords = set(re.findall(r'\w+', text))
            for kw in keywords:
                if len(kw) > 3: # Ignore small words
                    if kw not in index:
                        index[kw] = []
                    index[kw].append(node)
        return index

    def search(self, query):
        keywords = re.findall(r'\w+', query.lower())
        results = {}

        for kw in keywords:
            if kw in self.index:
                for node in self.index[kw]:
                    node_id = node['id']
                    if node_id not in results:
                        results[node_id] = {"node": node, "score": 0}
                    results[node_id]["score"] += 1

        # Sort by score
        sorted_results = sorted(results.values(), key=lambda x: x['score'], reverse=True)
        return [r['node'] for r in sorted_results[:5]]

    def run_simulation(self, name):
        if name in self.simulations:
            print(f"Running simulation: {name}...")
            try:
                subprocess.run(self.simulations[name].split(), check=True)
                print(f"Simulation {name} completed successfully.")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error running simulation {name}: {e}")
                return False
        else:
            print(f"Simulation '{name}' not found. Available: {', '.join(self.simulations.keys())}")
            return False

    def chat(self, user_input):
        user_input = user_input.strip()

        # Check for commands
        if user_input.lower().startswith("run "):
            sim_name = user_input[4:].strip()
            self.run_simulation(sim_name)
            return "Simulation executed."

        if user_input.lower() in ["help", "list", "commands"]:
            return f"Available simulations: {', '.join(self.simulations.keys())}. You can also ask me about the system architecture."

        # Search Knowledge Graph
        results = self.search(user_input)

        if not results:
            return "I couldn't find anything in the System Knowledge Graph related to that."

        response = "Here is what I found in the Knowledge Graph:\n"
        for node in results:
            response += f"- **{node['label']}** ({node.get('group', 'unknown')}): {node.get('title', '')}\n"
            if node.get('preview'):
                preview = node['preview'].replace('\n', ' ')[:100]
                response += f"  > {preview}...\n"

        return response

def main():
    print("Initializing System Guide Agent...")
    agent = SystemGuideAgent()
    print("Agent Ready. Type 'exit' to quit.")
    print("Example commands: 'run market_coding_bridge', 'search RiskGuardian'")

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            response = agent.chat(user_input)
            print(f"Agent: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
