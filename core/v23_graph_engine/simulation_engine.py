"""
Crisis Simulation Engine for Adam v23.5.
Parses simulation prompts and applies graph-theoretic shocks to the Knowledge Graph.
"""

import json
import re
import networkx as nx
from typing import Dict, List, Optional, Any
from core.system.provenance_logger import ProvenanceLogger, ActivityType

class CrisisSimulationEngine:
    # Compiled patterns
    RE_TITLE = re.compile(r"^# PROMPT:\s*(.*)", re.MULTILINE)
    RE_MARKET_IMPACT = re.compile(r"\*\*Market Impact:\*\*(.*?)(##|$)", re.DOTALL)
    RE_IMPACT_LINE = re.compile(r"\*\s*\*\*(.*?):\*\*\s*(.*)")
    RE_IMPACT_DETAILS = re.compile(r"([A-Z]+)\s*([+-]?\d+%)")

    def __init__(self, knowledge_graph: Optional[nx.DiGraph] = None, logger: Optional[ProvenanceLogger] = None):
        self.kg = knowledge_graph if knowledge_graph else nx.DiGraph()
        self.logger = logger if logger else ProvenanceLogger()
        self.active_scenarios = []
        self._regex_cache = {}

    def load_scenario_from_markdown(self, filepath: str) -> Dict[str, Any]:
        """Parses a structured markdown prompt file into a scenario dictionary."""
        with open(filepath, 'r') as f:
            content = f.read()

        scenario = {
            "id": self._extract_field(content, "ID"),
            "title": self._extract_title(content),
            "tags": self._extract_list(content, "Tags"),
            "shocks": self._parse_shocks(content),
            "raw_content": content
        }

        self.active_scenarios.append(scenario)
        return scenario

    def _extract_title(self, content: str) -> str:
        # Matches "# PROMPT: Title"
        match = self.RE_TITLE.search(content)
        return match.group(1).strip() if match else "UNKNOWN"

    def _get_regex(self, label: str, is_list: bool = False) -> re.Pattern:
        key = (label, is_list)
        if key not in self._regex_cache:
            if is_list:
                pattern = r"\*\*?" + re.escape(label) + r":?\*\*?\s*\[(.*)\]"
            else:
                pattern = r"\*\*?" + re.escape(label) + r":?\*\*?\s*(.*)"
            self._regex_cache[key] = re.compile(pattern, re.IGNORECASE)
        return self._regex_cache[key]

    def _extract_field(self, content: str, label: str) -> str:
        # Matches "**Label:** Value"
        regex = self._get_regex(label, is_list=False)
        match = regex.search(content)
        return match.group(1).strip() if match else "UNKNOWN"

    def _extract_list(self, content: str, label: str) -> List[str]:
        regex = self._get_regex(label, is_list=True)
        match = regex.search(content)
        if match:
            return [x.strip() for x in match.group(1).split(',')]
        return []

    def _parse_shocks(self, content: str) -> List[Dict[str, Any]]:
        # Heuristic parsing of "Market Impact" section
        shocks = []
        impact_section = self.RE_MARKET_IMPACT.search(content)
        if impact_section:
            lines = impact_section.group(1).strip().split('\n')
            for line in lines:
                if '*' in line:
                    # e.g., "* Indices: NDX -15%, SPX -8%" or "* **Indices:** ..."
                    # Flexible regex:
                    # 1. start with *
                    # 2. optional whitespace
                    # 3. **Category:**
                    # 4. Details
                    match = self.RE_IMPACT_LINE.match(line.strip())
                    if match:
                        category = match.group(1)
                        details = match.group(2)
                        # Parse details like "NDX -15%"
                        impacts = self.RE_IMPACT_DETAILS.findall(details)
                        for ticker, change in impacts:
                            shocks.append({
                                "target": ticker,
                                "change": change,
                                "category": category
                            })
        return shocks

    def run_simulation(self, scenario_id: str) -> Dict[str, Any]:
        """Applies the shocks of a loaded scenario to the graph and calculates propagation."""
        scenario = next((s for s in self.active_scenarios if s['id'] == scenario_id), None)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found.")

        simulation_log = {
            "scenario": scenario['title'],
            "direct_impacts": [],
            "second_order_effects": []
        }

        # 1. Direct Impact
        for shock in scenario['shocks']:
            target = shock['target']
            magnitude = float(shock['change'].replace('%', '')) / 100.0

            simulation_log['direct_impacts'].append({
                "node": target,
                "shock": magnitude
            })

            # Update Graph State (Mocking property update)
            if target in self.kg.nodes:
                current_val = self.kg.nodes[target].get('valuation', 100) # Default mock value
                self.kg.nodes[target]['valuation'] = current_val * (1 + magnitude)

        # 2. Second Order Effects (Simple Neighbors Propagation)
        # Assuming edges have weight representing correlation or dependency
        for impact in simulation_log['direct_impacts']:
            node = impact['node']
            shock_mag = impact['shock']

            if node in self.kg.nodes:
                for neighbor in self.kg.neighbors(node):
                    # Decay factor for propagation
                    decay = 0.5
                    propagated_shock = shock_mag * decay
                    simulation_log['second_order_effects'].append({
                        "source": node,
                        "target": neighbor,
                        "impact": propagated_shock
                    })

        # Log via Provenance
        self.logger.log_activity(
            agent_id="CrisisEngine",
            activity_type=ActivityType.SIMULATION,
            input_data={"scenario_id": scenario_id},
            output_data=simulation_log
        )

        return simulation_log

if __name__ == "__main__":
    # Mock Usage
    kg = nx.DiGraph()
    kg.add_node("NDX", type="Index", valuation=15000)
    kg.add_node("NVDA", type="Equity", valuation=1000)
    kg.add_edge("NDX", "NVDA", weight=0.8) # NVDA depends on NDX sentiment

    engine = CrisisSimulationEngine(kg)
    # Assuming the file exists from previous turn
    try:
        engine.load_scenario_from_markdown("prompt_library/AOPL-v1.0/simulation/semiconductor_supply_shock.md")
        result = engine.run_simulation("SIM-SC-004")
        print(json.dumps(result, indent=2))
    except FileNotFoundError:
        print("Scenario file not found for test.")
