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
    # Pre-compiled regex patterns for performance
    TITLE_PATTERN = re.compile(r"^# PROMPT:\s*(.*)", re.MULTILINE)
    ID_PATTERN = re.compile(r"\*\*?ID:?\*\*?\s*(.*)", re.IGNORECASE)
    TAGS_PATTERN = re.compile(r"\*\*?Tags:?\*\*?\s*\[(.*)\]", re.IGNORECASE)
    MARKET_IMPACT_PATTERN = re.compile(r"\*\*Market Impact:\*\*(.*?)(##|$)", re.DOTALL)
    IMPACT_LINE_PATTERN = re.compile(r"\*\s*\*\*(.*?):\*\*\s*(.*)")
    IMPACT_DETAIL_PATTERN = re.compile(r"([A-Z]+)\s*([+-]?\d+%)")

    def __init__(self, knowledge_graph: Optional[nx.DiGraph] = None, logger: Optional[ProvenanceLogger] = None):
        self.kg = knowledge_graph if knowledge_graph else nx.DiGraph()
        self.logger = logger if logger else ProvenanceLogger()
        self.active_scenarios = []

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
        match = self.TITLE_PATTERN.search(content)
        return match.group(1).strip() if match else "UNKNOWN"

    def _extract_field(self, content: str, label: str) -> str:
        if label == "ID":
            match = self.ID_PATTERN.search(content)
            return match.group(1).strip() if match else "UNKNOWN"

        # Matches "**Label:** Value"
        pattern = r"\*\*?" + re.escape(label) + r":?\*\*?\s*(.*)"
        match = re.search(pattern, content, re.IGNORECASE)
        return match.group(1).strip() if match else "UNKNOWN"

    def _extract_list(self, content: str, label: str) -> List[str]:
        if label == "Tags":
            match = self.TAGS_PATTERN.search(content)
            if match:
                return [x.strip() for x in match.group(1).split(',')]
            return []

        pattern = r"\*\*?" + re.escape(label) + r":?\*\*?\s*\[(.*)\]"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return [x.strip() for x in match.group(1).split(',')]
        return []

    def _parse_shocks(self, content: str) -> List[Dict[str, Any]]:
        # Heuristic parsing of "Market Impact" section
        shocks = []
        impact_section = self.MARKET_IMPACT_PATTERN.search(content)
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
                    match = self.IMPACT_LINE_PATTERN.match(line.strip())
                    if match:
                        category = match.group(1)
                        details = match.group(2)
                        # Parse details like "NDX -15%"
                        impacts = self.IMPACT_DETAIL_PATTERN.findall(details)
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
                current_val = self.kg.nodes[target].get('valuation', 100)  # Default mock value
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
    kg.add_edge("NDX", "NVDA", weight=0.8)  # NVDA depends on NDX sentiment

    engine = CrisisSimulationEngine(kg)
    # Assuming the file exists from previous turn
    try:
        engine.load_scenario_from_markdown("prompt_library/AOPL-v1.0/simulation/semiconductor_supply_shock.md")
        result = engine.run_simulation("SIM-SC-004")
        print(json.dumps(result, indent=2))
    except FileNotFoundError:
        print("Scenario file not found for test.")
