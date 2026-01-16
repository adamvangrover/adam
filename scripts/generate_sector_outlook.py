import json
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Specialists
# Note: In a real environment, we'd use the factory pattern from IndustrySpecialistAgent
# But for this data generation script, direct instantiation is simpler and safer.
from core.agents.industry_specialists.technology import TechnologySpecialist
from core.agents.industry_specialists.financials import FinancialsSpecialist
from core.agents.industry_specialists.energy import EnergySpecialist
from core.agents.analytics.impact_analysis_agent import ImpactAnalysisAgent

OUTPUT_FILE = 'showcase/data/sector_outlook.json'

def main():
    print("Generating Sector Outlook Data...")

    # Config placeholder
    config = {'data_sources': {}}

    specialists = [
        TechnologySpecialist(config),
        FinancialsSpecialist(config),
        EnergySpecialist(config)
    ]

    outlooks = []

    # Map for easy access by name
    outlook_map = {}

    for s in specialists:
        try:
            data = s.generate_outlook()
            outlooks.append(data)
            outlook_map[data['sector']] = data
        except Exception as e:
            print(f"Error generating outlook for {s}: {e}")

    # Generate Impact Matrix
    impact_agent = ImpactAnalysisAgent()
    matrix = impact_agent.analyze_impact(outlook_map)

    final_data = {
        "timestamp": "2026-01-15T12:00:00Z",
        "sectors": outlooks,
        "matrix": matrix,
        "synthesis": "The swarm consensus indicates a 'Reflationary Boom'. Tech and Energy are positively correlated due to the AI-Energy nexus. Financials remain a risk vector due to CRE exposure."
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
