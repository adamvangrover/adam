from typing import Dict, List, Tuple

class ScenarioBuilder:
    """
    Constructs Base, Bull, and Bear case scenarios based on the current regime.
    """
    def __init__(self):
        # Default probabilities to ensure they sum to 100
        self.default_probs = {"Base": 60, "Bull": 20, "Bear": 20}

    def generate_scenarios(self, regime: str) -> List[Dict[str, str]]:
        """
        Generates scenario templates based on the detected macro regime.
        """
        scenarios = []

        if regime.lower() == "restrictive":
            scenarios = [
                {"name": "Base Case", "probability": 60, "description": "Rates stay elevated, growth slows moderately, resulting in a mild earnings recession and a shallow equity correction."},
                {"name": "Bull Case", "probability": 15, "description": "Inflation falls faster than expected, allowing the Fed to signal cuts, sparking a massive duration rally and equity melt-up."},
                {"name": "Bear Case", "probability": 25, "description": "High rates cause a credit event or banking stress, forcing emergency liquidity injections while inflation remains sticky."}
            ]
        elif regime.lower() == "early cycle (recovery)":
            scenarios = [
                {"name": "Base Case", "probability": 65, "description": "The Fed cuts rates 3-4 times. Growth remains positive but sluggish. Equities grind higher on multiple expansion."},
                {"name": "Bull Case", "probability": 20, "description": "Productivity gains boost earnings significantly, while inflation stays dead. The broad market rallies strongly."},
                {"name": "Bear Case", "probability": 15, "description": "The easing of financial conditions sparks a second wave of inflation, forcing central banks to reverse course."}
            ]
        else:
            # Generic fallback
            scenarios = [
                {"name": "Base Case", "probability": 60, "description": "Current trends continue with moderate volatility."},
                {"name": "Bull Case", "probability": 20, "description": "Upside surprise in growth and positive earnings revisions."},
                {"name": "Bear Case", "probability": 20, "description": "Unexpected shock to growth or liquidity causes a rapid sell-off."}
            ]

        return scenarios

    def format_markdown(self, scenarios: List[Dict[str, str]]) -> str:
        """
        Formats the generated scenarios into markdown for the report.
        """
        markdown = ""
        for s in scenarios:
            markdown += f"- **{s['name']} ({s['probability']}% Probability):** {s['description']}\n"
        return markdown

if __name__ == "__main__":
    builder = ScenarioBuilder()
    scenarios = builder.generate_scenarios("restrictive")
    print(builder.format_markdown(scenarios))