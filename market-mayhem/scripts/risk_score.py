from typing import Dict

class SystemicRiskCalculator:
    """
    Calculates the Systemic Risk Score based on six pillars defined in SKILL.md.
    """
    def __init__(self):
        # Weights according to SKILL.md
        self.weights = {
            'rates': 0.20,
            'credit': 0.20,
            'liquidity': 0.20,
            'inflation': 0.15,
            'volatility': 0.15,
            'geopolitics': 0.10
        }

    def calculate_score(self, components: Dict[str, float]) -> int:
        """
        Calculates the weighted average risk score (0-100).
        Expects a dictionary where keys match self.weights and values are 0-100.
        """
        total_score = 0.0

        for key, weight in self.weights.items():
            value = components.get(key, 0.0) # Default to 0 if missing
            # Ensure value is bounded
            value = max(0.0, min(100.0, float(value)))
            total_score += value * weight

        return int(round(total_score))

    def get_risk_label(self, score: int) -> str:
        """
        Translates the numerical score into a categorical label.
        """
        if score <= 30:
            return "Normal"
        elif score <= 50:
            return "Elevated"
        elif score <= 70:
            return "High Risk"
        else:
            return "Crisis"

if __name__ == "__main__":
    calculator = SystemicRiskCalculator()

    # Example: Late 2023 scenario
    inputs = {
        'rates': 80,       # High rates, inverted curve
        'credit': 40,      # Spreads widening slightly but manageable
        'liquidity': 30,   # Ample liquidity
        'inflation': 60,   # Sticky core
        'volatility': 30,  # VIX relatively low
        'geopolitics': 70  # Elevated tensions
    }

    score = calculator.calculate_score(inputs)
    label = calculator.get_risk_label(score)
    print(f"Risk Score: {score} ({label})")