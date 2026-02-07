from typing import List, Dict, Any, Tuple
import random
import statistics
from datetime import datetime, timedelta
import copy

class SovereignConflictSimulation:
    SECTORS = [
        "Technology", "Energy", "Finance", "Healthcare", "Industrials",
        "Services", "Media", "Telecom", "Consumer", "Retail",
        "Real Estate", "Shadow Banking", "Political"
    ]

    def __init__(self):
        self.scenarios = {
            "Semiconductor Blockade": {
                "phases": ["Diplomatic Posturing", "Trade Restrictions", "Naval Blockade", "Supply Chain Collapse"],
                "events": [
                    "New export controls announced on advanced lithography equipment.",
                    "Strait of Malacca naval exercises escalate tensions.",
                    "Major foundry reports 40% drop in raw material shipments.",
                    "Global tech stocks tumble as chip shortage fears grow.",
                    "Emergency nationalization of domestic semiconductor assets."
                ],
                "base_impact": {
                    "Technology": 0.9, "Industrials": 0.7, "Telecom": 0.6, "Consumer": 0.5, "Political": 0.4
                }
            },
            "Energy Shock": {
                "phases": ["Pipeline Sabotage", "Price Spike", "Grid Instability", "Rationing Protocols"],
                "events": [
                    "Unexplained pressure drop in trans-continental pipeline.",
                    "Oil futures spike to $150/barrel overnight.",
                    "Rolling blackouts initiated in major industrial zones.",
                    "Strategic petroleum reserves tapped by coalition nations.",
                    "Renewable energy infrastructure targeted by cyber attacks."
                ],
                "base_impact": {
                    "Energy": 0.95, "Industrials": 0.8, "Transport": 0.7, "Consumer": 0.6, "Political": 0.7, "Real Estate": 0.3
                }
            },
            "Cyber Infrastructure Attack": {
                "phases": ["Data Exfiltration", "Ransomware Deployment", "System Paralysis", "Financial Contagion"],
                "events": [
                    "Major banking SWIFT node goes offline intermittently.",
                    "Crypto exchanges halt withdrawals citing security breach.",
                    "Power grid control systems detect unauthorized root access.",
                    "National stock exchange suspends trading due to data corruption.",
                    "Central Bank digital currency ledger forked by unknown actors."
                ],
                "base_impact": {
                    "Finance": 0.9, "Shadow Banking": 0.8, "Technology": 0.7, "Services": 0.6, "Media": 0.5
                }
            },
            "Quantum Decryption Event": {
                "phases": ["Algorithm Breach", "Key Rotation Failure", "Ledger Collapse", "Trust Zero"],
                "events": [
                    "Nation-state actor demonstrates Shore's algorithm at scale.",
                    "Legacy RSA-2048 keys compromised across major banking sector.",
                    "Bitcoin blockchain suffers 51% attack via quantum supremacy.",
                    "Military comms revert to analog pad-cipher protocols.",
                    "Global internet trust anchors revoked; SSL/TLS collapse."
                ],
                "base_impact": {
                    "Finance": 1.0, "Shadow Banking": 1.0, "Technology": 0.9, "Political": 0.9, "Healthcare": 0.7, "Telecom": 0.8
                }
            }
        }

    def run(self, scenario_name: str, intensity: int = 5, duration_days: int = 30) -> Dict[str, Any]:
        """Runs a single deterministic simulation."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario_data = self.scenarios[scenario_name]
        start_date = datetime.now()
        timeline = []
        current_intensity = intensity

        # Initialize all sectors to 0
        sector_impact = {s: 0.0 for s in self.SECTORS}

        for day in range(0, duration_days, 2):
            current_date = start_date + timedelta(days=day)

            if random.random() < (current_intensity / 10.0):
                event_template = random.choice(scenario_data["events"])

                # Event Types
                if scenario_name == "Cyber Infrastructure Attack":
                     event_type = "Cyber" if random.random() > 0.3 else "Finance"
                elif scenario_name == "Quantum Decryption Event":
                     event_type = "Quantum" if random.random() > 0.2 else "Cyber"
                else:
                     event_type = "Geopolitical" if random.random() > 0.5 else "Economic"

                impact_score = random.randint(current_intensity * 5, current_intensity * 10)

                # Apply specific sector weights
                base_weights = scenario_data.get("base_impact", {})
                for sector in self.SECTORS:
                    weight = base_weights.get(sector, 0.1) # Default low correlation
                    # Add noise
                    noise = random.uniform(0.8, 1.2)
                    sector_impact[sector] += (impact_score * weight * 0.1 * noise)

                event = {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "day_offset": day,
                    "type": event_type,
                    "description": event_template,
                    "impact_score": impact_score,
                    "defcon_level": max(1, 5 - int(current_intensity / 2.5))
                }
                timeline.append(event)

                current_intensity += random.choice([-1, 0, 1])
                current_intensity = max(1, min(10, current_intensity))

        # Normalize
        for k in sector_impact:
            sector_impact[k] = min(100, int(sector_impact[k]))

        return {
            "scenario": scenario_name,
            "initial_intensity": intensity,
            "final_intensity": current_intensity,
            "timeline": timeline,
            "total_impact": sum(e["impact_score"] for e in timeline),
            "sector_impact": sector_impact
        }

    def run_monte_carlo(self, scenario_name: str, intensity: int = 5, iterations: int = 50) -> Dict[str, Any]:
        """
        Runs a Monte Carlo simulation to determine probability distribution of outcomes.
        """
        results = []
        for _ in range(iterations):
            results.append(self.run(scenario_name, intensity))

        # Aggregation logic
        total_impacts = [r["total_impact"] for r in results]
        avg_impact = statistics.mean(total_impacts)
        stdev_impact = statistics.stdev(total_impacts) if len(total_impacts) > 1 else 0
        min_impact = min(total_impacts)
        max_impact = max(total_impacts)

        # Sector Aggregation
        sector_agg = {s: [] for s in self.SECTORS}
        for r in results:
            for s, val in r["sector_impact"].items():
                sector_agg[s].append(val)

        sector_stats = {}
        for s, vals in sector_agg.items():
            sector_stats[s] = {
                "mean": int(statistics.mean(vals)),
                "max": int(max(vals)),
                "min": int(min(vals))
            }

        # Select the run closest to the mean as the "representative" timeline
        representative_run = min(results, key=lambda x: abs(x["total_impact"] - avg_impact))

        return {
            "scenario": scenario_name,
            "iterations": iterations,
            "statistics": {
                "mean_impact": int(avg_impact),
                "stdev_impact": int(stdev_impact),
                "min_impact": int(min_impact),
                "max_impact": int(max_impact),
                "confidence_95": [int(avg_impact - 2*stdev_impact), int(avg_impact + 2*stdev_impact)]
            },
            "sector_stats": sector_stats,
            "representative_timeline": representative_run["timeline"],
            "representative_sector_impact": representative_run["sector_impact"]
        }
