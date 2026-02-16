import json
import random
import datetime
import os
import math

OUTPUT_DIR = "showcase/data"

class ScenarioGenerator:
    def __init__(self, steps=100):
        self.steps = steps
        self.data = []

    def generate(self):
        raise NotImplementedError

class MarketScenario(ScenarioGenerator):
    def generate(self):
        spx = 5500.0
        ndx = 19500.0
        vix = 15.0

        for i in range(self.steps):
            # Market regime switching logic
            regime = "BULL" if i < 30 else ("BEAR" if i < 60 else "RECOVERY")

            vol_mult = 1.0
            if regime == "BULL":
                spx_chg = random.gauss(0.001, 0.005)
                vix = max(10, vix * 0.95 + random.gauss(0, 0.5))
            elif regime == "BEAR":
                spx_chg = random.gauss(-0.002, 0.015)
                vix = min(50, vix * 1.05 + random.gauss(0, 1.0))
                vol_mult = 2.0
            else: # RECOVERY
                spx_chg = random.gauss(0.0005, 0.008)
                vix = max(12, vix * 0.98 + random.gauss(0, 0.5))

            spx *= (1 + spx_chg)
            ndx *= (1 + spx_chg * 1.2 + random.gauss(0, 0.005)) # Tech beta > 1

            self.data.append({
                "step": i,
                "spx": round(spx, 2),
                "ndx": round(ndx, 2),
                "vix": round(vix, 2),
                "regime": regime
            })
        return self.data

class BankingScenario(ScenarioGenerator):
    def generate(self):
        # Simulating JPM Twin Data (Liquidity, Capital Adequacy, Systemic Risk)
        liquidity = 100.0 # Billions
        cet1_ratio = 13.5 # %
        systemic_risk = 20.0 # Index 0-100

        for i in range(self.steps):
            # Correlate with market phase (simplified)
            market_stress = 0 if i < 30 else (1 if i < 60 else 0.5)

            liquidity_shock = random.gauss(0, 1.0) - (market_stress * 2.0)
            liquidity = max(50, liquidity + liquidity_shock)

            cet1_chg = random.gauss(0, 0.05) - (market_stress * 0.1)
            cet1_ratio = max(10.0, cet1_ratio + cet1_chg)

            risk_shock = random.gauss(0, 1.0) + (market_stress * 5.0)
            systemic_risk = max(0, min(100, systemic_risk + risk_shock))

            self.data.append({
                "step": i,
                "liquidity_coverage": round(liquidity, 2),
                "cet1_ratio": round(cet1_ratio, 2),
                "systemic_risk_score": round(systemic_risk, 2)
            })
        return self.data

class RiskScenario(ScenarioGenerator):
    def generate(self):
        # VaR and Stress Tests
        var_95 = 1.5 # %
        stress_loss = 0.0 # Billions

        for i in range(self.steps):
            market_stress = 0 if i < 30 else (1 if i < 60 else 0.5)

            var_95 = max(1.0, var_95 * (1 + random.gauss(0, 0.05) + market_stress * 0.1))
            stress_loss = max(0, stress_loss * 0.9 + (market_stress * random.uniform(0, 5)))

            self.data.append({
                "step": i,
                "var_95": round(var_95, 2),
                "stress_test_loss_est": round(stress_loss, 2)
            })
        return self.data

class MacroScenario(ScenarioGenerator):
    def generate(self):
        cpi = 3.2
        gdp_growth = 2.1
        fed_funds = 5.25

        for i in range(self.steps):
            # Slow moving macro indicators
            cpi += random.gauss(0, 0.02)
            gdp_growth += random.gauss(0, 0.01)

            # Simple Fed reaction function
            if cpi > 3.5:
                fed_funds = min(6.0, fed_funds + 0.01)
            elif cpi < 2.0 or gdp_growth < 0:
                fed_funds = max(2.0, fed_funds - 0.02)

            self.data.append({
                "step": i,
                "cpi_yoy": round(cpi, 2),
                "gdp_growth_qoq": round(gdp_growth, 2),
                "fed_funds_rate": round(fed_funds, 2)
            })
        return self.data

class AgentScenario(ScenarioGenerator):
    def generate(self):
        agents = ["FundamentalAnalyst", "TechnicalAnalyst", "RiskGuardian", "SentimentSwarm", "MacroOverseer"]
        actions = ["ANALYZING", "IDLE", "ALERTING", "EXECUTING", "OPTIMIZING"]
        topics = ["NVDA", "Inflation", "Liquidity", "Geopolitics", "MeanReversion"]

        for i in range(self.steps):
            daily_logs = []
            num_events = random.randint(1, 5)
            for _ in range(num_events):
                agent = random.choice(agents)
                action = random.choice(actions)
                topic = random.choice(topics)
                severity = "INFO"
                if action == "ALERTING": severity = "HIGH"
                if action == "EXECUTING": severity = "MEDIUM"

                log = {
                    "agent": agent,
                    "action": action,
                    "topic": topic,
                    "severity": severity,
                    "message": f"{agent} is {action} on {topic} signal."
                }
                daily_logs.append(log)

            self.data.append({
                "step": i,
                "logs": daily_logs,
                "active_agents": random.randint(3, 5)
            })
        return self.data

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    steps = 90
    print(f"Generating Enterprise Simulation Data ({steps} steps)...")

    simulation_data = {
        "metadata": {
            "simulation_id": "ENT-SIM-2026-ALPHA",
            "generated_at": datetime.datetime.now().isoformat(),
            "scenarios_included": ["Market", "Banking", "Risk", "Macro", "Agents"]
        },
        "market": MarketScenario(steps).generate(),
        "banking": BankingScenario(steps).generate(),
        "risk": RiskScenario(steps).generate(),
        "macro": MacroScenario(steps).generate(),
        "agents": AgentScenario(steps).generate()
    }

    filepath = os.path.join(OUTPUT_DIR, "enterprise_simulation_data.json")
    with open(filepath, "w") as f:
        json.dump(simulation_data, f, indent=2)

    print(f"Success! Data saved to {filepath}")

if __name__ == "__main__":
    main()
