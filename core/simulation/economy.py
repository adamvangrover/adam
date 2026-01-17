from typing import List, Dict, Any
from pydantic import BaseModel
import random
from .sovereign import Sovereign, Resource

class Industry(BaseModel):
    name: str
    dependence_map: Dict[str, float]  # Resource -> Importance (0-1)
    output: float = 1.0
    resilience: float = 0.5 # Ability to withstand shortages

class EconomyEngine:
    INDUSTRIES = [
        "Semiconductors", "Energy", "Agriculture", "Defense", "Biotech", "Fintech", "Manufacturing", "Services"
    ]

    @staticmethod
    def generate_industries(ideology: str) -> List[Industry]:
        industries = []
        if "Digital" in ideology or "Tech" in ideology:
            industries.append(Industry(name="Semiconductors", dependence_map={"Rare Earths": 0.9, "Water": 0.4, "Energy": 0.6}, output=1.5))
            industries.append(Industry(name="Fintech", dependence_map={"Compute": 0.8, "Energy": 0.2}, output=2.0))
        elif "Agrarian" in ideology:
            industries.append(Industry(name="Agriculture", dependence_map={"Water": 0.9, "Fertilizer": 0.7, "Energy": 0.3}, output=0.8))
            industries.append(Industry(name="Manufacturing", dependence_map={"Steel": 0.5, "Energy": 0.5}, output=1.0))
        else:
            industries.append(Industry(name="Energy", dependence_map={"Oil": 0.9}, output=1.8))
            industries.append(Industry(name="Defense", dependence_map={"Steel": 0.8, "Semiconductors": 0.7, "Energy": 0.5}, output=1.2))
            industries.append(Industry(name="Manufacturing", dependence_map={"Steel": 0.8, "Energy": 0.7}, output=1.0))

        return industries

    @staticmethod
    def simulate_global_market(sovereigns: List[Sovereign]):
        """
        Updates global resource prices based on Supply/Demand.
        """
        # 1. Aggregate Supply and Demand
        global_supply: Dict[str, float] = {}
        global_demand: Dict[str, float] = {}

        # Initialize
        all_resource_names = set()
        for sov in sovereigns:
            for r in sov.resources:
                all_resource_names.add(r.name)
                global_supply[r.name] = global_supply.get(r.name, 0.0) + r.amount

        # Estimate Demand based on Industries (Simplified)
        # Each industry unit consumes resources based on dependence
        for sov in sovereigns:
            # We assume sovereigns have industries implicitly based on ideology (but we didn't add industries to Sovereign model explicitly yet)
            # Let's generate ephemeral industries for calculation or check if we should add them to Sovereign.
            # For now, we regenerate them. Efficient? No. Simulation-theory accurate? Yes.
            industries = EconomyEngine.generate_industries(sov.ideology)
            for ind in industries:
                for res, dep in ind.dependence_map.items():
                    # Demand = Dependence * Output
                    global_demand[res] = global_demand.get(res, 0.0) + (dep * ind.output)

        # 2. Update Prices
        # Price equilibrium: Price moves proportional to (Demand / Supply)
        for res_name in all_resource_names:
            supply = max(0.1, global_supply.get(res_name, 1.0))
            demand = global_demand.get(res_name, 1.0)

            scarcity_ratio = demand / supply

            # Apply to all sovereigns' resources of this type
            for sov in sovereigns:
                for r in sov.resources:
                    if r.name == res_name:
                        # Price elasticity
                        change = (scarcity_ratio - 1.0) * r.volatility
                        r.market_price = max(0.1, r.market_price * (1.0 + change))

    @staticmethod
    def compute_economic_metrics(sov: Sovereign):
        """
        Updates GDP, Inflation, Trade Balance based on resources and market.
        """
        industries = EconomyEngine.generate_industries(sov.ideology)

        # 1. Calculate Potential Output (GDP)
        total_output = 0.0
        input_costs = 0.0

        for ind in industries:
            # Check resource availability for this industry
            bottleneck = 1.0
            ind_cost = 0.0

            for res_name, importance in ind.dependence_map.items():
                # Find if sovereign has resource
                res = next((r for r in sov.resources if r.name == res_name), None)
                if res:
                    amount = res.amount
                    price = res.market_price
                else:
                    # Import needed (assume 50% available via market but expensive)
                    amount = 0.5
                    price = 2.0 # Import premium (simplified)

                bottleneck = min(bottleneck, amount / importance) if importance > 0 else 1.0
                ind_cost += (importance * price)

            # Actual output limited by bottleneck
            actual_output = ind.output * bottleneck
            total_output += actual_output
            input_costs += ind_cost

        # 2. Update Metrics

        # GDP Growth: Compare to baseline (simplified random walk with bias)
        prev_gdp = 100.0 # Baseline
        current_gdp = total_output * 10.0 # Scaling factor
        growth = (current_gdp - prev_gdp) / prev_gdp

        # Smoothing
        sov.economy.gdp_growth = (sov.economy.gdp_growth * 0.8) + (growth * 0.2)

        # Inflation: Driven by input costs and currency strength
        inflation_pressure = (input_costs / max(1.0, total_output)) * 0.1
        sov.economy.inflation_rate = (sov.economy.inflation_rate * 0.9) + (inflation_pressure * 0.1)

        # Unemployment: Inverse to GDP growth
        if sov.economy.gdp_growth < 0:
            sov.economy.unemployment_rate += 0.005
        else:
            sov.economy.unemployment_rate = max(0.02, sov.economy.unemployment_rate - 0.002)

        # Trade Balance: Exports - Imports
        # Simplified: Surplus resources are exported, missing are imported
        exports = 0.0
        imports = 0.0

        for r in sov.resources:
            if r.amount > 1.0:
                exports += (r.amount - 1.0) * r.market_price

        # Imports inferred from industry needs (simplified for aggregate)
        # If input costs are high, we assume imports
        if input_costs > 5.0:
            imports += (input_costs - 5.0)

        sov.economy.trade_balance = exports - imports
