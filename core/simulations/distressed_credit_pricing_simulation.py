from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math
import logging

logger = logging.getLogger(__name__)

@dataclass
class CollateralPool:
    name: str
    value: float
    liquidation_factor: float = 0.8 # Haircut for forced sale

    @property
    def liquidation_value(self):
        return self.value * self.liquidation_factor

@dataclass
class Tranche:
    name: str
    principal: float
    interest_rate: float
    seniority: int  # 1 is highest (Senior)
    security_type: str = "Unsecured" # Secured, Unsecured
    secured_by: str = None # Name of collateral pool
    current_price: float = 100.0
    recovery_rate: float = 0.0
    expected_loss: float = 0.0

    @property
    def annual_interest(self):
        return self.principal * self.interest_rate

@dataclass
class CapitalStructure:
    tranches: List[Tranche] = field(default_factory=list)
    collateral_pools: Dict[str, CollateralPool] = field(default_factory=dict)
    equity_value: float = 0.0

    def add_tranche(self, tranche: Tranche):
        self.tranches.append(tranche)
        self.tranches.sort(key=lambda x: x.seniority)

    def add_collateral(self, pool: CollateralPool):
        self.collateral_pools[pool.name] = pool

    def total_debt(self):
        return sum(t.principal for t in self.tranches)

    def total_interest(self):
        return sum(t.annual_interest for t in self.tranches)

class DistressedCreditPricingSimulation:
    """
    Simulates credit pricing events for LBOs and distressed underwrites.
    Models leverage, PD, LGD, SNC Ratings, and recovery waterfalls.
    Updated for AVG framework with Collateral Pools and EL estimates.
    """

    def __init__(self):
        self.name = "AVG Distressed Credit Pricing Simulation"

    def calculate_leverage_metrics(self, ebitda: float, cap_structure: CapitalStructure):
        """
        Calculates leverage and coverage ratios.
        """
        total_debt = cap_structure.total_debt()
        interest_expense = cap_structure.total_interest()

        leverage_ratio = total_debt / ebitda if ebitda > 0 else float('inf')
        interest_coverage = ebitda / interest_expense if interest_expense > 0 else float('inf')

        return {
            "total_debt": total_debt,
            "ebitda": ebitda,
            "leverage_ratio": leverage_ratio,
            "interest_coverage": interest_coverage
        }

    def estimate_pd_lgd(self, leverage: float, coverage: float, industry_risk_score: float = 5.0):
        """
        Estimates Probability of Default (PD).
        LGD is now tranche-specific in calculate_expected_loss.
        """
        # Base score
        score = (leverage * 0.5) + ((10 - coverage) * 0.5) + (industry_risk_score * 0.3)
        pd = 1 / (1 + math.exp(-(score - 8))) # Centered around score 8

        # SNC Rating Derivation
        if pd < 0.01: snc_rating = "Pass"
        elif pd < 0.05: snc_rating = "Special Mention"
        elif pd < 0.20: snc_rating = "Substandard"
        elif pd < 0.50: snc_rating = "Doubtful"
        else: snc_rating = "Loss"

        return {
            "pd": pd,
            "snc_rating": snc_rating,
            "risk_score": score
        }

    def calculate_expected_loss(self, cap_structure: CapitalStructure, pd: float, accounting_standard: str = "IFRS9"):
        """
        Calculates Expected Loss (EL) for each tranche based on Collateral coverage.

        Standards:
        - IFRS9 / CECL: EL = PD * LGD * EAD. Considers forward-looking scenarios (simplified here).
        - GAAP (Legacy): Incurred Loss. Often 0 unless PD is very high (Probable & Estimable).
        """
        results = []

        # Calculate LGD per tranche based on collateral
        # Simplified: Secured tranches get first dip on specific collateral, then general pool.

        total_collateral_value = sum(p.liquidation_value for p in cap_structure.collateral_pools.values())

        # Remaining collateral for unsecured
        unencumbered_collateral = total_collateral_value

        # 1. Secured Allocation
        secured_claims = {}
        for tranche in cap_structure.tranches:
            if tranche.security_type == "Secured" and tranche.secured_by in cap_structure.collateral_pools:
                pool = cap_structure.collateral_pools[tranche.secured_by]
                pool_val = pool.liquidation_value

                # Check if multiple tranches claim same pool? assuming 1-to-1 or senior takes all for now
                # Subtract from unencumbered (assuming strictly segregated for this model)
                if pool.name in cap_structure.collateral_pools:
                     unencumbered_collateral -= pool_val

                coverage = min(pool_val, tranche.principal)
                lgd_pct = 1.0 - (coverage / tranche.principal)
                secured_claims[tranche.name] = lgd_pct

        # 2. Unsecured Allocation (Waterfall style for LGD estimation)
        # Remaining unencumbered collateral flows down seniority for LGD purposes
        remaining_gen_collateral = max(0, unencumbered_collateral)

        for tranche in cap_structure.tranches:
            lgd = 1.0

            if tranche.name in secured_claims:
                lgd = secured_claims[tranche.name]
            else:
                # Unsecured: take from remaining general pool
                coverage = min(remaining_gen_collateral, tranche.principal)
                lgd = 1.0 - (coverage / tranche.principal)
                remaining_gen_collateral -= coverage

            # EL Calculation
            if accounting_standard == "GAAP_Legacy":
                # Incurred loss: Only book if PD > 0.5 (Probable)
                el = (pd * lgd * tranche.principal) if pd > 0.5 else 0.0
            else:
                # IFRS9 / CECL: Lifetime EL (simplified as 1-year PD * LGD here)
                el = pd * lgd * tranche.principal

            tranche.expected_loss = el
            results.append({
                "tranche": tranche.name,
                "lgd_pct": lgd,
                "el": el,
                "accounting": accounting_standard
            })

        return results

    def simulate_restructuring(self, enterprise_value: float, cap_structure: CapitalStructure):
        """
        Runs a waterfall analysis to determine recovery for each tranche.
        """
        remaining_value = enterprise_value
        results = []

        for tranche in cap_structure.tranches:
            if remaining_value >= tranche.principal:
                # Full recovery
                recovery = tranche.principal
                recovery_pct = 1.0
                remaining_value -= tranche.principal
            else:
                # Partial recovery
                recovery = remaining_value
                recovery_pct = recovery / tranche.principal if tranche.principal > 0 else 0
                remaining_value = 0

            tranche.recovery_rate = recovery_pct
            tranche.current_price = recovery_pct * 100.0

            results.append({
                "tranche": tranche.name,
                "principal": tranche.principal,
                "recovery_amount": recovery,
                "recovery_pct": recovery_pct,
                "price": tranche.current_price
            })

        return {
            "enterprise_value": enterprise_value,
            "waterfall": results,
            "equity_recovery": remaining_value
        }

    def run(self, input_data: Dict = None):
        """
        Main execution method.
        """
        if not input_data: input_data = {}

        ebitda = input_data.get("ebitda", 50_000_000)
        ev_multiple = input_data.get("enterprise_value_multiple", 6.0)
        industry_risk = input_data.get("industry_risk", 5)
        accounting = input_data.get("accounting_standard", "IFRS9")

        # Build Cap Structure
        cs = CapitalStructure()

        # Collateral
        collateral_data = input_data.get("collateral", [])
        for c in collateral_data:
            cs.add_collateral(CollateralPool(c["name"], c["value"], c.get("haircut", 0.8)))

        # Tranches
        tranche_data = input_data.get("capital_structure", [])
        for t in tranche_data:
            cs.add_tranche(Tranche(
                name=t["name"],
                principal=t["amount"],
                interest_rate=t["rate"],
                seniority=t["seniority"],
                security_type=t.get("security_type", "Unsecured"),
                secured_by=t.get("secured_by", None)
            ))

        # 1. Metrics
        metrics = self.calculate_leverage_metrics(ebitda, cs)

        # 2. Risk (PD)
        risk = self.estimate_pd_lgd(metrics["leverage_ratio"], metrics["interest_coverage"], industry_risk)

        # 3. Expected Loss (EL)
        el_analysis = self.calculate_expected_loss(cs, risk["pd"], accounting)

        # 4. Restructuring
        enterprise_value = ebitda * ev_multiple
        waterfall = self.simulate_restructuring(enterprise_value, cs)

        result = {
            "simulation": self.name,
            "metrics": metrics,
            "risk_analysis": risk,
            "el_analysis": el_analysis,
            "restructuring_analysis": waterfall,
            "status": "Completed"
        }

        logger.info(f"Ran AVG Credit Sim: Leverage {metrics['leverage_ratio']:.1f}x, SNC: {risk['snc_rating']}")
        return result
