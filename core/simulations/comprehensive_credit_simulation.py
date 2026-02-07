from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math
import logging
import numpy as np

logger = logging.getLogger(__name__)

# --- Data Structures ---

@dataclass
class CollateralAsset:
    asset_type: str # 'AR', 'Inventory', 'M&E', 'RealEstate', 'Cash', 'Securities'
    book_value: float
    ineligible_amount: float = 0.0
    advance_rate: float = 0.85
    liquidation_factor: float = 0.8

    @property
    def net_eligible(self):
        return max(0, self.book_value - self.ineligible_amount)

    @property
    def borrowing_base_value(self):
        return self.net_eligible * self.advance_rate

    @property
    def liquidation_value(self):
        return self.book_value * self.liquidation_factor

@dataclass
class LoanTranche:
    name: str
    principal: float
    interest_rate: float
    seniority: int
    loan_type: str # 'Term', 'Revolver', 'DIP', 'Mezzanine', 'Unitranche'
    security_type: str = "Unsecured"
    secured_by: List[str] = field(default_factory=list) # List of asset types
    covenants: Dict[str, float] = field(default_factory=dict) # e.g. {'MaxLeverage': 4.0}

    @property
    def annual_interest(self):
        return self.principal * self.interest_rate

@dataclass
class DerivativePosition:
    product_type: str # 'Swap', 'Option', 'Forward'
    notional: float
    mtm_value: float # Mark to Market (can be negative)
    counterparty_rating: str
    margin_posted: float = 0.0

    def calculate_cva(self, pd: float, lgd: float):
        """Credit Valuation Adjustment"""
        # Simplified CVA = EAD * PD * LGD
        exposure = max(0, self.mtm_value - self.margin_posted)
        return exposure * pd * lgd

class ComprehensiveCreditSimulation:
    """
    A unified engine for:
    1. Distressed/LBO Pricing (Cash Flow)
    2. Asset Based Lending (ABL)
    3. Derivative/Flow Risk
    4. AVG-Optimized Restructuring
    """

    def __init__(self):
        self.name = "AVG Unified Credit Engine"

    # --- 1. ABL Logic ---
    def calculate_borrowing_base(self, assets: List[CollateralAsset]):
        total_bb = 0.0
        details = {}
        for asset in assets:
            val = asset.borrowing_base_value
            total_bb += val
            details[asset.asset_type] = {
                "gross": asset.book_value,
                "net": asset.net_eligible,
                "advance_rate": asset.advance_rate,
                "availability": val
            }
        return total_bb, details

    # --- 2. Cash Flow / LBO Logic ---
    def calculate_cash_flow_metrics(self, ebitda: float, capex: float, tax_rate: float, debt_stack: List[LoanTranche]):
        total_debt = sum(t.principal for t in debt_stack)
        total_interest = sum(t.annual_interest for t in debt_stack)

        # Free Cash Flow proxy
        fcf = (ebitda - capex) * (1 - tax_rate)

        leverage = total_debt / ebitda if ebitda else 999.0
        dscr = (ebitda - capex) / total_interest if total_interest else 999.0

        return {
            "total_debt": total_debt,
            "leverage": leverage,
            "dscr": dscr,
            "fcf_pre_debt": fcf
        }

    # --- 3. Derivative / Flow Logic ---
    def calculate_counterparty_risk(self, positions: List[DerivativePosition], pd_lookup: Dict[str, float]):
        total_cva = 0.0
        total_exposure = 0.0

        for pos in positions:
            pd = pd_lookup.get(pos.counterparty_rating, 0.02)
            lgd = 0.6 # Standard assumption
            cva = pos.calculate_cva(pd, lgd)
            total_cva += cva
            total_exposure += max(0, pos.mtm_value - pos.margin_posted)

        return {
            "total_exposure": total_exposure,
            "cva_charge": total_cva
        }

    # --- 4. AVG Recovery & Restructuring ---
    def run_avg_restructuring_search(self, enterprise_value: float, debt_stack: List[LoanTranche], iterations: int = 100):
        """
        Simulates an AVG (Adam-Van-Grover) search for the optimal restructuring support agreement (RSA).
        Objective: Maximize total value recovery while minimizing litigation risk (energy).
        """
        # This is a meta-simulation. In a real system, this would call the Quantum Retrieval Agent.
        # Here we simulate the output of such a search.

        logger.info(f"Initiating AVG Search for Restructuring Scenarios. EV={enterprise_value}")

        # 1. Standard Waterfall (The baseline state)
        remaining = enterprise_value
        baseline_recovery = {}
        for tranche in sorted(debt_stack, key=lambda x: x.seniority):
            rec = min(remaining, tranche.principal)
            baseline_recovery[tranche.name] = rec
            remaining -= rec

        # 2. Simulated Optimization
        # Finds a "better" scenario via e.g. Debt-for-Equity swap which might preserve EV better
        # (avoiding bankruptcy costs).

        # Bankruptcy cost assumption: 20% of EV if purely litigated.
        litigation_ev = enterprise_value * 0.8

        # AVG "Found" Solution: Consensual restructure
        # Preserves 95% of EV.
        consensual_ev = enterprise_value * 0.95

        # Allocate consensual EV
        # Senior gets Par (if covered)
        # Junior gets Equity

        optimized_structure = []
        remaining_opt = consensual_ev

        for tranche in sorted(debt_stack, key=lambda x: x.seniority):
            if remaining_opt >= tranche.principal:
                # Full reinstatement
                optimized_structure.append({
                    "tranche": tranche.name,
                    "action": "Reinstate",
                    "recovery_pct": 1.0,
                    "instrument": "Debt"
                })
                remaining_opt -= tranche.principal
            else:
                # Equitize
                rec_val = remaining_opt
                rec_pct = rec_val / tranche.principal
                optimized_structure.append({
                    "tranche": tranche.name,
                    "action": "Equitize",
                    "recovery_pct": rec_pct,
                    "instrument": "New Equity"
                })
                remaining_opt = 0 # All equity value given to fulcrum security

        return {
            "baseline_ev": litigation_ev,
            "optimized_ev": consensual_ev,
            "value_added": consensual_ev - litigation_ev,
            "proposal": optimized_structure,
            "avg_confidence": 0.94 # Simulated confidence score
        }

    def run_comprehensive_analysis(self, inputs: Dict):
        """
        Master orchestrator.
        """
        # Parse Assets
        assets = [CollateralAsset(**a) for a in inputs.get('assets', [])]

        # Parse Debt
        debt = [LoanTranche(**d) for d in inputs.get('debt', [])]

        # Parse Derivatives
        derivs = [DerivativePosition(**p) for p in inputs.get('derivatives', [])]

        # 1. ABL Analysis
        bb_total, bb_details = self.calculate_borrowing_base(assets)

        # 2. Cash Flow Analysis
        cf_metrics = self.calculate_cash_flow_metrics(
            inputs.get('ebitda', 0),
            inputs.get('capex', 0),
            inputs.get('tax_rate', 0.25),
            debt
        )

        # 3. CVA
        cva_metrics = self.calculate_counterparty_risk(derivs, inputs.get('pd_map', {}))

        # 4. Valuation & Restructuring (AVG)
        ev_multiple = inputs.get('ev_multiple', 5.0)
        enterprise_value = inputs.get('ebitda', 0) * ev_multiple

        restructuring = self.run_avg_restructuring_search(enterprise_value, debt)

        return {
            "abl_status": {
                "borrowing_base": bb_total,
                "details": bb_details,
                "surplus_deficit": bb_total - cf_metrics['total_debt'] # Simplified check against total debt
            },
            "credit_metrics": cf_metrics,
            "market_risk": cva_metrics,
            "avg_recovery": restructuring
        }
