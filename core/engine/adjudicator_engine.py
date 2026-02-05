import random
from typing import Dict, Any, List
from core.engine.states import DoubleCrisisState, init_double_crisis_state

class AdjudicatorEngine:
    """
    The 'Master Prompt' Engine for the High-Fidelity Crisis Simulation.
    Simulates the mechanical linkages between Sovereign Debt, Banking Solvency,
    and Wholesale Funding Markets (Repo).
    """

    def __init__(self, state: DoubleCrisisState = None):
        self.state = state if state else init_double_crisis_state()
        self.active_injects = {}  # ID -> Inject

    def get_state(self) -> DoubleCrisisState:
        return self.state

    def reset(self):
        self.state = init_double_crisis_state()
        self.active_injects = {}
        # Act I trigger immediately
        self._trigger_act_i()

    def _trigger_act_i(self):
        """Act I: The Sovereign Shock"""
        self.state['history'].append("ACT I: THE SOVEREIGN SHOCK BEGINS.")

        inject = {
            "id": "INJECT_001",
            "type": "Standard",
            "title": "Sovereign Rating Downgrade",
            "message": "Monday, 08:00 AM. Rating Agency S&P downgrades Country Z by two notches to BBB-. Outlook Negative. Cites 'undeclared fiscal liabilities'.",
            "options": [
                {"id": "A1_OPT1", "text": "Do Nothing (Wait for clarity)", "consequence": "Spread widens significantly."},
                {"id": "A1_OPT2", "text": "Issue Reassurance Statement", "consequence": "Spread widens moderately. Trust Score -5."},
                {"id": "A1_OPT3", "text": "Sell $10B Sovereign Bonds (De-risk)", "consequence": "Immediate Fire Sale Loss. Spread widens maximally."}
            ]
        }
        self.state['injects'].append(inject)
        self.active_injects[inject['id']] = inject

    def resolve_action(self, inject_id: str, option_id: str):
        """Handles player decision."""
        if inject_id not in self.active_injects:
            return {"error": "Invalid or expired inject."}

        # Remove inject from active
        self.state['injects'] = [i for i in self.state['injects'] if i['id'] != inject_id]
        del self.active_injects[inject_id]

        # Log decision
        self.state['history'].append(f"Player chose option {option_id} for {inject_id}")

        # Logic for specific choices
        if inject_id == "INJECT_001":
            if option_id == "A1_OPT1": # Do nothing
                self.state['sovereign_spread'] += 50
                self.state['history'].append("Market interprets silence as weakness. Spreads +50bps.")
            elif option_id == "A1_OPT2": # Reassure
                self.state['sovereign_spread'] += 25
                self.state['market_trust'] -= 5
                self.state['history'].append("Statement perceived as hollow. Spreads +25bps. Trust falls.")
            elif option_id == "A1_OPT3": # Sell
                self.state['sovereign_spread'] += 100
                self.state['history'].append("Fire sale triggers market panic. Spreads +100bps.")
                self._apply_fire_sale_loss(volume_bn=10.0, price_impact=0.05)

            # Transition to next phase automatically after decision
            self._update_market_mechanics()
            self._trigger_act_ii()

        elif inject_id == "INJECT_002": # Repo Squeeze
            if option_id == "A2_OPT1": # Pay from Buffer
                self.state['intraday_liquidity'] -= self.state['funding_gap']
                self.state['history'].append(f"Paid ${self.state['funding_gap']:.1f}M from buffer.")
            elif option_id == "A2_OPT2": # Fire Sale Assets
                self._apply_fire_sale_loss(volume_bn=5.0, price_impact=0.08)
                cash_raised = 5000 * (1 - 0.08) # Simplification
                self.state['intraday_liquidity'] += cash_raised
                self.state['intraday_liquidity'] -= self.state['funding_gap']
                self.state['history'].append("Asset sale executed at steep discount.")

            self._update_market_mechanics()
            self._trigger_act_iii()

        elif inject_id == "INJECT_003": # Counterparty Failure
            if option_id == "A3_OPT1": # Call Default
                self.state['history'].append("Default called. Counterparty Alpha files for bankruptcy.")
                self.state['counterparty_cds'] = 10000 # Defaulted
                self._apply_cva_shock(default=True)
            elif option_id == "A3_OPT2": # Extend Credit
                self.state['history'].append("Credit extended. Governance breach flagged.")
                self.state['market_trust'] -= 20
                # Delayed failure
                self.state['history'].append("24 hours later: Counterparty Alpha fails anyway. Loss doubled.")
                self._apply_cva_shock(default=True, multiplier=2.0)

            self._update_market_mechanics()
            self._trigger_act_iv()

        elif inject_id == "INJECT_004": # Resolution
            self.state['game_over'] = True
            if option_id == "A4_OPT1": # Bail-in
                self.state['history'].append("Bail-in executed. Bondholders wiped out. Bank survives.")
                self.state['score']['survival'] = 100
            elif option_id == "A4_OPT2": # State Aid
                self.state['history'].append("State Aid requested. CEO replaced. Bank nationalized.")
                self.state['score']['survival'] = 50

        # Update Metrics after any action
        self._recalculate_metrics()

    def _trigger_act_ii(self):
        """Act II: The Repo Squeeze"""
        # Calculate Haircut Spike
        # Base logic: Haircut correlates with Volatility/Spread
        # If Spread > 500, Haircut -> 15%
        new_haircut = 2.0
        if self.state['sovereign_spread'] > 500:
            new_haircut = 15.0
        elif self.state['sovereign_spread'] > 470:
            new_haircut = 10.0

        self.state['repo_haircut'] = new_haircut

        # Calculate Liquidity Gap
        # Assume $50B Repo Book
        repo_book = 50000 # $50B
        old_haircut = 0.02
        # Cash needed = (New Haircut - Old Haircut) * Book
        gap = repo_book * ((new_haircut / 100) - old_haircut)
        self.state['funding_gap'] = gap

        self.state['history'].append(f"ACT II: THE REPO SQUEEZE. Haircuts spike to {new_haircut}%. Gap: ${gap:.1f}M")

        inject = {
            "id": "INJECT_002",
            "type": "Critical",
            "title": "Repo Margin Call",
            "message": f"Tuesday, 09:00 AM. Repo Desk reports money market funds demanding {new_haircut}% haircuts. Immediate margin call of ${gap:.0f}M required.",
            "options": [
                {"id": "A2_OPT1", "text": f"Pay from Intraday Buffer (Current: ${self.state['intraday_liquidity']:.0f}M)", "consequence": "Buffer depleted. LCR drops."},
                {"id": "A2_OPT2", "text": "Emergency Asset Sale", "consequence": "Realize losses. Further price impact."}
            ]
        }
        self.state['injects'].append(inject)
        self.active_injects[inject['id']] = inject

    def _trigger_act_iii(self):
        """Act III: Wrong-Way Risk"""
        # Counterparty CDS spikes due to correlation with Sovereign
        self.state['counterparty_cds'] = 1200 # 1200bps = Distress

        self.state['history'].append("ACT III: COUNTERPARTY CONTAGION.")

        inject = {
            "id": "INJECT_003",
            "type": "Critical",
            "title": "Counterparty Failure",
            "message": "Wednesday, 12:00 PM. Counterparty Alpha fails to meet a $500m margin call on ITM derivatives. They claim a 'technical error'.",
            "options": [
                {"id": "A3_OPT1", "text": "Declare Default & Seize Collateral", "consequence": "Immediate CVA Loss. Legal certainty."},
                {"id": "A3_OPT2", "text": "Grant 24h Waiver (Extend Credit)", "consequence": "Breach risk limits. Delay pain."}
            ]
        }
        self.state['injects'].append(inject)
        self.active_injects[inject['id']] = inject

    def _trigger_act_iv(self):
        """Act IV: Resolution"""
        self.state['history'].append("ACT IV: RESOLUTION.")

        status_msg = "Critical" if self.state['cet1'] < 10.0 else "Stressed"

        inject = {
            "id": "INJECT_004",
            "type": "Strategic",
            "title": "Resolution Authority Intervention",
            "message": f"Thursday. Capital ratios are {status_msg} (CET1: {self.state['cet1']:.1f}%). Regulator demands immediate recovery plan.",
            "options": [
                {"id": "A4_OPT1", "text": "Trigger Bail-in (Write down sub-debt)", "consequence": "Shareholders diluted/wiped. Bank stays open."},
                {"id": "A4_OPT2", "text": "Request Government Bail-out", "consequence": "Political firestorm. Nationalization."}
            ]
        }
        self.state['injects'].append(inject)
        self.active_injects[inject['id']] = inject

    def _update_market_mechanics(self):
        """Simulates background market drift."""
        # Random walk with drift based on trust
        drift = 0
        if self.state['market_trust'] < 90:
            drift = 5
        if self.state['market_trust'] < 50:
            drift = 15

        self.state['sovereign_spread'] += (random.randint(-2, 10) + drift)
        self.state['lcr'] -= 0.5 # Bleed liquidity

    def _apply_fire_sale_loss(self, volume_bn: float, price_impact: float):
        """Simulate P&L hit from fire sales."""
        # Loss = Volume * Price Impact
        loss_bn = volume_bn * price_impact
        # Impact on CET1 (Capital)
        # Assume RWA (Risk Weighted Assets) = $300B fixed for simplicity
        rwa = 300.0
        # CET1 impact in %
        cet1_hit = (loss_bn / rwa) * 100
        self.state['cet1'] -= cet1_hit
        self.state['history'].append(f"Fire Sale Loss: ${loss_bn:.2f}B. CET1 Impact: -{cet1_hit:.2f}%")

    def _apply_cva_shock(self, default=False, multiplier=1.0):
        """Simulate CVA (Credit Valuation Adjustment) losses."""
        # CVA charge roughly
        charge_bn = 2.0 * multiplier if default else 0.5
        rwa = 300.0
        cet1_hit = (charge_bn / rwa) * 100
        self.state['cet1'] -= cet1_hit
        self.state['history'].append(f"CVA Charge: ${charge_bn:.2f}B. CET1 Impact: -{cet1_hit:.2f}%")

    def _recalculate_metrics(self):
        """Updates derived metrics."""
        # LCR decay based on Intraday Liquidity
        # Base LCR = 115
        # If Intraday < 5000, LCR drops
        deficit = max(0, 5000 - self.state['intraday_liquidity'])
        # $1B deficit = -5% LCR roughly
        lcr_hit = (deficit / 1000) * 5
        self.state['lcr'] = 115.0 - lcr_hit
