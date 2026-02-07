from typing import List, Dict, Any, Literal, Optional
import uuid
import random
import logging
from datetime import datetime
from pydantic import BaseModel, Field

# --- Data Models ---

class GameAction(BaseModel):
    actor: Literal["RED_TEAM", "BLUE_TEAM"]
    action_type: str
    description: str
    impact_score: float = 0.0
    cost: float = 0.0
    timestamp: float

class WargameState(BaseModel):
    match_id: str
    turn: int = 0
    financial_health: float = 100.0  # 0 to 100
    market_stability: float = 100.0
    detected_threats: List[str] = Field(default_factory=list)
    capital_reserves: float = 1_000_000.0
    game_over: bool = False
    winner: Optional[str] = None
    log: List[GameAction] = Field(default_factory=list)

# --- Engine ---

class FinancialWargameEngine:
    """
    Orchestrates a turn-based Cyber-Financial Wargame.
    Red Team: Injects market shocks, fraud, and cyber events.
    Blue Team: Deploys capital, adjusts risk parameters, investigates anomalies.
    """

    def __init__(self, match_id: str = None):
        self.match_id = match_id or str(uuid.uuid4())
        self.state = WargameState(match_id=self.match_id)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("WargameEngine")

    def step(self):
        """Executes one turn (Red Move -> Blue Move -> Resolution)."""
        if self.state.game_over:
            return

        self.state.turn += 1
        self.logger.info(f"--- Turn {self.state.turn} ---")

        # 1. Red Team Move
        red_action = self._red_team_strategy()
        self._apply_action(red_action)

        # 2. Blue Team Move
        blue_action = self._blue_team_strategy(red_action)
        self._apply_action(blue_action)

        # 3. Resolution & Scoring
        self._resolve_turn()

    def _red_team_strategy(self) -> GameAction:
        """Determines Red Team action based on current state."""
        actions = [
            ("Flash Crash", 15.0, "Injects high-frequency sell orders to destabilize liquidity."),
            ("Ledger Fraud", 10.0, "Alters transaction records to hide losses."),
            ("DDoS Attack", 5.0, "Disrupts trading API availability."),
            ("Rumor Bomb", 8.0, "Spreads fake news about solvent banks.")
        ]

        # Simple heuristic: escalate if losing, probe if winning
        choice = random.choice(actions)
        impact = choice[1] * random.uniform(0.8, 1.2)

        return GameAction(
            actor="RED_TEAM",
            action_type=choice[0],
            description=choice[2],
            impact_score=impact,
            timestamp=datetime.now().timestamp()
        )

    def _blue_team_strategy(self, last_red_action: GameAction) -> GameAction:
        """Determines Blue Team action based on detected threats."""
        # Detection probability
        detection_roll = random.random()
        detected = detection_roll > 0.3  # 70% chance to detect

        if not detected:
            return GameAction(
                actor="BLUE_TEAM",
                action_type="Wait",
                description="No immediate threats detected. Monitoring...",
                impact_score=0.0,
                timestamp=datetime.now().timestamp()
            )

        # Counter-measures
        if last_red_action.action_type == "Flash Crash":
            return GameAction(
                actor="BLUE_TEAM",
                action_type="Liquidity Injection",
                description="Central Bank injects liquidity to stabilize spread.",
                impact_score=10.0,
                cost=50000.0,
                timestamp=datetime.now().timestamp()
            )
        elif last_red_action.action_type == "Ledger Fraud":
             return GameAction(
                actor="BLUE_TEAM",
                action_type="Forensic Audit",
                description="Forensic Accountant Agent scans ledger for Benford anomalies.",
                impact_score=12.0,
                cost=10000.0,
                timestamp=datetime.now().timestamp()
            )
        else:
             return GameAction(
                actor="BLUE_TEAM",
                action_type="System Hardening",
                description="Increasing firewall rules and risk thresholds.",
                impact_score=5.0,
                cost=5000.0,
                timestamp=datetime.now().timestamp()
            )

    def _apply_action(self, action: GameAction):
        """Updates state based on action."""
        self.state.log.append(action)
        self.logger.info(f"{action.actor}: {action.action_type} ({action.impact_score:.1f})")

        if action.actor == "RED_TEAM":
            damage = action.impact_score
            self.state.financial_health -= damage
            self.state.market_stability -= (damage * 0.5)

        elif action.actor == "BLUE_TEAM":
            mitigation = action.impact_score
            self.state.financial_health += mitigation
            self.state.market_stability += (mitigation * 0.5)
            self.state.capital_reserves -= action.cost

            if action.cost > 0:
                self.state.detected_threats.append(f"Mitigated {action.action_type}")

    def _resolve_turn(self):
        """Checks win conditions."""
        # Clamp values
        self.state.financial_health = max(0.0, min(100.0, self.state.financial_health))
        self.state.market_stability = max(0.0, min(100.0, self.state.market_stability))

        if self.state.financial_health <= 0:
            self.state.game_over = True
            self.state.winner = "RED_TEAM"
            self.logger.info("GAME OVER: Financial Collapse.")

        elif self.state.turn >= 20:
            self.state.game_over = True
            self.state.winner = "BLUE_TEAM"
            self.logger.info("GAME OVER: Survival.")

    def run_simulation(self, max_turns: int = 20) -> WargameState:
        while not self.state.game_over and self.state.turn < max_turns:
            self.step()
        return self.state
