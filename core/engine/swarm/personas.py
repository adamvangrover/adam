import random
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.engine.swarm.pheromone_board import PheromoneBoard


class PersonaAction(BaseModel):
    """Strongly typed output model for persona actions within the swarm."""
    agent_id: str = Field(..., description="Unique ID of the persona executing the action.")
    action_type: str = Field(..., description="Type of action (e.g., POST, TRADE, PANIC).")
    sentiment: str = Field(..., description="Current sentiment direction (Bullish, Bearish, Neutral).")
    rationale: str = Field(..., description="Reasoning behind the action.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction level of the action.")


class BasePersona:
    """
    Base class for MiroFish Swarm Personas.
    Simulates individual actor logic responding to the environment.
    """
    def __init__(self, agent_id: str, board: PheromoneBoard):
        self.agent_id = agent_id
        self.board = board
        self.memory: list[Dict[str, Any]] = []

    async def act(self, cycle: int) -> Optional[PersonaAction]:
        """
        Evaluate the environment and execute an action.
        """
        # Read the environment (e.g., global events or other agent actions)
        recent_events = await self.board.sniff("simulation.events.global")
        other_actions = await self.board.sniff("simulation.actions.agent")

        # Analyze and determine response
        action = self._determine_action(cycle, recent_events, other_actions)

        if action:
            self.memory.append(action.model_dump())
            # Publish action to the event bus
            await self.board.deposit(
                signal_type="simulation.actions.agent",
                data={"cycle": cycle, **action.model_dump()},
                intensity=action.confidence * 10,
                source=self.agent_id
            )

        return action

    def _determine_action(self, cycle: int, global_events: list, agent_actions: list) -> Optional[PersonaAction]:
        raise NotImplementedError("Subclasses must implement _determine_action")


class RetailInvestorPersona(BasePersona):
    """
    Simulates retail flow logic.
    High momentum bias, fast reaction to sentiment swings, low conviction.
    """
    def _determine_action(self, cycle: int, global_events: list, agent_actions: list) -> Optional[PersonaAction]:
        # Simple herd behavior simulation
        bearish_signals = sum(1 for a in agent_actions if a.data.get("sentiment") == "Bearish")
        bullish_signals = sum(1 for a in agent_actions if a.data.get("sentiment") == "Bullish")

        if bearish_signals > bullish_signals * 1.5:
            sentiment = "Bearish"
            action_type = "PANIC_SELL"
        elif bullish_signals > bearish_signals * 1.5:
            sentiment = "Bullish"
            action_type = "FOMO_BUY"
        else:
            sentiment = random.choice(["Bullish", "Bearish", "Neutral"])
            action_type = "POST"

        return PersonaAction(
            agent_id=self.agent_id,
            action_type=action_type,
            sentiment=sentiment,
            rationale="Following retail momentum trends.",
            confidence=random.uniform(0.2, 0.6)
        )


class InstitutionalTraderPersona(BasePersona):
    """
    Simulates quantitative and fundamental institutional capital.
    Slower reaction, higher conviction, fading retail extremes.
    """
    def _determine_action(self, cycle: int, global_events: list, agent_actions: list) -> Optional[PersonaAction]:
        # Fade the retail herd if it becomes too extreme
        bearish_signals = sum(1 for a in agent_actions if a.data.get("sentiment") == "Bearish")
        bullish_signals = sum(1 for a in agent_actions if a.data.get("sentiment") == "Bullish")
        total = max(1, len(agent_actions))

        if bearish_signals / total > 0.8:
            sentiment = "Bullish"
            action_type = "LIQUIDITY_PROVISION"
            rationale = "Fading extreme retail pessimism."
        elif bullish_signals / total > 0.8:
            sentiment = "Bearish"
            action_type = "SHORT_SALE"
            rationale = "Fading extreme retail exuberance."
        else:
            sentiment = "Neutral"
            action_type = "HOLD"
            rationale = "Awaiting structural market clarity."

        return PersonaAction(
            agent_id=self.agent_id,
            action_type=action_type,
            sentiment=sentiment,
            rationale=rationale,
            confidence=random.uniform(0.7, 0.95)
        )


class RegulatorPersona(BasePersona):
    """
    Simulates regulatory bodies (e.g., Fed, SEC).
    Reacts to extreme systemic stress with dampening actions.
    """
    def _determine_action(self, cycle: int, global_events: list, agent_actions: list) -> Optional[PersonaAction]:
        bearish_signals = sum(1 for a in agent_actions if a.data.get("sentiment") == "Bearish")
        total = max(1, len(agent_actions))

        if bearish_signals / total > 0.9:
            return PersonaAction(
                agent_id=self.agent_id,
                action_type="INTERVENTION",
                sentiment="Bullish",
                rationale="Systemic risk detected. Announcing liquidity facilities.",
                confidence=1.0
            )

        # Regulators are typically inactive unless stressed
        if random.random() < 0.1:
            return PersonaAction(
                agent_id=self.agent_id,
                action_type="OBSERVE",
                sentiment="Neutral",
                rationale="Monitoring market stability.",
                confidence=0.9
            )

        return None
