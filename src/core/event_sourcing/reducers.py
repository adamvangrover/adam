"""
Adam OS - Event Sourcing Reducers

Implements pure functions that take an aggregate state and a FinancialEvent,
and return a new, updated state based on the event payload.
These must be completely deterministic (no API calls, no random numbers).
"""

from typing import cast
import structlog
from pydantic import ValidationError

from src.core.event_sourcing.ledger import FinancialEvent
from src.core.event_sourcing.domain import (
    SecurityState, MarketState, SecurityIssuancePayload, RiskRatingPayload,
    TradingLevelPayload, PricingTargetPayload, MacroConditionPayload,
    NewsTriggerPayload
)

logger = structlog.get_logger(__name__)

def security_reducer(state: SecurityState, event: FinancialEvent) -> SecurityState:
    """
    Reduces events related to a specific financial security (aggregate_id = ticker).
    """
    # Create a fresh copy to ensure immutability during the replay process
    new_state = state.model_copy(deep=True)

    try:
        if event.event_type == "SECURITY_ISSUED":
            payload = SecurityIssuancePayload(**event.payload)
            new_state.asset_class = payload.asset_class
            new_state.shares_outstanding = payload.initial_shares
            new_state.current_price = payload.initial_price

        elif event.event_type == "RISK_RATING_UPDATED":
            payload = RiskRatingPayload(**event.payload)
            new_state.risk_ratings[payload.agency] = {
                "rating": payload.rating,
                "outlook": payload.outlook
            }

        elif event.event_type == "TRADING_LEVEL_CHANGED":
            payload = TradingLevelPayload(**event.payload)
            new_state.current_price = payload.price
            new_state.last_trading_volume = payload.volume

        elif event.event_type == "PRICING_TARGET_UPDATED":
            payload = PricingTargetPayload(**event.payload)
            new_state.analyst_targets[payload.analyst_id] = payload.target_price

        elif event.event_type == "NEWS_TRIGGERED":
            payload = NewsTriggerPayload(**event.payload)
            # Simple running average of sentiment
            total_sentiment = (new_state.news_sentiment_aggregate * new_state.news_count) + payload.sentiment_score
            new_state.news_count += 1
            new_state.news_sentiment_aggregate = total_sentiment / new_state.news_count

    except ValidationError as e:
        logger.error("reducer_payload_validation_failed", event_id=event.event_id, event_type=event.event_type, error=str(e))
        # In a strict CQRS system, if an event is in the ledger but invalid against the *current*
        # schema, we might need upcasting logic. For now, we skip applying the malformed event.
        return state

    return new_state


def market_reducer(state: MarketState, event: FinancialEvent) -> MarketState:
    """
    Reduces events related to global macroeconomic conditions.
    """
    new_state = state.model_copy(deep=True)

    try:
        if event.event_type == "MACRO_CONDITION_CHANGED":
            payload = MacroConditionPayload(**event.payload)
            new_state.indicators[payload.indicator] = payload.value
            new_state.current_regime = payload.regime

    except ValidationError as e:
        logger.error("reducer_payload_validation_failed", event_id=event.event_id, event_type=event.event_type, error=str(e))
        return state

    return new_state
