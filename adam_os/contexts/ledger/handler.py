from typing import List, Optional
import structlog
from adam_os.core.events import DomainEvent, LoanOriginated, DebtUpdated, AssetRevalued
from adam_os.contexts.ledger.commands import Command, OriginateLoanCommand, UpdateDebtCommand, RevalueAssetCommand
from adam_os.contexts.ledger.aggregate import FinancialEntity

logger = structlog.get_logger()

class CommandHandler:
    """Handles incoming commands and generates corresponding domain events."""

    def __init__(self) -> None:
        # In a real system, we'd inject an Event Store repository here
        # to load the entity's past events before processing the command.
        logger.info("initialized_command_handler")

    def handle(self, command: Command, previous_events: Optional[List[DomainEvent]] = None) -> List[DomainEvent]:
        """Processes a command and returns the newly generated domain events."""

        entity = FinancialEntity(entity_id=command.entity_id)
        if previous_events:
            entity.load_from_history(previous_events)

        logger.info("handling_command", command_type=type(command).__name__, entity_id=command.entity_id)

        if isinstance(command, OriginateLoanCommand):
            if entity.version > 0:
                raise ValueError("Cannot originate a loan that already exists.")

            event = LoanOriginated(
                entity_id=command.entity_id,
                principal_amount=command.principal_amount,
                asset_value=command.asset_value
            )
            entity.apply(event)

        elif isinstance(command, UpdateDebtCommand):
            if entity.version == 0:
                raise ValueError("Cannot update debt for an uninitialized loan.")

            event = DebtUpdated(
                entity_id=command.entity_id,
                new_debt_amount=command.new_debt_amount
            )
            entity.apply(event)

        elif isinstance(command, RevalueAssetCommand):
            if entity.version == 0:
                raise ValueError("Cannot revalue asset for an uninitialized loan.")

            event = AssetRevalued(
                entity_id=command.entity_id,
                new_asset_value=command.new_asset_value
            )
            entity.apply(event)

        else:
            raise ValueError(f"Unknown command type: {type(command).__name__}")

        uncommitted = entity.get_uncommitted_events()
        logger.info("command_handled", generated_events_count=len(uncommitted))
        return uncommitted
