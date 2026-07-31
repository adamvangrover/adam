from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from adam_os.contexts.ledger.commands import OriginateLoanCommand, UpdateDebtCommand, RevalueAssetCommand
from adam_os.contexts.ledger.handler import CommandHandler
from adam_os.core.events import DomainEvent

router = APIRouter(prefix="/ledger", tags=["ledger"])

def get_command_handler() -> CommandHandler:
    return CommandHandler()

class CommandResponse(BaseModel):
    message: str
    events_generated: int

@router.post("/originate", response_model=CommandResponse)
async def originate_loan(
    command: OriginateLoanCommand,
    handler: CommandHandler = Depends(get_command_handler)
) -> CommandResponse:
    try:
        events = handler.handle(command)
        return CommandResponse(
            message="Loan successfully originated.",
            events_generated=len(events)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/update_debt", response_model=CommandResponse)
async def update_debt(
    command: UpdateDebtCommand,
    handler: CommandHandler = Depends(get_command_handler)
) -> CommandResponse:
    try:
        # In a real app we'd load previous events from a DB
        # For this prototype we pass None which acts like a brand new but invalid state for updating.
        # This will raise an error as intended because it needs version > 0
        events = handler.handle(command)
        return CommandResponse(
            message="Debt successfully updated.",
            events_generated=len(events)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/revalue_asset", response_model=CommandResponse)
async def revalue_asset(
    command: RevalueAssetCommand,
    handler: CommandHandler = Depends(get_command_handler)
) -> CommandResponse:
    try:
        events = handler.handle(command)
        return CommandResponse(
            message="Asset successfully revalued.",
            events_generated=len(events)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
