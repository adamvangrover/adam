from pydantic import BaseModel, Field

class Command(BaseModel):
    """Base class for all commands."""
    entity_id: str

class OriginateLoanCommand(Command):
    """Command to originate a new loan."""
    principal_amount: float = Field(..., gt=0)
    asset_value: float = Field(..., gt=0)

class UpdateDebtCommand(Command):
    """Command to update the debt amount."""
    new_debt_amount: float = Field(..., ge=0)

class RevalueAssetCommand(Command):
    """Command to revalue the underlying asset."""
    new_asset_value: float = Field(..., gt=0)
