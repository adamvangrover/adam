from enum import Enum
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field

class ExecutionVenue(str, Enum):
    INTERNAL = "INTERNAL"
    NASDAQ = "NASDAQ"
    ONYX = "ONYX"
    LIINK = "LIINK"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"

class TokenizedAsset(BaseModel):
    """
    Represents an asset on the Unified Ledger, potentially tokenized on a Kinexys-like chain.
    """
    asset_id: str
    symbol: str
    name: str
    is_tokenized: bool = False
    chain_id: Optional[str] = None
    smart_contract_address: Optional[str] = None
    token_standard: Optional[str] = None # e.g. "ERC-20", "JPM-Coin"

class ParentOrder(BaseModel):
    """
    Represents the macro intent (Strategy/Client Level).
    """
    order_id: UUID = Field(default_factory=uuid4)
    client_id: str
    strategy_tag: Optional[str] = None # e.g. "Fortress_Rebalance"
    symbol: str
    side: OrderSide
    quantity: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "PENDING"

class ChildOrder(BaseModel):
    """
    Represents the specific execution instruction derived from a ParentOrder.
    """
    order_id: UUID = Field(default_factory=uuid4)
    parent_id: UUID
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None # Limit price, None for Market
    order_type: OrderType
    venue: ExecutionVenue
    desk_id: str # The IB desk taking the risk or routing it
    internalization_flag: bool = False
    time_in_force: TimeInForce = TimeInForce.GTC
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "PENDING"
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0

class LedgerEntry(BaseModel):
    """
    Double-entry accounting log for the Unified Ledger.
    """
    entry_id: UUID = Field(default_factory=uuid4)
    transaction_id: UUID # Links related entries (Debit/Credit)
    account_id: str
    asset_id: str
    amount: float # Positive for Debit, Negative for Credit (or vice versa depending on convention)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = {}
