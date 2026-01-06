from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from uuid import UUID
from datetime import datetime

class UnifiedOrder(BaseModel):
    """
    Table 1: Unified Ledger Order Schema Hierarchy
    """
    order_id: UUID = Field(..., description="Unique identifier for the instruction.")
    parent_id: Optional[UUID] = Field(None, description="Reference to the aggregate strategy order (AM).")
    client_id: str = Field(..., description="Identifier for the ultimate beneficiary (WM).")
    desk_id: str = Field(..., description="Identifier for the trading desk taking the risk (IB).")
    strategy_tag: Optional[str] = Field(None, description="Algo strategy ID (e.g., 'VWAP_01').")
    intent_side: Literal["Buy", "Sell"] = Field(..., description="Buy/Sell intent.")
    internalization_flag: bool = Field(False, description="True if filled against internal inventory.")
    symbol: str
    price: Optional[float]
    quantity: float

    # Bitemporal Fields (Chronos)
    valid_time_start: datetime
    valid_time_end: Optional[datetime] = None
    transaction_time_start: datetime
    transaction_time_end: Optional[datetime] = None

class RefactorProposal(BaseModel):
    """
    Appendix A: MCP Tool Definition for Evolutionary Architect
    """
    file_path: str = Field(..., description="Path to the python file to be refactored.")
    target_node_type: str = Field(..., description="The specific AST node type to target (e.g., 'For', 'FunctionDef', 'Call').")
    optimization_goal: Literal["latency", "memory", "readability"] = Field(..., description="The primary metric to improve.")

class MCPResource(BaseModel):
    uri: str
    mime_type: str
    name: str

class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: dict
