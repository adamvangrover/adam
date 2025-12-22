from __future__ import annotations
from datetime import date
from typing import Optional, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class FundMaster(BaseModel):
    cik: str = Field(..., max_length=10, description="The unique Central Index Key")
    fund_name: str = Field(..., max_length=255)
    fund_style: Literal['Hedge Fund', 'Family Office', 'Quant', 'Bank', 'Other'] = Field('Other')
    manager_name: Optional[str] = Field(None, max_length=255)
    whitelist_status: bool = Field(False, description="Flag to include in high-priority trend analysis")

    model_config = ConfigDict(from_attributes=True)


class FilingEvent(BaseModel):
    filing_id: UUID
    cik: str
    report_period: date
    filing_date: date
    accession_number: str = Field(..., max_length=25)
    is_amendment: bool = False

    model_config = ConfigDict(from_attributes=True)


class HoldingDetail(BaseModel):
    holding_id: UUID
    filing_id: UUID
    cusip: str = Field(..., max_length=9)
    ticker: Optional[str] = Field(None, max_length=10)
    shares: int
    value: int = Field(..., description="Reported value in thousands (x$1000)")
    put_call: Optional[Literal['PUT', 'CALL']] = None
    vote_sole: int = 0

    model_config = ConfigDict(from_attributes=True)


class SecurityMaster(BaseModel):
    cusip: str = Field(..., max_length=9)
    ticker: str = Field(..., max_length=10)
    name: str = Field(..., max_length=255)
    sector: Optional[str] = Field(None, max_length=50)
    industry: Optional[str] = Field(None, max_length=50)

    model_config = ConfigDict(from_attributes=True)
