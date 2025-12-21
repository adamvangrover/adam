from __future__ import annotations

import uuid
from datetime import date
from typing import Optional

from sqlalchemy import BigInteger, Boolean, Date, ForeignKey, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.types import CHAR, TypeDecorator


# SQLite compatible UUID type
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses CHAR(36) on generic platforms (SQLite).
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value

class Base(DeclarativeBase):
    pass

class FundMasterDB(Base):
    __tablename__ = 'fund_master'

    cik: Mapped[str] = mapped_column(String(10), primary_key=True)
    fund_name: Mapped[str] = mapped_column(String(255))
    fund_style: Mapped[str] = mapped_column(String(50)) # 'Hedge Fund', 'Family Office', etc.
    manager_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    whitelist_status: Mapped[bool] = mapped_column(Boolean, default=False)

    filings: Mapped[list["FilingEventDB"]] = relationship(back_populates="fund")

class FilingEventDB(Base):
    __tablename__ = 'filing_event'

    filing_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    cik: Mapped[str] = mapped_column(ForeignKey("fund_master.cik"))
    report_period: Mapped[date] = mapped_column(Date)
    filing_date: Mapped[date] = mapped_column(Date)
    accession_number: Mapped[str] = mapped_column(String(25))
    is_amendment: Mapped[bool] = mapped_column(Boolean, default=False)

    fund: Mapped["FundMasterDB"] = relationship(back_populates="filings")
    holdings: Mapped[list["HoldingDetailDB"]] = relationship(back_populates="filing")

class HoldingDetailDB(Base):
    __tablename__ = 'holdings_detail'

    holding_id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    filing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("filing_event.filing_id"))
    cusip: Mapped[str] = mapped_column(String(9))
    ticker: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    shares: Mapped[int] = mapped_column(BigInteger)
    value: Mapped[int] = mapped_column(BigInteger) # In thousands
    put_call: Mapped[Optional[str]] = mapped_column(String(4), nullable=True) # 'PUT', 'CALL', or None
    vote_sole: Mapped[int] = mapped_column(BigInteger, default=0)

    filing: Mapped["FilingEventDB"] = relationship(back_populates="holdings")

class SecurityMasterDB(Base):
    __tablename__ = 'securities_master'

    cusip: Mapped[str] = mapped_column(String(9), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10))
    name: Mapped[str] = mapped_column(String(255))
    sector: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

# Database Setup
engine = create_engine("sqlite:///core/institutional_radar/radar.db", echo=False)

def init_db():
    Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
