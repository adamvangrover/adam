"""
Market Mayhem - WhaleScanner Module
Author: Principal Software Architect
Description: Core logic for detecting institutional accumulation of distressed assets
             via 13F filings, utilizing edgartools and robust error handling.
"""

import pandas as pd
from edgar import Company, Filing, set_identity
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import logging
import time

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for Type Safety ---
class Holding(BaseModel):
    """
    Represents a single position in a 13F filing.
    Strict typing prevents data corruption from XML parsing artifacts.
    """
    issuer: str = Field(..., alias="nameOfIssuer")
    cusip: str
    ticker: Optional[str] = None
    value: float
    shares: float
    share_type: str  # 'SH' or 'PRN'
    discretion: str

    @validator('cusip')
    def validate_cusip(cls, v):
        if len(v)!= 9:
            raise ValueError(f"Invalid CUSIP length: {v}")
        return v

class WhaleSignal(BaseModel):
    """
    Represents a detected signal of activity (Entry, Exit, Accumulation).
    """
    fund_name: str
    ticker: str
    signal_type: str  # 'NEW_ENTRY', 'ACCUMULATION', 'LIQUIDATION'
    change_pct: float
    conviction_score: float  # Value / Total Portfolio (Simplified)
    description: str
    share_type: str # Critical for distinguishing Debt (PRN) from Equity (SH)

# --- Core Logic Class ---
class WhaleScanner:
    def __init__(self, user_agent: str):
        """
        Initialize the WhaleScanner with SEC identity.

        Args:
            user_agent: 'Name <email>' string for SEC EDGAR compliance.
        """
        set_identity(user_agent)
        # Known Vulture/Distressed Debt Funds CIKs
        self.vulture_ciks = {
            "OAKTREE": "0000949509",
            "APOLLO": "0001411494",
            "CENTERBRIDGE": "0001484836",
            "BAUPOST": "0001061768",
            "ARES": "0001176948"
        }

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    def _fetch_filings(self, cik: str, limit: int = 2) -> List[Filing]:
        """
        Fetches 13F-HR filings with exponential backoff for rate limits.
        Filters strictly for 13F-HR to avoid 13F-NT double counting.

        Args:
            cik: Central Index Key of the fund.
            limit: Number of recent filings to retrieve.
        """
        logger.info(f"Fetching 13F-HR filings for CIK: {cik}")
        company = Company(cik)
        # Explicitly filter for 13F-HR. Do not process 13F-NT.
        filings = company.get_filings(form="13F-HR").latest(limit)

        # edgartools returns a single object if limit=1, ensure list
        if not isinstance(filings, list):
             # Depending on edgartools version, latest(n) might return a Filings object which is iterable
             # or a single Filing. We need to handle this.
             # For the purpose of this architecture, we assume an iterable list of Filing objects.
             if hasattr(filings, '__iter__'):
                 return list(filings)
             else:
                 return [filings]

        return filings

    def _parse_13f_xml(self, filing: Filing) -> pd.DataFrame:
        """
        Extracts the Information Table XML into a normalized DataFrame.
        Uses edgartools 'smart object' parsing capabilities.
        """
        try:
            # Edgartools.obj() parses the XML automatically into a ThirteenF object
            thirteen_f = filing.obj()
            if not thirteen_f or not thirteen_f.infotable:
                logger.warning(f"No info table found for filing {filing.accession_no}")
                return pd.DataFrame()

            # Access the infotable (list of holdings)
            # The library handles the XML tags: nameOfIssuer, cusip, value, sshPrnamt
            df = thirteen_f.infotable.to_dataframe()

            # Normalize column names to match our internal schema
            # Edgartools dataframe columns are typically: Issuer, CUSIP, Value, Shares, etc.
            df = df.rename(columns={
                "Issuer": "issuer",
                "Cusip": "cusip",
                "Value": "value",
                "Shares": "shares",
                "Type": "share_type", # maps to sshPrnamtType
                "InvestmentDiscretion": "discretion",
                "Ticker": "ticker" # edgartools attempts resolution
            })

            return df
        except Exception as e:
            logger.error(f"Failed to parse infotable for filing {filing.accession_no}: {e}")
            return pd.DataFrame()

    def calculate_fund_sentiment(self, fund_key: str, lookback: int = 2) -> List:
        """
        Analyzes Quarter-over-Quarter (QoQ) changes to detect VULTURE_ENTRY signals.
        """
        cik = self.vulture_ciks.get(fund_key)
        if not cik:
            raise ValueError(f"Unknown fund key: {fund_key}")

        # Fetch filings
        filings = self._fetch_filings(cik, limit=lookback)
        if len(filings) < 2:
            logger.warning(f"Insufficient filings for {fund_key} QoQ comparison.")
            return []

        # Current Quarter (Q0) and Previous Quarter (Q-1)
        # Assuming filings are returned in descending date order (newest first)
        q0_df = self._parse_13f_xml(filings[0])
        q1_df = self._parse_13f_xml(filings[1])

        if q0_df.empty or q1_df.empty:
            return []

        # Logic for Distress Detection
        signals = []

        # Merge on CUSIP to compare positions
        # Using CUSIP is more reliable than Ticker for distressed/delisted assets
        merged = pd.merge(
            q0_df,
            q1_df,
            on='cusip',
            how='outer',
            suffixes=('_q0', '_q1'),
            indicator=True
        )

        for _, row in merged.iterrows():
            # Resolve Ticker: Q0 > Q1 > CUSIP fallback
            ticker = row.get('ticker_q0') or row.get('ticker_q1') or row['cusip']
            val_q0 = row.get('value_q0', 0)
            share_type = row.get('share_type_q0') or row.get('share_type_q1')
            issuer = row.get('issuer_q0') or row.get('issuer_q1')

            # 1. Detect New Entries (Vulture Entering)
            if row['_merge'] == 'left_only':
                desc = f"New Position: {issuer}"
                if share_type == 'PRN':
                    desc += " (DEBT/CONVERTIBLE - HIGH CONVICTION)"

                signals.append(WhaleSignal(
                    fund_name=fund_key,
                    ticker=str(ticker),
                    signal_type="VULTURE_ENTRY",
                    change_pct=100.0,
                    conviction_score=float(val_q0),
                    description=desc,
                    share_type=str(share_type)
                ))

            # 2. Detect Aggressive Accumulation
            elif row['_merge'] == 'both':
                shares_q0 = row['shares_q0']
                shares_q1 = row['shares_q1']

                if shares_q1 > 0:
                    pct_change = ((shares_q0 - shares_q1) / shares_q1) * 100
                    if pct_change > 20: # 20% Threshold for "Aggressive"
                        signals.append(WhaleSignal(
                            fund_name=fund_key,
                            ticker=str(ticker),
                            signal_type="ACCUMULATION",
                            change_pct=pct_change,
                            conviction_score=float(val_q0),
                            description=f"Increased position by {pct_change:.1f}%",
                            share_type=str(share_type)
                        ))

        return signals
