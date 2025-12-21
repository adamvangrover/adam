import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from uuid import UUID
from datetime import date
from sqlalchemy import func
from scipy.stats import zscore
from core.institutional_radar.database import SessionLocal, FilingEventDB, HoldingDetailDB, FundMasterDB, SecurityMasterDB
from core.utils.logging_utils import get_logger

logger = get_logger("institutional_radar.analytics")


class InstitutionalRadarAnalytics:
    def __init__(self, session=None):
        self.session = session or SessionLocal()

    def get_quarterly_holdings(self, year: int, quarter: int) -> pd.DataFrame:
        """
        Retrieves all holdings for a specific quarter.
        Applies the 'Long-Only' filter (put_call is NULL).
        """
        # Calculate approx date range
        # Q1: Jan-Mar (Filed by May 15) -> Report Period Mar 30
        # This logic is a bit loose on dates, usually we select by report_period

        # report_period is usually end of quarter
        q_end_month = quarter * 3
        report_date = date(year, q_end_month, 30)
        if q_end_month in [3, 12]:
            report_date = date(year, q_end_month, 31)
        elif q_end_month in [6, 9]:
            report_date = date(year, q_end_month, 30)

        # Allow some wiggle room or select exact
        stmt = (
            self.session.query(
                HoldingDetailDB.cusip,
                HoldingDetailDB.shares,
                HoldingDetailDB.value,
                FundMasterDB.fund_style,
                FundMasterDB.fund_name,
                SecurityMasterDB.ticker,
                SecurityMasterDB.sector
            )
            .join(FilingEventDB, HoldingDetailDB.filing_id == FilingEventDB.filing_id)
            .join(FundMasterDB, FilingEventDB.cik == FundMasterDB.cik)
            .outerjoin(SecurityMasterDB, HoldingDetailDB.cusip == SecurityMasterDB.cusip)
            .filter(FilingEventDB.report_period == report_date)
            .filter(HoldingDetailDB.put_call == None)  # Long Only
        )

        df = pd.read_sql(stmt.statement, self.session.bind)
        return df

    def calculate_crowding_score(self, year: int, quarter: int) -> pd.DataFrame:
        """
        Calculates Crowding Score (S_crowd) for every stock.
        S = w1*N + w2*C + w3*L
        """
        df = self.get_quarterly_holdings(year, quarter)
        if df.empty:
            logger.warning("No data found for crowding score calculation")
            return pd.DataFrame()

        # Breadth (N): Unique funds holding the security
        breadth = df.groupby('cusip')['fund_name'].nunique().rename('N')

        # Concentration (C): % held by top 50 active hedge funds
        # Filter for hedge funds
        hf_df = df[df['fund_style'] == 'Hedge Fund']
        concentration = hf_df.groupby('cusip')['shares'].sum().rename('C_raw')

        # We need Total Shares Outstanding to calculate %
        # Assume we have it in SecurityMaster or mock it.
        # For now, we'll use raw shares as a proxy or skip normalization if TSO is missing.
        # Ideally: C = (Sum Shares of Top 50) / TSO

        # Liquidity (L): Days to Liquidate. Requires Avg Volume.
        # We will mock L for now or use a placeholder.

        metrics = pd.concat([breadth, concentration], axis=1).fillna(0)

        # Normalize metrics (0-100 scale or Z-score)
        metrics['N_score'] = (metrics['N'] - metrics['N'].min()) / (metrics['N'].max() - metrics['N'].min())
        metrics['C_score'] = (metrics['C_raw'] - metrics['C_raw'].min()) / \
            (metrics['C_raw'].max() - metrics['C_raw'].min())
        metrics['L_score'] = 0.5  # Placeholder

        w1, w2, w3 = 0.4, 0.4, 0.2
        metrics['crowding_score'] = (w1 * metrics['N_score'] + w2 * metrics['C_score'] + w3 * metrics['L_score']) * 100

        return metrics[['crowding_score', 'N', 'C_raw']]

    def calculate_sector_flows(self, year: int, quarter: int) -> pd.DataFrame:
        """
        Calculates Net Flow per sector compared to previous quarter.
        """
        curr_df = self.get_quarterly_holdings(year, quarter)

        prev_year = year
        prev_q = quarter - 1
        if prev_q == 0:
            prev_q = 4
            prev_year -= 1

        prev_df = self.get_quarterly_holdings(prev_year, prev_q)

        if curr_df.empty or prev_df.empty:
            logger.warning("Insufficient data for flow analysis")
            return pd.DataFrame()

        curr_sector = curr_df.groupby('sector')['value'].sum()
        prev_sector = prev_df.groupby('sector')['value'].sum()

        flows = (curr_sector - prev_sector).rename('net_flow')

        # Calculate Z-Score relative to history (mocking history here)
        # Ideally fetch 12 quarters
        # For now, just return raw flow

        return flows.reset_index()

    def detect_cluster_buys(self, year: int, quarter: int, whitelist_only: bool = True) -> pd.DataFrame:
        """
        Detects simultaneous entry by multiple funds.
        """
        curr_df = self.get_quarterly_holdings(year, quarter)

        # Logic: Find funds that hold it now but didn't hold it last quarter
        # This requires checking previous quarter holdings per fund/cusip

        prev_year = year
        prev_q = quarter - 1
        if prev_q == 0:
            prev_q = 4
            prev_year -= 1

        prev_df = self.get_quarterly_holdings(prev_year, prev_q)

        # Create set of (fund, cusip) for previous quarter
        prev_holdings = set(zip(prev_df['fund_name'], prev_df['cusip']))

        new_positions = []
        for _, row in curr_df.iterrows():
            if (row['fund_name'], row['cusip']) not in prev_holdings:
                new_positions.append(row)

        new_pos_df = pd.DataFrame(new_positions)
        if new_pos_df.empty:
            return pd.DataFrame()

        # Group by CUSIP to find clusters
        clusters = new_pos_df.groupby(['cusip', 'ticker', 'sector'])['fund_name'].agg(['count', list])
        clusters = clusters.rename(columns={'count': 'fund_count', 'list': 'funds'})

        # Filter for threshold > 3
        significant_clusters = clusters[clusters['fund_count'] >= 3].sort_values('fund_count', ascending=False)

        return significant_clusters

    def close(self):
        self.session.close()
