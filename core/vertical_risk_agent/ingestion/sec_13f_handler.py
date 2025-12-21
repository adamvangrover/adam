import logging
import xml.etree.ElementTree as ET

import pandas as pd

logger = logging.getLogger(__name__)

class Sec13FHandler:
    """
    Handles fetching and parsing of SEC 13F filings to track institutional holdings.
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "Adam/2.0 (internal-research@adam-project.io)"
        }
        # Mock data for demonstration purposes (Q3 2025 context)
        self.mock_data = {
            "0001067983": { # Berkshire Hathaway
                "2025-Q3": [
                    {"ticker": "AAPL", "value": 80000000000, "shares": 350000000, "action": "REDUCE"},
                    {"ticker": "GOOGL", "value": 4300000000, "shares": 17850000, "action": "NEW"},
                    {"ticker": "DPZ", "value": 150000000, "shares": 348000, "action": "NEW"}
                ],
                "2025-Q2": [
                    {"ticker": "AAPL", "value": 140000000000, "shares": 600000000, "action": "HOLD"}
                ]
            },
            "0001037389": { # Renaissance Technologies
                "2025-Q3": [
                    {"ticker": "PLTR", "value": 0, "shares": 0, "action": "EXIT"},
                    {"ticker": "GOOGL", "value": 633000000, "shares": 3500000, "action": "ADD"},
                    {"ticker": "VRTX", "value": 200000000, "shares": 500000, "action": "NEW"}
                ],
                "2025-Q2": [
                    {"ticker": "PLTR", "value": 120000000, "shares": 4900000, "action": "HOLD"}
                ]
            }
        }

    def fetch_holdings(self, cik: str, period: str = "2025-Q3") -> pd.DataFrame:
        """
        Fetches holdings for a given CIK and period.
        In a real scenario, this would query SEC EDGAR.
        For now, it returns mock data for the Q3 2025 scenario.
        """
        logger.info(f"Fetching 13F holdings for CIK: {cik}, Period: {period}")

        # Simulate network delay or check
        if cik in self.mock_data and period in self.mock_data[cik]:
            data = self.mock_data[cik][period]
            df = pd.DataFrame(data)
            return df

        # Fallback to empty if not in mock
        logger.warning(f"No data found for {cik} in {period}, returning empty DataFrame.")
        return pd.DataFrame(columns=["ticker", "value", "shares", "action"])

    def parse_xml_file(self, file_path: str) -> pd.DataFrame:
        """
        Parses a raw XML 13F file.
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Namespace handling
            ns_map = {}
            if '}' in root.tag:
                ns_url = root.tag.split('}')[0].strip('{')
                ns_map = {'ns': ns_url}

            rows = []
            # Find infoTable elements
            # Usually <infoTable> is the repeating element
            # We might need to handle different XML structures

            tables = root.findall('.//ns:infoTable', ns_map) if ns_map else root.findall('.//infoTable')

            for table in tables:
                def get_text(tag):
                    elem = table.find(f'ns:{tag}', ns_map) if ns_map else table.find(tag)
                    return elem.text if elem is not None else None

                issuer = get_text('nameOfIssuer')
                cusip = get_text('cusip')
                value = get_text('value')
                shrs_elem = table.find('ns:shrsOrPrnAmt', ns_map) if ns_map else table.find('shrsOrPrnAmt')
                shares = None
                if shrs_elem is not None:
                    sshPrnamt = shrs_elem.find('ns:sshPrnamt', ns_map) if ns_map else shrs_elem.find('sshPrnamt')
                    shares = sshPrnamt.text if sshPrnamt is not None else None

                rows.append({
                    "issuer": issuer,
                    "cusip": cusip,
                    "value": float(value) * 1000 if value else 0, # Usually in thousands
                    "shares": float(shares) if shares else 0
                })

            return pd.DataFrame(rows)

        except Exception as e:
            logger.error(f"Failed to parse XML file {file_path}: {e}")
            return pd.DataFrame()

    def calculate_delta(self, current_df: pd.DataFrame, previous_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the change in holdings between two quarters.
        Expects DataFrames with 'ticker' (or 'cusip') and 'shares' columns.
        """
        if current_df.empty and previous_df.empty:
            return pd.DataFrame()

        # Ensure we have a join key, prefer ticker if available, else cusip
        join_key = 'ticker' if 'ticker' in current_df.columns and 'ticker' in previous_df.columns else 'cusip'

        if join_key not in current_df.columns:
            # If mocking without cusip/ticker properly, just return current
            return current_df

        merged = pd.merge(
            current_df,
            previous_df,
            on=join_key,
            how='outer',
            suffixes=('_curr', '_prev')
        )

        merged['shares_curr'] = merged['shares_curr'].fillna(0)
        merged['shares_prev'] = merged['shares_prev'].fillna(0)
        merged['share_change'] = merged['shares_curr'] - merged['shares_prev']
        merged['pct_change'] = (merged['share_change'] / merged['shares_prev']).replace([float('inf'), -float('inf')], 0).fillna(0)

        def determine_action(row):
            if row['shares_prev'] == 0 and row['shares_curr'] > 0:
                return 'NEW'
            elif row['shares_curr'] == 0 and row['shares_prev'] > 0:
                return 'EXIT'
            elif row['share_change'] > 0:
                return 'ADD'
            elif row['share_change'] < 0:
                return 'REDUCE'
            else:
                return 'HOLD'

        merged['action_calculated'] = merged.apply(determine_action, axis=1)

        # Cleanup
        cols = [join_key, 'shares_curr', 'shares_prev', 'share_change', 'pct_change', 'action_calculated']
        if 'value_curr' in merged.columns:
            cols.append('value_curr')

        return merged[cols]
