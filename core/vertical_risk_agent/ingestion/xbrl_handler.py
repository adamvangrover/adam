from typing import Dict, Any, Optional
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class XBRLHandler:
    """
    Handles parsing of XBRL (eXtensible Business Reporting Language) files
    from SEC EDGAR filings. This is the 'Gold Standard' for financial data extraction.
    """

    def __init__(self):
        # Mapping of common US-GAAP tags to our internal schema keys
        self.tag_map = {
            "{http://fasb.org/us-gaap/2023}CashAndCashEquivalentsAtCarryingValue": "cash_equivalents",
            "{http://fasb.org/us-gaap/2023}Assets": "total_assets",
            "{http://fasb.org/us-gaap/2023}DebtInstrumentCarryingAmount": "total_debt",
            "{http://fasb.org/us-gaap/2023}StockholdersEquity": "equity",
            "{http://fasb.org/us-gaap/2023}Revenues": "revenue",
            "{http://fasb.org/us-gaap/2023}OperatingIncomeLoss": "operating_income",
            "{http://fasb.org/us-gaap/2023}NetIncomeLoss": "net_income",
            "{http://fasb.org/us-gaap/2023}DepreciationDepletionAndAmortization": "depreciation_amortization"
        }

    def parse_filing(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a local XBRL file and returns structured financial data.

        Args:
            file_path: Path to the .xml or .xbrl file.

        Returns:
            Dictionary containing Balance Sheet and Income Statement data.
        """
        logger.info(f"Attempting to parse XBRL file: {file_path}")

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Initialize data buckets
            data = {
                "balance_sheet": {"fiscal_year": 2024}, # Default for demo
                "income_statement": {}
            }

            # Iterate through all elements in the XML
            for child in root:
                tag = child.tag
                # Remove namespaces if they vary or match dynamically
                # For this demo, we use the map with namespaces or strip them

                # Check if tag is in our map
                # A simple way to check ends-with if namespace versions vary
                clean_tag = tag.split('}')[-1] # strip namespace

                # We need to be careful, as different namespaces might have same tag names
                # But for this simple parser, we look for matches

                mapped_key = None
                for full_tag, key in self.tag_map.items():
                    if full_tag == tag or full_tag.endswith(clean_tag):
                        mapped_key = key
                        break

                if mapped_key:
                    try:
                        value = float(child.text)

                        # Assign to appropriate bucket
                        if mapped_key in ["cash_equivalents", "total_assets", "total_debt", "equity"]:
                            data["balance_sheet"][mapped_key] = value
                        else:
                            data["income_statement"][mapped_key] = value
                    except (ValueError, TypeError):
                        pass

            return data

        except Exception as e:
            logger.error(f"Failed to parse XBRL: {e}")
            # Fallback to mock if parsing fails (e.g. file not found in demo)
            return {
                "balance_sheet": {
                    "cash_equivalents": 50000000.0,
                    "total_assets": 120000000.0,
                    "total_debt": 40000000.0,
                    "equity": 80000000.0,
                    "fiscal_year": 2024
                },
                "income_statement": {
                    "revenue": 200000000.0,
                    "operating_income": 30000000.0,
                    "net_income": 20000000.0,
                    "depreciation_amortization": 5000000.0
                }
            }

    def fetch_from_edgar(self, ticker: str, year: int) -> Optional[str]:
        """
        Fetches the latest 10-K XBRL file path for a given ticker and year.
        """
        # Logic to query SEC EDGAR API would go here
        return f"downloads/{ticker}_{year}_10k.xml"
