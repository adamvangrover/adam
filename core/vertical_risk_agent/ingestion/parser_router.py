import logging
from typing import Any, Dict

from .xbrl_handler import XBRLHandler

logger = logging.getLogger(__name__)

class ParserRouter:
    """
    Routes parsing requests to the most appropriate engine:
    1. XBRL (Gold Standard) - if available.
    2. Vision/LlamaParse (Fallback) - for PDFs/Images.
    """

    def __init__(self):
        self.xbrl_handler = XBRLHandler()

    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """
        Determines the file type and routes to the correct parser.
        """
        if self._is_xbrl(file_path):
            logger.info("Routing to XBRL Handler")
            return self.xbrl_handler.parse_filing(file_path)
        else:
            logger.info("Routing to Vision Parser (LlamaParse)")
            return self._parse_with_vision(file_path)

    def _is_xbrl(self, file_path: str) -> bool:
        return file_path.endswith('.xml') or file_path.endswith('.xbrl')

    def _parse_with_vision(self, file_path: str) -> Dict[str, Any]:
        """
        Fallback parser using Vision-Language Models (e.g. LlamaParse).
        """
        # Integration with LlamaParse would go here
        # result = LlamaParse(result_type="markdown").load_data(file_path)

        # Mock return
        return {
            "balance_sheet": {
                "cash_equivalents": 48000000.0, # Slightly worse accuracy simulated
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
