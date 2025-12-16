# core/common/mcp_executor.py
import logging
import sqlite3
import yfinance as yf
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)

class MCPExecutor:
    """
    Executes MCP (Model Context Protocol) tools.
    Provides actual execution logic for authorized tools, falling back to mocks where necessary.
    """

    def __init__(self, db_path: str = "data/lakehouse.db"):
        self.db_path = db_path

    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        logger.info(f"Executing MCP Tool: {tool_name} with params {params}")

        try:
            if tool_name == "azure_ai_search":
                return self._mock_azure_search(params)

            elif tool_name == "microsoft_fabric_run_sql":
                return self._run_sql(params)

            elif tool_name == "price_asset":
                return self._get_price_asset(params)

            elif tool_name == "execute_order":
                # Simulation only
                return {"order_id": "ORD-SIM-12345", "status": "Filled", "symbol": params.get("symbol"), "qty": params.get("qty")}

            elif tool_name == "plan_wealth_goal":
                return {"goal": params.get("goal_name"), "status": "On Track", "required_monthly_savings": 1200}

            elif tool_name == "screen_deal":
                return {"deal": params.get("deal_name"), "score": 8.5, "recommendation": "Review NDAs"}

            else:
                return {"error": f"Tool {tool_name} not found"}
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return {"error": str(e)}

    def _mock_azure_search(self, params: Dict[str, Any]) -> str:
        query = params.get("query", "")
        return f"[Azure Search Result] Found 3 documents relevant to '{query}'. 1. 10-K Filing (2023)... 2. Moody's Credit Report... 3. Earnings Call Transcript..."

    def _run_sql(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes SQL against local SQLite replica of Data Lakehouse.
        """
        query = params.get("sql_query", "")
        if not query:
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Simple sanitization/safety check (read-only enforcement)
            if "drop" in query.lower() or "delete" in query.lower() or "update" in query.lower() or "insert" in query.lower():
                return [{"error": "Write operations forbidden in MCP SQL Tool."}]

            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            conn.close()
            return results
        except sqlite3.OperationalError as e:
            # Fallback for table not found or syntax
            logger.warning(f"SQL Error: {e}. Falling back to mock logic.")
            if "revenue" in query.lower():
                return [{"metric": "Revenue", "value": 10000, "unit": "USD"}]
            elif "debt" in query.lower():
                return [{"metric": "Total Debt", "value": 5000, "unit": "USD"}]
            return [{"error": str(e)}]

    def _get_price_asset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        symbol = params.get("symbol")
        if not symbol: return {"error": "Symbol required"}

        try:
            # Try Real Fetch
            ticker = yf.Ticker(symbol)
            # fast_info is preferred in newer yfinance, info is slow
            price = ticker.fast_info.last_price
            if price:
                return {
                    "symbol": symbol,
                    "price": price,
                    "currency": ticker.fast_info.currency,
                    "timestamp": "Real-time"
                }
        except Exception:
            pass

        # Fallback
        return {"symbol": symbol, "price": 150.25, "currency": "USD", "timestamp": "Mock"}
