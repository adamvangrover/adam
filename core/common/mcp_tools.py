# core/common/mcp_tools.py

"""
Defines the strictly enforced MCP (Model Context Protocol) tools for Adam v23.5.
Natural language tools are FORBIDDEN.
"""

MCP_TOOLS_DEFINITIONS = [
    {
        "name": "azure_ai_search",
        "description": "Searches and retrieves excerpts from unstructured documents (rating reports, filings).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Highly specific keyword query (e.g. 'NextEra Energy S&P FFO/Debt')"
                },
                "top_k": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of document chunks to return."
                },
                "filter": {
                    "type": "string",
                    "description": "OData filter string for metadata (e.g. \"category eq 'Report'\")"
                }
            },
            "required": [
                "query"
            ]
        }
    },
    {
        "name": "microsoft_fabric_run_sql",
        "description": "Executes read-only SQL queries against the Data Lakehouse for structured financials.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "Valid T-SQL SELECT statement. (e.g. 'SELECT * FROM metrics WHERE ticker = ''NEE''')"
                }
            },
            "required": [
                "sql_query"
            ]
        }
    },
    # --- Expanded FO Super-App Tools ---
    {
        "name": "price_asset",
        "description": "Get real-time price of an asset.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Asset ticker (e.g. AAPL)"},
                "side": {"type": "string", "enum": ["bid", "ask", "mid"], "default": "mid"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "execute_order",
        "description": "Execute a trade order via the Execution Router.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "qty": {"type": "number"},
                "type": {"type": "string", "enum": ["market", "limit"], "default": "market"}
            },
            "required": ["symbol", "side", "qty"]
        }
    },
    {
        "name": "plan_wealth_goal",
        "description": "Generate a wealth plan for a specific goal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal_name": {"type": "string"},
                "target_amount": {"type": "number"},
                "horizon_years": {"type": "integer"},
                "current_savings": {"type": "number", "default": 0}
            },
            "required": ["goal_name", "target_amount"]
        }
    },
    {
        "name": "screen_deal",
        "description": "Screen a private equity or venture deal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "deal_name": {"type": "string"},
                "sector": {"type": "string"},
                "valuation": {"type": "number"},
                "ebitda": {"type": "number"}
            },
            "required": ["deal_name", "sector"]
        }
    }
]
