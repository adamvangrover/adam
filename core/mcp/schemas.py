from typing import Dict, Any

# Tool Schemas for Model Context Protocol (MCP) compliance.
# These schemas define the interface for agents to call core system tools.

AZURE_AI_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "azure_ai_search",
        "description": "Searches and retrieves excerpts from unstructured documents (e.g., rating agency reports, company filings) from the Azure AI Search index.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A highly specific keyword query. Example: 'NextEra Energy S&P FFO/Debt downgrade threshold'"
                },
                "top_k": {
                    "type": "integer",
                    "description": "The number of top document chunks to return. Default is 3.",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}

MICROSOFT_FABRIC_RUN_SQL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "microsoft_fabric_run_sql",
        "description": "Executes a read-only SQL query against the Microsoft Fabric data lakehouse to retrieve structured, time-series financial data and key credit metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "A valid SQL query to be executed. Must be a SELECT statement. Example: 'SELECT Date, FFO_to_Debt FROM credit_metrics WHERE Ticker = \\'NEE\\' AND Date >= \\'2021-01-01\\' ORDER BY Date DESC'"
                }
            },
            "required": ["sql_query"]
        }
    }
}

REQUEST_USER_CONFIRMATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "request_user_confirmation",
        "description": "Pauses execution and asks the user for explicit confirmation before proceeding with a potentially risky or costly action. Use this for any tool call that modifies data or is marked as high-risk.",
        "parameters": {
            "type": "object",
            "properties": {
                "action_description": {
                    "type": "string",
                    "description": "A clear, concise description of the action that requires confirmation. Example: 'About to execute a complex query against the entire financial history table. This may incur significant compute costs. Proceed?'"
                }
            },
            "required": ["action_description"]
        }
    }
}

# Mapping of tool name to schema
TOOL_SCHEMAS = {
    "azure_ai_search": AZURE_AI_SEARCH_SCHEMA,
    "microsoft_fabric_run_sql": MICROSOFT_FABRIC_RUN_SQL_SCHEMA,
    "request_user_confirmation": REQUEST_USER_CONFIRMATION_SCHEMA
}

def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """Retrieves the JSON schema for a given tool name."""
    return TOOL_SCHEMAS.get(tool_name, {})
