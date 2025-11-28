try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # Fallback for environments without MCP installed
    class FastMCP:
        def __init__(self, name): self.name = name
        def resource(self, path): return lambda f: f
        def tool(self): return lambda f: f
        def run(self): print("MCP Server Mock Run")

import sqlite3
import pandas as pd
from typing import List

# Initialize the MCP Server
mcp = FastMCP("Financial Data Room")

DB_PATH = "finance_data.db"

@mcp.resource("finance://{ticker}/{year}/10k")
def get_10k_filing(ticker: str, year: str) -> str:
    """
    Returns the full text of the 10-K filing for the given ticker and year.
    Resource URI: finance://AAPL/2023/10k
    """
    # In a real app, this would read from the file system or S3
    return f"Full 10-K text content for {ticker} in {year}..."

@mcp.resource("finance://{ticker}/ratios")
def get_financial_ratios(ticker: str) -> str:
    """
    Returns key financial ratios as a CSV string.
    Resource URI: finance://AAPL/ratios
    """
    return "Ratio,Value\nDebt/EBITDA,2.5x\nInterest Coverage,5.0x"

@mcp.tool()
def query_sql(query: str) -> str:
    """
    Executes a read-only SQL query against the local financial database.
    Use this to aggregate data or find specific numerical facts.
    """
    # Security check: only allow SELECT
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_markdown()
    except Exception as e:
        return f"SQL Error: {str(e)}"

@mcp.tool()
def get_covenant_definitions(doc_id: str) -> str:
    """
    Retrieves the legal definitions of financial covenants from a specific credit agreement.
    """
    # logic to parse the credit agreement text
    return """
    Section 7.1. Financial Covenants.
    (a) Consolidated Leverage Ratio. The Borrower shall not permit the Consolidated Leverage Ratio
        as of the end of any Fiscal Quarter to be greater than 4.50 to 1.00.
    """

if __name__ == "__main__":
    mcp.run()
