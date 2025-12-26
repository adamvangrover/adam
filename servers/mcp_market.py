"""
Market Mayhem - MCP Server Interface
Author: Principal Software Architect
Description: Exposes WhaleScanner tools to the AI agent via Model Context Protocol.
             Implements the 'scan_vulture_activity' tool.
"""

from fastmcp import FastMCP
from src.market_mayhem.scanners import WhaleScanner
import json
import logging

# Initialize Logging
logger = logging.getLogger("mcp_server")

# Initialize FastMCP Server
# "Market Mayhem" is the server name exposed to the client
mcp = FastMCP("Market Mayhem - Alpha Engine")

# Initialize the Scanner with a compliant User-Agent
# In production, this should come from environment variables
scanner = WhaleScanner(user_agent="Market Mayhem Bot <bot@marketmayhem.com>")

@mcp.tool()
def scan_vulture_activity(fund_name: str) -> str:
    """
    Scans a specific Vulture Fund (OAKTREE, APOLLO, CENTERBRIDGE, BAUPOST, ARES)
    for recent distressed asset accumulation and new entries.

    This tool performs a Quarter-over-Quarter analysis of 13F-HR filings to identify
    new positions (VULTURE_ENTRY) or significant accumulation. It specifically
    looks for 'PRN' (Principal) share types which may indicate convertible debt
    positions in distressed companies.

    Args:
        fund_name: The internal key for the fund (e.g., 'OAKTREE', 'APOLLO').
                   Use 'get_supported_vultures' to see valid keys.

    Returns:
        Markdown formatted summary of detected signals and positions.
    """
    fund_key = fund_name.upper()
    try:
        logger.info(f"Received request to scan vulture activity for {fund_key}")
        signals = scanner.calculate_fund_sentiment(fund_key)

        if not signals:
            return f"No significant signals detected for {fund_key} in the latest filings."

        # Format output as Markdown for the LLM
        # Tables allow the LLM to easily parse and reference specific data points
        md_output = f"# 🚨 Vulture Signals Detected: {fund_key}\n\n"
        md_output += f"**Analysis Type:** 13F-HR QoQ Comparison\n"
        md_output += f"**Signal Count:** {len(signals)}\n\n"

        md_output += "| Ticker | Signal | Type | Change % | Description |\n"
        md_output += "| :--- | :--- | :--- | :--- | :--- |\n"

        for s in signals:
            # Highlight Debt positions
            type_display = "**DEBT (PRN)**" if s.share_type == 'PRN' else "Equity (SH)"
            md_output += f"| **{s.ticker}** | {s.signal_type} | {type_display} | {s.change_pct:.1f}% | {s.description} |\n"

        return md_output

    except ValueError as ve:
        return f"Input Error: {str(ve)}. Please check the fund name."
    except Exception as e:
        logger.error(f"System error scanning {fund_key}: {e}")
        return f"Internal Error scanning fund {fund_key}: {str(e)}"

@mcp.tool()
def get_supported_vultures() -> str:
    """
    Returns the list of distressed debt funds currently monitored by the engine.
    Useful for discovery before calling scan_vulture_activity.
    """
    return json.dumps(list(scanner.vulture_ciks.keys()))

if __name__ == "__main__":
    # Run via Stdio for local agent integration (e.g. Claude Desktop)
    # For production, we would use transport='sse'
    print("Starting Market Mayhem MCP Server...")
    mcp.run(transport="stdio")
