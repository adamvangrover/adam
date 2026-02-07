from langchain.tools import tool
from langchain.tools.python.tool import PythonAstREPLTool
from typing import Optional, Type, List, Dict, Any
from pydantic import BaseModel, Field
import yfinance as yf
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain import hub

# Define a REPL tool for advanced analysis
repl_tool = PythonAstREPLTool()

@tool
def get_stock_price(ticker: str) -> float:
    """Returns the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.history(period="1d")["Close"].iloc[-1]

@tool
def calculate_pe_ratio(price: float, eps: float) -> float:
    """Calculates the Price-to-Earnings (P/E) ratio."""
    return price / eps

@tool
def get_company_info(ticker: str) -> str:
    """Returns basic company information for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return f"Name: {info.get('longName', 'N/A')}\nSector: {info.get('sector', 'N/A')}\nIndustry: {info.get('industry', 'N/A')}\nDescription: {info.get('longBusinessSummary', 'N/A')}"

# Define a list of tools for the agent
tools = [
    get_stock_price,
    calculate_pe_ratio,
    get_company_info,
    repl_tool,
    # ... (add other tools as needed)
]

class FundamentalAnalyst:
    """
    An agent that performs fundamental analysis on a company using various tools.
    """

    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        """
        Initializes the FundamentalAnalyst agent with necessary configurations and tools.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.tools = tools
        self.prompt = hub.pull("hwchase17/openai-tools-agent")
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def analyze(self, ticker: str) -> str:
        """
        Performs a fundamental analysis on the given ticker symbol.
        """
        query = f"Perform a comprehensive fundamental analysis of {ticker}. Include its current price, P/E ratio (if available), company overview, and any other relevant financial metrics. Provide a summary of your findings."
        return self.agent_executor.invoke({"input": query})["output"]
