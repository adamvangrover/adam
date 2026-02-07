# core/analysis/fundamental_analysis.py

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, List, Dict, Any, Union

from langchain.tools import tool, Tool
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain import hub

# ==========================================
# 1. Tool Definitions
# ==========================================

# Define a REPL tool for advanced analysis
python_repl = PythonAstREPLTool()

@tool
def python_repl_ast(query: str) -> str:
    """A Python shell that can handle any instructions provided in the prompt.
    Use this to execute Python code, manipulate data, or perform complex calculations.
    """
    return python_repl.run(query)

@tool
def get_stock_price(ticker: str) -> float:
    """Returns the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    # Fetch 1 day history to get the latest close
    history = stock.history(period="1d")
    if history.empty:
        return 0.0
    return history["Close"].iloc[-1]

@tool
def calculate_pe_ratio(price: float, eps: float) -> float:
    """Calculates the Price-to-Earnings (P/E) ratio."""
    if eps == 0:
        return 0.0
    return price / eps

@tool
def get_company_info(ticker: str) -> str:
    """Returns basic company information for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return (
        f"Name: {info.get('longName', 'N/A')}\n"
        f"Sector: {info.get('sector', 'N/A')}\n"
        f"Industry: {info.get('industry', 'N/A')}\n"
        f"Description: {info.get('longBusinessSummary', 'N/A')}"
    )

# Consolidate tools for the agent
global_tools = [
    get_stock_price,
    calculate_pe_ratio,
    get_company_info,
    python_repl_ast,
]

# ==========================================
# 2. Main Analyst Class
# ==========================================

class FundamentalAnalyst:
    """
    Hybrid Agent specialized in fundamental analysis. 
    Capable of both LLM-driven agentic research (via yfinance) and 
    deterministic financial modeling (DCF, Ratio Analysis) using provided data.
    """

    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None, 
        model_name: str = "gpt-4-turbo-preview"
    ):
        """
        Initializes the FundamentalAnalyst agent.

        Args:
            config: Dictionary containing configurations for manual analysis 
                    (data_sources, knowledge_graph, etc.).
            model_name: The OpenAI model to use for the agentic interface.
        """
        self.config = config or {}
        
        # Manual Analysis Components
        self.data_sources = self.config.get('data_sources', {})
        self.knowledge_graph = self.config.get('knowledge_graph', None) # Placeholder object
        
        # Agentic Components
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.tools = global_tools
        
        # Initialize Agent
        # Pull the standard OpenAI tools agent prompt
        self.prompt = hub.pull("hwchase17/openai-tools-agent")
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    # ---------------------------------------------------------
    # Mode A: Agentic Analysis (LLM + Tools)
    # ---------------------------------------------------------
    
    def analyze_ticker_with_agent(self, ticker: str) -> str:
        """
        Performs a fundamental analysis on the given ticker symbol using the LLM Agent.
        This method fetches live data via tools.
        """
        query = (
            f"Perform a comprehensive fundamental analysis of {ticker}. "
            f"Include its current price, P/E ratio (if available), company overview, "
            f"and any other relevant financial metrics. Provide a summary of your findings."
        )
        return self.agent_executor.invoke({"input": query})["output"]

    # ---------------------------------------------------------
    # Mode B: Deterministic/Manual Analysis (Math + Data)
    # ---------------------------------------------------------

    def analyze_company_data(self, company_data: Dict[str, Any], analysis_modules: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Performs a comprehensive fundamental analysis using provided structured company data.
        
        Args:
            company_data: Dictionary containing 'financial_statements', 'name', etc.
            analysis_modules: List of modules to execute (e.g., ['liquidity', 'dcf']).
            **kwargs: Additional arguments for specific modules.
        """
        print(f"Analyzing company fundamentals for {company_data.get('name', 'Unknown')}...")
        financial_statements = company_data.get('financial_statements', {})
        analysis_results = {}

        # 1. Profitability
        if analysis_modules is None or 'profitability' in analysis_modules:
            analysis_results['profitability'] = self.analyze_profitability(financial_statements)

        # 2. Liquidity
        if analysis_modules is None or 'liquidity' in analysis_modules:
            analysis_results['liquidity'] = self.analyze_liquidity(financial_statements)

        # 3. Solvency
        if analysis_modules is None or 'solvency' in analysis_modules:
            analysis_results['solvency'] = self.analyze_solvency(financial_statements)

        # 4. DCF Valuation
        if analysis_modules is None or 'dcf_valuation' in analysis_modules:
            analysis_results['dcf_valuation'] = self.calculate_dcf_valuation(company_data, **kwargs)

        # 5. Comparable Company Analysis
        if analysis_modules is None or 'comparable_company_analysis' in analysis_modules:
            analysis_results['comparable_company_analysis'] = self.perform_comparable_company_analysis(
                company_data, **kwargs)

        # 6. Precedent Transaction Analysis
        if analysis_modules is None or 'precedent_transaction_analysis' in analysis_modules:
            analysis_results['precedent_transaction_analysis'] = self.perform_precedent_transaction_analysis(
                company_data, **kwargs)

        return analysis_results

    # --- Financial Ratio Methods ---

    def analyze_profitability(self, financial_statements: Dict) -> Dict[str, float]:
        """Calculates key profitability ratios."""
        try:
            inc = financial_statements['income_statement']
            bal = financial_statements['balance_sheet']
            
            revenue = inc.get('revenue', 0)
            net_income = inc.get('net_income', 0)
            total_assets = bal.get('total_assets', 0)
            shareholder_equity = bal.get('shareholder_equity', 0)

            return {
                'profit_margin': net_income / revenue if revenue else 0,
                'roe': net_income / shareholder_equity if shareholder_equity else 0,
                'roa': net_income / total_assets if total_assets else 0
            }
        except KeyError as e:
            return {"error": f"Missing data for profitability: {str(e)}"}

    def analyze_liquidity(self, financial_statements: Dict) -> Dict[str, float]:
        """Calculates key liquidity ratios."""
        try:
            bal = financial_statements['balance_sheet']
            current_assets = bal.get('current_assets', 0)
            current_liabilities = bal.get('current_liabilities', 0)
            inventory = bal.get('inventory', 0)

            return {
                'current_ratio': current_assets / current_liabilities if current_liabilities else 0,
                'quick_ratio': (current_assets - inventory) / current_liabilities if current_liabilities else 0
            }
        except KeyError as e:
            return {"error": f"Missing data for liquidity: {str(e)}"}

    def analyze_solvency(self, financial_statements: Dict) -> Dict[str, float]:
        """Calculates key solvency ratios."""
        try:
            bal = financial_statements['balance_sheet']
            total_debt = bal.get('total_debt', 0)
            total_assets = bal.get('total_assets', 0)
            shareholder_equity = bal.get('shareholder_equity', 0)

            return {
                'debt_to_equity': total_debt / shareholder_equity if shareholder_equity else 0,
                'debt_to_assets': total_debt / total_assets if total_assets else 0
            }
        except KeyError as e:
            return {"error": f"Missing data for solvency: {str(e)}"}

    # --- Valuation Methods ---

    def calculate_dcf_valuation(self, company_data, discount_rate=None, growth_rate=None, terminal_growth_rate=None):
        """Calculates the DCF valuation."""
        # Retrieve parameters from knowledge graph if available and not provided
        industry = company_data.get('industry')
        
        if self.knowledge_graph:
            if discount_rate is None:
                discount_rate = self.knowledge_graph.get_discount_rate(industry)
            if growth_rate is None:
                growth_rate = self.knowledge_graph.get_growth_rate(industry)
            if terminal_growth_rate is None:
                terminal_growth_rate = self.knowledge_graph.get_terminal_growth_rate(industry)

        # Fallback defaults if KG fails
        discount_rate = discount_rate or 0.10
        growth_rate = growth_rate or 0.05
        terminal_growth_rate = terminal_growth_rate or 0.02

        fcf_projections = self.project_fcf(company_data, growth_rate)
        
        # Calculate Terminal Value
        last_fcf = fcf_projections[-1]
        terminal_value = last_fcf * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)

        # Discount Cash Flows
        present_values = [fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(fcf_projections)]
        present_value_terminal = terminal_value / ((1 + discount_rate) ** len(fcf_projections))

        enterprise_value = sum(present_values) + present_value_terminal
        
        # Equity Value
        net_debt = company_data['financial_statements']['balance_sheet'].get('net_debt', 0)
        equity_value = enterprise_value - net_debt
        
        shares = company_data.get('shares_outstanding', 1)
        return equity_value / shares

    def project_fcf(self, company_data, growth_rate):
        """Projects free cash flows (Simplified)."""
        historical_fcf = company_data['financial_statements']['cash_flow_statement'].get('free_cash_flow', 0)
        # Project for 10 years
        return [historical_fcf * ((1 + growth_rate) ** i) for i in range(1, 11)]

    def perform_comparable_company_analysis(self, company_data, peer_group=None, valuation_multiples=None):
        """Placeholder for Comparable Company Analysis logic."""
        # Implementation depends heavily on external data availability
        return {"status": "Not implemented - requires rich peer data source"}

    def perform_precedent_transaction_analysis(self, company_data, transaction_data=None, valuation_multiples=None):
        """Placeholder for Precedent Transaction Analysis logic."""
        return {"status": "Not implemented - requires rich transaction data source"}
