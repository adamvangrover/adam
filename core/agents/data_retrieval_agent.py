# core/agents/data_retrieval_agent.py
import logging
import json
import os
import asyncio
from typing import Optional, Union, List, Dict, Any

from core.agents.agent_base import AgentBase
from core.utils.data_utils import load_data
from core.system.knowledge_base import KnowledgeBase
from core.system.error_handler import DataNotFoundError, FileReadError 
from semantic_kernel import Kernel
from core.data_sources.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataRetrievalAgent(AgentBase):
    """
    Agent responsible for retrieving data from various configured sources.
    Now integrates with DataFetcher for live market data.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.persona = self.config.get('persona', "Data Retrieval Specialist")
        self.description = self.config.get('description', "Retrieves data from various configured sources based on structured requests.")
        self.expertise = self.config.get('expertise', ["data access", "file retrieval", "knowledge base query"])

        self.risk_ratings_path = self.config.get('risk_ratings_file_path', 'data/risk_rating_mapping.json')
        self.market_baseline_path = self.config.get('market_baseline_file_path', 'data/adam_market_baseline.json')
        
        try:
            self.knowledge_base = KnowledgeBase()
        except Exception as e:
            logging.error(f"Failed to initialize KnowledgeBase in DataRetrievalAgent: {e}")
            self.knowledge_base = None

        # Initialize the Live Data Connector
        self.data_fetcher = DataFetcher()

    async def execute(self, request: Dict[str, Any]) -> Optional[Any]:
        data_type = request.get('data_type')
        logging.info(f"DataRetrievalAgent executing request for data_type: {data_type} with params: {request}")

        try:
            if data_type == 'get_risk_rating':
                company_id = request.get('company_id')
                if not company_id: return None
                return self.get_risk_rating(company_id)
            elif data_type == 'get_market_data':
                return self.get_market_data()
            elif data_type == 'access_knowledge_base':
                query = request.get('query')
                if not query: return None
                return self.access_knowledge_base(query)
            elif data_type == 'access_knowledge_graph':
                query = request.get('query')
                if not query: return None
                return self.access_knowledge_graph(query)
            elif data_type == 'get_company_financials':
                company_id = request.get('company_id')
                if not company_id: return None
                return self._get_company_financial_data(company_id)
            elif data_type == 'get_company_news':
                company_id = request.get('company_id')
                if not company_id: return None
                return self.data_fetcher.fetch_news(company_id)
            elif data_type == 'get_company_recommendations':
                company_id = request.get('company_id')
                if not company_id: return None
                return self.data_fetcher.fetch_recommendations(company_id)
            else:
                logging.warning(f"Unknown data_type requested: {data_type}")
                return None
        except Exception as e:
            logging.exception(f"Error during execute for '{data_type}': {e}")
            return None

    def _get_company_financial_data(self, company_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves company financial data. Uses live data if possible, falls back to mock for testing.
        """
        if company_id == "ABC_TEST":
            return self._get_mock_abc_test_data()

        # Use Live Data Fetcher
        return self._fetch_real_company_data(company_id)

    def _fetch_real_company_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetches real data using DataFetcher and maps it to the expected schema.
        """
        try:
            # 1. Fetch Market Data (Snapshot)
            market_data = self.data_fetcher.fetch_market_data(ticker)
            if not market_data:
                logging.warning(f"Could not fetch market data for {ticker}")
                return None

            # 2. Fetch Financials
            financials = self.data_fetcher.fetch_financials(ticker)

            # 3. Construct the response object

            def transpose_financials(fin_data_by_date):
                if not fin_data_by_date: return {}
                metrics = {}
                sorted_dates = sorted(fin_data_by_date.keys())
                if not sorted_dates: return {}

                # Use keys from the first available date
                first_date = sorted_dates[0]
                sample_metrics = fin_data_by_date[first_date].keys()

                for metric in sample_metrics:
                    series = []
                    for d in sorted_dates:
                        val = fin_data_by_date[d].get(metric)
                        # Handle basic types, defaulting to 0 for missing numeric data
                        if val is None:
                            series.append(0)
                        else:
                            series.append(val)
                    metrics[metric] = series
                return metrics

            income_transposed = transpose_financials(financials.get('income_statement', {}))
            balance_transposed = transpose_financials(financials.get('balance_sheet', {}))
            cashflow_transposed = transpose_financials(financials.get('cash_flow', {}))

            def get_mapped_series(transposed_data, possible_keys):
                for k in possible_keys:
                    if k in transposed_data:
                        return transposed_data[k]
                return []

            # Mapping logic
            revenue = get_mapped_series(income_transposed, ["Total Revenue", "Revenue", "TotalRevenue"])
            net_income = get_mapped_series(income_transposed, ["Net Income", "NetIncome"])
            ebitda = get_mapped_series(income_transposed, ["EBITDA", "Normalized EBITDA"])

            total_assets = get_mapped_series(balance_transposed, ["Total Assets", "TotalAssets"])
            total_liabilities = get_mapped_series(balance_transposed, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
            equity = get_mapped_series(balance_transposed, ["Stockholders Equity", "Total Equity Gross Minority Interest"])

            financial_data_detailed = {
                "income_statement": {
                    "revenue": revenue,
                    "net_income": net_income,
                    "ebitda": ebitda,
                },
                "balance_sheet": {
                    "total_assets": total_assets,
                    "total_liabilities": total_liabilities,
                    "shareholders_equity": equity,
                    "short_term_debt": get_mapped_series(balance_transposed, ["Current Debt", "Short Term Debt"]),
                    "long_term_debt": get_mapped_series(balance_transposed, ["Long Term Debt"]),
                    "cash_and_equivalents": get_mapped_series(balance_transposed, ["Cash And Cash Equivalents"])
                },
                "cash_flow_statement": {
                    "free_cash_flow": get_mapped_series(cashflow_transposed, ["Free Cash Flow"])
                },
                "market_data": {
                    "share_price": market_data.get("current_price"),
                    "shares_outstanding": None
                },
                "key_ratios": {
                    "debt_to_equity_ratio": market_data.get("debtToEquity"),
                    "net_profit_margin": market_data.get("profitMargins")
                },
                "dcf_assumptions": {
                    "fcf_projection_years_total": 10,
                    "initial_high_growth_period_years": 5,
                    "initial_high_growth_rate": 0.10,
                    "stable_growth_rate": 0.05,
                    "discount_rate": 0.09,
                    "terminal_growth_rate": 0.025,
                    "terminal_growth_rate_perpetuity": 0.025
                }
            }

            # Enrich market data
            if market_data.get("market_cap") and market_data.get("current_price"):
                 financial_data_detailed["market_data"]["shares_outstanding"] = market_data["market_cap"] / market_data["current_price"]

            result_data = {
                "company_info": {
                    "name": ticker,
                    "industry_sector": market_data.get("sector"),
                    "country": market_data.get("country", "Unknown")
                },
                "financial_data_detailed": financial_data_detailed,
                "qualitative_company_info": {
                     "description": market_data.get("description")
                },
                "industry_data_context": {
                     "industry": market_data.get("industry")
                },
                "economic_data_context": {},
                "collateral_and_debt_details": {}
            }

            self._save_to_cache(ticker, result_data)
            return result_data

        except Exception as e:
            logging.error(f"Error fetching real data for {ticker}: {e}")
            return self._load_from_cache(ticker)

    def _get_mock_abc_test_data(self):
        return {
                "company_info": {
                    "name": "ABC_TEST Corp",
                    "industry_sector": "Technology", 
                    "country": "USA"
                },
                "financial_data_detailed": { 
                    "income_statement": { 
                        "revenue": [1000, 1100, 1250], "cogs": [400, 440, 500], "gross_profit": [600, 660, 750],
                        "operating_expenses": [300, 320, 350], "ebitda": [300, 340, 400], "depreciation_amortization": [50, 55, 60],
                        "ebit": [250, 285, 340], "interest_expense": [30, 28, 25], "income_before_tax": [220, 257, 315],
                        "taxes": [44, 51, 63], "net_income": [176, 206, 252]
                    },
                    "balance_sheet": { 
                        "cash_and_equivalents": [200, 250, 300], "accounts_receivable": [150, 160, 170], "inventory": [100, 110, 120],
                        "total_current_assets": [450, 520, 590], "property_plant_equipment_net": [1500, 1550, 1600],
                        "total_assets": [1950, 2070, 2190],
                        "accounts_payable": [120, 130, 140], "short_term_debt": [100, 80, 60], "total_current_liabilities": [220, 210, 200],
                        "long_term_debt": [500, 450, 400], "total_liabilities": [720, 660, 600],
                        "shareholders_equity": [1230, 1410, 1590]
                    },
                    "cash_flow_statement": { 
                        "net_income_cf": [176, 206, 252], "depreciation_amortization_cf": [50, 55, 60], "change_in_working_capital": [-20, -15, -10],
                        "cash_flow_from_operations": [206, 246, 302], "capital_expenditures": [-70, -75, -80],
                        "cash_flow_from_investing": [-70, -75, -80], "debt_issued_repaid": [-50, -20, -20], "equity_issued_repaid": [0, 0, 0],
                        "dividends_paid": [-30, -35, -40], "cash_flow_from_financing": [-80, -55, -60],
                        "net_change_in_cash": [56, 116, 162],
                        "free_cash_flow": [136, 171, 222] 
                    },
                    "key_ratios": { 
                        "debt_to_equity_ratio": 0.58, "net_profit_margin": 0.20,
                        "current_ratio": 2.95, "interest_coverage_ratio": 13.6 
                    },
                    "dcf_assumptions": {
                        "fcf_projection_years_total": 10,
                        "initial_high_growth_period_years": 5,
                        "initial_high_growth_rate": 0.10, 
                        "stable_growth_rate": 0.05,       
                        "discount_rate": 0.09,            
                        "terminal_growth_rate": 0.025,
                        "terminal_growth_rate_perpetuity": 0.025 
                    },
                    "market_data": { 
                        "share_price": 65.00, "shares_outstanding": 10000000 
                    }
                },
                "qualitative_company_info": { 
                    "management_assessment": "Experienced team with a clear strategy.",
                    "corporate_governance_notes": "Standard governance practices in place.",
                    "business_model_strength": "Strong recurring revenue streams.",
                    "competitive_advantages": "Proprietary technology, strong brand.",
                    "customer_supplier_concentration": "No significant concentration risks identified.",
                    "legal_regulatory_issues": "None outstanding."
                },
                "industry_data_context": { 
                    "name": "Software Development",
                    "outlook": "Positive growth expected, moderate competition.",
                    "key_drivers": ["Cloud adoption", "AI integration"],
                    "benchmark_ratios": {"avg_debt_to_equity": 0.5, "avg_net_margin": 0.18}
                },
                "economic_data_context": { 
                    "overall_outlook": "Stable with moderate growth.",
                    "gdp_growth_forecast": "2.5%",
                    "inflation_rate": "3.0%",
                    "interest_rate_outlook": "Stable to slightly increasing."
                },
                "collateral_and_debt_details":{ 
                    "collateral_type": "Accounts Receivable, Intellectual Property",
                    "collateral_valuation": 1200000, 
                    "loan_to_value_ratio": 0.5,
                    "debt_tranches": [
                        {"type": "Senior Secured Term Loan", "amount": 300, "interest_rate": "SOFR+300", "maturity": "2028-12-31"},
                        {"type": "Revolving Credit Facility", "amount": 200, "interest_rate": "SOFR+250", "maturity": "2027-12-31"}
                    ],
                    "guarantees_exist": True,
                    "other_credit_enhancements": "None noted."
                }
            }

    def get_risk_rating(self, company_id: str) -> Optional[str]:
        logging.info(f"Attempting to retrieve risk rating for company_id: {company_id} from {self.risk_ratings_path}")
        try:
            source_config = {'type': 'json', 'path': self.risk_ratings_path}
            data = load_data(source_config, cache=False)
            if data and isinstance(data, dict):
                rating = data.get(company_id)
                if rating is None:
                    logging.warning(f"Risk rating for company_id '{company_id}' not found in {self.risk_ratings_path}.")
                return rating
            else:
                logging.error(f"Failed to load risk ratings data from {self.risk_ratings_path}")
                return None
        except FileReadError as e:
            logging.error(f"FileReadError while reading risk ratings: {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error in get_risk_rating for {company_id}: {e}")
            return None

    def get_market_data(self) -> Optional[Dict[str, Any]]:
        logging.info(f"Attempting to retrieve market data from {self.market_baseline_path}")
        try:
            source_config = {'type': 'json', 'path': self.market_baseline_path}
            market_data = load_data(source_config, cache=False)
            if market_data is None:
                 logging.error(f"Failed to load market data from {self.market_baseline_path}.")
            return market_data
        except FileReadError as e:
            logging.error(f"FileReadError while reading market data: {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error in get_market_data: {e}")
            return None

    def access_knowledge_base(self, query: str) -> Optional[str]:
        if self.knowledge_base:
            try:
                logging.info(f"Querying KnowledgeBase with: {query}")
                return self.knowledge_base.query(query)
            except Exception as e:
                logging.error(f"Error accessing Knowledge Base: {e}")
                return None
        else:
            logging.warning("KnowledgeBase is not initialized in DataRetrievalAgent.")
            return None

    def access_knowledge_graph(self, query: str) -> str:
        logging.warning("Knowledge Graph access is not yet implemented. Query: {query}")
        return "Knowledge Graph access is not yet implemented."

    async def receive_message(self, sender_agent: str, message: Dict[str, Any]) -> Optional[Any]:
        logging.info(f"DataRetrievalAgent received message from {sender_agent}: {message}")
        return await self.execute(request=message)

    def _save_to_cache(self, company_id: str, data: Dict[str, Any]):
        try:
            cache_dir = "data/cache"
            os.makedirs(cache_dir, exist_ok=True)
            file_path = f"{cache_dir}/{company_id}_financials.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to cache data for {company_id}: {e}")

    def _load_from_cache(self, company_id: str) -> Optional[Dict[str, Any]]:
        try:
            file_path = f"data/cache/{company_id}_financials.json"
            if os.path.exists(file_path):
                logging.info(f"Loading cached data for {company_id} from {file_path}")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['source'] = 'cache'
                    return data
        except Exception as e:
            logging.warning(f"Failed to load cache for {company_id}: {e}")
        return None

if __name__ == '__main__':
    # Create dummy config for the agent
    dummy_agent_config = {
        'persona': 'Test Data Retriever',
        'risk_ratings_file_path': 'data/dummy_risk_ratings.json',
        'market_baseline_file_path': 'data/dummy_market_baseline.json'
    }
    # Create dummy data files
    os.makedirs('data', exist_ok=True)
    dummy_risk_data = {"AAPL": "Low", "MSFT": "Low", "GOOG": "Medium"}
    with open(dummy_agent_config['risk_ratings_file_path'], 'w') as f:
        json.dump(dummy_risk_data, f)
    dummy_market_data = {"market_index": "S&P 500", "current_value": 4500.67, "trend": "bullish"}
    with open(dummy_agent_config['market_baseline_file_path'], 'w') as f:
        json.dump(dummy_market_data, f)

    dra_agent = DataRetrievalAgent(config=dummy_agent_config)

    async def main_test():
        # Test 1: ABC_TEST (Mock)
        print("\n--- Test 1: ABC_TEST (Mock) ---")
        result1 = await dra_agent.execute({'data_type': 'get_company_financials', 'company_id': 'ABC_TEST'})
        assert result1 is not None
        assert result1['company_info']['name'] == "ABC_TEST Corp"
        print("Test 1 Passed.")

        # Test 2: Real Data (AAPL) - assuming yfinance works
        print("\n--- Test 2: Real Data (AAPL) ---")
        # NOTE: This will only work if yfinance can reach the internet and fetch data.
        # Since we are in a sandbox with internet access, it should work.
        result2 = await dra_agent.execute({'data_type': 'get_company_financials', 'company_id': 'AAPL'})
        if result2:
            print(f"Fetched AAPL Data. Name: {result2['company_info']['name']}")
            print(f"Revenue samples: {result2['financial_data_detailed']['income_statement']['revenue'][:5]}")
            assert result2['company_info']['name'] == 'AAPL'
        else:
            print("Failed to fetch AAPL data (network might be down or API changed).")

    try:
        asyncio.run(main_test())
    finally:
        if os.path.exists(dummy_agent_config['risk_ratings_file_path']): os.remove(dummy_agent_config['risk_ratings_file_path'])
        if os.path.exists(dummy_agent_config['market_baseline_file_path']): os.remove(dummy_agent_config['market_baseline_file_path'])
        if os.path.exists('data') and not os.listdir('data'): os.rmdir('data')
