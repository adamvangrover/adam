# core/agents/data_retrieval_agent.py
import logging
import json
import os
import asyncio
from typing import Optional, Union, List, Dict, Any

from core.agents.agent_base import AgentBase
from core.utils.data_utils import load_data # Assuming load_data is suitable
from core.system.knowledge_base import KnowledgeBase
# Error classes might be used by load_data or if we add more specific error handling
from core.system.error_handler import DataNotFoundError, FileReadError 
from semantic_kernel import Kernel # For AgentBase type hinting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataRetrievalAgent(AgentBase):
    """
    Agent responsible for retrieving data from various configured sources.
    Supports:
    - Local files (JSON, CSV, YAML) via data_utils.load_data.
    - A Knowledge Base.
    - Placeholder for future Knowledge Graph integration.
    - Asynchronous A2A communication via receive_message.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Kernel] = None):
        """
        Initializes the DataRetrievalAgent.

        Args:
            config: Agent-specific configuration dictionary.
            kernel: Optional Semantic Kernel instance.
        """
        super().__init__(config, kernel)
        self.persona = self.config.get('persona', "Data Retrieval Specialist")
        self.description = self.config.get('description', "Retrieves data from various configured sources based on structured requests.")
        self.expertise = self.config.get('expertise', ["data access", "file retrieval", "knowledge base query"])

        # File paths from agent-specific configuration
        self.risk_ratings_path = self.config.get('risk_ratings_file_path', 'data/risk_rating_mapping.json')
        self.market_baseline_path = self.config.get('market_baseline_file_path', 'data/adam_market_baseline.json')
        
        # Initialize KnowledgeBase
        try:
            self.knowledge_base = KnowledgeBase()
        except Exception as e:
            logging.error(f"Failed to initialize KnowledgeBase in DataRetrievalAgent: {e}")
            self.knowledge_base = None

    async def execute(self, request: Dict[str, Any]) -> Optional[Any]:
        """
        Executes data retrieval based on a structured request.

        Args:
            request: A dictionary with 'data_type' and other parameters.
                     Example: {'data_type': 'get_risk_rating', 'company_id': 'ABC'}
                              {'data_type': 'get_market_data'}
                              {'data_type': 'access_knowledge_base', 'query': 'some query'}

        Returns:
            The retrieved data, or None if an error occurs or data_type is unknown.
        """
        data_type = request.get('data_type')
        logging.info(f"DataRetrievalAgent executing request for data_type: {data_type} with params: {request}")

        try:
            if data_type == 'get_risk_rating':
                company_id = request.get('company_id')
                if not company_id:
                    logging.error("Request for 'get_risk_rating' missing 'company_id'.")
                    return None
                return self.get_risk_rating(company_id)
            elif data_type == 'get_market_data':
                return self.get_market_data()
            elif data_type == 'access_knowledge_base':
                query = request.get('query')
                if not query:
                    logging.error("Request for 'access_knowledge_base' missing 'query'.")
                    return None
                return self.access_knowledge_base(query)
            elif data_type == 'access_knowledge_graph':
                query = request.get('query')
                if not query:
                    logging.error("Request for 'access_knowledge_graph' missing 'query'.")
                    return None
                return self.access_knowledge_graph(query)
            elif data_type == 'get_company_financials':
                company_id = request.get('company_id')
                if not company_id:
                    logging.error("Request for 'get_company_financials' missing 'company_id'.")
                    return None
                return self._get_company_financial_data(company_id)
            else:
                logging.warning(f"Unknown data_type requested from DataRetrievalAgent: {data_type}")
                return None
        except Exception as e:
            logging.exception(f"Error during DataRetrievalAgent.execute for data_type '{data_type}': {e}")
            return None

    def _get_company_financial_data(self, company_id: str) -> Optional[Dict[str, Any]]:
        """
        Placeholder method to simulate fetching comprehensive company financial data.
        """
        logging.info(f"Fetching comprehensive financial data for company_id: {company_id}")
        if company_id == "ABC_TEST": # Use a specific ID for test data
            return {
                "company_info": {
                    "name": f"{company_id} Corp",
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
                    "dcf_assumptions": { # Renamed from direct keys for clarity
                        "growth_rate": 0.05, "discount_rate": 0.10, "terminal_growth_rate": 0.03
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
        else:
            logging.warning(f"No mock data found for company_id: {company_id} in DataRetrievalAgent._get_company_financial_data")
            return None

    def get_risk_rating(self, company_id: str) -> Optional[str]:
        """Retrieves the risk rating for a given company."""
        logging.info(f"Attempting to retrieve risk rating for company_id: {company_id} from {self.risk_ratings_path}")
        try:
            # load_data expects a config dict for the source
            source_config = {'type': 'json', 'path': self.risk_ratings_path}
            data = load_data(source_config, cache=False) # Disable cache for this potentially dynamic data
            if data and isinstance(data, dict):
                rating = data.get(company_id)
                if rating is None:
                    logging.warning(f"Risk rating for company_id '{company_id}' not found in {self.risk_ratings_path}.")
                return rating
            else:
                logging.error(f"Failed to load or parse risk ratings data from {self.risk_ratings_path}, or data is not a dictionary.")
                return None
        except FileReadError as e:
            logging.error(f"FileReadError while reading risk ratings from {self.risk_ratings_path}: {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error in get_risk_rating for {company_id}: {e}")
            return None

    def get_market_data(self) -> Optional[Dict[str, Any]]:
        """Retrieves general market data."""
        logging.info(f"Attempting to retrieve market data from {self.market_baseline_path}")
        try:
            source_config = {'type': 'json', 'path': self.market_baseline_path}
            market_data = load_data(source_config, cache=False)
            if market_data is None:
                 logging.error(f"Failed to load market data from {self.market_baseline_path}.")
            return market_data
        except FileReadError as e:
            logging.error(f"FileReadError while reading market data from {self.market_baseline_path}: {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error in get_market_data: {e}")
            return None

    def access_knowledge_base(self, query: str) -> Optional[str]:
        """Accesses the Knowledge Base to retrieve information."""
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
        """Placeholder for knowledge graph access."""
        logging.warning("Knowledge Graph access is not yet implemented. Query: {query}")
        return "Knowledge Graph access is not yet implemented."

    async def receive_message(self, sender_agent: str, message: Dict[str, Any]) -> Optional[Any]:
        """
        Handles incoming A2A messages by processing them as data retrieval requests.
        """
        logging.info(f"DataRetrievalAgent received message from {sender_agent}: {message}")
        # Assuming the message is a valid request structure for the execute method
        return await self.execute(request=message)

if __name__ == '__main__':
    # Create dummy config for the agent
    dummy_agent_config = {
        'persona': 'Test Data Retriever',
        'risk_ratings_file_path': 'data/dummy_risk_ratings.json',
        'market_baseline_file_path': 'data/dummy_market_baseline.json'
    }

    # Create dummy data files for the example
    os.makedirs('data', exist_ok=True)
    dummy_risk_data = {
        "AAPL": "Low",
        "MSFT": "Low",
        "GOOG": "Medium"
    }
    with open(dummy_agent_config['risk_ratings_file_path'], 'w') as f:
        json.dump(dummy_risk_data, f)

    dummy_market_data = {
        "market_index": "S&P 500",
        "current_value": 4500.67,
        "trend": "bullish"
    }
    with open(dummy_agent_config['market_baseline_file_path'], 'w') as f:
        json.dump(dummy_market_data, f)

    # Instantiate the agent
    dra_agent = DataRetrievalAgent(config=dummy_agent_config, kernel=None) # kernel=None for this example

    async def main_test():
        # Test 1: Get risk rating for an existing company
        print("\n--- Test 1: Get Risk Rating (AAPL) ---")
        request1 = {'data_type': 'get_risk_rating', 'company_id': 'AAPL'}
        result1 = await dra_agent.execute(request1)
        print(f"Result for {request1}: {result1}")
        assert result1 == "Low"

        # Test 2: Get risk rating for a non-existing company
        print("\n--- Test 2: Get Risk Rating (XYZ) ---")
        request2 = {'data_type': 'get_risk_rating', 'company_id': 'XYZ'}
        result2 = await dra_agent.execute(request2)
        print(f"Result for {request2}: {result2}")
        assert result2 is None

        # Test 3: Get market data
        print("\n--- Test 3: Get Market Data ---")
        request3 = {'data_type': 'get_market_data'}
        result3 = await dra_agent.execute(request3)
        print(f"Result for {request3}: {result3}")
        assert result3 == dummy_market_data

        # Test 4: Access Knowledge Base (assuming default KB is empty or has some known test data)
        print("\n--- Test 4: Access Knowledge Base ---")
        request4 = {'data_type': 'access_knowledge_base', 'query': 'What is Adam?'}
        result4 = await dra_agent.execute(request4) # Made execute async
        print(f"Result for {request4}: {result4}")
        # Add assertion based on expected KB content if any

        # Test 5: Unknown data type
        print("\n--- Test 5: Unknown Data Type ---")
        request5 = {'data_type': 'get_weather_forecast'}
        result5 = await dra_agent.execute(request5)
        print(f"Result for {request5}: {result5}")
        assert result5 is None
        
        # Test 6: Missing company_id for get_risk_rating
        print("\n--- Test 6: Missing company_id for get_risk_rating ---")
        request6 = {'data_type': 'get_risk_rating'}
        result6 = await dra_agent.execute(request6)
        print(f"Result for {request6}: {result6}")
        assert result6 is None

        # Test 7: Get company financials for ABC_TEST
        print("\n--- Test 7: Get Company Financials (ABC_TEST) ---")
        request7 = {'data_type': 'get_company_financials', 'company_id': 'ABC_TEST'}
        result7 = await dra_agent.execute(request7)
        print(f"Company Financials for ABC_TEST (first level keys): {list(result7.keys()) if result7 else None}")
        assert result7 is not None
        assert "company_info" in result7
        assert result7["company_info"]["name"] == "ABC_TEST Corp"
        assert "financial_data_detailed" in result7
        assert "income_statement" in result7["financial_data_detailed"]
        assert "qualitative_company_info" in result7
        assert "industry_data_context" in result7
        assert "economic_data_context" in result7
        assert "collateral_and_debt_details" in result7
        assert result7["financial_data_detailed"]["key_ratios"]["debt_to_equity_ratio"] == 0.58


        # Test 8: Get company financials for NON_EXISTENT
        print("\n--- Test 8: Get Company Financials (NON_EXISTENT) ---")
        request8 = {'data_type': 'get_company_financials', 'company_id': 'NON_EXISTENT'}
        result8 = await dra_agent.execute(request8)
        print(f"Result for {request8}: {result8}")
        assert result8 is None
        
        # Test 9: Missing company_id for get_company_financials
        print("\n--- Test 9: Missing company_id for get_company_financials ---")
        request9 = {'data_type': 'get_company_financials'}
        result9 = await dra_agent.execute(request9)
        print(f"Result for {request9}: {result9}")
        assert result9 is None


    try:
        asyncio.run(main_test())
    finally:
        # Clean up dummy files
        if os.path.exists(dummy_agent_config['risk_ratings_file_path']):
            os.remove(dummy_agent_config['risk_ratings_file_path'])
        if os.path.exists(dummy_agent_config['market_baseline_file_path']):
            os.remove(dummy_agent_config['market_baseline_file_path'])
        # Clean up 'data' directory if it was created by this script and is empty
        if os.path.exists('data') and not os.listdir('data'): # Check if empty before removing
            try:
                os.rmdir('data') # This will fail if 'data' contains other files.
            except OSError as e:
                logging.warning(f"Could not remove 'data' directory (it might not be empty or access denied): {e}")
        print("\nExample execution finished and dummy files (if created by this script) cleaned up.")
