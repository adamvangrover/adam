# core/agents/SNC_analyst_agent.py

from enum import Enum

class SNCRating(Enum):
    PASS = "Pass"
    SPECIAL_MENTION = "Special Mention"
    SUBSTANDARD = "Substandard"
    DOUBTFUL = "Doubtful"
    LOSS = "Loss"

class SNCAnalystAgent:
    def __init__(self, knowledge_base_path="knowledge_base/Knowledge_Graph.json"):
        """
        Initializes the SNC Analyst Agent with knowledge of the
        Comptroller's Handbook and OCC guidelines. It can operate
        independently or integrate with the broader system.

        Args:
            knowledge_base_path (str): Path to the knowledge base file.
        """
        # Load relevant sections from Comptroller's Handbook and OCC guidelines
        # This could involve loading pre-processed data or using a document retrieval system
        # For this example, we'll hardcode some key elements for demonstration purposes
        self.comptrollers_handbook = {
            "SNC": {
                "primary_repayment_source": "sustainable source of cash under the borrower's control",
                "substandard_definition": "inadequately protected by the current sound worth and paying capacity of the obligor or of the collateral pledged",
                "doubtful_definition": "all the weaknesses inherent in one classified substandard with the added characteristic that the weaknesses make collection or liquidation in full, highly questionable and improbable",
                "loss_definition": "uncollectible and of such little value that their continuance as bankable assets is not warranted",
                # ... other relevant sections
            }
        }
        self.occ_guidelines = {
            "SNC": {
                "nonaccrual_status": "asset is maintained on a cash basis because of deterioration in the financial condition of the borrower",
                "capitalization_of_interest": "interest may be capitalized only when the borrower is creditworthy and has the ability to repay the debt in the normal course of business",
                # ... other relevant guidelines
            }
        }
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        Loads the knowledge base from the JSON file.

        Returns:
            dict: The knowledge base data.
        """
        try:
            with open(self.knowledge_base_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Knowledge base file not found: {self.knowledge_base_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding knowledge base JSON: {self.knowledge_base_path}")
            return {}

    def analyze_snc(self, company_name, financial_data=None, industry_data=None, economic_data=None):
        """
        Analyzes a Shared National Credit (SNC) and assigns a risk rating based on
        the Comptroller's Handbook and OCC guidelines. Acts as an independent
        examiner persona. Can receive data directly or pull from the knowledge
        base if integrated with the system.

        Args:
            company_name (str): The name of the company.
            financial_data (dict, optional): Financial data of the company.
            industry_data (dict, optional): Industry-specific data.
            economic_data (dict, optional): Macroeconomic data.

        Returns:
            tuple: (SNCRating, str): The SNC rating and a detailed rationale for the rating.
        """

        # If data is not provided directly, retrieve from the knowledge base
        if not financial_data:
            financial_data = self.get_company_financial_data(company_name)
        if not industry_data:
            industry_data = self.get_industry_data(company_name)
        if not economic_data:
            economic_data = self.get_economic_data()

        # 1. Financial Statement Analysis
        # Analyze financial data based on Comptroller's Handbook guidelines
        # Assess cash flow, liquidity, leverage, profitability, and other relevant metrics
        # Identify trends and potential weaknesses
        # ... (Implementation of financial analysis logic)

        # 2. Qualitative Analysis
        # Evaluate management quality, industry outlook, and economic conditions
        # Consider factors such as competitive landscape, regulatory environment, and
        # technological advancements
        # ... (Implementation of qualitative analysis logic)

        # 3. Credit Risk Mitigation
        # Assess the presence and effectiveness of credit risk mitigants, such as:
        # - Collateral: Evaluate collateral type, quality, and value
        # - Guarantees: Analyze guarantor strength and willingness to perform
        # - Other mitigants: Consider credit insurance, letters of credit, etc.
        # ... (Implementation of credit risk mitigation assessment logic)

        # 4. Rating Determination
        # Assign SNC rating based on a combination of quantitative and qualitative factors
        # Consider the probability of default and the severity of loss given default
        rating, rationale = self._determine_rating(company_name, financial_data, industry_data, economic_data)

        return rating, rationale

    def _determine_rating(self, company_name, financial_data, industry_data, economic_data):
        """
        Determines the SNC rating based on a comprehensive assessment of
        credit risk, incorporating quantitative and qualitative factors.

        Args:
            company_name (str): The name of the company.
            financial_data (dict): Financial data of the company.
            industry_data (dict): Industry-specific data.
            economic_data (dict): Macroeconomic data.

        Returns:
            tuple: (SNCRating, str): The SNC rating and a detailed rationale for the rating.
        """

        # Implement complex rating logic based on the Comptroller's Handbook
        # and OCC guidelines.
        # This logic should include:
        # - Assessment of repayment capacity over a 7-year period
        # - Probability-based assessment for each rating category
        # - Non-accrual designation based on interest coverage and valuation/debt ratios
        # - Consideration of qualitative factors and credit risk mitigants
        # - Detailed rationale for the assigned rating

        # Example logic (replace with actual implementation based on the guidelines):
        if financial_data.get("debt_to_equity", 0) > 3.0 and financial_data.get("profitability", 0) < 0:
            rating = SNCRating.LOSS
            rationale = "High debt-to-equity ratio and negative profitability indicate significant risk of loss, aligning with the Comptroller's Handbook definition of 'Loss'."
        elif financial_data.get("debt_to_equity", 0) > 2.0 and financial_data.get("profitability", 0) < 0.1:
            rating = SNCRating.DOUBTFUL
            rationale = "Elevated debt-to-equity ratio and low profitability raise concerns about repayment capacity, suggesting a 'Doubtful' rating as per the Comptroller's Handbook."
        # ... other rating logic based on the guidelines ...
        elif financial_data.get("debt_to_equity", 0) <= 1.0 and financial_data.get("profitability", 0) >= 0.3:
            rating = SNCRating.PASS
            rationale = "Strong financial condition with low debt-to-equity ratio and healthy profitability, meeting the criteria for a 'Pass' rating."
        else:
            rating = SNCRating.SPECIAL_MENTION
            rationale = "Potential weaknesses require further monitoring, warranting a 'Special Mention' rating."

        return rating, rationale

    def get_company_financial_data(self, company_name):
        """
        Retrieves company financial data from the knowledge base.

        Args:
            company_name (str): Name of the company.

        Returns:
            dict: Financial data of the company.
        """
        # Placeholder for knowledge base interaction
        # Replace with actual data retrieval logic
        return self.knowledge_base.get("companies", {}).get(company_name, {})

    def get_industry_data(self, company_name):
        """
        Retrieves industry data for the company's industry from the knowledge base.

        Args:
            company_name (str): Name of the company.

        Returns:
            dict: Industry-specific data.
        """
        # Placeholder for knowledge base interaction
        # Replace with actual data retrieval logic
        industry = self.knowledge_base.get("companies", {}).get(company_name, {}).get("industry", None)
        if industry:
            return self.knowledge_base.get("industries", {}).get(industry, {})
        else:
            return {}

    def get_economic_data(self):
        """
        Retrieves macroeconomic data from the knowledge base.

        Returns:
            dict: Macroeconomic data.
        """
        # Placeholder for knowledge base interaction
        # Replace with actual data retrieval logic
        return self.knowledge_base.get("macroeconomic_data", {})



#WIP ///////////////////////////////////////////////////////////////////////

import json
from enum import Enum


class SNCRating(Enum):
    PASS = "Pass"
    SPECIAL_MENTION = "Special Mention"
    SUBSTANDARD = "Substandard"
    DOUBTFUL = "Doubtful"
    LOSS = "Loss"


class SNCAnalystAgent:
    def __init__(self, knowledge_base_path="knowledge_base/Knowledge_Graph.json"):
        """
        Initializes the SNC Analyst Agent with knowledge of the
        Comptroller's Handbook and OCC guidelines. It can operate
        independently or integrate with the broader system.

        Args:
            knowledge_base_path (str): Path to the knowledge base file.
        """
        # Load relevant sections from Comptroller's Handbook and OCC guidelines
        self.comptrollers_handbook = {
            "SNC": {
                "primary_repayment_source": "sustainable source of cash under the borrower's control",
                "substandard_definition": "inadequately protected by the current sound worth and paying capacity of the obligor or of the collateral pledged",
                "doubtful_definition": "all the weaknesses inherent in one classified substandard with the added characteristic that the weaknesses make collection or liquidation in full, highly questionable and improbable",
                "loss_definition": "uncollectible and of such little value that their continuance as bankable assets is not warranted",
                # Additional sections as per Comptroller's Handbook
                "repayment_capacity_period": 7,
                "nonaccrual_status": "asset is maintained on a cash basis because of deterioration in the financial condition of the borrower",
                "capitalization_of_interest": "interest may be capitalized only when the borrower is creditworthy and has the ability to repay the debt in the normal course of business",
            }
        }

        self.occ_guidelines = {
            "SNC": {
                "nonaccrual_status": "asset is maintained on a cash basis because of deterioration in the financial condition of the borrower",
                "capitalization_of_interest": "interest may be capitalized only when the borrower is creditworthy and has the ability to repay the debt in the normal course of business",
                # Additional OCC guidelines could be added here
            }
        }
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        Loads the knowledge base from the JSON file.

        Returns:
            dict: The knowledge base data.
        """
        try:
            with open(self.knowledge_base_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Knowledge base file not found: {self.knowledge_base_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding knowledge base JSON: {self.knowledge_base_path}")
            return {}

    def analyze_snc(self, company_name, financial_data=None, industry_data=None, economic_data=None):
        """
        Analyzes a Shared National Credit (SNC) and assigns a risk rating based on
        the Comptroller's Handbook and OCC guidelines. Acts as an independent
        examiner persona. Can receive data directly or pull from the knowledge
        base if integrated with the system.

        Args:
            company_name (str): The name of the company.
            financial_data (dict, optional): Financial data of the company.
            industry_data (dict, optional): Industry-specific data.
            economic_data (dict, optional): Macroeconomic data.

        Returns:
            tuple: (SNCRating, str): The SNC rating and a detailed rationale for the rating.
        """

        # If data is not provided directly, retrieve from the knowledge base
        if not financial_data:
            financial_data = self.get_company_financial_data(company_name)
        if not industry_data:
            industry_data = self.get_industry_data(company_name)
        if not economic_data:
            economic_data = self.get_economic_data()

        # 1. Financial Statement Analysis
        # Analyze financial data based on Comptroller's Handbook guidelines
        financial_analysis_result = self._perform_financial_analysis(financial_data)

        # 2. Qualitative Analysis
        qualitative_analysis_result = self._perform_qualitative_analysis(company_name, industry_data, economic_data)

        # 3. Credit Risk Mitigation
        credit_risk_mitigation_result = self._evaluate_credit_risk_mitigation(financial_data)

        # 4. Rating Determination
        # Assign SNC rating based on a combination of quantitative and qualitative factors
        rating, rationale = self._determine_rating(financial_analysis_result, qualitative_analysis_result, credit_risk_mitigation_result)

        return rating, rationale

    def _perform_financial_analysis(self, financial_data):
        """
        Perform in-depth financial statement analysis based on Comptroller's Handbook criteria.

        Args:
            financial_data (dict): Financial data of the company.

        Returns:
            dict: Analysis results containing financial performance.
        """
        # Financial analysis metrics and thresholds based on the Comptroller's Handbook
        analysis_result = {
            "debt_to_equity": financial_data.get("debt_to_equity", 0),
            "profitability": financial_data.get("profitability", 0),
            "cash_flow": financial_data.get("cash_flow", 0),
            "liquidity_ratio": financial_data.get("liquidity_ratio", 0),
            "interest_coverage": financial_data.get("interest_coverage", 0)
        }

        return analysis_result

    def _perform_qualitative_analysis(self, company_name, industry_data, economic_data):
        """
        Evaluate qualitative factors including industry outlook, management quality, and economic context.

        Args:
            company_name (str): Name of the company.
            industry_data (dict): Industry-specific data.
            economic_data (dict): Macroeconomic data.

        Returns:
            dict: Qualitative analysis results.
        """
        qualitative_result = {
            "management_quality": "Strong" if industry_data.get("management_quality", "Strong") == "Strong" else "Weak",
            "industry_outlook": industry_data.get("outlook", "Neutral"),
            "economic_conditions": economic_data.get("economic_conditions", "Stable")
        }

        return qualitative_result

    def _evaluate_credit_risk_mitigation(self, financial_data):
        """
        Evaluate the effectiveness of credit risk mitigants such as collateral, guarantees, etc.

        Args:
            financial_data (dict): Financial data of the company.

        Returns:
            dict: Credit risk mitigation factors.
        """
        mitigation_result = {
            "collateral_quality": financial_data.get("collateral_quality", "Low"),
            "guarantees": financial_data.get("guarantees", "None"),
            "other_mitigants": financial_data.get("other_mitigants", "None")
        }

        return mitigation_result

    def _determine_rating(self, financial_analysis, qualitative_analysis, credit_risk_mitigation):
        """
        Determines the SNC rating based on a comprehensive assessment of
        credit risk, incorporating quantitative and qualitative factors.

        Args:
            financial_analysis (dict): Financial performance analysis results.
            qualitative_analysis (dict): Qualitative analysis results.
            credit_risk_mitigation (dict): Credit risk mitigation factors.

        Returns:
            tuple: (SNCRating, str): The SNC rating and a detailed rationale for the rating.
        """
        debt_to_equity = financial_analysis["debt_to_equity"]
        profitability = financial_analysis["profitability"]
        liquidity_ratio = financial_analysis["liquidity_ratio"]
        cash_flow = financial_analysis["cash_flow"]
        interest_coverage = financial_analysis["interest_coverage"]
        collateral_quality = credit_risk_mitigation["collateral_quality"]
        management_quality = qualitative_analysis["management_quality"]
        economic_conditions = qualitative_analysis["economic_conditions"]

        # Rating logic based on the Comptroller's Handbook and OCC guidelines
        if debt_to_equity > 3.0 and profitability < 0:
            rating = SNCRating.LOSS
            rationale = f"High debt-to-equity ratio and negative profitability suggest a 'Loss' rating as per Comptroller's definition."
        elif debt_to_equity > 2.0 and profitability < 0.1:
            rating = SNCRating.DOUBTFUL
            rationale = f"Elevated debt-to-equity ratio and low profitability imply a 'Doubtful' rating."
        elif liquidity_ratio < 1.0 and interest_coverage < 1.0:
            rating = SNCRating.SUBSTANDARD
            rationale = f"Liquidity ratio and interest coverage are insufficient, aligning with 'Substandard' criteria."
        elif collateral_quality == "Low" and management_quality == "Weak":
            rating = SNCRating.SPECIAL_MENTION
            rationale = "Weak management and poor collateral quality necessitate closer monitoring, 'Special Mention'."
        elif debt_to_equity <= 1.0 and profitability >= 0.3 and economic_conditions == "Stable":
            rating = SNCRating.PASS
            rationale = "Strong financial performance and favorable economic conditions, warranting a 'Pass' rating."
        else:
            rating = SNCRating.SPECIAL_MENTION
            rationale = "Potential weaknesses require monitoring, suggesting a 'Special Mention'."

        return rating, rationale

    def get_company_financial_data(self, company_name):
        """
        Retrieves company financial data from the knowledge base.

        Args:
            company_name (str): Name of the company.

        Returns:
            dict: Financial data of the company.
        """
        return self.knowledge_base.get("companies", {}).get(company_name, {})

    def get_industry_data(self, company_name):
        """
        Retrieves industry data for the company's industry from the knowledge base.

        Args:
            company_name (str): Name of the company.

        Returns:
            dict: Industry-specific data.
        """
        industry = self.knowledge_base.get("companies", {}).get(company_name, {}).get("industry", None)
        if industry:
            return self.knowledge_base.get("industries", {}).get(industry, {})
        else:
            return {}

    def get_economic_data(self):
        """
        Retrieves macroeconomic data from the knowledge base.

        Returns:
            dict: Macroeconomic data.
        """
        return self.knowledge_base.get("macroeconomic_data", {})


##################################

# core/agents/SNC_analyst_agent.py
import logging
import json 
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from unittest.mock import patch # Added for example usage

from core.agents.agent_base import AgentBase
from semantic_kernel import Kernel # For AgentBase type hinting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# For XAI debug logs, ensure the logger level is set to DEBUG if you want to see them.
# Example: logging.getLogger().setLevel(logging.DEBUG) in the main application or test setup.

class SNCRating(Enum):
    PASS = "Pass"
    SPECIAL_MENTION = "Special Mention"
    SUBSTANDARD = "Substandard"
    DOUBTFUL = "Doubtful"
    LOSS = "Loss"

class SNCAnalystAgent(AgentBase):
    """
    Agent for performing Shared National Credit (SNC) analysis.
    This agent analyzes company data based on regulatory guidelines to assign an SNC rating.
    It retrieves data via A2A communication with DataRetrievalAgent and can use SK skills.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Kernel] = None):
        super().__init__(config, kernel)
        self.persona = self.config.get('persona', "SNC Analyst Examiner")
        self.description = self.config.get('description', "Analyzes Shared National Credits based on regulatory guidelines by retrieving data via A2A and using Semantic Kernel skills.")
        self.expertise = self.config.get('expertise', ["SNC analysis", "regulatory compliance", "credit risk assessment"])

        self.comptrollers_handbook_snc = self.config.get('comptrollers_handbook_SNC', {})
        if not self.comptrollers_handbook_snc:
            logging.warning("Comptroller's Handbook SNC guidelines not found in agent configuration.")
        
        self.occ_guidelines_snc = self.config.get('occ_guidelines_SNC', {})
        if not self.occ_guidelines_snc:
            logging.warning("OCC Guidelines SNC not found in agent configuration.")

    async def execute(self, **kwargs) -> Optional[Tuple[Optional[SNCRating], str]]:
        company_id = kwargs.get('company_id')
        logging.info(f"Executing SNC analysis for company_id: {company_id}")
        logging.debug(f"SNC_ANALYSIS_EXECUTE_INPUT: company_id='{company_id}', all_kwargs={kwargs}")

        if not company_id:
            error_msg = "Company ID not provided for SNC analysis."
            logging.error(error_msg)
            return None, error_msg

        if 'DataRetrievalAgent' not in self.peer_agents:
            error_msg = "DataRetrievalAgent not found in peer agents for SNC_analyst_agent."
            logging.error(error_msg)
            return None, error_msg
        
        dra_request = {'data_type': 'get_company_financials', 'company_id': company_id}
        logging.debug(f"SNC_ANALYSIS_A2A_REQUEST: Requesting data from DataRetrievalAgent: {dra_request}")
        company_data_package = await self.send_message('DataRetrievalAgent', dra_request)
        logging.debug(f"SNC_ANALYSIS_A2A_RESPONSE: Received data package: {company_data_package is not None}")

        if not company_data_package:
            error_msg = f"Failed to retrieve company data package for {company_id} from DataRetrievalAgent."
            logging.error(error_msg)
            return None, error_msg

        company_info = company_data_package.get('company_info', {})
        financial_data_detailed = company_data_package.get('financial_data_detailed', {})
        qualitative_company_info = company_data_package.get('qualitative_company_info', {})
        industry_data_context = company_data_package.get('industry_data_context', {})
        economic_data_context = company_data_package.get('economic_data_context', {})
        collateral_and_debt_details = company_data_package.get('collateral_and_debt_details', {})
        
        logging.debug(f"SNC_ANALYSIS_DATA_EXTRACTED: CompanyInfo: {company_info.keys()}, FinancialDetailed: {financial_data_detailed.keys()}, Qualitative: {qualitative_company_info.keys()}, Industry: {industry_data_context.keys()}, Economic: {economic_data_context.keys()}, Collateral: {collateral_and_debt_details.keys()}")

        financial_analysis_inputs_for_sk = self._prepare_financial_inputs_for_sk(financial_data_detailed)
        qualitative_analysis_inputs_for_sk = self._prepare_qualitative_inputs_for_sk(qualitative_company_info)
        
        # These methods provide both numerical data for Python logic & string data for SK skills
        financial_analysis_result = self._perform_financial_analysis(financial_data_detailed, financial_analysis_inputs_for_sk)
        qualitative_analysis_result = self._perform_qualitative_analysis(
            company_info.get('name', company_id), 
            qualitative_company_info, 
            industry_data_context, 
            economic_data_context,
            qualitative_analysis_inputs_for_sk 
        )
        credit_risk_mitigation_info = self._evaluate_credit_risk_mitigation(collateral_and_debt_details)
        
        rating, rationale = await self._determine_rating( 
            company_info.get('name', company_id), 
            financial_analysis_result, 
            qualitative_analysis_result, 
            credit_risk_mitigation_info, 
            economic_data_context
        )
        logging.debug(f"SNC_ANALYSIS_EXECUTE_OUTPUT: Rating='{rating.value if rating else 'N/A'}', Rationale='{rationale}'")
        return rating, rationale

    def _prepare_financial_inputs_for_sk(self, financial_data_detailed: Dict[str, Any]) -> Dict[str, str]:
        """Prepares stringified financial inputs required by SK skills."""
        cash_flow_statement = financial_data_detailed.get("cash_flow_statement", {})
        key_ratios = financial_data_detailed.get("key_ratios", {})
        # Using .get for placeholders to avoid error if market_data or dcf_assumptions are missing
        market_data = financial_data_detailed.get("market_data", {})
        dcf_assumptions = financial_data_detailed.get("dcf_assumptions", {})


        return {
            "historical_fcf_str": str(cash_flow_statement.get('free_cash_flow', [])),
            "historical_cfo_str": str(cash_flow_statement.get('cash_flow_from_operations', [])),
            "annual_debt_service_str": str(market_data.get("annual_debt_service_placeholder", "Not Available")), # Placeholder
            "ratios_summary_str": json.dumps(key_ratios),
            "projected_fcf_str": str(dcf_assumptions.get("projected_fcf_placeholder", "Not Available")), # Placeholder
            "payment_history_status_str": str(market_data.get("payment_history_placeholder", "Current")), # Placeholder
            "interest_capitalization_status_str": str(market_data.get("interest_capitalization_placeholder", "No")) # Placeholder
        }

    def _prepare_qualitative_inputs_for_sk(self, qualitative_company_info: Dict[str, Any]) -> Dict[str, str]:
        """Prepares stringified qualitative inputs required by SK skills."""
        return {
            "qualitative_notes_stability_str": qualitative_company_info.get("revenue_cashflow_stability_notes_placeholder", "Management reports stable customer contracts."), # Placeholder
            "notes_financial_deterioration_str": qualitative_company_info.get("financial_deterioration_notes_placeholder", "No significant deterioration noted recently.") # Placeholder
        }

    def _perform_financial_analysis(self, financial_data_detailed: Dict[str, Any], sk_financial_inputs: Dict[str, str]) -> Dict[str, Any]:
        logging.debug(f"SNC_FIN_ANALYSIS_INPUT: financial_data_detailed keys: {financial_data_detailed.keys()}, sk_inputs: {sk_financial_inputs.keys()}")
        key_ratios = financial_data_detailed.get("key_ratios", {})
        
        analysis_result = {
            "debt_to_equity": key_ratios.get("debt_to_equity_ratio"),
            "profitability": key_ratios.get("net_profit_margin"),
            "liquidity_ratio": key_ratios.get("current_ratio"),
            "interest_coverage": key_ratios.get("interest_coverage_ratio"),
            **sk_financial_inputs # Add stringified inputs for SK
        }
        logging.debug(f"SNC_FIN_ANALYSIS_OUTPUT: {analysis_result}")
        return analysis_result

    def _perform_qualitative_analysis(self, 
                                      company_name: str, 
                                      qualitative_company_info: Dict[str, Any], 
                                      industry_data_context: Dict[str, Any], 
                                      economic_data_context: Dict[str, Any],
                                      sk_qualitative_inputs: Dict[str, str]) -> Dict[str, Any]:
        logging.debug(f"SNC_QUAL_ANALYSIS_INPUT: company_name='{company_name}', qualitative_info_keys={qualitative_company_info.keys()}, industry_keys={industry_data_context.keys()}, economic_keys={economic_data_context.keys()}, sk_qual_inputs: {sk_qualitative_inputs.keys()}")
        qualitative_result = {
            "management_quality": qualitative_company_info.get("management_assessment", "Not Assessed"),
            "industry_outlook": industry_data_context.get("outlook", "Neutral"),
            "economic_conditions": economic_data_context.get("overall_outlook", "Stable"),
            "business_model_strength": qualitative_company_info.get("business_model_strength", "N/A"),
            "competitive_advantages": qualitative_company_info.get("competitive_advantages", "N/A"),
            **sk_qualitative_inputs # Add stringified inputs for SK
        }
        logging.debug(f"SNC_QUAL_ANALYSIS_OUTPUT: {qualitative_result}")
        return qualitative_result

    def _evaluate_credit_risk_mitigation(self, collateral_and_debt_details: Dict[str, Any]) -> Dict[str, Any]:
        logging.debug(f"SNC_CREDIT_MITIGATION_INPUT: collateral_and_debt_details_keys={collateral_and_debt_details.keys()}")
        ltv = collateral_and_debt_details.get("loan_to_value_ratio")
        collateral_quality_assessment = "Low" 
        if ltv is not None:
            try:
                ltv_float = float(ltv)
                if ltv_float < 0.5: collateral_quality_assessment = "High"
                elif ltv_float < 0.75: collateral_quality_assessment = "Medium"
            except ValueError: logging.warning(f"Could not parse LTV ratio '{ltv}' as float.")
        
        mitigation_result = {
            "collateral_quality_fallback": collateral_quality_assessment, 
            "collateral_summary_for_sk": collateral_and_debt_details.get("collateral_type", "Not specified."),
            "loan_to_value_ratio": str(ltv) if ltv is not None else "Not specified.",
            "collateral_notes_for_sk": collateral_and_debt_details.get("other_credit_enhancements", "None."),
            "collateral_valuation": collateral_and_debt_details.get("collateral_valuation"),
            "guarantees_present": collateral_and_debt_details.get("guarantees_exist", False)
        }
        logging.debug(f"SNC_CREDIT_MITIGATION_OUTPUT: {mitigation_result}")
        return mitigation_result

    async def _determine_rating(self, company_name: str, 
                               financial_analysis: Dict[str, Any], 
                               qualitative_analysis: Dict[str, Any], 
                               credit_risk_mitigation: Dict[str, Any],
                               economic_data_context: Dict[str, Any] # Retained for direct use if needed
                               # financial_data_detailed is no longer passed as its parts are in financial_analysis
                               ) -> Tuple[Optional[SNCRating], str]:
        logging.debug(f"SNC_DETERMINE_RATING_INPUT: company='{company_name}', financial_analysis={financial_analysis.keys()}, qualitative_analysis={qualitative_analysis.keys()}, credit_mitigation={credit_risk_mitigation.keys()}, economic_context={economic_data_context.keys()}")
        
        rationale_parts = []
        collateral_sk_assessment_str = None
        collateral_sk_justification = ""
        repayment_sk_assessment_str = None
        repayment_sk_justification = ""
        repayment_sk_concerns = ""
        nonaccrual_sk_assessment_str = None
        nonaccrual_sk_justification = ""

        if self.kernel and hasattr(self.kernel, 'skills'):
            # 1. AssessCollateralRisk (existing)
            try:
                sk_input_vars_collateral = {
                    "guideline_substandard_collateral": self.comptrollers_handbook_snc.get('substandard_definition', "Collateral is inadequately protective."),
                    "guideline_repayment_source": self.comptrollers_handbook_snc.get('primary_repayment_source', "Primary repayment should come from a sustainable source of cash under borrower control."),
                    "collateral_description": credit_risk_mitigation.get('collateral_summary_for_sk', "Not specified."),
                    "ltv_ratio": credit_risk_mitigation.get('loan_to_value_ratio', "Not specified."),
                    "other_collateral_notes": credit_risk_mitigation.get('collateral_notes_for_sk', "None.")
                }
                logging.debug(f"SNC_DETERMINE_RATING_SK_INPUT_Collateral: {sk_input_vars_collateral}")
                sk_response_collateral = await self.run_semantic_kernel_skill("SNCRatingAssistSkill", "CollateralRiskAssessment", sk_input_vars_collateral)
                lines = sk_response_collateral.strip().splitlines()
                if lines:
                    if "Assessment:" in lines[0]: collateral_sk_assessment_str = lines[0].split("Assessment:", 1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]: collateral_sk_justification = lines[1].split("Justification:", 1)[1].strip()
                logging.debug(f"SNC_DETERMINE_RATING_SK_OUTPUT_Collateral: Assessment='{collateral_sk_assessment_str}', Justification='{collateral_sk_justification}'")
                if collateral_sk_justification: rationale_parts.append(f"SK Collateral Assessment ({collateral_sk_assessment_str}): {collateral_sk_justification}")
            except Exception as e: logging.error(f"Error in CollateralRiskAssessment SK skill: {e}")

            # 2. AssessRepaymentCapacity
            try:
                sk_input_vars_repayment = {
                    "guideline_repayment_source": self.comptrollers_handbook_snc.get('primary_repayment_source', "Default guideline..."),
                    "guideline_substandard_paying_capacity": self.comptrollers_handbook_snc.get('substandard_definition', "Default substandard..."), # This might need a more specific guideline part
                    "repayment_capacity_period_years": str(self.comptrollers_handbook_snc.get('repayment_capacity_period', 7)),
                    "historical_fcf": financial_analysis.get('historical_fcf_str', "Not available"),
                    "historical_cfo": financial_analysis.get('historical_cfo_str', "Not available"),
                    "annual_debt_service": financial_analysis.get('annual_debt_service_str', "Not available"), # Corrected key
                    "relevant_ratios": financial_analysis.get('ratios_summary_str', "Not available"),
                    "projected_fcf": financial_analysis.get('projected_fcf_str', "Not available"),
                    "qualitative_notes_stability": qualitative_analysis.get('qualitative_notes_stability_str', "None provided.")
                }
                logging.debug(f"SNC_DETERMINE_RATING_SK_INPUT_Repayment: {sk_input_vars_repayment}")
                sk_response_repayment = await self.run_semantic_kernel_skill("SNCRatingAssistSkill", "AssessRepaymentCapacity", sk_input_vars_repayment)
                lines = sk_response_repayment.strip().splitlines()
                if lines:
                    if "Assessment:" in lines[0]: repayment_sk_assessment_str = lines[0].split("Assessment:",1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]: repayment_sk_justification = lines[1].split("Justification:",1)[1].strip()
                    if len(lines) > 2 and "Concerns:" in lines[2]: repayment_sk_concerns = lines[2].split("Concerns:",1)[1].strip()
                logging.debug(f"SNC_DETERMINE_RATING_SK_OUTPUT_Repayment: Assessment='{repayment_sk_assessment_str}', Justification='{repayment_sk_justification}', Concerns='{repayment_sk_concerns}'")
                if repayment_sk_justification: rationale_parts.append(f"SK Repayment Capacity ({repayment_sk_assessment_str}): {repayment_sk_justification}. Concerns: {repayment_sk_concerns}")
            except Exception as e: logging.error(f"Error in AssessRepaymentCapacity SK skill: {e}")

            # 3. AssessNonAccrualStatusIndication
            try:
                sk_input_vars_nonaccrual = {
                    "guideline_nonaccrual_status": self.occ_guidelines_snc.get('nonaccrual_status', "Default non-accrual..."),
                    "guideline_interest_capitalization": self.occ_guidelines_snc.get('capitalization_of_interest', "Default interest cap..."),
                    "payment_history_status": financial_analysis.get('payment_history_status_str', "Current"),
                    "relevant_ratios": financial_analysis.get('ratios_summary_str', "Not available"),
                    "repayment_capacity_assessment": repayment_sk_assessment_str if repayment_sk_assessment_str else "Adequate", # Use output from previous skill
                    "notes_financial_deterioration": qualitative_analysis.get('notes_financial_deterioration_str', "None noted."),
                    "interest_capitalization_status": financial_analysis.get('interest_capitalization_status_str', "No")
                }
                logging.debug(f"SNC_DETERMINE_RATING_SK_INPUT_NonAccrual: {sk_input_vars_nonaccrual}")
                sk_response_nonaccrual = await self.run_semantic_kernel_skill("SNCRatingAssistSkill", "AssessNonAccrualStatusIndication", sk_input_vars_nonaccrual)
                lines = sk_response_nonaccrual.strip().splitlines()
                if lines:
                    if "Assessment:" in lines[0]: nonaccrual_sk_assessment_str = lines[0].split("Assessment:",1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]: nonaccrual_sk_justification = lines[1].split("Justification:",1)[1].strip()
                logging.debug(f"SNC_DETERMINE_RATING_SK_OUTPUT_NonAccrual: Assessment='{nonaccrual_sk_assessment_str}', Justification='{nonaccrual_sk_justification}'")
                if nonaccrual_sk_justification: rationale_parts.append(f"SK Non-Accrual Assessment ({nonaccrual_sk_assessment_str}): {nonaccrual_sk_justification}")
            except Exception as e: logging.error(f"Error in AssessNonAccrualStatusIndication SK skill: {e}")
        
        debt_to_equity = financial_analysis.get("debt_to_equity")
        profitability = financial_analysis.get("profitability")
        rating = SNCRating.PASS 

        logging.debug(f"SNC_RATING_INITIAL_PARAMS: DtE={debt_to_equity}, Profitability={profitability}, SKCollateral='{collateral_sk_assessment_str}', SKRepayment='{repayment_sk_assessment_str}', SKNonAccrual='{nonaccrual_sk_assessment_str}', FallbackCollateral='{credit_risk_mitigation.get('collateral_quality_fallback')}', ManagementQuality='{qualitative_analysis.get('management_quality')}'")

        # Incorporate SK outputs into rating logic
        if repayment_sk_assessment_str == "Unsustainable" or (nonaccrual_sk_assessment_str == "Non-Accrual Warranted" and repayment_sk_assessment_str == "Weak"):
            logging.debug(f"SNC_RATING_RULE_TRIGGERED: LOSS - SKRepayment: {repayment_sk_assessment_str}, SKNonAccrual: {nonaccrual_sk_assessment_str}")
            rating = SNCRating.LOSS
            rationale_parts.append("Loss rating driven by SK assessment of unsustainable repayment or non-accrual with weak repayment.")
        elif repayment_sk_assessment_str == "Weak" or (collateral_sk_assessment_str == "Substandard" and repayment_sk_assessment_str == "Adequate"):
            logging.debug(f"SNC_RATING_RULE_TRIGGERED: DOUBTFUL - SKRepayment: {repayment_sk_assessment_str}, SKCollateral: {collateral_sk_assessment_str}")
            rating = SNCRating.DOUBTFUL
            rationale_parts.append("Doubtful rating influenced by SK assessment of weak repayment or substandard collateral with adequate repayment.")
        elif nonaccrual_sk_assessment_str == "Non-Accrual Warranted" or collateral_sk_assessment_str == "Substandard" or repayment_sk_assessment_str == "Adequate": # Simplified
            logging.debug(f"SNC_RATING_RULE_TRIGGERED: SUBSTANDARD - SKNonAccrual: {nonaccrual_sk_assessment_str}, SKCollateral: {collateral_sk_assessment_str}, SKRepayment: {repayment_sk_assessment_str}")
            rating = SNCRating.SUBSTANDARD # This is a broad rule, refine with more specific conditions
            rationale_parts.append("Substandard rating influenced by SK assessments (Non-Accrual, Collateral, or Repayment).")
        
        # Fallback/Original Python-based logic if SK results are not decisive or available
        if rating == SNCRating.PASS: # Only apply python logic if SK hasn't set a more severe rating
            if debt_to_equity is not None and profitability is not None:
                if debt_to_equity > 3.0 and profitability < 0:
                    # This was Loss in python logic, SK might have already determined this.
                    # If SK didn't determine Loss, but python logic does, we might need to reconcile or prioritize.
                    # For now, let SK take precedence if it made a call.
                    if rating == SNCRating.PASS: # Only if SK didn't already make it worse
                        logging.debug(f"SNC_RATING_RULE_TRIGGERED (Python Fallback): LOSS - DtE ({debt_to_equity}) > 3.0 and Profitability ({profitability}) < 0")
                        rating = SNCRating.LOSS
                        rationale_parts.append("Fallback: High D/E ratio and negative profitability.")
                # ... (other original python rules can be adapted here as fallbacks or complementary checks) ...
                elif (collateral_sk_assessment_str is None and credit_risk_mitigation.get("collateral_quality_fallback") == "Low") and \
                     qualitative_analysis.get("management_quality") == "Weak":
                    logging.debug(f"SNC_RATING_RULE_TRIGGERED (Python Fallback): SPECIAL_MENTION - Fallback Collateral: {credit_risk_mitigation.get('collateral_quality_fallback')}, Management: {qualitative_analysis.get('management_quality')}")
                    rating = SNCRating.SPECIAL_MENTION
                    rationale_parts.append(f"Fallback: Collateral concerns (Fallback: {credit_risk_mitigation.get('collateral_quality_fallback')}) and weak management warrant Special Mention.")
                elif debt_to_equity <= 1.0 and profitability >= 0.3 and qualitative_analysis.get("economic_conditions") == "Stable":
                    logging.debug(f"SNC_RATING_RULE_TRIGGERED (Python Fallback): PASS - DtE ({debt_to_equity}) <= 1.0, Profitability ({profitability}) >= 0.3, Econ Conditions: {qualitative_analysis.get('economic_conditions')}")
                    # rating remains PASS
                    rationale_parts.append("Fallback: Strong financials and stable economic conditions.")
                else: 
                    if rating == SNCRating.PASS: 
                        logging.debug(f"SNC_RATING_RULE_TRIGGERED (Python Fallback): SPECIAL_MENTION - Fallback/Mixed Indicators. Initial DtE: {debt_to_equity}, Profitability: {profitability}")
                        rating = SNCRating.SPECIAL_MENTION
                        rationale_parts.append("Fallback: Mixed financial indicators or other unaddressed concerns warrant monitoring.")
            elif rating == SNCRating.PASS : # If still pass and key metrics were missing
                logging.debug("SNC_RATING_RULE_TRIGGERED (Python Fallback): UNDETERMINED - Missing key financial metrics (DtE or Profitability)")
                rating = None # Cannot determine rating
                rationale_parts.append("Fallback: Cannot determine rating due to missing key financial metrics (debt-to-equity or profitability).")


        rationale_parts.append(f"Regulatory guidance: Comptroller's Handbook SNC v{self.comptrollers_handbook_snc.get('version', 'N/A')}, OCC Guidelines v{self.occ_guidelines_snc.get('version', 'N/A')}.")
        final_rationale = " ".join(filter(None, rationale_parts))
        
        logging.debug(f"SNC_DETERMINE_RATING_OUTPUT: Final Rating='{rating.value if rating else 'Undetermined'}', Rationale='{final_rationale}'")
        logging.info(f"SNC rating for {company_name}: {rating.value if rating else 'Undetermined'}. Rationale: {final_rationale}")
        return rating, final_rationale

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG) # Enable DEBUG logs for example
    
    dummy_snc_agent_config = {
        'persona': "Test SNC Analyst",
        'comptrollers_handbook_SNC': {
            "version": "2024.Q1_test",
            "substandard_definition": "Collateral is inadequately protective if its value doesn't cover the loan or if perfection issues exist. Paying capacity is inadequate if primary repayment source is not sustainable.",
            "primary_repayment_source": "Repayment should primarily come from the borrower's sustainable cash flow, under their control.",
            "repayment_capacity_period": 7
        },
        'occ_guidelines_SNC': {
            "version": "2024-03_test",
            "nonaccrual_status": "Loan is maintained on a cash basis due to financial deterioration of borrower; payment of principal or interest is not expected.",
            "capitalization_of_interest": "Permissible only if borrower is creditworthy and can repay in normal course of business."
            },
        'peers': ['DataRetrievalAgent'] 
    }

    mock_data_package_template = {
        "company_info": {"name": "TestCompany Corp", "industry_sector": "Tech", "country": "USA"},
        "financial_data_detailed": {
            "key_ratios": {"debt_to_equity_ratio": 1.5, "net_profit_margin": 0.15, "current_ratio": 1.8, "interest_coverage_ratio": 3.0},
            "cash_flow_statement": {"free_cash_flow": [100,110,120], "cash_flow_from_operations": [150,160,170]},
            "market_data": {"annual_debt_service_placeholder": "60", "payment_history_placeholder": "30 days past due", "interest_capitalization_placeholder": "Yes"},
            "dcf_assumptions": {"projected_fcf_placeholder": "[130, 140, 150]"}
        },
        "qualitative_company_info": {
            "management_assessment": "Average", 
            "business_model_strength": "Moderate",
            "revenue_cashflow_stability_notes_placeholder": "Revenue streams are moderately diverse.",
            "financial_deterioration_notes_placeholder": "Recent downturn in quarterly earnings."
            },
        "industry_data_context": {"outlook": "Stable"},
        "economic_data_context": {"overall_outlook": "Stable"},
        "collateral_and_debt_details": {
            "collateral_type": "Accounts Receivable, Inventory", 
            "collateral_valuation": 750000, 
            "loan_to_value_ratio": 0.6, 
            "guarantees_exist": False,
            "other_credit_enhancements": "Standard covenants in place."
        }
    }

    async def mock_send_message(target_agent_name, message):
        if target_agent_name == 'DataRetrievalAgent' and message.get('data_type') == 'get_company_financials':
            company_id = message.get('company_id')
            data = json.loads(json.dumps(mock_data_package_template)) # Deep copy
            data["company_info"]["name"] = f"{company_id} Corp"
            if company_id == "TEST_COMPANY_REPAY_WEAK":
                data["financial_data_detailed"]["key_ratios"]["interest_coverage_ratio"] = 0.8
                data["financial_data_detailed"]["cash_flow_statement"]["free_cash_flow"] = [10, 5, -20]
            return data
        return None

    class MockSKFunction:
        def __init__(self, skill_name):
            self.skill_name = skill_name

        async def invoke(self, variables=None): 
            class MockSKResult:
                def __init__(self, value_str): self._value = value_str
                def __str__(self): return self._value
            
            if self.skill_name == "CollateralRiskAssessment":
                ltv_str = variables.get("ltv_ratio", "Not specified.")
                assessment = "Pass"; justification = "Collateral LTV is acceptable."
                try:
                    ltv = float(ltv_str)
                    if ltv > 0.7: assessment = "Substandard"; justification = "High LTV."
                    elif ltv > 0.5: assessment = "Special Mention"; justification = "LTV needs monitoring."
                except ValueError: pass
                return MockSKResult(f"Assessment: {assessment}\nJustification: {justification}")
            elif self.skill_name == "AssessRepaymentCapacity":
                assessment = "Adequate"; justification = "Repayment capacity seems adequate."; concerns="None."
                if "0.8" in variables.get("relevant_ratios",""): # Crude check for RiskyCorp from test_agents
                     assessment = "Weak"; justification = "Repayment capacity is weak based on ratios."; concerns="Debt service coverage."
                return MockSKResult(f"Assessment: {assessment}\nJustification: {justification}\nConcerns: {concerns}")
            elif self.skill_name == "AssessNonAccrualStatusIndication":
                assessment = "Accrual Appropriate"; justification = "Currently performing."
                if variables.get("payment_history_status") == "90 days past due" or variables.get("repayment_capacity_assessment") == "Weak":
                    assessment = "Non-Accrual Warranted"; justification = "Deterioration noted."
                return MockSKResult(f"Assessment: {assessment}\nJustification: {justification}")
            return MockSKResult("Unknown mock skill called.")


    class MockSKSkillsCollection:
        def get_function(self, skill_collection_name, skill_name):
            if skill_collection_name == "SNCRatingAssistSkill":
                return MockSKFunction(skill_name) # Pass skill_name to distinguish in invoke
            return None
        
    class MockKernel:
        def __init__(self): self.skills = MockSKSkillsCollection()
        async def run_async(self, sk_function, input_vars=None, **kwargs): # Changed to run_async for consistency
            if sk_function: return await sk_function.invoke(variables=input_vars)
            return "Mock kernel run_async failed: No function"

    async def run_tests():
        logging.getLogger().setLevel(logging.DEBUG) # Enable DEBUG logs for this example run
        
        mock_kernel_instance = MockKernel()
        snc_agent_with_kernel = SNCAnalystAgent(config=dummy_snc_agent_config, kernel=mock_kernel_instance)
        
        with patch.object(snc_agent_with_kernel, 'send_message', new=mock_send_message):
            print("\n--- Test Case: Good Financials (TEST_COMPANY_SK_PASS) ---")
            rating_pass, rationale_pass = await snc_agent_with_kernel.execute(company_id="TEST_COMPANY_SK_PASS")
            print(f"Rating: {rating_pass.value if rating_pass else 'N/A'}, Rationale: {rationale_pass}")

            print("\n--- Test Case: Weak Repayment (TEST_COMPANY_REPAY_WEAK) ---")
            rating_repay_weak, rationale_repay_weak = await snc_agent_with_kernel.execute(company_id="TEST_COMPANY_REPAY_WEAK")
            print(f"Rating: {rating_repay_weak.value if rating_repay_weak else 'N/A'}, Rationale: {rationale_repay_weak}")
            
    if __name__ == '__main__':
        asyncio.run(run_tests())


