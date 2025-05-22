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
        """
        Initializes the SNC Analyst Agent.

        Args:
            config: Agent-specific configuration dictionary. Expected keys:
                    'persona', 'description', 'expertise', 
                    'comptrollers_handbook_SNC' (dict), 'occ_guidelines_SNC' (dict),
                    and 'peers' (list, for A2A communication).
            kernel: Optional Semantic Kernel instance.
        """
        super().__init__(config, kernel)
        self.persona = self.config.get('persona', "SNC Analyst Examiner")
        self.description = self.config.get('description', "Analyzes Shared National Credits based on regulatory guidelines by retrieving data via A2A and using Semantic Kernel skills.")
        self.expertise = self.config.get('expertise', ["SNC analysis", "regulatory compliance", "credit risk assessment"])

        # Load regulatory guidelines from agent configuration
        self.comptrollers_handbook_snc = self.config.get('comptrollers_handbook_SNC', {})
        if not self.comptrollers_handbook_snc:
            logging.warning("Comptroller's Handbook SNC guidelines not found in agent configuration.")
        
        self.occ_guidelines_snc = self.config.get('occ_guidelines_SNC', {})
        if not self.occ_guidelines_snc:
            logging.warning("OCC Guidelines SNC not found in agent configuration.")

    async def execute(self, **kwargs) -> Optional[Tuple[Optional[SNCRating], str]]:
        """
        Main execution entry point for the SNC Analyst Agent.
        Performs SNC analysis on a company identified by `company_id`.
        Data is fetched via A2A communication with DataRetrievalAgent.

        Args:
            **kwargs: Expected keyword arguments:
                company_id (str): The ID of the company to analyze.

        Returns:
            A tuple containing:
                - SNCRating (Optional[SNCRating]): The determined SNC rating, or None if an error occurs.
                - str: A rationale for the rating or an error message.
        """
        company_id = kwargs.get('company_id')
        logging.info(f"Executing SNC analysis for company_id: {company_id}")
        # XAI Log: Record initial input to the execute method
        logging.debug(f"SNC_ANALYSIS_EXECUTE_INPUT: company_id='{company_id}', all_kwargs={kwargs}")


        if not company_id:
            error_msg = "Company ID not provided for SNC analysis."
            logging.error(error_msg)
            return None, error_msg

        # A2A Communication: Check for DataRetrievalAgent peer
        if 'DataRetrievalAgent' not in self.peer_agents:
            error_msg = "DataRetrievalAgent not found in peer agents for SNC_analyst_agent."
            logging.error(error_msg)
            return None, error_msg
        
        # Prepare and send request to DataRetrievalAgent
        dra_request = {'data_type': 'get_company_financials', 'company_id': company_id}
        logging.debug(f"SNC_ANALYSIS_A2A_REQUEST: Requesting data from DataRetrievalAgent: {dra_request}")
        company_data_package = await self.send_message('DataRetrievalAgent', dra_request)
        logging.debug(f"SNC_ANALYSIS_A2A_RESPONSE: Received data package: {company_data_package is not None}")


        if not company_data_package:
            error_msg = f"Failed to retrieve company data package for {company_id} from DataRetrievalAgent."
            logging.error(error_msg)
            return None, error_msg

        # Extract structured data from the received package
        company_info = company_data_package.get('company_info', {})
        financial_data_detailed = company_data_package.get('financial_data_detailed', {})
        qualitative_company_info = company_data_package.get('qualitative_company_info', {})
        industry_data_context = company_data_package.get('industry_data_context', {})
        economic_data_context = company_data_package.get('economic_data_context', {})
        collateral_and_debt_details = company_data_package.get('collateral_and_debt_details', {})
        
        # XAI Log: Record the keys of extracted data sections to verify data presence
        logging.debug(f"SNC_ANALYSIS_DATA_EXTRACTED: CompanyInfo: {company_info.keys()}, FinancialDetailed: {financial_data_detailed.keys()}, Qualitative: {qualitative_company_info.keys()}, Industry: {industry_data_context.keys()}, Economic: {economic_data_context.keys()}, Collateral: {collateral_and_debt_details.keys()}")

        # Perform various analysis steps using helper methods
        financial_analysis_result = self._perform_financial_analysis(financial_data_detailed)
        qualitative_analysis_result = self._perform_qualitative_analysis(
            company_info.get('name', company_id), 
            qualitative_company_info, 
            industry_data_context, 
            economic_data_context
        )
        credit_risk_mitigation_info = self._evaluate_credit_risk_mitigation(collateral_and_debt_details)
        
        # Asynchronously determine the final rating and rationale
        rating, rationale = await self._determine_rating( 
            company_info.get('name', company_id), 
            financial_analysis_result, 
            qualitative_analysis_result, 
            credit_risk_mitigation_info, 
            economic_data_context
        )
        # XAI Log: Record the final output of the execute method
        logging.debug(f"SNC_ANALYSIS_EXECUTE_OUTPUT: Rating='{rating.value if rating else 'N/A'}', Rationale='{rationale}'")
        return rating, rationale

    def _perform_financial_analysis(self, financial_data_detailed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs financial analysis based on detailed financial data.

        Args:
            financial_data_detailed (Dict[str, Any]): A dictionary containing detailed financial data,
                                                     expected to have a 'key_ratios' sub-dictionary.
        Returns:
            Dict[str, Any]: A dictionary of key financial metrics relevant for SNC analysis.
        """
        # XAI Log: Record input for financial analysis
        logging.debug(f"SNC_FIN_ANALYSIS_INPUT: financial_data_detailed keys: {financial_data_detailed.keys()}")
        key_ratios = financial_data_detailed.get("key_ratios", {})
        analysis_result = {
            "debt_to_equity": key_ratios.get("debt_to_equity_ratio"),
            "profitability": key_ratios.get("net_profit_margin"),
            "liquidity_ratio": key_ratios.get("current_ratio"),
            "interest_coverage": key_ratios.get("interest_coverage_ratio")
        }
        # XAI Log: Record output of financial analysis
        logging.debug(f"SNC_FIN_ANALYSIS_OUTPUT: {analysis_result}")
        return analysis_result

    def _perform_qualitative_analysis(self, 
                                      company_name: str, 
                                      qualitative_company_info: Dict[str, Any], 
                                      industry_data_context: Dict[str, Any], 
                                      economic_data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs qualitative analysis based on company, industry, and economic context.

        Args:
            company_name (str): Name of the company.
            qualitative_company_info (Dict[str, Any]): Dictionary with qualitative company details.
            industry_data_context (Dict[str, Any]): Dictionary with industry context.
            economic_data_context (Dict[str, Any]): Dictionary with economic context.

        Returns:
            Dict[str, Any]: A dictionary of key qualitative assessment factors.
        """
        # XAI Log: Record inputs for qualitative analysis
        logging.debug(f"SNC_QUAL_ANALYSIS_INPUT: company_name='{company_name}', qualitative_info_keys={qualitative_company_info.keys()}, industry_keys={industry_data_context.keys()}, economic_keys={economic_data_context.keys()}")
        qualitative_result = {
            "management_quality": qualitative_company_info.get("management_assessment", "Not Assessed"),
            "industry_outlook": industry_data_context.get("outlook", "Neutral"),
            "economic_conditions": economic_data_context.get("overall_outlook", "Stable"),
            "business_model_strength": qualitative_company_info.get("business_model_strength", "N/A"),
            "competitive_advantages": qualitative_company_info.get("competitive_advantages", "N/A")
        }
        # XAI Log: Record output of qualitative analysis
        logging.debug(f"SNC_QUAL_ANALYSIS_OUTPUT: {qualitative_result}")
        return qualitative_result

    def _evaluate_credit_risk_mitigation(self, collateral_and_debt_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates credit risk mitigants based on collateral and debt details.
        Prepares data for both SK skill and fallback logic.

        Args:
            collateral_and_debt_details (Dict[str, Any]): Dictionary with collateral and debt information.

        Returns:
            Dict[str, Any]: A dictionary containing evaluated mitigation factors and data for SK skill.
        """
        # XAI Log: Record input for credit risk mitigation evaluation
        logging.debug(f"SNC_CREDIT_MITIGATION_INPUT: collateral_and_debt_details_keys={collateral_and_debt_details.keys()}")
        
        ltv = collateral_and_debt_details.get("loan_to_value_ratio")
        collateral_quality_assessment = "Low" # Default for fallback logic
        if ltv is not None:
            try:
                ltv_float = float(ltv)
                if ltv_float < 0.5: collateral_quality_assessment = "High"
                elif ltv_float < 0.75: collateral_quality_assessment = "Medium"
            except ValueError:
                logging.warning(f"Could not parse LTV ratio '{ltv}' as float.")
        
        # Prepare data for SK skill and for fallback logic
        mitigation_result = {
            "collateral_quality_fallback": collateral_quality_assessment, 
            "collateral_summary_for_sk": collateral_and_debt_details.get("collateral_type", "Not specified."),
            "loan_to_value_ratio": str(ltv) if ltv is not None else "Not specified.", # Ensure string for SK
            "collateral_notes_for_sk": collateral_and_debt_details.get("other_credit_enhancements", "None."),
            "collateral_valuation": collateral_and_debt_details.get("collateral_valuation"),
            "guarantees_present": collateral_and_debt_details.get("guarantees_exist", False)
        }
        # XAI Log: Record output of credit risk mitigation evaluation
        logging.debug(f"SNC_CREDIT_MITIGATION_OUTPUT: {mitigation_result}")
        return mitigation_result

    async def _determine_rating(self, company_name: str, 
                               financial_analysis: Dict[str, Any], 
                               qualitative_analysis: Dict[str, Any], 
                               credit_risk_mitigation: Dict[str, Any], 
                               economic_data_context: Dict[str, Any]
                               ) -> Tuple[Optional[SNCRating], str]:
        """
        Determines the final SNC rating by integrating various analysis components.
        Uses Semantic Kernel for collateral assessment if available, otherwise falls back
        to Python-based logic.

        Args:
            company_name (str): Name of the company.
            financial_analysis (Dict[str, Any]): Output from _perform_financial_analysis.
            qualitative_analysis (Dict[str, Any]): Output from _perform_qualitative_analysis.
            credit_risk_mitigation (Dict[str, Any]): Output from _evaluate_credit_risk_mitigation.
            economic_data_context (Dict[str, Any]): Economic context data.

        Returns:
            A tuple: (SNCRating, str) representing the SNC rating and a detailed rationale.
        """
        # XAI Log: Record all inputs to the rating determination logic
        logging.debug(f"SNC_DETERMINE_RATING_INPUT: company='{company_name}', financial_analysis={financial_analysis}, qualitative_analysis={qualitative_analysis}, credit_mitigation={credit_risk_mitigation}, economic_context={economic_data_context}")
        
        rationale_parts = []
        collateral_sk_assessment_str = None 
        collateral_sk_justification = ""

        # Semantic Kernel Integration for Collateral Assessment
        if self.kernel and hasattr(self.kernel, 'skills'): # Check if kernel and skills collection exist
            try:
                # Prepare inputs for the SK skill "CollateralRiskAssessment"
                sk_input_vars = {
                    "guideline_substandard_collateral": self.comptrollers_handbook_snc.get('substandard_definition', "Collateral is inadequately protective."),
                    "guideline_repayment_source": self.comptrollers_handbook_snc.get('primary_repayment_source', "Primary repayment should come from a sustainable source of cash under borrower control."),
                    "collateral_description": credit_risk_mitigation.get('collateral_summary_for_sk', "Not specified."),
                    "ltv_ratio": credit_risk_mitigation.get('loan_to_value_ratio', "Not specified."),
                    "other_collateral_notes": credit_risk_mitigation.get('collateral_notes_for_sk', "None.")
                }
                # XAI Log: Inputs to the SK skill
                logging.debug(f"SNC_DETERMINE_RATING_SK_INPUT: Calling CollateralRiskAssessment SK skill for {company_name} with inputs: {sk_input_vars}")
                
                sk_response_str = await self.run_semantic_kernel_skill(
                    "SNCRatingAssistSkill", 
                    "CollateralRiskAssessment", 
                    sk_input_vars
                )
                
                # Parse SK response (expected format: "Assessment: [Rating]\nJustification: [Text]")
                lines = sk_response_str.strip().splitlines()
                if lines:
                    assessment_line = lines[0]
                    if "Assessment:" in assessment_line: # Robust parsing
                        collateral_sk_assessment_str = assessment_line.split("Assessment:", 1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]:
                        collateral_sk_justification = lines[1].split("Justification:", 1)[1].strip()
                
                # XAI Log: Output from the SK skill
                logging.debug(f"SNC_DETERMINE_RATING_SK_OUTPUT: SK CollateralRiskAssessment for {company_name}: Assessment='{collateral_sk_assessment_str}', Justification='{collateral_sk_justification}'")
                if collateral_sk_justification: # Add SK justification to rationale
                    rationale_parts.append(f"SK Collateral Assessment ({collateral_sk_assessment_str}): {collateral_sk_justification}")

            except Exception as e:
                logging.error(f"Error calling CollateralRiskAssessment SK skill for {company_name}: {e}")
                # Fallback: collateral_sk_assessment_str remains None, Python logic will use collateral_quality_fallback
        
        # Python-based rating logic, potentially augmented/influenced by SK assessment
        debt_to_equity = financial_analysis.get("debt_to_equity")
        profitability = financial_analysis.get("profitability")
        rating = SNCRating.PASS # Default rating

        # XAI Log: Initial values before rule-based logic
        logging.debug(f"SNC_RATING_INITIAL_PARAMS: DtE={debt_to_equity}, Profitability={profitability}, SKCollateralAssessment='{collateral_sk_assessment_str}', FallbackCollateral='{credit_risk_mitigation.get('collateral_quality_fallback')}', ManagementQuality='{qualitative_analysis.get('management_quality')}'")

        if debt_to_equity is not None and profitability is not None:
            # Rule 1: High D/E and negative profitability
            if debt_to_equity > 3.0 and profitability < 0:
                logging.debug(f"SNC_RATING_RULE_TRIGGERED: LOSS - DtE ({debt_to_equity}) > 3.0 and Profitability ({profitability}) < 0")
                rating = SNCRating.LOSS
                rationale_parts.append("High D/E ratio and negative profitability.")
            # Rule 2: Elevated D/E and low profitability
            elif debt_to_equity > 2.0 and profitability < 0.1:
                logging.debug(f"SNC_RATING_RULE_TRIGGERED: DOUBTFUL - DtE ({debt_to_equity}) > 2.0 and Profitability ({profitability}) < 0.1")
                rating = SNCRating.DOUBTFUL
                rationale_parts.append("Elevated D/E ratio and low profitability.")
            # Rule 3: Insufficient liquidity and interest coverage
            elif financial_analysis.get("liquidity_ratio", 0) < 1.0 and financial_analysis.get("interest_coverage", 0) < 1.0:
                logging.debug(f"SNC_RATING_RULE_TRIGGERED: SUBSTANDARD - Liquidity ({financial_analysis.get('liquidity_ratio')}) < 1.0 and Interest Coverage ({financial_analysis.get('interest_coverage')}) < 1.0")
                rating = SNCRating.SUBSTANDARD
                rationale_parts.append("Insufficient liquidity and interest coverage.")
            # Rule 4: Collateral concerns and weak management
            elif (collateral_sk_assessment_str == "Substandard" or 
                  (collateral_sk_assessment_str is None and credit_risk_mitigation.get("collateral_quality_fallback") == "Low")) and \
                 qualitative_analysis.get("management_quality") == "Weak":
                logging.debug(f"SNC_RATING_RULE_TRIGGERED: SPECIAL_MENTION - SK Collateral: {collateral_sk_assessment_str}, Fallback Collateral: {credit_risk_mitigation.get('collateral_quality_fallback')}, Management: {qualitative_analysis.get('management_quality')}")
                rating = SNCRating.SPECIAL_MENTION
                rationale_parts.append(f"Collateral concerns (SK: {collateral_sk_assessment_str}, Fallback: {credit_risk_mitigation.get('collateral_quality_fallback')}) and weak management warrant Special Mention.")
            # Rule 5: Strong financials and stable economic conditions
            elif debt_to_equity <= 1.0 and profitability >= 0.3 and qualitative_analysis.get("economic_conditions") == "Stable":
                logging.debug(f"SNC_RATING_RULE_TRIGGERED: PASS - DtE ({debt_to_equity}) <= 1.0, Profitability ({profitability}) >= 0.3, Econ Conditions: {qualitative_analysis.get('economic_conditions')}")
                rating = SNCRating.PASS 
                rationale_parts.append("Strong financials and stable economic conditions.")
            # Rule 6: Fallback to Special Mention if no other critical rule met but not clearly Pass
            else: 
                if rating == SNCRating.PASS: # Only if not already downgraded by a more severe rule
                    logging.debug(f"SNC_RATING_RULE_TRIGGERED: SPECIAL_MENTION - Fallback/Mixed Indicators. Initial DtE: {debt_to_equity}, Profitability: {profitability}")
                    rating = SNCRating.SPECIAL_MENTION
                    rationale_parts.append("Mixed financial indicators or other unaddressed concerns warrant monitoring.")
        else: # If key financial metrics are missing
            logging.debug("SNC_RATING_RULE_TRIGGERED: UNDETERMINED - Missing key financial metrics (DtE or Profitability)")
            rating = None
            rationale_parts.append("Cannot determine rating due to missing key financial metrics (debt-to-equity or profitability).")

        # Append regulatory guidance reference
        rationale_parts.append(f"Regulatory guidance: Comptroller's Handbook SNC v{self.comptrollers_handbook_snc.get('version', 'N/A')}, OCC Guidelines v{self.occ_guidelines_snc.get('version', 'N/A')}.")
        final_rationale = " ".join(filter(None, rationale_parts)) # Filter out potential None/empty strings
        
        # XAI Log: Final rating and combined rationale
        logging.debug(f"SNC_DETERMINE_RATING_OUTPUT: Final Rating='{rating.value if rating else 'Undetermined'}', Rationale='{final_rationale}'")
        logging.info(f"SNC rating for {company_name}: {rating.value if rating else 'Undetermined'}. Rationale: {final_rationale}")
        return rating, final_rationale

if __name__ == '__main__':
    # To see XAI debug logs for the example, you might need to set the root logger level:
    # logging.getLogger().setLevel(logging.DEBUG) 
    
    dummy_snc_agent_config = {
        'persona': "Test SNC Analyst",
        'comptrollers_handbook_SNC': {
            "version": "2024.Q1_test",
            "substandard_definition": "Collateral is inadequately protective if its value doesn't cover the loan or if perfection issues exist.",
            "primary_repayment_source": "Repayment should primarily come from the borrower's sustainable cash flow."
        },
        'occ_guidelines_SNC': {"version": "2024-03_test"},
        'peers': ['DataRetrievalAgent'] 
    }

    mock_data_package_template = {
        "company_info": {"name": "TestCompany Corp", "industry_sector": "Tech", "country": "USA"},
        "financial_data_detailed": {
            "key_ratios": {"debt_to_equity_ratio": 1.5, "net_profit_margin": 0.15, "current_ratio": 1.8, "interest_coverage_ratio": 3.0}
        },
        "qualitative_company_info": {"management_assessment": "Average", "business_model_strength": "Moderate"},
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
        # logging.info(f"MOCKED send_message to {target_agent_name} with {message}") # Already logged by AgentBase
        if target_agent_name == 'DataRetrievalAgent' and message.get('data_type') == 'get_company_financials':
            company_id = message.get('company_id')
            if company_id == "TEST_COMPANY_SK_PASS":
                data = json.loads(json.dumps(mock_data_package_template)) 
                data["company_info"]["name"] = "SKPass Corp"
                data["collateral_and_debt_details"]["loan_to_value_ratio"] = 0.4 
                return data
            elif company_id == "TEST_COMPANY_SK_SUB":
                data = json.loads(json.dumps(mock_data_package_template)) 
                data["company_info"]["name"] = "SKSub Corp"
                data["collateral_and_debt_details"]["loan_to_value_ratio"] = 0.8 
                data["collateral_and_debt_details"]["collateral_type"] = "Outdated machinery"
                data["qualitative_company_info"]["management_assessment"] = "Weak" # To trigger specific rule
                return data
        return None

    class MockSKFunction:
        async def invoke(self, variables=None): 
            class MockSKResult:
                def __init__(self, value_str): self._value = value_str
                def __str__(self): return self._value
            
            ltv_str = variables.get("ltv_ratio", "Not specified.")
            assessment = "Pass"
            justification = "Collateral appears adequate based on LTV."
            try:
                ltv = float(ltv_str)
                if ltv > 0.7:
                    assessment = "Substandard"
                    justification = "Collateral LTV is high, indicating potential under-collateralization."
                elif ltv > 0.5:
                    assessment = "Special Mention"
                    justification = "Collateral LTV warrants monitoring."
            except ValueError: pass 
            return MockSKResult(f"Assessment: {assessment}\nJustification: {justification}")

    class MockSKSkillsCollection:
        def get_function(self, skill_collection_name, skill_name):
            if skill_collection_name == "SNCRatingAssistSkill" and skill_name == "CollateralRiskAssessment":
                return MockSKFunction()
            return None

    class MockKernel:
        def __init__(self): self.skills = MockSKSkillsCollection()
        async def run_async(self, sk_function, input_vars=None, **kwargs):
            if sk_function: return await sk_function.invoke(variables=input_vars)
            return "Mock kernel run_async failed: No function"

    async def run_tests():
        # logging.getLogger().setLevel(logging.DEBUG) # Example: Uncomment to see XAI debug logs
        
        snc_agent_no_kernel = SNCAnalystAgent(config=dummy_snc_agent_config, kernel=None)
        with patch.object(snc_agent_no_kernel, 'send_message', new=mock_send_message):
            print("\n--- Test Case: GoodCorp (No SK Kernel - Fallback) ---")
            rating_no_sk, rationale_no_sk = await snc_agent_no_kernel.execute(company_id="TEST_COMPANY_SK_PASS")
            print(f"Rating: {rating_no_sk.value if rating_no_sk else 'N/A'}, Rationale: {rationale_no_sk}")

        mock_kernel_instance = MockKernel()
        snc_agent_with_kernel = SNCAnalystAgent(config=dummy_snc_agent_config, kernel=mock_kernel_instance)
        with patch.object(snc_agent_with_kernel, 'send_message', new=mock_send_message):
            print("\n--- Test Case: GoodCorp (With SK Kernel - Expects SK Collateral Assessment) ---")
            rating_sk_pass, rationale_sk_pass = await snc_agent_with_kernel.execute(company_id="TEST_COMPANY_SK_PASS")
            print(f"Rating: {rating_sk_pass.value if rating_sk_pass else 'N/A'}, Rationale: {rationale_sk_pass}")
            assert "SK Collateral Assessment" in rationale_sk_pass

            print("\n--- Test Case: RiskyCorp (With SK Kernel - Expects SK Collateral Assessment) ---")
            rating_sk_sub, rationale_sk_sub = await snc_agent_with_kernel.execute(company_id="TEST_COMPANY_SK_SUB")
            print(f"Rating: {rating_sk_sub.value if rating_sk_sub else 'N/A'}, Rationale: {rationale_sk_sub}")
            assert "SK Collateral Assessment (Substandard)" in rationale_sk_sub

    if __name__ == '__main__':
        asyncio.run(run_tests())
