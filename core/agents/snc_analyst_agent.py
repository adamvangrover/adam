# core/agents/snc_analyst_agent.py
import os
import sys

# Add the project root to sys.path to allow imports like 'from core...'
# when running this script directly for its __main__ block.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')) # core/agents/snc_analyst_agent.py -> core/agents -> core -> /app
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
import json
import logging
import os  # For os.path.exists and os.remove
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch

from semantic_kernel import Kernel

from core.agents.agent_base import AgentBase

# Note: Initial basicConfig is removed as it will be handled in __main__

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
            error_msg = "DataRetrievalAgent not found in peer agents for snc_analyst_agent."
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
        
        logging.debug(f"SNC_ANALYSIS_DATA_EXTRACTED: CompanyInfo: {list(company_info.keys())}, FinancialDetailed: {list(financial_data_detailed.keys())}, Qualitative: {list(qualitative_company_info.keys())}, Industry: {list(industry_data_context.keys())}, Economic: {list(economic_data_context.keys())}, Collateral: {list(collateral_and_debt_details.keys())}")

        financial_analysis_inputs_for_sk = self._prepare_financial_inputs_for_sk(financial_data_detailed)
        qualitative_analysis_inputs_for_sk = self._prepare_qualitative_inputs_for_sk(qualitative_company_info)
        
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
        market_data = financial_data_detailed.get("market_data", {})
        dcf_assumptions = financial_data_detailed.get("dcf_assumptions", {}) 

        return {
            "historical_fcf_str": str(cash_flow_statement.get('free_cash_flow', ["N/A"])),
            "historical_cfo_str": str(cash_flow_statement.get('cash_flow_from_operations', ["N/A"])),
            "annual_debt_service_str": str(market_data.get("annual_debt_service_placeholder", "Not Available")), 
            "ratios_summary_str": json.dumps(key_ratios) if key_ratios else "Not available",
            "projected_fcf_str": str(dcf_assumptions.get("projected_fcf_placeholder", "Not Available")), 
            "payment_history_status_str": str(market_data.get("payment_history_placeholder", "Current")), 
            "interest_capitalization_status_str": str(market_data.get("interest_capitalization_placeholder", "No")) 
        }

    def _prepare_qualitative_inputs_for_sk(self, qualitative_company_info: Dict[str, Any]) -> Dict[str, str]:
        """Prepares stringified qualitative inputs required by SK skills."""
        return {
            "qualitative_notes_stability_str": qualitative_company_info.get("revenue_cashflow_stability_notes_placeholder", "Management reports stable customer contracts."),
            "notes_financial_deterioration_str": qualitative_company_info.get("financial_deterioration_notes_placeholder", "No significant deterioration noted recently.")
        }

    def _perform_financial_analysis(self, financial_data_detailed: Dict[str, Any], sk_financial_inputs: Dict[str, str]) -> Dict[str, Any]:
        logging.debug(f"SNC_FIN_ANALYSIS_INPUT: financial_data_detailed keys: {list(financial_data_detailed.keys())}, sk_inputs keys: {list(sk_financial_inputs.keys())}")
        key_ratios = financial_data_detailed.get("key_ratios", {})
        
        analysis_result = {
            "debt_to_equity": key_ratios.get("debt_to_equity_ratio"),
            "profitability": key_ratios.get("net_profit_margin"),
            "liquidity_ratio": key_ratios.get("current_ratio"),
            "interest_coverage": key_ratios.get("interest_coverage_ratio"),
            **sk_financial_inputs 
        }
        logging.debug(f"SNC_FIN_ANALYSIS_OUTPUT: {analysis_result}")
        return analysis_result

    def _perform_qualitative_analysis(self, 
                                      company_name: str, 
                                      qualitative_company_info: Dict[str, Any], 
                                      industry_data_context: Dict[str, Any], 
                                      economic_data_context: Dict[str, Any],
                                      sk_qualitative_inputs: Dict[str, str]) -> Dict[str, Any]:
        logging.debug(f"SNC_QUAL_ANALYSIS_INPUT: company_name='{company_name}', qualitative_info_keys={list(qualitative_company_info.keys())}, industry_keys={list(industry_data_context.keys())}, economic_keys={list(economic_data_context.keys())}, sk_qual_inputs keys: {list(sk_qualitative_inputs.keys())}")
        qualitative_result = {
            "management_quality": qualitative_company_info.get("management_assessment", "Not Assessed"),
            "industry_outlook": industry_data_context.get("outlook", "Neutral"),
            "economic_conditions": economic_data_context.get("overall_outlook", "Stable"),
            "business_model_strength": qualitative_company_info.get("business_model_strength", "N/A"),
            "competitive_advantages": qualitative_company_info.get("competitive_advantages", "N/A"),
            **sk_qualitative_inputs
        }
        logging.debug(f"SNC_QUAL_ANALYSIS_OUTPUT: {qualitative_result}")
        return qualitative_result

    def _evaluate_credit_risk_mitigation(self, collateral_and_debt_details: Dict[str, Any]) -> Dict[str, Any]:
        logging.debug(f"SNC_CREDIT_MITIGATION_INPUT: collateral_and_debt_details_keys={list(collateral_and_debt_details.keys())}")
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

    def _rate_from_sk_assessments(self, repayment_assessment: Optional[str], collateral_assessment: Optional[str], nonaccrual_assessment: Optional[str]) -> Tuple[Optional[SNCRating], str]:
        """Determines a rating based purely on the string outputs from SK skills."""
        if repayment_assessment == "Unsustainable":
            return SNCRating.LOSS, "Repayment capacity is assessed as unsustainable."
        if nonaccrual_assessment == "Non-Accrual Warranted" and repayment_assessment == "Weak":
            return SNCRating.LOSS, "Non-accrual status is warranted alongside weak repayment capacity."
        
        if repayment_assessment == "Weak":
            return SNCRating.DOUBTFUL, "Repayment capacity is weak, casting doubt on the ability to service debt."
        if collateral_assessment == "Substandard" and repayment_assessment == "Adequate":
             return SNCRating.DOUBTFUL, "Collateral is substandard, and repayment capacity is only adequate, not strong."

        if nonaccrual_assessment == "Non-Accrual Warranted":
            return SNCRating.SUBSTANDARD, "Non-accrual status is warranted, indicating significant financial distress."
        if collateral_assessment == "Substandard":
            return SNCRating.SUBSTANDARD, "Collateral is substandard, providing inadequate support for the credit."
        if repayment_assessment == "Adequate" and collateral_assessment == "Special Mention":
            return SNCRating.SUBSTANDARD, "Repayment capacity is merely adequate and not supported by Pass-rated collateral."

        if collateral_assessment == "Special Mention" or repayment_assessment == "Adequate":
            return SNCRating.SPECIAL_MENTION, "Weaknesses are noted in collateral or repayment that require monitoring."

        if repayment_assessment == "Strong" and collateral_assessment == "Pass" and nonaccrual_assessment == "Accrual Appropriate":
            return SNCRating.PASS, "Strong repayment, Pass-rated collateral, and appropriate accrual status indicate a Pass rating."

        return None, "SK assessments did not map to a definitive rating."

    def _rate_from_fallback_logic(self, financial_analysis: Dict[str, Any], qualitative_analysis: Dict[str, Any], credit_risk_mitigation: Dict[str, Any]) -> Tuple[Optional[SNCRating], str]:
        """Provides a rating based on hardcoded financial metrics if SK fails."""
        debt_to_equity = financial_analysis.get("debt_to_equity")
        profitability = financial_analysis.get("profitability")

        if debt_to_equity is None or profitability is None:
            return None, "Fallback rating failed: Missing key financial metrics (debt-to-equity or profitability)."

        if debt_to_equity > 3.0 and profitability < 0:
            return SNCRating.LOSS, "Fallback: High D/E ratio (> 3.0) and negative profitability."
        if debt_to_equity > 2.0 and profitability < 0.1:
            return SNCRating.DOUBTFUL, "Fallback: Elevated D/E ratio (> 2.0) and low profitability (< 10%)."
        if financial_analysis.get("liquidity_ratio", 1.0) < 1.0 and financial_analysis.get("interest_coverage", 1.0) < 1.0:
            return SNCRating.SUBSTANDARD, "Fallback: Insufficient liquidity (< 1.0) and interest coverage (< 1.0)."
        if credit_risk_mitigation.get("collateral_quality_fallback") == "Low" and qualitative_analysis.get("management_quality") == "Weak":
            return SNCRating.SPECIAL_MENTION, "Fallback: Combination of low-quality collateral and weak management."
        
        return SNCRating.PASS, "Fallback: Financial metrics do not meet any adverse classification triggers."

    def _synthesize_rationale(self, company_name: str, rating: SNCRating, sk_rationale_summary: str, 
                              financial_analysis: Dict[str, Any], qualitative_analysis: Dict[str, Any], credit_risk_mitigation: Dict[str, Any],
                              repayment_assessment: Optional[str], collateral_assessment: Optional[str], nonaccrual_assessment: Optional[str]) -> str:
        """Constructs a coherent, narrative rationale for the final rating."""
        
        header = f"Executive Summary for {company_name}: The credit has been assigned a rating of {rating.value}."
        
        sk_section_header = "Primary Justification (AI Skill-Based Analysis):"
        sk_section_body = sk_rationale_summary
        
        sk_details = []
        if repayment_assessment: sk_details.append(f"- Repayment Capacity Assessment: {repayment_assessment}")
        if collateral_assessment: sk_details.append(f"- Collateral Risk Assessment: {collateral_assessment}")
        if nonaccrual_assessment: sk_details.append(f"- Non-Accrual Status Assessment: {nonaccrual_assessment}")
        sk_section_details = "\n".join(sk_details)

        supporting_factors_header = "Supporting Quantitative and Qualitative Factors:"
        
        factors = []
        d_to_e = financial_analysis.get('debt_to_equity')
        profit = financial_analysis.get('profitability')
        mgmt_quality = qualitative_analysis.get('management_quality')
        econ_outlook = qualitative_analysis.get('economic_conditions')

        if d_to_e is not None: factors.append(f"- Debt/Equity Ratio: {d_to_e:.2f}")
        if profit is not None: factors.append(f"- Profit Margin: {profit:.2%}")
        if mgmt_quality: factors.append(f"- Management Quality: {mgmt_quality}")
        if econ_outlook: factors.append(f"- Economic Outlook: {econ_outlook}")
        
        supporting_factors_body = "\n".join(factors)

        footer = f"Regulatory guidance considered: Comptroller's Handbook SNC v{self.comptrollers_handbook_snc.get('version', 'N/A')}, OCC Guidelines v{self.occ_guidelines_snc.get('version', 'N/A')}."

        full_rationale = "\n\n".join([header, sk_section_header, sk_section_body, sk_section_details, supporting_factors_header, supporting_factors_body, footer])
        return full_rationale

    async def _determine_rating(self, company_name: str, 
                               financial_analysis: Dict[str, Any], 
                               qualitative_analysis: Dict[str, Any], 
                               credit_risk_mitigation: Dict[str, Any],
                               economic_data_context: Dict[str, Any]
                               ) -> Tuple[Optional[SNCRating], str]:
        logging.debug(f"SNC_DETERMINE_RATING_INPUT: company='{company_name}'")
        
        collateral_sk_assessment_str, repayment_sk_assessment_str, nonaccrual_sk_assessment_str = None, None, None
        collateral_sk_justification, repayment_sk_justification, nonaccrual_sk_justification = "", "", ""
        repayment_sk_concerns = ""

        if self.kernel and hasattr(self.kernel, 'skills'):
            # 1. AssessCollateralRisk
            try:
                sk_input_vars_collateral = {
                    "guideline_substandard_collateral": self.comptrollers_handbook_snc.get('substandard_definition', "Collateral is inadequately protective."),
                    "guideline_repayment_source": self.comptrollers_handbook_snc.get('primary_repayment_source', "Primary repayment should come from a sustainable source of cash under borrower control."),
                    "collateral_description": credit_risk_mitigation.get('collateral_summary_for_sk', "Not specified."),
                    "ltv_ratio": credit_risk_mitigation.get('loan_to_value_ratio', "Not specified."),
                    "other_collateral_notes": credit_risk_mitigation.get('collateral_notes_for_sk', "None.")
                }
                logging.debug(f"SNC_XAI:SK_INPUT:AssessCollateralRisk: {sk_input_vars_collateral}")
                sk_response_collateral = await self.run_semantic_kernel_skill("SNCRatingAssistSkill", "CollateralRiskAssessment", sk_input_vars_collateral)
                lines = sk_response_collateral.strip().splitlines()
                if lines:
                    if "Assessment:" in lines[0]: collateral_sk_assessment_str = lines[0].split("Assessment:", 1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]: collateral_sk_justification = lines[1].split("Justification:", 1)[1].strip()
                logging.debug(f"SNC_XAI:SK_OUTPUT:AssessCollateralRisk: Assessment='{collateral_sk_assessment_str}', Justification='{collateral_sk_justification}'")
            except Exception as e: logging.error(f"Error in CollateralRiskAssessment SK skill for {company_name}: {e}")

            # 2. AssessRepaymentCapacity
            try:
                sk_input_vars_repayment = {
                    "guideline_repayment_source": self.comptrollers_handbook_snc.get('primary_repayment_source', "Default guideline..."),
                    "guideline_substandard_paying_capacity": self.comptrollers_handbook_snc.get('substandard_definition', "Default substandard..."),
                    "repayment_capacity_period_years": str(self.comptrollers_handbook_snc.get('repayment_capacity_period', 7)),
                    "historical_fcf": financial_analysis.get('historical_fcf_str', "Not available"),
                    "historical_cfo": financial_analysis.get('historical_cfo_str', "Not available"),
                    "annual_debt_service": financial_analysis.get('annual_debt_service_str', "Not available"),
                    "relevant_ratios": financial_analysis.get('ratios_summary_str', "Not available"),
                    "projected_fcf": financial_analysis.get('projected_fcf_str', "Not available"),
                    "qualitative_notes_stability": qualitative_analysis.get('qualitative_notes_stability_str', "None provided.")
                }
                logging.debug(f"SNC_XAI:SK_INPUT:AssessRepaymentCapacity: {sk_input_vars_repayment}")
                sk_response_repayment = await self.run_semantic_kernel_skill("SNCRatingAssistSkill", "AssessRepaymentCapacity", sk_input_vars_repayment)
                lines = sk_response_repayment.strip().splitlines()
                if lines:
                    if "Assessment:" in lines[0]: repayment_sk_assessment_str = lines[0].split("Assessment:",1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]: repayment_sk_justification = lines[1].split("Justification:",1)[1].strip()
                    if len(lines) > 2 and "Concerns:" in lines[2]: repayment_sk_concerns = lines[2].split("Concerns:",1)[1].strip()
                logging.debug(f"SNC_XAI:SK_OUTPUT:AssessRepaymentCapacity: Assessment='{repayment_sk_assessment_str}', Justification='{repayment_sk_justification}', Concerns='{repayment_sk_concerns}'")
            except Exception as e: logging.error(f"Error in AssessRepaymentCapacity SK skill for {company_name}: {e}")

            # 3. AssessNonAccrualStatusIndication
            try:
                sk_input_vars_nonaccrual = {
                    "guideline_nonaccrual_status": self.occ_guidelines_snc.get('nonaccrual_status', "Default non-accrual..."),
                    "guideline_interest_capitalization": self.occ_guidelines_snc.get('capitalization_of_interest', "Default interest cap..."),
                    "payment_history_status": financial_analysis.get('payment_history_status_str', "Current"),
                    "relevant_ratios": financial_analysis.get('ratios_summary_str', "Not available"),
                    "repayment_capacity_assessment": repayment_sk_assessment_str if repayment_sk_assessment_str else "Adequate", 
                    "notes_financial_deterioration": qualitative_analysis.get('notes_financial_deterioration_str', "None noted."),
                    "interest_capitalization_status": financial_analysis.get('interest_capitalization_status_str', "No")
                }
                logging.debug(f"SNC_XAI:SK_INPUT:AssessNonAccrualStatusIndication: {sk_input_vars_nonaccrual}")
                sk_response_nonaccrual = await self.run_semantic_kernel_skill("SNCRatingAssistSkill", "AssessNonAccrualStatusIndication", sk_input_vars_nonaccrual)
                lines = sk_response_nonaccrual.strip().splitlines()
                if lines:
                    if "Assessment:" in lines[0]: nonaccrual_sk_assessment_str = lines[0].split("Assessment:",1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]: nonaccrual_sk_justification = lines[1].split("Justification:",1)[1].strip()
                logging.debug(f"SNC_XAI:SK_OUTPUT:AssessNonAccrualStatusIndication: Assessment='{nonaccrual_sk_assessment_str}', Justification='{nonaccrual_sk_justification}'")
            except Exception as e: logging.error(f"Error in AssessNonAccrualStatusIndication SK skill for {company_name}: {e}")

        # Primary rating path: Use SK skill outputs
        rating, sk_rationale = self._rate_from_sk_assessments(
            repayment_sk_assessment_str,
            collateral_sk_assessment_str,
            nonaccrual_sk_assessment_str
        )

        final_rationale = ""
        # Fallback path: If SK skills did not yield a rating, use hardcoded financial logic
        if rating is None:
            logging.warning(f"SK-based rating was inconclusive for {company_name}. Using fallback logic.")
            rating, final_rationale = self._rate_from_fallback_logic(
                financial_analysis,
                qualitative_analysis,
                credit_risk_mitigation
            )
        else:
            # Synthesize a comprehensive rationale if the primary path was successful
            final_rationale = self._synthesize_rationale(
                company_name=company_name,
                rating=rating,
                sk_rationale_summary=sk_rationale,
                financial_analysis=financial_analysis,
                qualitative_analysis=qualitative_analysis,
                credit_risk_mitigation=credit_risk_mitigation,
                repayment_assessment=repayment_sk_assessment_str,
                collateral_assessment=collateral_sk_assessment_str,
                nonaccrual_assessment=nonaccrual_sk_assessment_str
            )

        logging.debug(f"SNC_DETERMINE_RATING_OUTPUT: Final Rating='{rating.value if rating else 'Undetermined'}', Rationale='{final_rationale}'")
        logging.info(f"SNC rating for {company_name}: {rating.value if rating else 'Undetermined'}.")
        return rating, final_rationale

if __name__ == '__main__':
    # Configure file logging for XAI trace at the START of the __main__ block
    log_file_name = 'snc_xai_test_run.log'
    if os.path.exists(log_file_name): 
        os.remove(log_file_name)
    
    # Setup basic logging
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
    
    logging.basicConfig(level=logging.DEBUG, 
                        filename=log_file_name, 
                        filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)

    logging.debug("XAI DEBUG Logging to file is configured for this script run.")

    # Definitions needed by run_tests, kept in the __main__ scope
    dummy_snc_agent_config = {
        'persona': "Test SNC Analyst",
        'comptrollers_handbook_SNC': {
            "version": "2024.Q1_test",
            "substandard_definition": "Collateral is inadequately protective...",
            "primary_repayment_source": "Repayment should come from sustainable cash flow...",
            "repayment_capacity_period": 7
        },
        'occ_guidelines_SNC': {
            "version": "2024-03_test",
            "nonaccrual_status": "Loan is on cash basis due to financial deterioration...",
            "capitalization_of_interest": "Permissible only if borrower is creditworthy..."
        },
        'peers': ['DataRetrievalAgent'] 
    }

    mock_data_package_template = {
        "company_info": {"name": "TestCompany Corp"},
        "financial_data_detailed": {
            "key_ratios": {"debt_to_equity_ratio": 1.5, "net_profit_margin": 0.15, "current_ratio": 1.8, "interest_coverage_ratio": 3.0},
            "cash_flow_statement": {"free_cash_flow": [100,110,120], "cash_flow_from_operations": [150,160,170]},
            "market_data": {"annual_debt_service_placeholder": "60", "payment_history_placeholder": "Current", "interest_capitalization_placeholder": "No"},
            "dcf_assumptions": {}
        },
        "qualitative_company_info": {"management_assessment": "Average"},
        "industry_data_context": {"outlook": "Stable"},
        "economic_data_context": {"overall_outlook": "Stable"},
        "collateral_and_debt_details": {"loan_to_value_ratio": 0.6}
    }

    async def mock_send_message(target_agent_name, message):
        company_id = message.get('company_id')
        data = json.loads(json.dumps(mock_data_package_template)) 
        data["company_info"]["name"] = f"{company_id} Corp"
        if company_id == "TEST_COMPANY_REPAY_WEAK":
            data["financial_data_detailed"]["key_ratios"]["interest_coverage_ratio"] = 0.8
            data["financial_data_detailed"]["cash_flow_statement"]["free_cash_flow"] = [10, 5, -20]
            data["financial_data_detailed"]["market_data"]["payment_history_placeholder"] = "90 days past due"
        elif company_id == "TEST_COMPANY_FALLBACK":
            # Data that will cause SK to return None, to test fallback
            pass # Use default template
        return data

    class MockSKFunction:
        def __init__(self, skill_name, test_case_id):
            self.skill_name = skill_name
            self.test_case_id = test_case_id

        async def invoke(self, variables=None): 
            class MockSKResult:
                def __init__(self, value_str): self._value = value_str
                def __str__(self): return self._value

            if self.test_case_id == "TEST_COMPANY_FALLBACK":
                return MockSKResult("Assessment: \nJustification: ") # Empty response

            if self.skill_name == "CollateralRiskAssessment":
                assessment, justification = "Pass", "Collateral LTV is acceptable."
                if self.test_case_id == "TEST_COMPANY_REPAY_WEAK":
                    assessment, justification = "Special Mention", "LTV needs monitoring."
                return MockSKResult(f"Assessment: {assessment}\nJustification: {justification}")

            elif self.skill_name == "AssessRepaymentCapacity":
                if self.test_case_id == "TEST_COMPANY_REPAY_WEAK":
                    return MockSKResult(
                        "Assessment: Weak\n"
                        "DSCR_Calculation: 5,000 (Latest FCF) / 60,000 (Debt Service) = 0.08x\n"
                        "Justification: Repayment capacity is weak due to negative FCF trend and very low DSCR.\n"
                        "Concerns: Debt service coverage low, negative FCF trend."
                    )
                return MockSKResult(
                    "Assessment: Strong\n"
                    "DSCR_Calculation: 110,000 (Avg FCF) / 60,000 (Debt Service) = 1.83x\n"
                    "Justification: Repayment capacity is strong with positive FCF and adequate DSCR.\n"
                    "Concerns: None."
                )

            elif self.skill_name == "AssessNonAccrualStatusIndication":
                if self.test_case_id == "TEST_COMPANY_REPAY_WEAK":
                    return MockSKResult(
                        "Assessment: Non-Accrual Warranted\n"
                        "Triggers: 90 days past due, Repayment capacity assessed as Weak.\n"
                        "Mitigants: None Identified.\n"
                        "Justification: Delinquency and weak repayment capacity warrant non-accrual status."
                    )
                return MockSKResult(
                    "Assessment: Accrual Appropriate\n"
                    "Triggers: None Identified.\n"
                    "Mitigants: N/A.\n"
                    "Justification: The loan is performing and repayment capacity is strong."
                )
            return MockSKResult("Unknown mock skill called.")

    class MockSKSkillsCollection:
        def __init__(self, test_case_id):
            self.test_case_id = test_case_id
        def get_function(self, skill_collection_name, skill_name):
            if skill_collection_name == "SNCRatingAssistSkill":
                return MockSKFunction(skill_name, self.test_case_id) 
            return None
        
    class MockKernel(Kernel):
        def __init__(self, test_case_id="default"):
            self.test_case_id = test_case_id
            self._skills = MockSKSkillsCollection(test_case_id)
        
        @property
        def skills(self):
            return self._skills

        async def run_semantic_kernel_skill(self, skill_collection_name, skill_name, sk_input_vars):
            # This method now needs to be part of the mock to properly simulate behavior
            # In the actual agent, `run_semantic_kernel_skill` is a helper that calls kernel.run_async
            # Here, we simulate that by directly invoking our mock function
            mock_function = self.skills.get_function(skill_collection_name, skill_name)
            if mock_function:
                result = await mock_function.invoke(variables=sk_input_vars)
                return str(result)
            return ""

    async def run_tests():
        # --- Test Case 1: Good Financials ---
        print("\n--- Test Case 1: Good Financials (Primary SK Path) ---")
        mock_kernel_good = MockKernel(test_case_id="TEST_COMPANY_GOOD")
        agent_good = SNCAnalystAgent(config=dummy_snc_agent_config, kernel=mock_kernel_good)
        agent_good.peer_agents['DataRetrievalAgent'] = object()
        # Patch the agent's own A2A and SK methods for this test
        with patch.object(agent_good, 'send_message', new=mock_send_message), \
             patch.object(agent_good, 'run_semantic_kernel_skill', new=mock_kernel_good.run_semantic_kernel_skill):
            rating, rationale = await agent_good.execute(company_id="TEST_COMPANY_GOOD")
            print(f"Rating: {rating.value if rating else 'N/A'}")
            print(f"Rationale:\n{rationale}")
            assert rating == SNCRating.PASS
            assert "Executive Summary" in rationale and "DSCR_Calculation" in rationale

        # --- Test Case 2: Weak Repayment ---
        print("\n--- Test Case 2: Weak Repayment (Primary SK Path, Adverse Rating) ---")
        mock_kernel_weak = MockKernel(test_case_id="TEST_COMPANY_REPAY_WEAK")
        agent_weak = SNCAnalystAgent(config=dummy_snc_agent_config, kernel=mock_kernel_weak)
        agent_weak.peer_agents['DataRetrievalAgent'] = object()
        with patch.object(agent_weak, 'send_message', new=mock_send_message), \
             patch.object(agent_weak, 'run_semantic_kernel_skill', new=mock_kernel_weak.run_semantic_kernel_skill):
            rating, rationale = await agent_weak.execute(company_id="TEST_COMPANY_REPAY_WEAK")
            print(f"Rating: {rating.value if rating else 'N/A'}")
            print(f"Rationale:\n{rationale}")
            assert rating == SNCRating.LOSS
            assert "Triggers: 90 days past due" in rationale

        # --- Test Case 3: SK Failure Fallback ---
        print("\n--- Test Case 3: SK Failure (Fallback Logic Path) ---")
        mock_kernel_fallback = MockKernel(test_case_id="TEST_COMPANY_FALLBACK")
        agent_fallback = SNCAnalystAgent(config=dummy_snc_agent_config, kernel=mock_kernel_fallback)
        agent_fallback.peer_agents['DataRetrievalAgent'] = object()
        with patch.object(agent_fallback, 'send_message', new=mock_send_message), \
             patch.object(agent_fallback, 'run_semantic_kernel_skill', new=mock_kernel_fallback.run_semantic_kernel_skill):
            # Modify data to trigger a specific fallback rule (e.g., Substandard)
            global mock_data_package_template
            original_data = json.loads(json.dumps(mock_data_package_template))
            mock_data_package_template["financial_data_detailed"]["key_ratios"]["liquidity_ratio"] = 0.8
            mock_data_package_template["financial_data_detailed"]["key_ratios"]["interest_coverage_ratio"] = 0.5
            
            rating, rationale = await agent_fallback.execute(company_id="TEST_COMPANY_FALLBACK")
            print(f"Rating: {rating.value if rating else 'N/A'}")
            print(f"Rationale:\n{rationale}")
            assert rating == SNCRating.SUBSTANDARD
            assert "Fallback: Insufficient liquidity" in rationale
            
            # Restore original data for any subsequent tests
            mock_data_package_template = original_data

        # --- Test Case 4: Missing Company ID ---
        print("\n--- Test Case 4: Missing Company ID ---")
        agent_missing_id = SNCAnalystAgent(config=dummy_snc_agent_config, kernel=MockKernel())
        rating, rationale = await agent_missing_id.execute() # No company_id
        print(f"Rating: {'N/A'}, Rationale: {rationale}")
        assert rating is None

    # The asyncio.run call should be the final line in the if __name__ == '__main__': block
    asyncio.run(run_tests())
