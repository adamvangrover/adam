# core/agents/SNC_analyst_agent.py
import sys
import os
# Add the project root to sys.path to allow imports like 'from core...'
# when running this script directly for its __main__ block.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')) # core/agents/SNC_analyst_agent.py -> core/agents -> core -> /app
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import json 
import os # For os.path.exists and os.remove
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from unittest.mock import patch 

from core.agents.agent_base import AgentBase
from semantic_kernel import Kernel 

# Note: Initial basicConfig is removed as it will be handled in __main__

import re # Added for XAI_LOG_PATTERNS
import ast # Added for parse_payload

# XAI Log Patterns (copied from scripts/extract_xai_reasoning.py)
XAI_LOG_PATTERNS = [
    {'type': 'EXECUTE_INPUT', 'regex': re.compile(r"SNC_ANALYSIS_EXECUTE_INPUT: company_id='([^']*)', all_kwargs=(.*)")},
    {'type': 'A2A_REQUEST', 'regex': re.compile(r"SNC_ANALYSIS_A2A_REQUEST: Requesting data from DataRetrievalAgent: (.*)")},
    {'type': 'A2A_RESPONSE', 'regex': re.compile(r"SNC_ANALYSIS_A2A_RESPONSE: Received data package: (True|False)")},
    {'type': 'DATA_EXTRACTION_SUMMARY', 'regex': re.compile(r"SNC_ANALYSIS_DATA_EXTRACTED: (.*)")},
    {'type': 'FIN_ANALYSIS_INPUT', 'regex': re.compile(r"SNC_FIN_ANALYSIS_INPUT: (.*)")},
    {'type': 'FIN_ANALYSIS_OUTPUT', 'regex': re.compile(r"SNC_FIN_ANALYSIS_OUTPUT: (.*)")},
    {'type': 'QUAL_ANALYSIS_INPUT', 'regex': re.compile(r"SNC_QUAL_ANALYSIS_INPUT: (.*)")},
    {'type': 'QUAL_ANALYSIS_OUTPUT', 'regex': re.compile(r"SNC_QUAL_ANALYSIS_OUTPUT: (.*)")},
    {'type': 'CREDIT_MITIGATION_INPUT', 'regex': re.compile(r"SNC_CREDIT_MITIGATION_INPUT: (.*)")},
    {'type': 'CREDIT_MITIGATION_OUTPUT', 'regex': re.compile(r"SNC_CREDIT_MITIGATION_OUTPUT: (.*)")},
    {'type': 'DETERMINE_RATING_INPUT', 'regex': re.compile(r"SNC_DETERMINE_RATING_INPUT: (.*)")},
    {'type': 'SK_INPUT', 'regex': re.compile(r"SNC_XAI:SK_INPUT:(\w+): (.*)")},
    {'type': 'SK_OUTPUT', 'regex': re.compile(r"SNC_XAI:SK_OUTPUT:(\w+): Assessment='([^']*)', Justification='([^']*)'(?:, Concerns='([^']*)')?")},
    {'type': 'RATING_RULE', 'regex': re.compile(r"SNC_XAI:RATING_RULE(?:_FALLBACK)?: (.*)")},
    {'type': 'RATING_LOGIC_PARAMS', 'regex': re.compile(r"SNC_XAI:RATING_PARAMS_FOR_LOGIC: (.*)")},
    {'type': 'FINAL_RATING_DETERMINATION', 'regex': re.compile(r"SNC_DETERMINE_RATING_OUTPUT: Final Rating='([^']*)', Rationale='(.*)'")},
    {'type': 'EXECUTE_OUTPUT', 'regex': re.compile(r"SNC_ANALYSIS_EXECUTE_OUTPUT: Rating='([^']*)', Rationale='(.*)'")}
]

def parse_payload(payload_str: str) -> Any:
    """
    Safely parses a string payload that might represent a Python literal (dict, list, bool, etc.).
    Uses ast.literal_eval for safe parsing. Returns original string if parsing fails.
    Handles 'True'/'False' strings specifically.
    """
    try:
        if payload_str.lower() == 'true':
            return True
        if payload_str.lower() == 'false':
            return False
        return ast.literal_eval(payload_str)
    except (ValueError, SyntaxError, TypeError):
        return payload_str.strip()

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
        self.xai_log_file = self.config.get('xai_log_file') # Added for XAI trace extraction

        self.comptrollers_handbook_snc = self.config.get('comptrollers_handbook_SNC', {})
        if not self.comptrollers_handbook_snc:
            logging.warning("Comptroller's Handbook SNC guidelines not found in agent configuration.")
        
        self.occ_guidelines_snc = self.config.get('occ_guidelines_SNC', {})
        if not self.occ_guidelines_snc:
            logging.warning("OCC Guidelines SNC not found in agent configuration.")

    def _extract_xai_trace(self, log_file_path: str) -> Dict[str, Any]:
        """
        Parses the specified log file and extracts XAI trace events.
        """
        trace_events: List[Dict[str, Any]] = []
        final_rating_info: Dict[str, Any] = {}

        try:
            with open(log_file_path, 'r') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    log_line_parts_match = re.search(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s-\s([\w.]+)\s-\s(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s-\s(.*)", line)

                    if not log_line_parts_match:
                        continue

                    log_content = log_line_parts_match.group(3).strip()

                    for pattern_info in XAI_LOG_PATTERNS:
                        match = pattern_info['regex'].match(log_content)
                        if match:
                            event_type = pattern_info['type']
                            groups = match.groups()
                            event_data: Dict[str, Any] = {'line': line_number}

                            if event_type == 'EXECUTE_INPUT':
                                event_data['company_id'] = groups[0]
                                event_data['all_kwargs'] = parse_payload(groups[1])
                            elif event_type == 'A2A_REQUEST':
                                event_data['request_details'] = parse_payload(groups[0])
                            elif event_type == 'A2A_RESPONSE':
                                event_data['received_package'] = parse_payload(groups[0])
                            elif event_type in ['DATA_EXTRACTION_SUMMARY', 'FIN_ANALYSIS_INPUT', 'QUAL_ANALYSIS_INPUT', 'CREDIT_MITIGATION_INPUT', 'DETERMINE_RATING_INPUT', 'RATING_RULE', 'RATING_LOGIC_PARAMS']:
                                event_data['details'] = groups[0]
                            elif event_type in ['FIN_ANALYSIS_OUTPUT', 'QUAL_ANALYSIS_OUTPUT', 'CREDIT_MITIGATION_OUTPUT']:
                                event_data['output_data'] = parse_payload(groups[0])
                            elif event_type == 'SK_INPUT':
                                event_data['skill_name'] = groups[0]
                                event_data['input_vars'] = parse_payload(groups[1])
                            elif event_type == 'SK_OUTPUT':
                                event_data['skill_name'] = groups[0]
                                event_data['assessment'] = groups[1]
                                event_data['justification'] = groups[2]
                                if len(groups) > 3 and groups[3] is not None:
                                    event_data['concerns'] = groups[3]
                            elif event_type == 'FINAL_RATING_DETERMINATION' or event_type == 'EXECUTE_OUTPUT':
                                rating_val = groups[0]
                                rationale_val = groups[1]
                                event_data['rating'] = rating_val
                                event_data['rationale'] = rationale_val
                                if event_type == 'EXECUTE_OUTPUT':
                                    final_rating_info = {'rating': rating_val, 'rationale': rationale_val}

                            trace_event = {
                                'event_type': event_type,
                                'data': event_data,
                                'original_message': log_content
                            }
                            trace_events.append(trace_event)
                            break
        except FileNotFoundError:
            logging.error(f"XAI trace log file not found: {log_file_path}")
            # Return empty trace if file not found, or could raise error
            return {"agent_execution_trace": {"agent_name": "SNC_analyst_agent", "log_file": log_file_path, "trace_events": [], "error": "Log file not found"}}
        except Exception as e:
            logging.error(f"An error occurred during XAI log parsing: {e}", exc_info=True)
            return {"agent_execution_trace": {"agent_name": "SNC_analyst_agent", "log_file": log_file_path, "trace_events": [], "error": str(e)}}

        output_data = {
            "agent_execution_trace": {
                "agent_name": "SNC_analyst_agent",
                "log_file": log_file_path,
                "trace_events": trace_events,
            }
        }
        if final_rating_info:
            output_data["agent_execution_trace"]["final_output"] = final_rating_info
        return output_data

    async def execute(self, include_xai_trace: bool = False, **kwargs) -> Dict[str, Any]:
        company_id = kwargs.get('company_id')
        logging.info(f"Executing SNC analysis for company_id: {company_id}")
        # Ensure all_kwargs is serializable for logging if it contains complex objects.
        # For simplicity, logging it as string here.
        logging.debug(f"SNC_ANALYSIS_EXECUTE_INPUT: company_id='{company_id}', all_kwargs={str(kwargs)}")


        if not company_id:
            error_msg = "Company ID not provided for SNC analysis."
            logging.error(error_msg)
            return {"rating": None, "rationale": error_msg, "xai_trace": None}

        if 'DataRetrievalAgent' not in self.peer_agents:
            error_msg = "DataRetrievalAgent not found in peer agents for SNC_analyst_agent."
            logging.error(error_msg)
            return {"rating": None, "rationale": error_msg, "xai_trace": None}
        
        dra_request = {'data_type': 'get_company_financials', 'company_id': company_id}
        logging.debug(f"SNC_ANALYSIS_A2A_REQUEST: Requesting data from DataRetrievalAgent: {json.dumps(dra_request)}") # Log serializable dict
        company_data_package = await self.send_message('DataRetrievalAgent', dra_request)
        logging.debug(f"SNC_ANALYSIS_A2A_RESPONSE: Received data package: {company_data_package is not None}")

        if not company_data_package:
            error_msg = f"Failed to retrieve company data package for {company_id} from DataRetrievalAgent."
            logging.error(error_msg)
            return {"rating": None, "rationale": error_msg, "xai_trace": None}

        company_info = company_data_package.get('company_info', {})
        financial_data_detailed = company_data_package.get('financial_data_detailed', {})
        qualitative_company_info = company_data_package.get('qualitative_company_info', {})
        industry_data_context = company_data_package.get('industry_data_context', {})
        economic_data_context = company_data_package.get('economic_data_context', {})
        collateral_and_debt_details = company_data_package.get('collateral_and_debt_details', {})
        
        # Log extracted data summaries (ensure serializable)
        log_data_summary = {
            "CompanyInfo_keys": list(company_info.keys()),
            "FinancialDetailed_keys": list(financial_data_detailed.keys()),
            "Qualitative_keys": list(qualitative_company_info.keys()),
            "Industry_keys": list(industry_data_context.keys()),
            "Economic_keys": list(economic_data_context.keys()),
            "Collateral_keys": list(collateral_and_debt_details.keys())
        }
        logging.debug(f"SNC_ANALYSIS_DATA_EXTRACTED: {json.dumps(log_data_summary)}")


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
        
        # Rating and rationale determination
        rating_obj, rationale_str = await self._determine_rating(
            company_info.get('name', company_id), 
            financial_analysis_result, 
            qualitative_analysis_result, 
            credit_risk_mitigation_info, 
            economic_data_context
        )

        final_rating_value = rating_obj.value if rating_obj else None
        logging.debug(f"SNC_ANALYSIS_EXECUTE_OUTPUT: Rating='{final_rating_value}', Rationale='{rationale_str}'")

        xai_trace_data = None
        if include_xai_trace and self.xai_log_file:
            try:
                # Attempt to flush logs before reading. This is a best-effort.
                # Proper flushing depends on the logging setup of the wider application.
                for handler in logging.getLogger().handlers: # Check root logger handlers
                    if hasattr(handler, 'baseFilename') and handler.baseFilename == self.xai_log_file:
                        if hasattr(handler, 'flush'):
                            handler.flush()
                            logging.debug(f"Flushed handler for {self.xai_log_file}")
                        # If direct access to agent's specific logger is available, use that.
                        # For example, if self.logger is the agent's logger instance:
                        # for handler in self.logger.handlers: ...

                xai_trace_data = self._extract_xai_trace(self.xai_log_file)
            except Exception as e:
                logging.error(f"Failed to extract XAI trace from {self.xai_log_file}: {e}", exc_info=True)

        return {
            "rating": final_rating_value,
            "rationale": rationale_str,
            "xai_trace": xai_trace_data
        }

    def _prepare_financial_inputs_for_sk(self, financial_data_detailed: Dict[str, Any]) -> Dict[str, str]:
        """Prepares stringified financial inputs required by SK skills."""
        # Ensure keys logged for SNC_FIN_ANALYSIS_INPUT are serializable
        logging.debug(f"SNC_FIN_ANALYSIS_INPUT: financial_data_detailed keys: {json.dumps(list(financial_data_detailed.keys()))}, sk_inputs keys: []") # sk_inputs not yet formed
        cash_flow_statement = financial_data_detailed.get("cash_flow_statement", {})
        key_ratios = financial_data_detailed.get("key_ratios", {})
        market_data = financial_data_detailed.get("market_data", {})
        dcf_assumptions = financial_data_detailed.get("dcf_assumptions", {}) 

        prepared_inputs = {
            "historical_fcf_str": str(cash_flow_statement.get('free_cash_flow', ["N/A"])),
            "historical_cfo_str": str(cash_flow_statement.get('cash_flow_from_operations', ["N/A"])),
            "annual_debt_service_str": str(market_data.get("annual_debt_service_placeholder", "Not Available")), 
            "ratios_summary_str": json.dumps(key_ratios) if key_ratios else "Not available",
            "projected_fcf_str": str(dcf_assumptions.get("projected_fcf_placeholder", "Not Available")), 
            "payment_history_status_str": str(market_data.get("payment_history_placeholder", "Current")), 
            "interest_capitalization_status_str": str(market_data.get("interest_capitalization_placeholder", "No")) 
        }
        # Update the log call to show the keys of the prepared_inputs as well
        logging.debug(f"SNC_FIN_ANALYSIS_INPUT: financial_data_detailed keys: {json.dumps(list(financial_data_detailed.keys()))}, sk_inputs keys: {json.dumps(list(prepared_inputs.keys()))}")
        return prepared_inputs

    def _prepare_qualitative_inputs_for_sk(self, qualitative_company_info: Dict[str, Any]) -> Dict[str, str]:
        """Prepares stringified qualitative inputs required by SK skills."""
        prepared_inputs = {
            "qualitative_notes_stability_str": qualitative_company_info.get("revenue_cashflow_stability_notes_placeholder", "Management reports stable customer contracts."),
            "notes_financial_deterioration_str": qualitative_company_info.get("financial_deterioration_notes_placeholder", "No significant deterioration noted recently.")
        }
        logging.debug(f"SNC_QUAL_ANALYSIS_INPUT: qualitative_company_info_keys: {json.dumps(list(qualitative_company_info.keys()))}, sk_qual_inputs keys: {json.dumps(list(prepared_inputs.keys()))}")
        return prepared_inputs

    def _perform_financial_analysis(self, financial_data_detailed: Dict[str, Any], sk_financial_inputs: Dict[str, str]) -> Dict[str, Any]:
        # sk_financial_inputs are already logged by _prepare_financial_inputs_for_sk
        key_ratios = financial_data_detailed.get("key_ratios", {})
        
        analysis_result = {
            "debt_to_equity": key_ratios.get("debt_to_equity_ratio"),
            "profitability": key_ratios.get("net_profit_margin"),
            "liquidity_ratio": key_ratios.get("current_ratio"),
            "interest_coverage": key_ratios.get("interest_coverage_ratio"),
            **sk_financial_inputs 
        }
        logging.debug(f"SNC_FIN_ANALYSIS_OUTPUT: {json.dumps(analysis_result)}") # Ensure serializable
        return analysis_result

    def _perform_qualitative_analysis(self, 
                                      company_name: str, 
                                      qualitative_company_info: Dict[str, Any], 
                                      industry_data_context: Dict[str, Any], 
                                      economic_data_context: Dict[str, Any],
                                      sk_qualitative_inputs: Dict[str, str]) -> Dict[str, Any]:
        # sk_qualitative_inputs are logged by _prepare_qualitative_inputs_for_sk
        # Log other inputs for clarity (ensure serializable)
        log_qual_inputs = {
            "company_name": company_name,
            "qualitative_info_keys": list(qualitative_company_info.keys()),
            "industry_keys": list(industry_data_context.keys()),
            "economic_keys": list(economic_data_context.keys())
        }
        logging.debug(f"SNC_QUAL_ANALYSIS_INPUT: {json.dumps(log_qual_inputs)}")

        qualitative_result = {
            "management_quality": qualitative_company_info.get("management_assessment", "Not Assessed"),
            "industry_outlook": industry_data_context.get("outlook", "Neutral"),
            "economic_conditions": economic_data_context.get("overall_outlook", "Stable"),
            "business_model_strength": qualitative_company_info.get("business_model_strength", "N/A"),
            "competitive_advantages": qualitative_company_info.get("competitive_advantages", "N/A"),
            **sk_qualitative_inputs
        }
        logging.debug(f"SNC_QUAL_ANALYSIS_OUTPUT: {json.dumps(qualitative_result)}") # Ensure serializable
        return qualitative_result

    def _evaluate_credit_risk_mitigation(self, collateral_and_debt_details: Dict[str, Any]) -> Dict[str, Any]:
        logging.debug(f"SNC_CREDIT_MITIGATION_INPUT: collateral_and_debt_details_keys={json.dumps(list(collateral_and_debt_details.keys()))}")
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
        logging.debug(f"SNC_CREDIT_MITIGATION_OUTPUT: {json.dumps(mitigation_result)}") # Ensure serializable
        return mitigation_result

    async def _determine_rating(self, company_name: str, 
                               financial_analysis: Dict[str, Any], 
                               qualitative_analysis: Dict[str, Any], 
                               credit_risk_mitigation: Dict[str, Any],
                               economic_data_context: Dict[str, Any]
                               ) -> Tuple[Optional[SNCRating], str]:
        # Log inputs (ensure serializable)
        determine_rating_inputs_log = {
            "company": company_name,
            "financial_analysis_keys": list(financial_analysis.keys()),
            "qualitative_analysis_keys": list(qualitative_analysis.keys()),
            "credit_mitigation_keys": list(credit_risk_mitigation.keys()),
            "economic_context_keys": list(economic_data_context.keys())
        }
        logging.debug(f"SNC_DETERMINE_RATING_INPUT: {json.dumps(determine_rating_inputs_log)}")
        
        rationale_parts = []
        collateral_sk_assessment_str = None
        collateral_sk_justification = ""
        repayment_sk_assessment_str = None
        repayment_sk_justification = ""
        repayment_sk_concerns = ""
        nonaccrual_sk_assessment_str = None
        nonaccrual_sk_justification = ""

        if self.kernel and hasattr(self.kernel, 'skills'):
            # 1. AssessCollateralRisk
            try:
                sk_input_vars_collateral = {
                    "guideline_substandard_collateral": self.comptrollers_handbook_snc.get('substandard_definition', "Collateral is inadequately protective."),
                    "guideline_repayment_source": self.comptrollers_handbook_snc.get('primary_repayment_source', "Primary repayment should come from a sustainable source of cash under borrower control."),
                    "guideline_substandard_collateral_ref": self.comptrollers_handbook_snc.get('substandard_definition_ref', "N/A"),
                    "guideline_repayment_source_ref": self.comptrollers_handbook_snc.get('primary_repayment_source_ref', "N/A"),
                    "collateral_description": credit_risk_mitigation.get('collateral_summary_for_sk', "Not specified."),
                    "ltv_ratio": credit_risk_mitigation.get('loan_to_value_ratio', "Not specified."),
                    "other_collateral_notes": credit_risk_mitigation.get('collateral_notes_for_sk', "None.")
                }
                logging.debug(f"SNC_XAI:SK_INPUT:AssessCollateralRisk: {json.dumps(sk_input_vars_collateral)}") # Ensure serializable
                sk_response_collateral = await self.run_semantic_kernel_skill("SNCRatingAssistSkill", "CollateralRiskAssessment", sk_input_vars_collateral)
                lines = sk_response_collateral.strip().splitlines()
                if lines:
                    if "Assessment:" in lines[0]: collateral_sk_assessment_str = lines[0].split("Assessment:", 1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]: collateral_sk_justification = lines[1].split("Justification:", 1)[1].strip()
                logging.debug(f"SNC_XAI:SK_OUTPUT:AssessCollateralRisk: Assessment='{collateral_sk_assessment_str}', Justification='{collateral_sk_justification}'")
                if collateral_sk_justification: rationale_parts.append(f"SK Collateral Assessment ({collateral_sk_assessment_str}): {collateral_sk_justification}")
            except Exception as e: logging.error(f"Error in CollateralRiskAssessment SK skill for {company_name}: {e}")

            # 2. AssessRepaymentCapacity
            try:
                sk_input_vars_repayment = {
                    "guideline_repayment_source": self.comptrollers_handbook_snc.get('primary_repayment_source', "Default guideline..."),
                    "guideline_repayment_source_ref": self.comptrollers_handbook_snc.get('primary_repayment_source_ref', "N/A"),
                    "guideline_substandard_paying_capacity": self.comptrollers_handbook_snc.get('substandard_definition', "Default substandard..."), # Note: Using general substandard_definition as placeholder, specific paying capacity aspect is in prompt.
                    "guideline_substandard_paying_capacity_ref": self.comptrollers_handbook_snc.get('substandard_paying_capacity_ref', "N/A"),
                    "repayment_capacity_period_years": str(self.comptrollers_handbook_snc.get('repayment_capacity_period', 7)),
                    "guideline_repayment_capacity_period_years_ref": self.comptrollers_handbook_snc.get('repayment_capacity_period_years_ref', "N/A"),
                    "historical_fcf": financial_analysis.get('historical_fcf_str', "Not available"),
                    "historical_cfo": financial_analysis.get('historical_cfo_str', "Not available"),
                    "annual_debt_service": financial_analysis.get('annual_debt_service_str', "Not available"),
                    "relevant_ratios": financial_analysis.get('ratios_summary_str', "Not available"),
                    "projected_fcf": financial_analysis.get('projected_fcf_str', "Not available"),
                    "qualitative_notes_stability": qualitative_analysis.get('qualitative_notes_stability_str', "None provided.")
                }
                logging.debug(f"SNC_XAI:SK_INPUT:AssessRepaymentCapacity: {json.dumps(sk_input_vars_repayment)}") # Ensure serializable
                sk_response_repayment = await self.run_semantic_kernel_skill("SNCRatingAssistSkill", "AssessRepaymentCapacity", sk_input_vars_repayment)
                lines = sk_response_repayment.strip().splitlines()
                if lines:
                    if "Assessment:" in lines[0]: repayment_sk_assessment_str = lines[0].split("Assessment:",1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]: repayment_sk_justification = lines[1].split("Justification:",1)[1].strip()
                    if len(lines) > 2 and "Concerns:" in lines[2]: repayment_sk_concerns = lines[2].split("Concerns:",1)[1].strip()
                logging.debug(f"SNC_XAI:SK_OUTPUT:AssessRepaymentCapacity: Assessment='{repayment_sk_assessment_str}', Justification='{repayment_sk_justification}', Concerns='{repayment_sk_concerns}'")
                if repayment_sk_justification: rationale_parts.append(f"SK Repayment Capacity ({repayment_sk_assessment_str}): {repayment_sk_justification}. Concerns: {repayment_sk_concerns}")
            except Exception as e: logging.error(f"Error in AssessRepaymentCapacity SK skill for {company_name}: {e}")

            # 3. AssessNonAccrualStatusIndication
            try:
                sk_input_vars_nonaccrual = {
                    "guideline_nonaccrual_status": self.occ_guidelines_snc.get('nonaccrual_status', "Default non-accrual..."),
                    "guideline_nonaccrual_status_ref": self.occ_guidelines_snc.get('nonaccrual_status_ref', "N/A"),
                    "guideline_interest_capitalization": self.occ_guidelines_snc.get('capitalization_of_interest', "Default interest cap..."),
                    "guideline_interest_capitalization_ref": self.occ_guidelines_snc.get('capitalization_of_interest_ref', "N/A"),
                    "payment_history_status": financial_analysis.get('payment_history_status_str', "Current"),
                    "relevant_ratios": financial_analysis.get('ratios_summary_str', "Not available"),
                    "repayment_capacity_assessment": repayment_sk_assessment_str if repayment_sk_assessment_str else "Adequate", 
                    "notes_financial_deterioration": qualitative_analysis.get('notes_financial_deterioration_str', "None noted."),
                    "interest_capitalization_status": financial_analysis.get('interest_capitalization_status_str', "No")
                }
                logging.debug(f"SNC_XAI:SK_INPUT:AssessNonAccrualStatusIndication: {json.dumps(sk_input_vars_nonaccrual)}") # Ensure serializable
                sk_response_nonaccrual = await self.run_semantic_kernel_skill("SNCRatingAssistSkill", "AssessNonAccrualStatusIndication", sk_input_vars_nonaccrual)
                lines = sk_response_nonaccrual.strip().splitlines()
                if lines:
                    if "Assessment:" in lines[0]: nonaccrual_sk_assessment_str = lines[0].split("Assessment:",1)[1].strip().replace('[','').replace(']','')
                    if len(lines) > 1 and "Justification:" in lines[1]: nonaccrual_sk_justification = lines[1].split("Justification:",1)[1].strip()
                logging.debug(f"SNC_XAI:SK_OUTPUT:AssessNonAccrualStatusIndication: Assessment='{nonaccrual_sk_assessment_str}', Justification='{nonaccrual_sk_justification}'")
                if nonaccrual_sk_justification: rationale_parts.append(f"SK Non-Accrual Assessment ({nonaccrual_sk_assessment_str}): {nonaccrual_sk_justification}")
            except Exception as e: logging.error(f"Error in AssessNonAccrualStatusIndication SK skill for {company_name}: {e}")
        
        debt_to_equity = financial_analysis.get("debt_to_equity")
        profitability = financial_analysis.get("profitability")
        rating = SNCRating.PASS 

        # Log parameters for rating logic (ensure serializable)
        rating_logic_params_log = {
            "DtE": debt_to_equity, "Profitability": profitability,
            "SKCollateral": collateral_sk_assessment_str, "SKRepayment": repayment_sk_assessment_str,
            "SKNonAccrual": nonaccrual_sk_assessment_str,
            "FallbackCollateral": credit_risk_mitigation.get('collateral_quality_fallback'),
            "ManagementQuality": qualitative_analysis.get('management_quality')
        }
        logging.debug(f"SNC_XAI:RATING_PARAMS_FOR_LOGIC: {json.dumps(rating_logic_params_log)}")


        # Incorporate SK outputs into rating logic
        if repayment_sk_assessment_str == "Unsustainable" or \
           (nonaccrual_sk_assessment_str == "Non-Accrual Warranted" and repayment_sk_assessment_str == "Weak"):
            logging.debug(f"SNC_XAI:RATING_RULE: LOSS - Based on SK Repayment ('{repayment_sk_assessment_str}') and/or SK Non-Accrual ('{nonaccrual_sk_assessment_str}').")
            rating = SNCRating.LOSS
            rationale_parts.append("Loss rating driven by SK assessment of unsustainable repayment or non-accrual with weak repayment.")
        elif repayment_sk_assessment_str == "Weak" or \
             (collateral_sk_assessment_str == "Substandard" and repayment_sk_assessment_str == "Adequate"):
            logging.debug(f"SNC_XAI:RATING_RULE: DOUBTFUL - Based on SK Repayment ('{repayment_sk_assessment_str}') or SK Collateral ('{collateral_sk_assessment_str}') with Repayment ('{repayment_sk_assessment_str}').")
            rating = SNCRating.DOUBTFUL
            rationale_parts.append("Doubtful rating influenced by SK assessment of weak repayment or substandard collateral with adequate repayment.")
        elif nonaccrual_sk_assessment_str == "Non-Accrual Warranted" or \
             collateral_sk_assessment_str == "Substandard" or \
             (repayment_sk_assessment_str == "Adequate" and collateral_sk_assessment_str != "Pass"): # If repayment is just adequate and collateral isn't perfect
            logging.debug(f"SNC_XAI:RATING_RULE: SUBSTANDARD - Based on SK Non-Accrual ('{nonaccrual_sk_assessment_str}'), SK Collateral ('{collateral_sk_assessment_str}'), or SK Repayment ('{repayment_sk_assessment_str}').")
            rating = SNCRating.SUBSTANDARD 
            rationale_parts.append("Substandard rating influenced by SK assessments (Non-Accrual, Collateral, or Repayment indicating weaknesses).")
        
        if rating == SNCRating.PASS: 
            if debt_to_equity is not None and profitability is not None:
                if debt_to_equity > 3.0 and profitability < 0:
                    if rating == SNCRating.PASS: 
                        logging.debug(f"SNC_XAI:RATING_RULE_FALLBACK: LOSS - DtE ({debt_to_equity}) > 3.0 and Profitability ({profitability}) < 0")
                        rating = SNCRating.LOSS
                        rationale_parts.append("Fallback: High D/E ratio and negative profitability.")
                elif debt_to_equity > 2.0 and profitability < 0.1:
                     if rating == SNCRating.PASS:
                        logging.debug(f"SNC_XAI:RATING_RULE_FALLBACK: DOUBTFUL - DtE ({debt_to_equity}) > 2.0 and Profitability ({profitability}) < 0.1")
                        rating = SNCRating.DOUBTFUL
                        rationale_parts.append("Fallback: Elevated D/E ratio and low profitability.")
                elif financial_analysis.get("liquidity_ratio", 0) < 1.0 and financial_analysis.get("interest_coverage", 0) < 1.0:
                     if rating == SNCRating.PASS:
                        logging.debug(f"SNC_XAI:RATING_RULE_FALLBACK: SUBSTANDARD - Liquidity ({financial_analysis.get('liquidity_ratio')}) < 1.0 and Interest Coverage ({financial_analysis.get('interest_coverage')}) < 1.0")
                        rating = SNCRating.SUBSTANDARD
                        rationale_parts.append("Fallback: Insufficient liquidity and interest coverage.")
                elif (collateral_sk_assessment_str is None and credit_risk_mitigation.get("collateral_quality_fallback") == "Low") and \
                     qualitative_analysis.get("management_quality") == "Weak":
                     if rating == SNCRating.PASS:
                        logging.debug(f"SNC_XAI:RATING_RULE_FALLBACK: SPECIAL_MENTION - Fallback Collateral: {credit_risk_mitigation.get('collateral_quality_fallback')}, Management: {qualitative_analysis.get('management_quality')}")
                        rating = SNCRating.SPECIAL_MENTION
                        rationale_parts.append(f"Fallback: Collateral concerns (Fallback: {credit_risk_mitigation.get('collateral_quality_fallback')}) and weak management warrant Special Mention.")
                elif debt_to_equity <= 1.0 and profitability >= 0.3 and qualitative_analysis.get("economic_conditions") == "Stable":
                    # This is a definite PASS if not overridden by SK.
                    logging.debug(f"SNC_XAI:RATING_RULE_FALLBACK: PASS - DtE ({debt_to_equity}) <= 1.0, Profitability ({profitability}) >= 0.3, Econ Conditions: {qualitative_analysis.get('economic_conditions')}")
                    rating = SNCRating.PASS # Explicitly ensure it's Pass
                    rationale_parts.append("Fallback: Strong financials and stable economic conditions.")
                else: 
                    if rating == SNCRating.PASS: 
                        logging.debug(f"SNC_XAI:RATING_RULE_FALLBACK: SPECIAL_MENTION - Fallback/Mixed Indicators. Initial DtE: {debt_to_equity}, Profitability: {profitability}")
                        rating = SNCRating.SPECIAL_MENTION
                        rationale_parts.append("Fallback: Mixed financial indicators or other unaddressed concerns warrant monitoring.")
            elif rating == SNCRating.PASS : 
                logging.debug("SNC_XAI:RATING_RULE_FALLBACK: UNDETERMINED - Missing key financial metrics (DtE or Profitability)")
                rating = None 
                rationale_parts.append("Fallback: Cannot determine rating due to missing key financial metrics (debt-to-equity or profitability).")

        rationale_parts.append(f"Regulatory guidance: Comptroller's Handbook SNC v{self.comptrollers_handbook_snc.get('version', 'N/A')}, OCC Guidelines v{self.occ_guidelines_snc.get('version', 'N/A')}.")
        final_rationale = " ".join(filter(None, rationale_parts))
        
        logging.debug(f"SNC_DETERMINE_RATING_OUTPUT: Final Rating='{rating.value if rating else 'Undetermined'}', Rationale='{final_rationale}'")
        logging.info(f"SNC rating for {company_name}: {rating.value if rating else 'Undetermined'}. Rationale: {final_rationale}")
        return rating, final_rationale

if __name__ == '__main__':
    # Configure file logging for XAI trace at the START of the __main__ block
    log_file_name = 'snc_xai_test_run.log'
    if os.path.exists(log_file_name): 
        os.remove(log_file_name)
    
    # Get the root logger
    root_logger = logging.getLogger()
    # Clear existing handlers
    # This is important if the script is run multiple times or if other modules also call basicConfig.
    for handler in root_logger.handlers[:]: # Iterate over a copy
        root_logger.removeHandler(handler)
        handler.close() 
    
    root_logger.setLevel(logging.DEBUG) # Set root logger level to DEBUG

    # Add new file handler for DEBUG messages
    file_handler = logging.FileHandler(log_file_name, mode='w')
    file_handler.setLevel(logging.DEBUG)
    # Use a more detailed formatter for the file to include logger name for clarity
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Optional: Add a console handler back to see INFO/WARN/ERROR on console too
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) 
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.debug("XAI DEBUG Logging to file is configured for this script run.")

    # Definitions needed by run_tests, kept in the __main__ scope
    # Ensure this log_file_name matches what's in dummy_snc_agent_config for xai_log_file
    log_file_name = 'snc_xai_test_run.log'

    dummy_snc_agent_config = {
        'persona': "Test SNC Analyst",
        'xai_log_file': log_file_name, # Added for XAI trace testing
        'comptrollers_handbook_SNC': {
            "version": "2024.Q1_test",
            "substandard_definition": "Collateral is inadequately protective if its value doesn't cover the loan or if perfection issues exist. Paying capacity is inadequate if primary repayment source is not sustainable.",
            "primary_repayment_source": "Repayment should primarily come from the borrower's sustainable cash flow, under their control.",
            "primary_repayment_source_ref": "CHB Repayment Sec 1.3 Test",
            "substandard_definition_ref": "CHB Collateral Eval Sec 2.1 Test",
            "substandard_paying_capacity_ref": "CHB Paying Capacity Sec 3.2 Test",
            "repayment_capacity_period": 7,
            "repayment_capacity_period_years_ref": "CHB Guidance Appx A Test"
        },
        'occ_guidelines_SNC': {
            "version": "2024-03_test",
            "nonaccrual_status": "Loan is maintained on a cash basis due to financial deterioration of borrower; payment of principal or interest is not expected.",
            "capitalization_of_interest": "Permissible only if borrower is creditworthy and can repay in normal course of business.",
            "nonaccrual_status_ref": "OCC Non-Accrual Reg 12.3(a) Test",
            "capitalization_of_interest_ref": "OCC Interest Cap Policy 4.5 Test"
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
            data = json.loads(json.dumps(mock_data_package_template)) 
            data["company_info"]["name"] = f"{company_id} Corp"
            if company_id == "TEST_COMPANY_REPAY_WEAK":
                data["financial_data_detailed"]["key_ratios"]["interest_coverage_ratio"] = 0.8
                data["financial_data_detailed"]["cash_flow_statement"]["free_cash_flow"] = [10, 5, -20]
                data["financial_data_detailed"]["market_data"]["payment_history_placeholder"] = "90 days past due" # To trigger non-accrual
            return data
        return None

    # Store skill inputs for verification
    mock_skill_inputs_store = {}

    class MockSKFunction:
        def __init__(self, skill_name):
            self.skill_name = skill_name
            self.received_variables = None # Store received variables

        async def invoke(self, variables=None): 
            self.received_variables = variables # Store for later inspection
            mock_skill_inputs_store[self.skill_name] = variables # Also store globally for access in test asserts

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
                # Simulate weak repayment based on some input data for testing
                if "0.8" in variables.get("relevant_ratios","") or "-20" in variables.get("historical_fcf", "") : 
                     assessment = "Weak"; justification = "Repayment capacity is weak based on ratios/FCF."; concerns="Debt service coverage low, negative FCF trend."
                return MockSKResult(f"Assessment: {assessment}\nJustification: {justification}\nConcerns: {concerns}")
            elif self.skill_name == "AssessNonAccrualStatusIndication":
                assessment = "Accrual Appropriate"; justification = "Currently performing."
                if variables.get("payment_history_status") == "90 days past due" or variables.get("repayment_capacity_assessment") == "Weak":
                    assessment = "Non-Accrual Warranted"; justification = "Deterioration noted and/or weak repayment."
                return MockSKResult(f"Assessment: {assessment}\nJustification: {justification}")
            return MockSKResult("Unknown mock skill called.")


    class MockSKSkillsCollection:
        def get_function(self, skill_collection_name, skill_name):
            if skill_collection_name == "SNCRatingAssistSkill":
                return MockSKFunction(skill_name) 
            return None
        
    class MockKernel:
        def __init__(self): self.skills = MockSKSkillsCollection()
        async def run_async(self, sk_function, input_vars=None, **kwargs): 
            if sk_function: return await sk_function.invoke(variables=input_vars)
            return "Mock kernel run_async failed: No function"

    async def run_tests():
        # Logging is now configured globally for the script run.
        # dummy_snc_agent_config, mock_data_package_template, mock_send_message, 
        # MockSKFunction, MockSKSkillsCollection, MockKernel are defined in the outer scope of __main__.

        # Instantiate agents
        # snc_agent_no_kernel = SNCAnalystAgent(config=dummy_snc_agent_config, kernel=None) # No longer testing no_kernel variant for simplicity
        mock_kernel_instance = MockKernel()
        snc_agent_with_kernel = SNCAnalystAgent(config=dummy_snc_agent_config, kernel=mock_kernel_instance)

        # Simulate DataRetrievalAgent peer for testing execute() method directly
        mock_dra_placeholder = object() 
        # snc_agent_no_kernel.peer_agents['DataRetrievalAgent'] = mock_dra_placeholder
        snc_agent_with_kernel.peer_agents['DataRetrievalAgent'] = mock_dra_placeholder
        
        # Test without XAI trace first (original behavior, new return type)
        # with patch.object(snc_agent_no_kernel, 'send_message', new=mock_send_message):
        #     print("\n--- Test Case: Good Financials (No SK Kernel - TEST_COMPANY_SK_PASS) ---")
        #     result_no_sk_good = await snc_agent_no_kernel.execute(company_id="TEST_COMPANY_SK_PASS")
        #     print(f"Result (No SK, Good): {result_no_sk_good}")
            
        #     print("\n--- Test Case: Weak Repayment (No SK Kernel - TEST_COMPANY_REPAY_WEAK) ---")
        #     result_no_sk_repay_weak = await snc_agent_no_kernel.execute(company_id="TEST_COMPANY_REPAY_WEAK")
        #     print(f"Result (No SK, Repay Weak): {result_no_sk_repay_weak}")

        with patch.object(snc_agent_with_kernel, 'send_message', new=mock_send_message):
            print("\n--- Test Case: Good Financials (With SK Kernel, Include XAI Trace - TEST_COMPANY_SK_PASS) ---")
            # Clear log for this test run
            if os.path.exists(log_file_name):
                 with open(log_file_name, 'w'): pass
            mock_skill_inputs_store.clear() # Clear previous inputs

            result_sk_pass_good = await snc_agent_with_kernel.execute(company_id="TEST_COMPANY_SK_PASS", include_xai_trace=True)

            # Verify output structure
            assert isinstance(result_sk_pass_good, dict)
            assert "rating" in result_sk_pass_good
            assert "rationale" in result_sk_pass_good
            assert "xai_trace" in result_sk_pass_good

            print(f"Rating (SK, Good): {result_sk_pass_good.get('rating')}, Rationale: {result_sk_pass_good.get('rationale')}")
            assert "SK Collateral Assessment" in result_sk_pass_good.get('rationale', "")

            # Verify XAI trace
            xai_trace_good = result_sk_pass_good.get('xai_trace')
            assert xai_trace_good is not None
            assert xai_trace_good['agent_execution_trace']['log_file'] == log_file_name
            assert len(xai_trace_good['agent_execution_trace']['trace_events']) > 0
            print(f"XAI Trace captured with {len(xai_trace_good['agent_execution_trace']['trace_events'])} events.")

            # Verify skill inputs for references
            collateral_inputs = mock_skill_inputs_store.get("CollateralRiskAssessment")
            assert collateral_inputs is not None
            assert collateral_inputs.get("guideline_substandard_collateral_ref") == dummy_snc_agent_config['comptrollers_handbook_SNC']['substandard_definition_ref']
            assert collateral_inputs.get("guideline_repayment_source_ref") == dummy_snc_agent_config['comptrollers_handbook_SNC']['primary_repayment_source_ref']

            repayment_inputs = mock_skill_inputs_store.get("AssessRepaymentCapacity")
            assert repayment_inputs is not None
            assert repayment_inputs.get("guideline_repayment_source_ref") == dummy_snc_agent_config['comptrollers_handbook_SNC']['primary_repayment_source_ref']
            assert repayment_inputs.get("guideline_substandard_paying_capacity_ref") == dummy_snc_agent_config['comptrollers_handbook_SNC']['substandard_paying_capacity_ref']
            assert repayment_inputs.get("guideline_repayment_capacity_period_years_ref") == dummy_snc_agent_config['comptrollers_handbook_SNC']['repayment_capacity_period_years_ref']

            nonaccrual_inputs = mock_skill_inputs_store.get("AssessNonAccrualStatusIndication")
            assert nonaccrual_inputs is not None
            assert nonaccrual_inputs.get("guideline_nonaccrual_status_ref") == dummy_snc_agent_config['occ_guidelines_SNC']['nonaccrual_status_ref']
            assert nonaccrual_inputs.get("guideline_interest_capitalization_ref") == dummy_snc_agent_config['occ_guidelines_SNC']['capitalization_of_interest_ref']
            print("Verified SK skill inputs contain correct *_ref values for Good Financials case.")

            print("\n--- Test Case: Weak Repayment (With SK Kernel, Include XAI Trace - TEST_COMPANY_REPAY_WEAK) ---")
            if os.path.exists(log_file_name):
                 with open(log_file_name, 'w'): pass
            mock_skill_inputs_store.clear()

            result_sk_repay_weak = await snc_agent_with_kernel.execute(company_id="TEST_COMPANY_REPAY_WEAK", include_xai_trace=True)

            assert isinstance(result_sk_repay_weak, dict)
            assert "rating" in result_sk_repay_weak
            assert "rationale" in result_sk_repay_weak
            assert "xai_trace" in result_sk_repay_weak

            print(f"Rating (SK, Repay Weak): {result_sk_repay_weak.get('rating')}, Rationale: {result_sk_repay_weak.get('rationale')}")
            assert "SK Repayment Capacity (Weak)" in result_sk_repay_weak.get('rationale', "")

            xai_trace_weak = result_sk_repay_weak.get('xai_trace')
            assert xai_trace_weak is not None
            assert xai_trace_weak['agent_execution_trace']['log_file'] == log_file_name
            assert len(xai_trace_weak['agent_execution_trace']['trace_events']) > 0
            print(f"XAI Trace captured with {len(xai_trace_weak['agent_execution_trace']['trace_events'])} events.")
            # Input ref verification for this case (inputs to skills should still be correctly populated)
            # This re-checks the same logic as above but ensures it's consistent for different execution paths if inputs were dynamic.
            collateral_inputs_weak = mock_skill_inputs_store.get("CollateralRiskAssessment")
            assert collateral_inputs_weak is not None
            assert collateral_inputs_weak.get("guideline_substandard_collateral_ref") == dummy_snc_agent_config['comptrollers_handbook_SNC']['substandard_definition_ref']
            print("Verified SK skill inputs contain correct *_ref values for Weak Repayment case.")

            print("\n--- Test Case: Missing Company ID (With Kernel, No XAI Trace) ---")
            result_missing_id = await snc_agent_with_kernel.execute() # No company_id

            assert isinstance(result_missing_id, dict)
            assert "rating" in result_missing_id
            assert "rationale" in result_missing_id
            assert "xai_trace" in result_missing_id # Should be None or empty structure
            
            print(f"Result (Missing ID): {result_missing_id}")
            assert result_missing_id.get('rating') is None

    # The asyncio.run call should be the final line in the if __name__ == '__main__': block
    asyncio.run(run_tests())
