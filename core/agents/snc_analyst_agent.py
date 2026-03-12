# core/agents/snc_analyst_agent.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union
import logging
import asyncio
import os
import sys
import json
from enum import Enum
from datetime import datetime, timezone

from semantic_kernel import Kernel
from unittest.mock import patch

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput
from core.compliance.snc_validators import evaluate_compliance, RiskLevel

# Add project root to sys.path if running directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

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
    It can operate as part of a swarm (retrieving data from peers) or in standalone mode
    (accepting manual data or running as a portable prompt).
    """

    # Embedded default knowledge for portability
    DEFAULT_HANDBOOK = {
        "version": "Portable_v1",
        "substandard_definition": "Assets that are inadequately protected by the current sound worth and paying capacity of the obligor or of the collateral pledged, if any.",
        "primary_repayment_source": "The primary source of repayment is sustainable cash flow from operations.",
        "repayment_capacity_period": 7
    }

    DEFAULT_OCC_GUIDELINES = {
        "version": "Portable_v1",
        "nonaccrual_status": "Loans should be placed on nonaccrual status when full payment of principal and interest is not expected.",
        "capitalization_of_interest": "Capitalization of interest is generally inappropriate for a borrower in financial distress."
    }

    def __init__(self, config: Dict[str, Any], kernel: Optional[Kernel] = None, **kwargs):
        super().__init__(config, kernel=kernel, **kwargs)
        self.persona = self.config.get('persona', "SNC Analyst Examiner")
        self.description = self.config.get(
            'description', "Analyzes Shared National Credits based on regulatory guidelines.")

        # Use config or fallback to embedded defaults for portability
        self.comptrollers_handbook_snc = self.config.get('comptrollers_handbook_SNC') or self.DEFAULT_HANDBOOK
        self.occ_guidelines_snc = self.config.get('occ_guidelines_SNC') or self.DEFAULT_OCC_GUIDELINES

        self.audit_log = []

    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Logs an event to the internal audit trail."""
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "details": details
        })

    def generate_portable_prompt(self, context: Dict[str, Any] = None) -> str:
        """
        Generates a standalone prompt containing the agent's persona, instructions, and context.
        This allows the agent's logic to be ported to any LLM interface manually.
        """
        context_str = json.dumps(context, indent=2) if context else "No context provided. Please input financial data."

        prompt = f"""
        # ROLE: {self.persona}
        # MISSION: {self.description}

        # REGULATORY KNOWLEDGE BASE (Comptroller's Handbook):
        - Substandard Definition: {self.comptrollers_handbook_snc.get('substandard_definition')}
        - Primary Repayment Source: {self.comptrollers_handbook_snc.get('primary_repayment_source')}

        # OCC GUIDELINES:
        - Non-Accrual Status: {self.occ_guidelines_snc.get('nonaccrual_status')}

        # TASK:
        Analyze the provided financial data and assign a Shared National Credit (SNC) rating (Pass, Special Mention, Substandard, Doubtful, Loss).
        Provide a detailed rationale referencing the regulatory guidelines above.

        # CONTEXT DATA:
        {context_str}

        # OUTPUT FORMAT:
        Rating: [RATING]
        Rationale: [Detailed Explanation]
        """
        return prompt

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Tuple[Optional[SNCRating], str], AgentOutput]:
        """
        Executes SNC Analysis.
        """
        # Reset audit log
        self.audit_log = []

        # 1. Input Normalization
        company_id = ""
        is_standard_mode = False
        manual_data = None

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                company_id = query
                is_standard_mode = True
                # Check for manual data injection in context
                if input_data.context:
                    manual_data = input_data.context.get("manual_data") or input_data.context.get("financial_data")
            elif isinstance(input_data, str):
                company_id = input_data
            elif isinstance(input_data, dict):
                company_id = input_data.get("company_id", input_data.get("query"))
                manual_data = input_data.get("manual_data")
                kwargs.update(input_data)

        if not company_id:
            company_id = kwargs.get("company_id")

        logging.info(f"Executing SNC analysis for company_id: {company_id}")
        self._log_audit_event("SNC_ANALYSIS_START", {"company_id": company_id, "mode": "manual" if manual_data else "automated"})

        if not company_id:
            error_msg = "Company ID not provided for SNC analysis."
            logging.error(error_msg)
            if is_standard_mode:
                return AgentOutput(answer=error_msg, confidence=0.0, metadata={"error": error_msg})
            return None, error_msg

        # Data Retrieval Strategy: Manual > Peer Retrieval
        if manual_data:
            logging.info("Using manually provided data for analysis.")
            company_data_package = manual_data
            # Ensure basic structure if passed simply
            if "financial_data_detailed" not in company_data_package:
                 # Attempt to normalize simple dict to package structure
                 company_data_package = {
                     "company_info": {"name": company_id},
                     "financial_data_detailed": manual_data,
                     "qualitative_company_info": {},
                     "industry_data_context": {},
                     "economic_data_context": {},
                     "collateral_and_debt_details": {}
                 }
        else:
            company_data_package = await self._retrieve_data(company_id)

        if not company_data_package:
            error_msg = f"Failed to retrieve company data package for {company_id}."
            if is_standard_mode:
                return AgentOutput(answer=error_msg, confidence=0.0, metadata={"error": error_msg})
            return None, error_msg

        # Analysis Logic
        rating, rationale = await self._analyze_company(company_id, company_data_package)

        # Portable Prompt Generation (for meta-output)
        portable_prompt = self.generate_portable_prompt(context=company_data_package)

        if is_standard_mode:
            metadata = {
                "rating": rating.value if rating else None,
                "audit_log": self.audit_log,
                "portable_prompt": portable_prompt
            }
            return AgentOutput(
                answer=f"SNC Rating: {rating.value if rating else 'N/A'}\n\nRationale:\n{rationale}",
                sources=["SNC Handbook", "Company Financials", "OCC Guidelines"],
                confidence=0.9 if rating else 0.0,
                metadata=metadata
            )

        return rating, rationale

    async def _retrieve_data(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves data via A2A."""
        if 'DataRetrievalAgent' not in self.peer_agents:
            logging.warning("DataRetrievalAgent not found in peer agents. Cannot fetch external data.")
            return None

        dra_request = {'data_type': 'get_company_financials', 'company_id': company_id}
        return await self.send_message('DataRetrievalAgent', dra_request)

    async def _analyze_company(self, company_id: str, company_data_package: Dict[str, Any]) -> Tuple[Optional[SNCRating], str]:
        """Core analysis logic separated from execution wrapper."""

        company_info = company_data_package.get('company_info', {})
        financial_data_detailed = company_data_package.get('financial_data_detailed', {})
        qualitative_company_info = company_data_package.get('qualitative_company_info', {})
        industry_data_context = company_data_package.get('industry_data_context', {})
        economic_data_context = company_data_package.get('economic_data_context', {})
        collateral_and_debt_details = company_data_package.get('collateral_and_debt_details', {})

        # Prepare Inputs for SK or Logic
        financial_analysis_inputs_for_sk = self._prepare_financial_inputs_for_sk(financial_data_detailed)
        qualitative_analysis_inputs_for_sk = self._prepare_qualitative_inputs_for_sk(qualitative_company_info)

        # Perform Component Analysis
        financial_analysis_result = self._perform_financial_analysis(
            financial_data_detailed, financial_analysis_inputs_for_sk)
        qualitative_analysis_result = self._perform_qualitative_analysis(
            company_info.get('name', company_id),
            qualitative_company_info,
            industry_data_context,
            economic_data_context,
            qualitative_analysis_inputs_for_sk
        )
        credit_risk_mitigation_info = self._evaluate_credit_risk_mitigation(collateral_and_debt_details)

        # Determine Final Rating
        rating, rationale = await self._determine_rating(
            company_info.get('name', company_id),
            financial_analysis_result,
            qualitative_analysis_result,
            credit_risk_mitigation_info,
            economic_data_context,
            industry_data_context
        )

        self._log_audit_event("RATING_DETERMINED", {
            "rating": rating.value if rating else 'N/A',
            "rationale_preview": rationale[:100] + "..."
        })

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
        key_ratios = financial_data_detailed.get("key_ratios", {})
        analysis_result = {
            "debt_to_equity": key_ratios.get("debt_to_equity_ratio"),
            "profitability": key_ratios.get("net_profit_margin"),
            "liquidity_ratio": key_ratios.get("current_ratio"),
            "interest_coverage": key_ratios.get("interest_coverage_ratio"),
            "tier_1_capital_ratio": key_ratios.get("tier_1_capital_ratio"),
            **sk_financial_inputs
        }
        return analysis_result

    def _perform_qualitative_analysis(self, company_name: str, qualitative_company_info: Dict[str, Any], industry_data_context: Dict[str, Any], economic_data_context: Dict[str, Any], sk_qualitative_inputs: Dict[str, str]) -> Dict[str, Any]:
        qualitative_result = {
            "management_quality": qualitative_company_info.get("management_assessment", "Not Assessed"),
            "industry_outlook": industry_data_context.get("outlook", "Neutral"),
            "economic_conditions": economic_data_context.get("overall_outlook", "Stable"),
            "business_model_strength": qualitative_company_info.get("business_model_strength", "N/A"),
            "competitive_advantages": qualitative_company_info.get("competitive_advantages", "N/A"),
            **sk_qualitative_inputs
        }
        return qualitative_result

    def _evaluate_credit_risk_mitigation(self, collateral_and_debt_details: Dict[str, Any]) -> Dict[str, Any]:
        ltv = collateral_and_debt_details.get("loan_to_value_ratio")
        collateral_quality_assessment = "Low"
        if ltv is not None:
            try:
                ltv_float = float(ltv)
                if ltv_float < 0.5:
                    collateral_quality_assessment = "High"
                elif ltv_float < 0.75:
                    collateral_quality_assessment = "Medium"
            except ValueError:
                pass

        mitigation_result = {
            "collateral_quality_fallback": collateral_quality_assessment,
            "collateral_summary_for_sk": collateral_and_debt_details.get("collateral_type", "Not specified."),
            "loan_to_value_ratio": str(ltv) if ltv is not None else "Not specified.",
            "collateral_notes_for_sk": collateral_and_debt_details.get("other_credit_enhancements", "None."),
            "collateral_valuation": collateral_and_debt_details.get("collateral_valuation"),
            "guarantees_present": collateral_and_debt_details.get("guarantees_exist", False)
        }
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

    def _rate_from_fallback_logic(self, financial_analysis: Dict[str, Any], qualitative_analysis: Dict[str, Any], credit_risk_mitigation: Dict[str, Any], sector_name: str = "General", market_data: Dict[str, Any] = None) -> Tuple[Optional[SNCRating], str]:
        """Provides a rating based on hardcoded financial metrics if SK fails."""

        # 1. Use Compliance Validators first
        compliance_input = {
            "key_ratios": {
                "debt_to_equity_ratio": financial_analysis.get("debt_to_equity"),
                "net_profit_margin": financial_analysis.get("profitability"),
                "current_ratio": financial_analysis.get("liquidity_ratio"),
                "interest_coverage_ratio": financial_analysis.get("interest_coverage"),
                "tier_1_capital_ratio": financial_analysis.get("tier_1_capital_ratio")
            }
        }

        compliance_result = evaluate_compliance(compliance_input, sector_name=sector_name, market_data=market_data)
        self._log_audit_event("COMPLIANCE_CHECK", {
            "passed": compliance_result.passed,
            "violations": compliance_result.violations
        })

        if not compliance_result.passed:
             rationale = "Compliance Validation Failed:\n- " + "\n- ".join(compliance_result.violations)
             return SNCRating.SUBSTANDARD, rationale

        # 2. Existing Legacy Fallback Logic
        debt_to_equity = financial_analysis.get("debt_to_equity")
        profitability = financial_analysis.get("profitability")

        if debt_to_equity is None or profitability is None:
            return None, "Fallback rating failed: Missing key financial metrics."

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

        header = f"Executive Summary for {company_name}: The credit has been assigned a rating of {rating.value}."
        sk_section_header = "Primary Justification (AI Skill-Based Analysis):"
        sk_section_body = sk_rationale_summary

        factors = []
        d_to_e = financial_analysis.get('debt_to_equity')
        profit = financial_analysis.get('profitability')

        if d_to_e is not None:
            factors.append(f"- Debt/Equity Ratio: {d_to_e:.2f}")
        if profit is not None:
            factors.append(f"- Profit Margin: {profit:.2%}")

        supporting_factors = "\n".join(factors)
        full_rationale = f"{header}\n\n{sk_section_header}\n{sk_section_body}\n\nSupporting Factors:\n{supporting_factors}"
        return full_rationale

    async def _determine_rating(self, company_name: str,
                                financial_analysis: Dict[str, Any],
                                qualitative_analysis: Dict[str, Any],
                                credit_risk_mitigation: Dict[str, Any],
                                economic_data_context: Dict[str, Any],
                                industry_data_context: Dict[str, Any]
                                ) -> Tuple[Optional[SNCRating], str]:

        collateral_sk_assessment_str, repayment_sk_assessment_str, nonaccrual_sk_assessment_str = None, None, None
        collateral_sk_justification, repayment_sk_justification, nonaccrual_sk_justification = "", "", ""

        # SK Logic (Condensed for clarity)
        if self.kernel and hasattr(self.kernel, 'skills'):
            # Implementation of SK calls would go here, updating the _sk variables
            pass

        # Primary rating path: Use SK skill outputs
        rating, sk_rationale = self._rate_from_sk_assessments(
            repayment_sk_assessment_str,
            collateral_sk_assessment_str,
            nonaccrual_sk_assessment_str
        )

        final_rationale = ""
        # Fallback path
        if rating is None:
            sector_name = industry_data_context.get('sector', 'General')
            market_data = {
                "vix": economic_data_context.get('vix') or economic_data_context.get('volatility_index')
            }
            rating, final_rationale = self._rate_from_fallback_logic(
                financial_analysis,
                qualitative_analysis,
                credit_risk_mitigation,
                sector_name=sector_name,
                market_data=market_data
            )
        else:
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

        return rating, final_rationale

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    async def main():
        # Standalone Test
        print("\n--- Test Standalone Mode ---")
        agent_standalone = SNCAnalystAgent(config={})
        manual_data = {
            "key_ratios": {"debt_to_equity_ratio": 4.0, "net_profit_margin": -0.05, "liquidity_ratio": 0.5, "interest_coverage_ratio": 0.5}
        }
        result = await agent_standalone.execute(AgentInput(query="Distressed Corp", context={"manual_data": manual_data}))
        print(result.answer)
        print("\n--- Portable Prompt Preview ---")
        print(result.metadata.get("portable_prompt")[:300] + "...")

    asyncio.run(main())
