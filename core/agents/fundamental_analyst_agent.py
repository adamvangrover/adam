# core/agents/fundamental_analyst_agent.py

import csv
import os
# from core.utils.data_utils import send_message # send_message is not used in this file

import logging
import pandas as pd
import numpy as np
from scipy import stats  # For statistical calculations (e.g., for DCF)
from typing import Dict, Any, Optional, Union, List
from core.agents.agent_base import AgentBase
from core.system.memory_manager import VectorMemoryManager
from semantic_kernel import Kernel # Added for type hinting
import asyncio # Added import
import yaml # Added for example usage block
import json

# Placeholder for message queue interaction (replace with real implementation later)
# from core.system.message_queue import MessageQueue
from unittest.mock import patch # Added for example usage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# For XAI debug logs, ensure the logger level is set to DEBUG if you want to see them.
# Example: logging.getLogger().setLevel(logging.DEBUG) in the main application or test setup.

class DCFCalculator:
    """
    Helper class for DCF calculations.
    """
    @staticmethod
    def calculate_intrinsic_value(wacc: float, terminal_growth: float, last_fcf: float, projection_years: int = 5, growth_rate: float = 0.05) -> float:
        """
        Calculates intrinsic value using a simplified DCF model.
        """
        if wacc <= terminal_growth:
            return 0.0

        present_value = 0.0
        current_fcf = last_fcf

        # Projection Period
        for i in range(1, projection_years + 1):
            current_fcf *= (1 + growth_rate)
            present_value += current_fcf / ((1 + wacc) ** i)

        # Terminal Value
        terminal_value = current_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_terminal_value = terminal_value / ((1 + wacc) ** projection_years)

        return present_value + pv_terminal_value

class FundamentalAnalystAgent(AgentBase):
    """
    Agent for performing fundamental analysis of companies.

    This agent analyzes financial statements, calculates key financial ratios,
    performs valuation modeling (DCF and comparables), and assesses financial health.
    It relies on DataRetrievalAgent for fetching company data via A2A communication.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Kernel] = None):
        super().__init__(config, kernel=kernel)
        self.persona = self.config.get('persona', "Financial Analyst")
        self.description = self.config.get('description', "Performs fundamental company analysis.")
        self.memory_manager = VectorMemoryManager()
        # The 'peers' key in self.config (e.g., ['DataRetrievalAgent']) is used by AgentOrchestrator
        # to set up connections via self.add_peer_agent(peer_instance)


    async def execute(self, company_id: str) -> Dict[str, Any]:
        """
        Performs fundamental analysis on a given company.
        """
        logging.info(f"Executing fundamental analysis for company_id: {company_id}")
        logging.debug(f"FAA_XAI:EXECUTE_INPUT: company_id='{company_id}'")

        # Retrieve past analysis
        history = self.memory_manager.query_history(company_id, limit=3)
        history_context = ""
        if history:
             history_context = "\nPast Analysis (Same Company):\n" + "\n".join([f"- {h['timestamp']}: {h['analysis_summary'][:200]}..." for h in history])

        # Retrieve similar analysis
        if hasattr(self.memory_manager, 'search_similar'):
             similar_docs = self.memory_manager.search_similar(f"Analysis of {company_id}", limit=2)
             # Filter out self
             similar_docs = [d for d in similar_docs if d['company_id'] != company_id]
             if similar_docs:
                 similar_context = "\nSimilar Past Analyses (Peers/Context):\n" + "\n".join([f"- {d['company_id']} ({d.get('similarity_score',0):.2f}): {d['analysis_summary'][:200]}..." for d in similar_docs])
                 history_context += similar_context

        logging.info(f"Retrieved {len(history)} past analysis records for {company_id}")

        try:
            company_data = await self.retrieve_company_data(company_id) 
            if company_data is None:
                logging.error(f"Failed to retrieve company data for {company_id} via A2A.")
                return {"error": f"Could not retrieve data for company {company_id}"}
            
            logging.debug(f"FAA_XAI:EXECUTE_COMPANY_DATA_KEYS: {list(company_data.keys())}")


            financial_ratios = self.calculate_financial_ratios(company_data)
            dcf_valuation_result = self.calculate_dcf_valuation(company_data)
            comps_valuation = self.calculate_comps_valuation(company_data) # Placeholder
            enterprise_value_result = self.calculate_enterprise_value(company_data)
            financial_health = self.assess_financial_health(financial_ratios)
            raw_metrics = self.extract_raw_metrics(company_data)
            
            analysis_summary = await self.generate_analysis_summary(
                company_id, 
                financial_ratios, 
                dcf_valuation_result, 
                comps_valuation, 
                financial_health,
                enterprise_value_result,
                history_context=history_context
            )

            # Save to Memory
            self.memory_manager.save_analysis(company_id, analysis_summary, {
                "dcf": dcf_valuation_result,
                "health": financial_health,
                "ev": enterprise_value_result
            })

            result_package = {
                "company_id": company_id,
                "financial_ratios": financial_ratios,
                "raw_metrics": raw_metrics,
                "dcf_valuation": dcf_valuation_result,
                "comps_valuation": comps_valuation,
                "enterprise_value": enterprise_value_result,
                "financial_health": financial_health,
                "analysis_summary": analysis_summary,
                "error": None
            }
            logging.debug(f"FAA_XAI:EXECUTE_OUTPUT: {result_package}")
            return result_package

        except Exception as e:
            logging.exception(f"Error during fundamental analysis of {company_id}: {e}")
            return {"error": f"An error occurred during analysis: {e}"}

    async def retrieve_company_data(self, company_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves company data by sending a message to the DataRetrievalAgent.
        """
        if 'DataRetrievalAgent' in self.peer_agents:
            logging.info(f"Requesting company financials for {company_id} from DataRetrievalAgent.")
            request_data = {'data_type': 'get_company_financials', 'company_id': company_id}
            try:
                response = await self.send_message('DataRetrievalAgent', request_data)
                logging.debug(f"Data received from DataRetrievalAgent for {company_id}: {response is not None}")
                if response: 
                    return response 
                else:
                    logging.warning(f"DataRetrievalAgent returned no data for {company_id}.")
                    return None
            except Exception as e:
                logging.exception(f"Error sending message to DataRetrievalAgent for {company_id}: {e}")
                return None
        else:
            logging.error("DataRetrievalAgent not found in peer agents. Cannot retrieve company data.")
            return None

    def extract_raw_metrics(self, company_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts raw financial metrics (EBITDA, Debt, Cash, Revenue) for downstream agents.
        """
        metrics = {}
        try:
            financial_details = company_data.get('financial_data_detailed', {})
            income_statement = financial_details.get('income_statement', {})
            balance_sheet = financial_details.get('balance_sheet', {})

            ebitda_list = income_statement.get("ebitda", [])
            metrics["ebitda"] = float(ebitda_list[-1]) if ebitda_list else 0.0

            revenue_list = income_statement.get("revenue", [])
            metrics["revenue"] = float(revenue_list[-1]) if revenue_list else 0.0

            short_term_debt_values = balance_sheet.get('short_term_debt', [0])
            long_term_debt_values = balance_sheet.get('long_term_debt', [0])
            st_debt = float(short_term_debt_values[-1]) if short_term_debt_values else 0.0
            lt_debt = float(long_term_debt_values[-1]) if long_term_debt_values else 0.0
            metrics["total_debt"] = st_debt + lt_debt

            cash_values = balance_sheet.get('cash_and_equivalents', [0])
            metrics["cash"] = float(cash_values[-1]) if cash_values else 0.0

            assets_values = balance_sheet.get('total_assets', [0])
            metrics["total_assets"] = float(assets_values[-1]) if assets_values else 0.0

        except Exception as e:
            logging.warning(f"Failed to extract raw metrics: {e}")
            metrics = {"ebitda": 0.0, "revenue": 0.0, "total_debt": 0.0, "cash": 0.0, "total_assets": 0.0}

        return metrics

    def calculate_financial_ratios(self, company_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates key financial ratios.
        """
        logging.debug(f"FAA_XAI:CALC_RATIOS_INPUT: company_data keys: {list(company_data.keys())}")
        ratios = {}
        try:
            # Ensure financial_data_detailed and its sub-keys exist
            financial_details = company_data.get('financial_data_detailed', {})
            income_statement = financial_details.get('income_statement', {})
            balance_sheet = financial_details.get('balance_sheet', {})

            if not income_statement or not balance_sheet:
                logging.warning("FAA_XAI:CALC_RATIOS_WARN: Missing financial statements for ratio calculation.")
                return {}
            
            # Safely get values, assuming they are lists and we need the last element
            revenue_list = income_statement.get("revenue", [])
            net_income_list = income_statement.get("net_income", [])
            ebitda_list = income_statement.get("ebitda", [])
            
            total_assets_list = balance_sheet.get("total_assets", [])
            total_liabilities_list = balance_sheet.get("total_liabilities", [])
            shareholders_equity_list = balance_sheet.get("shareholders_equity", [])

            if not all([revenue_list, net_income_list, total_assets_list, total_liabilities_list, shareholders_equity_list]):
                logging.warning("FAA_XAI:CALC_RATIOS_WARN: Missing essential lists in financial statements for ratio calculation.")
                return {}

            revenue = revenue_list[-1] if revenue_list else 0
            net_income = net_income_list[-1] if net_income_list else 0
            ebitda = ebitda_list[-1] if ebitda_list else 0
            
            total_assets = total_assets_list[-1] if total_assets_list else 0
            total_liabilities = total_liabilities_list[-1] if total_liabilities_list else 0
            shareholders_equity = shareholders_equity_list[-1] if shareholders_equity_list else 0
            
            logging.debug(f"FAA_XAI:CALC_RATIOS_VALUES: Revenue={revenue}, NetIncome={net_income}, EBITDA={ebitda}, Assets={total_assets}, Liab={total_liabilities}, Equity={shareholders_equity}")


            ratios["revenue_growth"] = (revenue / revenue_list[-2] - 1) if len(revenue_list) > 1 and revenue_list[-2] !=0 else None
            ratios["net_profit_margin"] = net_income / revenue if revenue else None
            ratios["return_on_equity"] = net_income / shareholders_equity if shareholders_equity else None
            ratios["debt_to_equity"] = total_liabilities / shareholders_equity if shareholders_equity else None
            ratios["current_ratio"] = total_assets / total_liabilities if total_liabilities else None # Simplified current ratio
            ratios["ebitda_margin"] = ebitda / revenue if revenue and ebitda else None
            
        except (KeyError, IndexError, TypeError) as e:
            logging.exception(f"Error calculating financial ratios: {e}")
            logging.debug(f"FAA_XAI:CALC_RATIOS_ERROR: Exception {e}")
            return {} 
        
        logging.debug(f"FAA_XAI:CALC_RATIOS_OUTPUT: {ratios}")
        return ratios


    def calculate_comps_valuation(self, company_data: Dict[str, Any]) -> Optional[float]:
        logging.warning("Comps valuation not yet implemented.")
        return None

    def assess_financial_health(self, financial_ratios: Dict[str, float]) -> str:
        logging.debug(f"FAA_XAI:ASSESS_HEALTH_INPUT: financial_ratios={financial_ratios}")
        if not financial_ratios:
            logging.debug("FAA_XAI:ASSESS_HEALTH_RESULT: Ratios empty, returning Unknown.")
            return "Unknown"

        score = 0
        if financial_ratios.get("return_on_equity", 0) > 0.15: score += 2
        if financial_ratios.get("debt_to_equity", 1) < 1: score += 1
        if financial_ratios.get("current_ratio", 0) > 1.5: score += 1
        if financial_ratios.get("net_profit_margin", 0) > .1: score +=1
        if financial_ratios.get("revenue_growth") is not None and financial_ratios.get("revenue_growth", 0) > 0.05 : score +=1 # Added check for > 0.05

        logging.debug(f"FAA_XAI:ASSESS_HEALTH_SCORE: Score={score}")

        health_assessment = "Weak"
        if score >= 4: health_assessment = "Strong"
        elif score >= 2: health_assessment = "Moderate"
        
        logging.debug(f"FAA_XAI:ASSESS_HEALTH_OUTPUT: Health='{health_assessment}'")
        return health_assessment


    async def generate_analysis_summary(self, company_id: str, financial_ratios: Dict[str, float],
                                      dcf_valuation: Optional[float], comps_valuation: Optional[float],
                                      financial_health: str, enterprise_value: Optional[float],
                                      history_context: str = "") -> str:
        # ... (existing summary generation logic, SK or fallback) ...
        # This method's logging is mostly for SK call, fallback summary is straightforward
        # For XAI, the inputs to this are already logged by `execute` and prior calc methods.
        # Logging within the SK call path is already present.
        # Fallback summary construction is direct.

        if self.kernel and hasattr(self.kernel, 'skills'): # Check for skills attribute
            try:
                ratios_str_parts = []
                if financial_ratios:
                    for ratio, value in financial_ratios.items():
                        ratios_str_parts.append(f"  - {ratio}: {value:.2f}" if value is not None else f"  - {ratio}: N/A")
                ratios_summary_str = "\n".join(ratios_str_parts) if ratios_str_parts else "Not available"

                dcf_summary = f"Value: {dcf_valuation:.2f}" if dcf_valuation is not None else "Not available"
                comps_summary = f"Value: {comps_valuation:.2f}" if comps_valuation is not None else "Not available"
                enterprise_value_summary_str = f"Value: {enterprise_value:.2f}" if enterprise_value is not None else "Not available"
                
                user_prompt_for_conclusion = self.config.get(
                    "summarize_analysis_user_prompt", 
                    "Provide a brief overall conclusion based on the data."
                )

                input_vars = {
                    "company_id": company_id,
                    "financial_health": financial_health,
                    "ratios_summary": ratios_summary_str,
                    "dcf_valuation_summary": dcf_summary,
                    "comps_valuation_summary": comps_summary,
                    "enterprise_value_summary": enterprise_value_summary_str, 
                    "user_provided_key_insights_or_conclusion_prompt": user_prompt_for_conclusion,
                    "history_context": history_context
                }
                
                logging.info(f"Attempting to generate summary for {company_id} using Semantic Kernel skill 'FundamentalAnalysisSkill.SummarizeAnalysis'.")
                logging.debug(f"FAA_XAI:GEN_SUMMARY_SK_INPUT: {input_vars}")
                summary = await self.run_semantic_kernel_skill("FundamentalAnalysisSkill", "SummarizeAnalysis", input_vars)
                logging.debug(f"FAA_XAI:GEN_SUMMARY_SK_OUTPUT: '{summary}'")
                logging.info(f"Successfully generated summary for {company_id} using SK.")
                return summary
            except AttributeError as e: 
                logging.warning(f"Semantic Kernel or skill method not available, or attribute error: {e}. Falling back to string formatting for summary.")
            except ValueError as e: 
                logging.warning(f"Semantic Kernel skill 'FundamentalAnalysisSkill.SummarizeAnalysis' execution failed: {e}. Falling back to string formatting.")
            except Exception as e:
                logging.error(f"An unexpected error occurred while using Semantic Kernel for summary: {e}. Falling back to string formatting.")
        
        logging.warning(f"Generating summary for {company_id} using fallback string formatting.")
        summary = f"Fundamental Analysis for {company_id}:\n\n"
        if financial_ratios:
            summary += "Key Financial Ratios:\n"
            for ratio, value in financial_ratios.items():
                summary += f"  - {ratio}: {value:.2f}\n" if value is not None else f"  - {ratio}: N/A\n"
        else:
            summary += "  Financial ratios could not be calculated.\n"
        summary += f"Financial Health: {financial_health}\n"
        if dcf_valuation is not None:
            summary += f"DCF Valuation: {dcf_valuation:.2f}\n"
        else:
            summary += "DCF Valuation: Not available\n"
        if comps_valuation is not None:
            summary += f"Comps Valuation: {comps_valuation:.2f}\n"
        else:
            summary += "Comps Valuation: Not available\n"
        if enterprise_value is not None: 
            summary += f"Enterprise Value: {enterprise_value:.2f}\n"
        else:
            summary += "Enterprise Value: Not available\n"
        
        default_likelihood = self.estimate_default_likelihood(financial_ratios)
        if default_likelihood is not None:
             summary += f"Estimated Default Likelihood: {default_likelihood:.2%}\n"
        else:
             summary += f"Estimated Default Likelihood: N/A\n"

        distressed_metrics = self.calculate_distressed_metrics(financial_ratios if financial_ratios else {}) 
        if distressed_metrics: 
            summary += "Distressed Metrics:\n"
            for metric, value in distressed_metrics.items():
                summary += f"  - {metric}: {value:.2f}\n"

        recovery_rate = self.estimate_recovery_rate(financial_ratios if financial_ratios else {}, default_likelihood)
        if recovery_rate is not None:
            summary += f"Estimated Recovery Rate (in default): {recovery_rate:.2%}\n"
        
        summary += "\n(Summary generated using fallback string formatting.)"
        logging.debug(f"FAA_XAI:GEN_SUMMARY_FALLBACK_OUTPUT: '{summary}'")
        return summary

    def export_to_csv(self, data: Dict[str, List[Union[str, float]]], filename: str):
        # ... (existing code) ...
        pass


    def calculate_growth_rate(self, financial_statements: Dict[str, Any], metric: str) -> Optional[float]:
        # ... (existing code) ...
        pass


    def calculate_ebitda_margin(self, financial_statements: Dict[str, Any]) -> Optional[float]:
        # ... (existing code) ...
        pass


    def calculate_dcf_valuation(self, company_data: Dict[str, Any]) -> Optional[float]:
        """
        Calculates the Discounted Cash Flow (DCF) valuation of the company using a two-stage FCF projection model.

        Args:
            company_data (Dict[str, Any]): The comprehensive data package for the company,
                                         expected to contain 'financial_data_detailed'.
        """
        company_name_for_log = company_data.get('company_info', {}).get('name', 'Unknown')
        logging.debug(f"FAA_XAI:DCF_INPUT: company_id='{company_name_for_log}', company_data keys: {list(company_data.keys())}")
        try:
            financial_details = company_data.get('financial_data_detailed', {})
            cash_flow_statement = financial_details.get('cash_flow_statement', {})
            historical_fcf = cash_flow_statement.get('free_cash_flow', []) 

            if not historical_fcf:
                logging.warning(f"FAA_XAI:DCF_ABORT: No historical free cash flow data for {company_name_for_log}.")
                return None
            
            last_historical_fcf = historical_fcf[-1]
            if not isinstance(last_historical_fcf, (int, float)): # Ensure last FCF is usable
                logging.warning(f"FAA_XAI:DCF_ABORT: Last historical FCF for {company_name_for_log} ('{last_historical_fcf}') is not numeric.")
                return None

            dcf_assumptions = financial_details.get('dcf_assumptions', {})
            
            # Core rates from existing assumptions
            discount_rate = dcf_assumptions.get('discount_rate')
            # Terminal growth rate for perpetuity calculation (after explicit projection period)
            terminal_growth_rate_perpetuity = dcf_assumptions.get('terminal_growth_rate') 
            
            # New parameters for two-stage growth model
            fcf_projection_years_total = int(dcf_assumptions.get('fcf_projection_years_total', 10)) # Default 10 years
            initial_high_growth_period_years = int(dcf_assumptions.get('initial_high_growth_period_years', 5)) # Default 5 years
            initial_high_growth_rate = dcf_assumptions.get('initial_high_growth_rate', 0.10) # Default 10%
            stable_growth_rate = dcf_assumptions.get('stable_growth_rate', 0.05) # Default 5% for second stage
            
            # Ensure initial_high_growth_period_years is not more than fcf_projection_years_total
            initial_high_growth_period_years = min(initial_high_growth_period_years, fcf_projection_years_total)

            logging.debug(
                f"FAA_XAI:DCF_PARAMS_TWO_STAGE: LastHistFCF={last_historical_fcf}, DiscountRate={discount_rate}, "
                f"TerminalGrowthRatePerpetuity={terminal_growth_rate_perpetuity}, TotalProjectionYears={fcf_projection_years_total}, "
                f"HighGrowthYears={initial_high_growth_period_years}, HighGrowthRate={initial_high_growth_rate}, StableGrowthRate={stable_growth_rate}"
            )

            # Validate core rates and growth rates
            essential_rates = [discount_rate, terminal_growth_rate_perpetuity, initial_high_growth_rate, stable_growth_rate]
            if not all(isinstance(rate, (int, float)) for rate in essential_rates if rate is not None): # Allow None for rates not used if logic handles it
                logging.warning(f"FAA_XAI:DCF_ABORT: One or more DCF rates are non-numeric for {company_name_for_log}. Rates: DR={discount_rate}, TGR_P={terminal_growth_rate_perpetuity}, IHGR={initial_high_growth_rate}, SGR={stable_growth_rate}")
                return None
            if discount_rate is None or terminal_growth_rate_perpetuity is None: # These are always needed
                 logging.warning(f"FAA_XAI:DCF_ABORT: Discount rate or terminal perpetuity growth rate is missing for {company_name_for_log}.")
                 return None
            if discount_rate <= terminal_growth_rate_perpetuity: # Check against perpetuity growth rate
                logging.warning(f"FAA_XAI:DCF_ABORT: Discount rate ({discount_rate}) not > terminal perpetuity growth rate ({terminal_growth_rate_perpetuity}) for {company_name_for_log}.")
                return None

            projected_cash_flows = []
            current_fcf = last_historical_fcf

            for year_num in range(1, fcf_projection_years_total + 1):
                growth_rate_for_year = 0.0 
                if year_num <= initial_high_growth_period_years:
                    if initial_high_growth_rate is None: # Should be caught by all() check above if strictly required for all years
                        logging.warning(f"FAA_XAI:DCF_WARN: Missing initial_high_growth_rate for year {year_num}, using 0 growth for this year.")
                    else:
                        growth_rate_for_year = initial_high_growth_rate
                else: # Stable growth period (years > initial_high_growth_period_years up to fcf_projection_years_total)
                    if stable_growth_rate is None: # Should be caught by all() check if strictly required
                        logging.warning(f"FAA_XAI:DCF_WARN: Missing stable_growth_rate for year {year_num}, using 0 growth for this year.")
                    else:
                        growth_rate_for_year = stable_growth_rate
                
                current_fcf *= (1 + growth_rate_for_year)
                projected_cash_flows.append(current_fcf)
            
            logging.debug(f"FAA_XAI:DCF_PROJECTED_FCF_TWO_STAGE ({fcf_projection_years_total} years): {projected_cash_flows}")

            if not projected_cash_flows: 
                logging.warning(f"FAA_XAI:DCF_ABORT: No projected cash flows generated for {company_name_for_log}.")
                return None

            terminal_value_fcf_base = projected_cash_flows[-1]
            terminal_value = (terminal_value_fcf_base * (1 + terminal_growth_rate_perpetuity)) / (discount_rate - terminal_growth_rate_perpetuity)
            logging.debug(f"FAA_XAI:DCF_TERMINAL_VALUE: TV_FCF_Base={terminal_value_fcf_base}, TV={terminal_value}")
            
            present_values = []
            for i, fcf in enumerate(projected_cash_flows):
                pv = fcf / ((1 + discount_rate) ** (i + 1))
                present_values.append(pv)

            terminal_pv = terminal_value / ((1 + discount_rate) ** fcf_projection_years_total) 
            logging.debug(f"FAA_XAI:DCF_PV_TERMINAL_VALUE: PV_TV={terminal_pv}")
            
            dcf_valuation = sum(present_values) + terminal_pv
            logging.debug(f"FAA_XAI:DCF_OUTPUT: DCF_Valuation={dcf_valuation}")

            # Use Helper Class as well for comparison or verification (as per v23.5 requirements)
            simple_val = DCFCalculator.calculate_intrinsic_value(
                discount_rate, terminal_growth_rate_perpetuity, last_historical_fcf
            )
            logging.debug(f"FAA_XAI:DCF_SIMPLE_CALC: {simple_val}")

            return dcf_valuation

        except Exception as e: 
            logging.exception(f"Error calculating DCF valuation for {company_name_for_log}: {e}")
            logging.debug(f"FAA_XAI:DCF_ERROR: Exception {e}")
            return None

    def calculate_enterprise_value(self, company_data: Dict[str, Any]) -> Optional[float]:
        company_name_for_log = company_data.get('company_info', {}).get('name', 'Unknown')
        logging.debug(f"FAA_XAI:EV_INPUT: company_id='{company_name_for_log}', company_data keys: {list(company_data.keys())}")
        try:
            financial_details = company_data.get('financial_data_detailed', {})
            if not financial_details:
                logging.warning(f"FAA_XAI:EV_ABORT: 'financial_data_detailed' missing for {company_name_for_log}.")
                return None

            market_data_info = financial_details.get('market_data', {})
            balance_sheet_info = financial_details.get('balance_sheet', {})

            share_price = market_data_info.get('share_price')
            shares_outstanding = market_data_info.get('shares_outstanding')
            logging.debug(f"FAA_XAI:EV_MARKET_DATA: SharePrice={share_price}, SharesOutstanding={shares_outstanding}")
            
            market_cap = None
            if isinstance(share_price, (int, float)) and isinstance(shares_outstanding, (int, float)):
                market_cap = share_price * shares_outstanding
            else:
                logging.warning(f"FAA_XAI:EV_WARN: Market Cap could not be calculated for {company_name_for_log} due to missing/invalid share_price or shares_outstanding.")

            short_term_debt_values = balance_sheet_info.get('short_term_debt', [0])
            short_term_debt = short_term_debt_values[-1] if short_term_debt_values else 0
            long_term_debt_values = balance_sheet_info.get('long_term_debt', [0])
            long_term_debt = long_term_debt_values[-1] if long_term_debt_values else 0
            total_debt = short_term_debt + long_term_debt

            cash_and_equivalents_values = balance_sheet_info.get('cash_and_equivalents', [0])
            cash_and_equivalents = cash_and_equivalents_values[-1] if cash_and_equivalents_values else 0
            logging.debug(f"FAA_XAI:EV_COMPONENTS: MarketCap={market_cap}, TotalDebt={total_debt}, Cash={cash_and_equivalents}")


            if market_cap is not None: 
                enterprise_value = market_cap + total_debt - cash_and_equivalents
                logging.debug(f"FAA_XAI:EV_OUTPUT: EV={enterprise_value} for {company_name_for_log}")
                return enterprise_value
            else:
                logging.warning(f"FAA_XAI:EV_ABORT: Enterprise Value cannot be calculated as Market Cap is unavailable for {company_name_for_log}.")
                return None

        except Exception as e:
            logging.exception(f"Error calculating enterprise value for {company_name_for_log}: {e}")
            logging.debug(f"FAA_XAI:EV_ERROR: Exception {e}")
            return None

    def estimate_default_likelihood(self, financial_ratios: Dict[str, float]) -> Optional[float]:
        # ... (existing code) ...
        pass

    def calculate_distressed_metrics(self, financial_statements: Dict[str, Any]) -> Dict[str, float]: # Should be financial_ratios based on usage
        # ... (existing code, note the type hint vs usage in generate_analysis_summary) ...
        pass

    def estimate_recovery_rate(self, financial_statements, default_likelihood): # Should be financial_ratios
        # ... (existing code, note the type hint vs usage in generate_analysis_summary) ...
        pass

    def send_message(self, message: Dict[str, Any]):
        # ... (existing code) ...
        pass


# Example Usage (for testing):
if __name__ == '__main__':
    # To see XAI debug logs for the example, uncomment the next line:
    # logging.getLogger().setLevel(logging.DEBUG) 
    
    agent_specific_config = {
        "persona": "Test Fundamental Analyst",
        "description": "Test instance for fundamental analysis.",
        "summarize_analysis_user_prompt": "Provide a detailed conclusion based on the findings." 
    }
    
    mock_data_package_template = {
        "company_info": {"name": "TestCompany Corp", "industry_sector": "Tech", "country": "USA"},
        "financial_data_detailed": {
            "income_statement": {"revenue": [1000, 1100, 1250], "net_income": [100, 120, 150], "ebitda": [150, 170, 200]},
            "balance_sheet": {"total_assets": [2000, 2100, 2200], "total_liabilities": [800, 850, 900], 
                              "shareholders_equity": [1200, 1250, 1300], "cash_and_equivalents": [200, 250, 300], 
                              "short_term_debt": [50,50,50], "long_term_debt": [500,450, 400]},
            "cash_flow_statement": {"operating_cash_flow": [180, 200, 230], "investing_cash_flow": [-50, -60, -70], 
                                    "financing_cash_flow": [-30, -40, -50], "free_cash_flow": [130, 140, 160]},
            "key_ratios": {"debt_to_equity_ratio": 0.58, "net_profit_margin": 0.20, "current_ratio": 2.95, "interest_coverage_ratio": 13.6},
            "dcf_assumptions": {
                "fcf_projection_years_total": 10,
                "initial_high_growth_period_years": 5,
                "initial_high_growth_rate": 0.10,
                "stable_growth_rate": 0.05,
                "discount_rate": 0.09,
                "terminal_growth_rate_perpetuity": 0.025 # Matches DataRetrievalAgent
            },
            "market_data": {"share_price": 65.00, "shares_outstanding": 10000000} 
        },
        "qualitative_company_info": {"management_assessment": "Experienced", "competitive_advantages": "Strong IP"},
        "industry_data_context": {"outlook": "Positive"},
        "economic_data_context": {"overall_outlook": "Stable"},
        "collateral_and_debt_details": {"loan_to_value_ratio": 0.6} 
    }

    async def mock_send_message(target_agent_name, message):
        # logging.info(f"MOCKED send_message to {target_agent_name} with {message}") # Already logged by AgentBase
        if target_agent_name == 'DataRetrievalAgent' and message.get('data_type') == 'get_company_financials':
            company_id = message.get('company_id')
            if company_id == "ABC_TEST": # Changed to match example in prev subtask for DRA
                data = json.loads(json.dumps(mock_data_package_template)) 
                data["company_info"]["name"] = "ABC_TEST Corp"
                return data
            elif company_id == "FAIL_TEST": 
                return None
        return None

    class MockSKFunction:
        async def invoke(self, variables=None):
            class MockSKResult:
                def __init__(self, value_str): self._value = value_str
                def __str__(self): return self._value
            return MockSKResult(f"SK Summary for {variables['company_id']}: Health {variables['financial_health']}. EV: {variables['enterprise_value_summary']}.")

    class MockSKSkillsCollection:
        def get_function(self, skill_collection_name, skill_name):
            if skill_collection_name == "FundamentalAnalysisSkill" and skill_name == "SummarizeAnalysis":
                return MockSKFunction()
            return None
        
    class MockKernel:
        def __init__(self): self.skills = MockSKSkillsCollection()
        async def run_async(self, sk_function, input_vars=None, **kwargs):
            if sk_function: return await sk_function.invoke(variables=input_vars)
            return "Mock kernel run_async failed: No function"

    async def run_tests():
        # logging.getLogger().setLevel(logging.DEBUG) # Example: Uncomment to see XAI debug logs

        mock_kernel_instance = MockKernel()
        agent_with_kernel_and_mock_a2a = FundamentalAnalystAgent(config=agent_specific_config, kernel=mock_kernel_instance)
        
        with patch.object(agent_with_kernel_and_mock_a2a, 'send_message', new=mock_send_message):
            print("\n--- Test with SK Kernel and Mocked A2A (ABC_TEST) ---")
            analysis_result_sk_abc = await agent_with_kernel_and_mock_a2a.execute("ABC_TEST")
            print(f"Analysis Summary (SK): {analysis_result_sk_abc.get('analysis_summary', 'No summary available.')}")
            print(f"Calculated EV: {analysis_result_sk_abc.get('enterprise_value')}") 
            assert "EV: Value: " in analysis_result_sk_abc.get("analysis_summary", "") 
            assert analysis_result_sk_abc.get('enterprise_value') is not None 

            print("\n--- Test with SK Kernel and Mocked A2A (FAIL_TEST for data retrieval) ---")
            analysis_result_sk_fail = await agent_with_kernel_and_mock_a2a.execute("FAIL_TEST")
            print(f"Analysis Result (FAIL_TEST): {analysis_result_sk_fail}") 
            assert "Could not retrieve data for company FAIL_TEST" in analysis_result_sk_fail.get("error", "")

    if __name__ == '__main__':
        asyncio.run(run_tests())
