from __future__ import annotations
from typing import Any, Dict, Optional, List, Union
import logging
import json
import numpy as np
import datetime
import asyncio
from core.agents.agent_base import AgentBase

# Quantitative Rigor: Import Scipy for probability distributions
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RiskAssessmentAgent(AgentBase):
    """
    Agent responsible for assessing various types of investment risks,
    such as market risk, credit risk, and operational risk.

    Philosophy:
    Risk is not a number; it's a distribution. We strive to quantify the tails.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the Risk Assessment Agent.

        Args:
            config: A dictionary containing configuration parameters.
                    Expected keys: 'knowledge_base_path' (optional)
        """
        super().__init__(config, **kwargs)
        self.knowledge_base_path = self.config.get("knowledge_base_path", "data/risk_rating_mapping.json")
        self.knowledge_base = self._load_knowledge_base()
        self.debug_mode = self.knowledge_base.get("metadata", {}).get("debug_mode", False)

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """
        Loads the knowledge base from the JSON file.

        Returns:
            dict: The knowledge base data.
        """
        try:
            with open(self.knowledge_base_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"Knowledge base file not found: {self.knowledge_base_path}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding knowledge base JSON: {self.knowledge_base_path}")
            return {}

    async def execute(self, target_data: Dict[str, Any], risk_type: str = "investment", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the risk assessment.

        Args:
            target_data (dict): Data related to the target being assessed.
            risk_type (str): Type of risk assessment (e.g., "investment", "loan", "project").
            context (dict): Additional context for the risk assessment.

        Returns:
            dict: Risk assessment results.
        """
        logger.info(f"Starting risk assessment for type: {risk_type}")

        if context is None:
            context = {}

        # --- v23 Update: Check for Cyclical Reasoning Graph ---
        use_graph = self.config.get("use_v23_graph", False) or context.get("use_v23_graph", False)

        if risk_type == "investment" and use_graph:
            try:
                from core.engine.cyclical_reasoning_graph import cyclical_reasoning_app
                from core.engine.states import init_risk_state

                logger.info("Delegating to v23 CyclicalReasoningGraph...")
                ticker = target_data.get("company_name", "UNKNOWN")
                intent = context.get("user_intent", f"Assess risk for {ticker}")

                initial_state = init_risk_state(ticker, intent)
                config = {"configurable": {"thread_id": "1"}}

                if hasattr(cyclical_reasoning_app, 'ainvoke'):
                    final_state = await cyclical_reasoning_app.ainvoke(initial_state, config=config)
                else:
                    final_state = cyclical_reasoning_app.invoke(initial_state, config=config)

                return {
                    "overall_risk_score": final_state.get("quality_score", 0) * 100,
                    "risk_factors": {"Analysis": "See detailed report"},
                    "detailed_report": final_state.get("draft_analysis", ""),
                    "graph_state": final_state
                }

            except ImportError:
                logger.warning("v23 CyclicalReasoningGraph not available. Falling back to v21 logic.")
            except Exception as e:
                logger.error(f"Error executing v23 graph: {e}. Falling back to v21 logic.")

        # --- v21 Legacy Logic (Enhanced) ---
        # Note: We run these synchronously as they are CPU bound math operations,
        # but in a real system we might offload to a thread pool if they get heavy.
        if risk_type == "investment":
            result = self.assess_investment_risk(target_data.get("company_name"), target_data.get(
                "financial_data", {}), target_data.get("market_data", {}))
        elif risk_type == "loan":
            result = self.assess_loan_risk(target_data.get("loan_details", {}), target_data.get("borrower_data", {}))
        elif risk_type == "project":
            result = self.assess_project_risk(target_data.get("project_details", {}), context)
        else:
            logger.warning(f"Unknown risk type: {risk_type}")
            return {"error": "Unknown risk type."}

        logger.info(f"Risk assessment completed. Score: {result.get('overall_risk_score', 'N/A')}")
        return result

    def assess_investment_risk(self, company_name: str, financial_data: Dict, market_data: Dict) -> Dict:
        """
        Assesses the risk associated with a potential investment.
        """
        logger.info(f"Assessing investment risk for {company_name}...")
        risk_factors = {}

        risk_factors["market_risk"] = self._calculate_market_risk(market_data)
        risk_factors["credit_risk"] = self._calculate_credit_risk(financial_data)
        risk_factors["liquidity_risk"] = self._calculate_liquidity_risk(market_data)
        risk_factors["operational_risk"] = self._assess_operational_risk(company_name)
        risk_factors["geopolitical_risk"] = self._assess_geopolitical_risks(company_name)

        industry = financial_data.get("industry", "Unknown")
        risk_factors["industry_risk"] = self._assess_industry_risk(industry)

        risk_factors["economic_risk"] = self._assess_economic_risk()
        risk_factors["volatility_risk"] = self._assess_volatility_risk()
        risk_factors["currency_risk"] = self._assess_currency_risk()

        overall_risk_score = self._calculate_overall_risk_score(risk_factors)

        return {
            "overall_risk_score": overall_risk_score,
            "risk_factors": risk_factors
        }

    def assess_loan_risk(self, loan_details: Dict, borrower_data: Dict) -> Dict:
        """
        Assesses the risk associated with a loan.
        """
        logger.info(f"Assessing loan risk...")
        risk_factors = {}

        risk_factors["credit_risk"] = self._calculate_credit_risk(borrower_data)
        risk_factors["liquidity_risk"] = self._assess_borrower_liquidity(borrower_data)
        risk_factors["collateral_risk"] = self._assess_collateral_risk(loan_details)
        risk_factors["economic_risk"] = self._assess_economic_risk()
        risk_factors["interest_rate_risk"] = self._assess_interest_rate_risk(loan_details)

        overall_risk_score = self._calculate_overall_risk_score(risk_factors)

        return {
            "overall_risk_score": overall_risk_score,
            "risk_factors": risk_factors
        }

    def assess_project_risk(self, project_details: Dict, context: Dict) -> Dict:
        """
        Assesses the risk associated with a project.
        """
        logger.info(f"Assessing project risk...")
        risk_factors = {}

        risk_factors["project_management_risk"] = self._assess_project_management_risk(project_details)
        risk_factors["technical_risk"] = self._assess_technical_risk(project_details)
        risk_factors["market_risk"] = self._assess_project_market_risk(project_details, context)
        risk_factors["financial_risk"] = self._assess_project_financial_risk(project_details)
        risk_factors["regulatory_risk"] = self._assess_regulatory_risk(project_details)

        overall_risk_score = self._calculate_overall_risk_score(risk_factors)

        return {
            "overall_risk_score": overall_risk_score,
            "risk_factors": risk_factors
        }

    def _calculate_market_risk(self, market_data: Dict) -> float:
        """
        Calculates market risk using Volatility or Parametric VaR (Value at Risk).
        """
        if "price_data" in market_data:
            prices = np.array(market_data["price_data"])  # Ensure numpy array
            if len(prices) > 1:
                returns = np.log(prices[1:] / prices[:-1])
                std_dev = np.std(returns)
                volatility = std_dev * np.sqrt(252)

                # Quantitative Rigor: Parametric VaR (95%)
                if SCIPY_AVAILABLE:
                    # VaR = z_score * std_dev * sqrt(T)
                    # We assume 1 day horizon for this metric
                    z_score = norm.ppf(0.95)
                    var_95 = z_score * std_dev
                    logger.debug(f"Volatility: {volatility:.2f}, 1-Day VaR(95%): {var_95:.4f}")
                    # Normalize VaR to 0-1 scale for risk scoring (assuming > 5% daily drop is catastrophic)
                    return min(1.0, var_95 * 20)

                return min(1.0, float(volatility))
        return 0.2

    def _calculate_credit_risk(self, financial_data: Dict) -> float:
        credit_rating = financial_data.get("credit_rating")
        if credit_rating:
            return self._estimate_default_probability(credit_rating)
        return 0.1

    def _calculate_liquidity_risk(self, market_data: Dict) -> float:
        trading_volume = market_data.get("trading_volume", 0)
        return self._assess_liquidity(trading_volume)

    def _estimate_default_probability(self, credit_rating: str) -> float:
        credit_ratings = self.knowledge_base.get("credit_ratings", {})
        for rating_system in credit_ratings.values():
            if credit_rating in rating_system:
                return rating_system[credit_rating].get("default_probability", 0.3)
        return 0.3

    def _assess_liquidity(self, trading_volume: int) -> float:
        # Logarithmic scaling for better differentiation at low volumes
        if trading_volume <= 0:
            return 1.0

        # Assume 1M is "High Liquidity" (Risk = 0.1)
        # Assume 10k is "Low Liquidity" (Risk = 0.9)
        # Risk = 1 / log10(volume) roughly

        log_vol = np.log10(trading_volume)
        if log_vol >= 6:  # 1M
            return 0.1
        elif log_vol >= 5:  # 100k
            return 0.2
        elif log_vol >= 4:  # 10k
            return 0.5
        else:
            return 0.8

    def _assess_operational_risk(self, company_name: str) -> str:
        company_size = self.knowledge_base.get("companies", {}).get(company_name, {}).get("size", "Medium")
        return "Medium" if company_size == "Large" else "Low"

    def _assess_geopolitical_risks(self, company_name: str) -> List[str]:
        company_location = self.knowledge_base.get("companies", {}).get(company_name, {}).get("location", "US")
        if company_location == "US":
            return ["Trade tensions with China"]
        else:
            return ["Political instability in emerging markets"]

    def _assess_industry_risk(self, industry: str) -> str:
        if industry == "Technology":
            return "Medium"
        elif industry == "Financials":
            return "High"
        else:
            return "Low"

    def _calculate_overall_risk_score(self, risk_factors: Dict) -> float:
        weights = self.knowledge_base.get("risk_weights", {})
        weighted_scores = []

        for factor, score in risk_factors.items():
            weight = weights.get(factor, 1)

            numeric_score = 0.0
            if isinstance(score, (int, float)):
                numeric_score = float(score)
            elif isinstance(score, str):
                if score == "Low":
                    numeric_score = 0.1
                elif score == "Medium":
                    numeric_score = 0.5
                elif score == "High":
                    numeric_score = 0.9
            elif isinstance(score, list):
                # Heuristic: List implies presence of risk factors
                numeric_score = 0.5 if len(score) > 0 else 0.1

            weighted_scores.append(numeric_score * weight)

        # Normalization handled by weights if they sum to N, but here we just average
        # Ideally weights sum to 1. If not, we divide by sum of weights
        total_weight = sum([weights.get(f, 1) for f in risk_factors.keys()])

        if total_weight > 0:
            return sum(weighted_scores) / total_weight
        else:
            return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

    def _assess_economic_risk(self) -> float:
        economic_data = self.knowledge_base.get("economic_data", {})
        if not economic_data:
            return 0.5
        usa_data = economic_data.get("USA", {})
        gdp_growth = usa_data.get("GDP_growth", 0)
        inflation = usa_data.get("inflation", 0)
        unemployment = usa_data.get("unemployment", 0)

        # Stagflation Index
        risk = (1 - gdp_growth) + inflation + unemployment
        # Clip to 0-1
        return max(0.0, min(1.0, risk / 3))

    def _assess_volatility_risk(self) -> float:
        volatility_indices = self.knowledge_base.get("volatility_indices", {})
        if not volatility_indices:
            return 0.5
        vix = volatility_indices.get("VIX", {}).get("value", 20) / 40
        vxn = volatility_indices.get("VXN", {}).get("value", 25) / 40
        return min(1.0, (vix + vxn) / 2)

    def _assess_currency_risk(self) -> float:
        currency_par_values = self.knowledge_base.get("currency_par_values", {})
        if not currency_par_values:
            return 0.5
        usd_eur = abs(1 - currency_par_values.get("USD/EUR", 0.85))
        usd_jpy = abs(1 - currency_par_values.get("USD/JPY", 145) / 145)
        return min(1.0, (usd_eur + usd_jpy) / 2)

    def _assess_borrower_liquidity(self, borrower_data: Dict) -> float:
        liquidity_ratio = borrower_data.get("liquidity_ratio", 1)
        return 1 / liquidity_ratio if liquidity_ratio > 0 else 1.0

    def _assess_collateral_risk(self, loan_details: Dict) -> float:
        collateral_value = loan_details.get("collateral_value", 0)
        loan_amount = loan_details.get("loan_amount", 1)
        return max(0.0, 1 - (collateral_value / loan_amount)) if loan_amount > 0 else 1.0

    def _assess_interest_rate_risk(self, loan_details: Dict) -> float:
        interest_rate_type = loan_details.get("interest_rate_type", "fixed")
        return 0.6 if interest_rate_type == "variable" else 0.2

    def _assess_project_management_risk(self, project_details: Dict) -> float:
        project_manager_experience = project_details.get("project_manager_experience", "Medium")
        if project_manager_experience == "High":
            return 0.2
        elif project_manager_experience == "Medium":
            return 0.5
        else:
            return 0.8

    def _assess_technical_risk(self, project_details: Dict) -> float:
        technology_maturity = project_details.get("technology_maturity", "Medium")
        if technology_maturity == "Mature":
            return 0.2
        elif technology_maturity == "Established":
            return 0.5
        else:
            return 0.8

    def _assess_project_market_risk(self, project_details: Dict, context: Dict) -> float:
        market_demand = context.get("market_demand", "Medium")
        if market_demand == "High":
            return 0.2
        elif market_demand == "Medium":
            return 0.5
        else:
            return 0.8

    def _assess_project_financial_risk(self, project_details: Dict) -> float:
        project_budget = project_details.get("project_budget", 1000000)
        funding_secured = project_details.get("funding_secured", 500000)
        return max(0.0, 1 - (funding_secured / project_budget)) if project_budget > 0 else 1.0

    def _assess_regulatory_risk(self, project_details: Dict) -> float:
        regulatory_environment = project_details.get("regulatory_environment", "Medium")
        if regulatory_environment == "Favorable":
            return 0.2
        elif regulatory_environment == "Neutral":
            return 0.5
        else:
            return 0.8
