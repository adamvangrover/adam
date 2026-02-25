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

        # Robustness: In-memory cache for risk assessments (simple LRU-like via limit)
        self._risk_cache = {}
        self._cache_size_limit = 100

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

        # Check Cache
        company_name = target_data.get("company_name", "UNKNOWN")
        cache_key = f"{risk_type}:{company_name}"

        if cache_key in self._risk_cache:
            # Check expiry (e.g., 5 minutes)
            cached_result, timestamp = self._risk_cache[cache_key]
            if (datetime.datetime.now() - timestamp).total_seconds() < 300:
                logger.info(f"Returning cached risk assessment for {cache_key}")
                return cached_result

        # --- v23 Update: Check for Cyclical Reasoning Graph ---
        use_graph = self.config.get("use_v23_graph", False) or context.get("use_v23_graph", False)

        if risk_type == "investment" and use_graph:
            try:
                from core.engine.cyclical_reasoning_graph import cyclical_reasoning_app
                from core.engine.states import init_risk_state

                logger.info("Delegating to v23 CyclicalReasoningGraph...")
                ticker = company_name
                intent = context.get("user_intent", f"Assess risk for {ticker}")

                initial_state = init_risk_state(ticker, intent)
                config = {"configurable": {"thread_id": "1"}}

                if hasattr(cyclical_reasoning_app, 'ainvoke'):
                    final_state = await cyclical_reasoning_app.ainvoke(initial_state, config=config)
                else:
                    final_state = cyclical_reasoning_app.invoke(initial_state, config=config)

                result = {
                    "overall_risk_score": final_state.get("quality_score", 0) * 100,
                    "risk_factors": {"Analysis": "See detailed report"},
                    "detailed_report": final_state.get("draft_analysis", ""),
                    "graph_state": final_state
                }

                # Cache Result
                self._update_cache(cache_key, result)
                return result

            except ImportError:
                logger.warning("v23 CyclicalReasoningGraph not available. Falling back to v21 logic.")
            except Exception as e:
                logger.error(f"Error executing v23 graph: {e}. Falling back to v21 logic.")

        # --- v21 Legacy Logic (Enhanced) ---
        # Note: We run these synchronously as they are CPU bound math operations,
        # but in a real system we might offload to a thread pool if they get heavy.
        if risk_type == "investment":
            result = self.assess_investment_risk(company_name, target_data.get(
                "financial_data", {}), target_data.get("market_data", {}))
        elif risk_type == "loan":
            result = self.assess_loan_risk(target_data.get("loan_details", {}), target_data.get("borrower_data", {}))
        elif risk_type == "project":
            result = self.assess_project_risk(target_data.get("project_details", {}), context)
        elif risk_type == "leveraged_finance":
            result = self.assess_leveraged_finance_risk(company_name, target_data.get("financial_data", {}), target_data.get("deal_structure", {}))
        elif risk_type == "distressed_debt":
            result = self.assess_distressed_debt_risk(company_name, target_data.get("financial_data", {}), target_data.get("market_data", {}))
        else:
            logger.warning(f"Unknown risk type: {risk_type}")
            return {"error": "Unknown risk type."}

        # --- Layer 1: Auditor Agent Integration (v26.0) ---
        # If enabled, run the Auditor to verify the assessment.
        if self.config.get("enable_auditor", False) or context.get("enable_auditor", False):
            try:
                from core.evaluation.judge import AuditorAgent
                from core.evaluation.tracing import TraceLog

                logger.info("Engaging Auditor Agent for verification...")
                auditor = AuditorAgent(mock_mode=True)
                audit_score = auditor.evaluate(target_data, result)

                # Enhance result with audit score
                result["_audit"] = audit_score.model_dump()

                # Log trace
                tracer = TraceLog(session_id=f"risk-agent-{company_name}")
                tracer.log_event("RiskAgent", "Assessment Generated", result)
                tracer.log_event("Auditor", "Evaluation", result["_audit"])
                tracer.save_trace("demo_trace.jsonl") # Append to global demo trace for visualization

            except ImportError:
                logger.warning("Auditor modules not found. Skipping verification.")
            except Exception as e:
                logger.error(f"Auditor execution failed: {e}")

        # Cache Result
        self._update_cache(cache_key, result)

        logger.info(f"Risk assessment completed. Score: {result.get('overall_risk_score', 'N/A')}")
        return result

    def _update_cache(self, key: str, value: Any) -> None:
        """Updates the cache, managing size limits."""
        if len(self._risk_cache) >= self._cache_size_limit:
            # Remove oldest (first inserted in Python 3.7+ dict)
            iterator = iter(self._risk_cache)
            try:
                del self._risk_cache[next(iterator)]
            except StopIteration:
                pass
        self._risk_cache[key] = (value, datetime.datetime.now())

    def assess_investment_risk(self, company_name: str, financial_data: Dict, market_data: Dict) -> Dict:
        """
        Assesses the risk associated with a potential investment.
        REFACTORED v26.0: Focus on Institutional Credit Risk & Distressed Debt.
        """
        logger.info(f"Assessing investment risk for {company_name}...")

        # 1. Metrics Calculation
        default_prob = self._calculate_credit_risk(financial_data)
        runway_months = self._assess_liquidity_runway(financial_data)
        recovery_rate = self._estimate_recovery_rate(financial_data)
        market_risk = self._calculate_market_risk(market_data)

        # 2. Risk Scores for Calculation (0=Safe, 1=Risky)
        # Default Prob is already 0-1

        # Runway score: < 6 months = 1.0, > 24 months = 0.0
        score_runway = max(0.0, min(1.0, (24 - runway_months) / 18)) if runway_months < 24 else 0.1

        # Recovery score: 100% recovery = 0.0 risk, 0% recovery = 1.0 risk
        score_recovery = 1.0 - recovery_rate

        risk_scores_for_calc = {
            "default_risk": default_prob,
            "liquidity_risk": score_runway,
            "recovery_risk": score_recovery,
            "market_risk": market_risk
        }

        overall_risk_score = self._calculate_overall_risk_score(risk_scores_for_calc)

        # 3. Output Factors (Raw values for the report)
        final_factors = {
            "default_probability": default_prob,
            "liquidity_runway": runway_months,
            "recovery_rate": recovery_rate,
            "market_risk_score": market_risk,
            "geopolitical_risk": self._assess_geopolitical_risks(company_name) # Keep for context
        }

        return {
            "overall_risk_score": overall_risk_score,
            "risk_factors": final_factors
        }

    def _assess_liquidity_runway(self, financial_data: Dict) -> float:
        """
        Estimates liquidity runway in months.
        """
        cash = financial_data.get("cash", 0)
        burn_rate = financial_data.get("monthly_burn_rate", 0)

        # Fallback if burn rate not available: use liquidity ratio
        if burn_rate == 0:
            l_ratio = financial_data.get("liquidity_ratio", 0)
            if l_ratio > 1.5: return 24.0
            if l_ratio > 1.0: return 12.0
            return 6.0

        if burn_rate < 0:
             # Net positive cash flow
             return 999.0

        return cash / burn_rate

    def _estimate_recovery_rate(self, financial_data: Dict) -> float:
        """
        Estimates recovery rate based on Asset Coverage.
        Recovery = Total Assets / Total Debt (Capped at 1.0)
        """
        assets = financial_data.get("total_assets", 0)
        debt = financial_data.get("total_debt", 1) # Avoid div/0

        if debt == 0:
            return 1.0

        coverage = assets / debt
        # Simple heuristic: Senior secured often gets 70-100% if coverage > 1.
        # Unsecured gets less. We'll use a conservative estimate (80% of coverage).
        return min(1.0, coverage * 0.8)

    def assess_loan_risk(self, loan_details: Dict, borrower_data: Dict) -> Dict:
        """
        Assesses the risk associated with a loan.
        """
        logger.info(f"Assessing loan risk...")
        risk_factors = {}

        pd = self._calculate_credit_risk(borrower_data)
        risk_factors["credit_risk"] = pd
        risk_factors["liquidity_risk"] = self._assess_borrower_liquidity(borrower_data)

        # Extract inputs for LGD/RWA
        seniority = loan_details.get("seniority", "Senior Unsecured")
        collateral_value = loan_details.get("collateral_value", 0)
        loan_amount = loan_details.get("loan_amount", 1)
        collateral_coverage = collateral_value / loan_amount if loan_amount > 0 else 0

        risk_factors["collateral_risk"] = self._assess_collateral_risk(loan_details)
        risk_factors["economic_risk"] = self._assess_economic_risk()
        risk_factors["interest_rate_risk"] = self._assess_interest_rate_risk(loan_details)

        overall_risk_score = self._calculate_overall_risk_score(risk_factors)

        # --- Quantitative Metrics (New) ---
        lgd = self.calculate_lgd(seniority, collateral_coverage)

        interest_rate = loan_details.get("interest_rate", 0.05) # Default 5%
        if isinstance(interest_rate, str) and '%' in interest_rate:
             try:
                 interest_rate = float(interest_rate.strip('%')) / 100
             except:
                 interest_rate = 0.05

        rwa = self.calculate_rwa(pd, lgd, loan_amount)
        # Ratio for RAROC calc
        rwa_ratio = rwa / loan_amount if loan_amount > 0 else 1.0

        expected_return = self.calculate_expected_return(interest_rate, pd, lgd, capital_cost=0.10, rwa_ratio=rwa_ratio)

        return {
            "overall_risk_score": overall_risk_score,
            "risk_factors": risk_factors,
            "risk_quant_metrics": {
                "PD": pd,
                "LGD": lgd,
                "EAD": loan_amount,
                "RWA": rwa,
                "Expected_Loss": pd * lgd * loan_amount,
                "RAROC": expected_return
            }
        }

    def assess_leveraged_finance_risk(self, company_name: str, financial_data: Dict, deal_structure: Dict) -> Dict:
        """
        Specialized risk assessment for Leveraged Finance (LBOs, M&A financing).
        Focuses on Cash Flow, De-leveraging ability, and Covenant headroom.
        """
        logger.info(f"Assessing LevFin risk for {company_name}...")

        # 1. Key Metrics
        ebitda = financial_data.get("ebitda", 1.0)
        total_debt = financial_data.get("total_debt", 0.0)
        cash_flow_available_for_debt = financial_data.get("fcf", 0.0)
        interest_expense = financial_data.get("interest_expense", 1.0)

        # Ratios
        if ebitda > 0:
            leverage_ratio = total_debt / ebitda
        else:
            # Negative or zero EBITDA implies extremely high leverage risk
            leverage_ratio = 99.0

        interest_coverage = ebitda / interest_expense if interest_expense else 99.0
        dscr = cash_flow_available_for_debt / interest_expense if interest_expense else 0.0 # Debt Service Coverage

        # 2. Risk Scoring (Institutional Grade thresholds)
        # Leverage: > 6.0x is High Risk (Speculative), < 3.0x is Low Risk
        # If leverage_ratio is 99.0 (distressed), risk is 1.0
        risk_leverage = min(1.0, max(0.0, (leverage_ratio - 3.0) / 4.0))

        # Interest Coverage: < 2.0x is High Risk, > 5.0x is Low Risk
        risk_coverage = min(1.0, max(0.0, (5.0 - interest_coverage) / 3.0))

        # DSCR: < 1.0 is Critical Risk, > 1.5 is Safe
        risk_dscr = min(1.0, max(0.0, (1.5 - dscr) / 0.5))

        # 3. Structural Risk (Covenants)
        covenant_cushion = deal_structure.get("covenant_cushion", 0.15) # 15% default
        risk_covenant = min(1.0, max(0.0, (0.25 - covenant_cushion) / 0.25))

        risk_factors = {
            "leverage_risk": risk_leverage,
            "coverage_risk": risk_coverage,
            "cash_flow_risk": risk_dscr,
            "structural_risk": risk_covenant
        }

        overall_score = self._calculate_overall_risk_score(risk_factors)

        return {
            "overall_risk_score": overall_score,
            "risk_factors": risk_factors,
            "metrics": {
                "leverage_ratio": leverage_ratio,
                "interest_coverage": interest_coverage,
                "dscr": dscr
            }
        }

    def assess_distressed_debt_risk(self, company_name: str, financial_data: Dict, market_data: Dict) -> Dict:
        """
        Specialized assessment for Distressed Debt / Special Situations.
        Focuses on Recovery Rates, Liquidation Value, and Legal/Process risk.
        """
        logger.info(f"Assessing Distressed Debt risk for {company_name}...")

        # 1. Asset Coverage & Liquidation Value
        total_assets = financial_data.get("total_assets", 0.0)
        secured_debt = financial_data.get("secured_debt", 0.0)
        unsecured_debt = financial_data.get("unsecured_debt", 0.0)

        # Haircuts for liquidation (Simulated)
        # Cash: 100%, Receivables: 75%, Inventory: 50%, PP&E: 40%, Intangibles: 0%
        # Simplified: weighted average haircut of 50% on total assets
        liquidation_value = total_assets * 0.5

        # 2. Recovery Waterfall
        remaining_value = liquidation_value

        # Senior / Secured
        if secured_debt > 0:
            secured_recovery = min(1.0, remaining_value / secured_debt)
            remaining_value = max(0.0, remaining_value - secured_debt)
        else:
            secured_recovery = 1.0 # No secured debt

        # Unsecured
        if unsecured_debt > 0:
            unsecured_recovery = min(1.0, remaining_value / unsecured_debt)
        else:
            unsecured_recovery = 1.0 if remaining_value > 0 else 0.0

        # 3. Market Pricing vs Intrinsic Recovery
        # If bond trades at 40c and we model 60c recovery, that's "Low Risk" (Opportunity)
        # If bond trades at 80c and we model 60c recovery, that's "High Risk"
        market_price = market_data.get("bond_price", 0.5) # % of par
        model_recovery = unsecured_recovery # Assuming we look at unsecured for distressed

        upside_cushion = model_recovery - market_price
        # Risk score: High if Price > Recovery (Negative cushion)
        # Score 0.5 is neutral (Fairly priced).
        # Score 1.0 is Overpriced (Risk of loss).
        # Score 0.0 is Underpriced (Margin of safety).
        valuation_risk = min(1.0, max(0.0, 0.5 - upside_cushion)) # Inverted logic

        return {
            "overall_risk_score": valuation_risk,
            "recovery_analysis": {
                "liquidation_value": liquidation_value,
                "secured_recovery": secured_recovery,
                "unsecured_recovery": unsecured_recovery
            },
            "risk_factors": {
                "valuation_risk": valuation_risk,
                "liquidation_risk": 1.0 - unsecured_recovery
            }
        }

    def calculate_lgd(self, seniority: str, collateral_coverage: float) -> float:
        """
        Estimates Loss Given Default (LGD).
        """
        # Base LGD by seniority
        base_lgd = {
            "Senior Secured": 0.35,
            "Senior Unsecured": 0.50,
            "Subordinated": 0.75,
            "Equity": 1.00
        }.get(seniority, 0.50)

        # Adjust for collateral
        # LGD = Base * (1 - min(1, collateral_coverage))
        # We'll assume a minimum floor of 5% even with full collateral
        adjusted_lgd = base_lgd * (1 - min(1.0, collateral_coverage))
        return max(0.05, adjusted_lgd)

    def calculate_rwa(self, pd: float, lgd: float, ead: float) -> float:
        """
        Estimates Risk-Weighted Assets (RWA) using a simplified heuristic.
        """
        # Heuristic: RW = 20% + (PD * 20) + (LGD * 1.5)
        # This is a mock function for demonstration purposes
        rw_percentage = 0.20 + (pd * 20) + (lgd * 1.5)
        rw_percentage = min(15.0, max(0.0, rw_percentage)) # Cap at 1500%

        rwa = ead * rw_percentage
        return rwa

    def calculate_expected_return(self, interest_rate: float, pd: float, lgd: float, capital_cost: float, rwa_ratio: float = 1.0) -> float:
        """
        Calculates Risk-Adjusted Return on Capital (RAROC).
        """
        expected_loss = pd * lgd
        risk_adjusted_income = interest_rate - expected_loss

        economic_capital = rwa_ratio * 0.08 # Assumes Basel standard 8% capital

        if economic_capital <= 0:
            return 0.0

        raroc = (risk_adjusted_income - (economic_capital * capital_cost)) / economic_capital
        return raroc

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