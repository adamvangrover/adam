# core/analysis/risk_assessment.py

import numpy as np
from scipy.stats import norm
import pandas as pd  # For data manipulation
from typing import Dict, Any, Tuple, List  # For type hinting
from datetime import datetime

class RiskAssessor:
    """
    Comprehensive risk assessment tool for investments, integrated with Adam v19.1's agent framework and knowledge graph.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the RiskAssessor with configuration parameters.
        """
        self.data_sources = config.get('data_sources', {})
        self.geopolitical_risk_agent = config.get('geopolitical_risk_agent', None)
        self.industry_specialist_agent = config.get('industry_specialist_agent', None)
        self.prediction_market_agent = config.get('prediction_market_agent', None) # Integrate prediction market agent
        self.macroeconomic_analysis_agent = config.get('macroeconomic_analysis_agent', None) # Integrate macroeconomic agent
        self.alternative_data_agent = config.get('alternative_data_agent', None) # Integrate alternative data agent
        self.risk_weights = config.get('risk_weights', {
            'market_risk': 0.2,
            'credit_risk': 0.3,
            'liquidity_risk': 0.15,
            'operational_risk': 0.1,
            'geopolitical_risk': 0.15,
            'industry_risk': 0.1
        })  # Weights for risk factors
        self.default_probability_mapping = {
            "AAA": 0.0001, "AA+": 0.0002, "AA": 0.0003, "AA-": 0.0005,
            "A+": 0.0008, "A": 0.0012, "A-": 0.0018, "BBB+": 0.0027,
            "BBB": 0.0040, "BBB-": 0.0060, "BB+": 0.0090, "BB": 0.0130,
            "BB-": 0.0190, "B+": 0.0270, "B": 0.0380, "B-": 0.0530,
            "CCC+": 0.0750, "CCC": 0.1000, "CCC-": 0.1500, "CC": 0.2000,
            "C": 0.3000, "D": 1.0000,
        }

    def assess_risk(self, investment: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Performs a comprehensive risk assessment of an investment.

        Args:
            investment: Dictionary containing investment information.

        Returns:
            A tuple containing a dictionary of risk factors and an overall risk score.
        """
        print(f"Assessing risk for {investment.get('name', 'Unknown Investment')}...")
        risk_factors = {}

        # 1. Market Risk (Enhanced)
        if 'price_data' in investment and investment['price_data']:
            prices = np.array(investment['price_data'])
            returns = np.log(prices[1:] / prices[:-1])
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            beta = self._calculate_beta(returns)
            risk_factors['market_risk'] = {'volatility': volatility, 'beta': beta}
            print(f"Market Risk - Volatility: {volatility:.2f}, Beta: {beta:.2f}")

        # 2. Credit Risk (Using Mapping)
        if 'credit_rating' in investment:
            credit_rating = investment['credit_rating'].upper()
            default_probability = self.default_probability_mapping.get(credit_rating, 0.1)
            risk_factors['credit_risk'] = default_probability
            print(f"Credit Risk - Default Probability: {default_probability:.4f}")

        # 3. Liquidity Risk (Enhanced)
        liquidity_score = self._calculate_liquidity_score(investment)
        risk_factors['liquidity_risk'] = liquidity_score
        print(f"Liquidity Risk - Score: {liquidity_score:.2f}")

        # 4. Operational Risk (Qualitative with Scoring)
        operational_risk = self._assess_operational_risk(investment)
        risk_factors['operational_risk'] = operational_risk
        print(f"Operational Risk - Assessment: {operational_risk}")

        # 5. Geopolitical Risk (Agent Integration)
        if self.geopolitical_risk_agent:
            geopolitical_risks = self.geopolitical_risk_agent.assess_geopolitical_risks()
            relevant_risks = self._filter_relevant_geopolitical_risks(investment, geopolitical_risks)
            risk_factors['geopolitical_risk'] = relevant_risks
            print(f"Geopolitical Risk - Relevant Risks: {relevant_risks}")

        # 6. Industry-Specific Risk (Agent Integration)
        if self.industry_specialist_agent:
            industry_risk = self.industry_specialist_agent.analyze_industry_risk(investment)
            risk_factors['industry_risk'] = industry_risk
            print(f"Industry Risk - Assessment: {industry_risk}")

        # 7. Overall Risk Assessment (Weighted)
        overall_risk_score = self._calculate_overall_risk_score(risk_factors)
        print(f"Overall Risk Score: {overall_risk_score:.2f}")

        return risk_factors, overall_risk_score

    def _calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray = None) -> float:
        """
        Calculates the beta of an asset using its returns and market returns.
        """
        if market_returns is None:
            # Placeholder: fetch market returns from data sources.
            market_returns = np.random.normal(0, 0.01, len(asset_returns))  # replace with real market returns
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        variance = np.var(market_returns)
        return covariance / variance if variance != 0 else 0.0

    def _calculate_liquidity_score(self, investment: Dict[str, Any]) -> float:
        """
        Calculates a liquidity score based on trading volume, bid-ask spread, etc.
        """
        # Placeholder: Implement logic to calculate liquidity score.
        # This could involve fetching real-time market data.
        return np.random.rand()

    def _assess_operational_risk(self, investment: Dict[str, Any]) -> str:
        """
        Qualitative assessment of operational risk.
        """
        # Placeholder: Implement logic to assess operational risk.
        # This could involve analyzing company filings, news, and other qualitative data.
        return "Medium"

    def _filter_relevant_geopolitical_risks(self, investment: Dict[str, Any], geopolitical_risks: List[str]) -> List[str]:
        """
        Filters and assesses relevant geopolitical risks for the investment.
        """
        # Placeholder: Implement logic to filter relevant risks.
        # This could involve matching keywords or using a more sophisticated NLP model.
        return geopolitical_risks[:2] if geopolitical_risks else []

    def _calculate_overall_risk_score(self, risk_factors: Dict[str, Any]) -> float:
        """
        Combines individual risk factors into an overall risk score using weights.
        """
        weighted_scores = []
        for risk_type, risk_value in risk_factors.items():
            weight = self.risk_weights.get(risk_type, 0)
            if isinstance(risk_value, dict):
                # if risk factor is a dictionary, average the values.
                score = np.mean(list(risk_value.values()))
            else:
                score = risk_value
            weighted_scores.append(score * weight)
        return sum(weighted_scores)

    # Enhanced risk assessment methods
    def _generate_probability_weighted_scenarios(self, investment: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Generates probability-weighted scenarios by combining macroeconomic forecasts, geopolitical risks, 
        and prediction market data.
        """
        scenarios = []
        # 1. Fetch macroeconomic scenarios from macroeconomic_analysis_agent
        macro_scenarios = self.macroeconomic_analysis_agent.generate_scenarios() if self.macroeconomic_analysis_agent else []
        # 2. Fetch geopolitical risks from geopolitical_risk_agent
        geopolitical_risks = self.geopolitical_risk_agent.assess_geopolitical_risks() if self.geopolitical_risk_agent else []
        # 3. Combine scenarios and risks
        for macro_scenario in macro_scenarios:
            for risk in geopolitical_risks:
                scenario_description = f"{macro_scenario} with {risk}"
                # 4. Fetch probability from prediction_market_agent
                probability = self.prediction_market_agent.get_scenario_probability(scenario_description) if self.prediction_market_agent else np.random.rand()
                scenarios.append((scenario_description, probability))
        return scenarios

    def _identify_early_warning_signals(self, investment: Dict[str, Any]) -> List[str]:
        """
        Identifies early warning signals of potential risks by analyzing alternative data sources.
        """
        signals = []
        # 1. Fetch relevant data from alternative_data_agent
        alternative_data = self.alternative_data_agent.get_data_for_asset(investment) if self.alternative_data_agent else {}
        # 2. Analyze data for signals (example - social media sentiment)
        if 'social_media_sentiment' in alternative_data:
            sentiment = alternative_data['social_media_sentiment']
            if sentiment < 0.2:
                signals.append("Negative social media sentiment detected.")
        # ... Add more logic for other alternative data sources
        return signals

    def _generate_risk_mitigation_strategies(self, investment: Dict[str, Any], risk_factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates risk mitigation strategies based on identified risk factors.
        """
        mitigation_strategies = {}
        # 1. Analyze risk factors and generate strategies
        if risk_factors.get('market_risk', {}).get('volatility') > 0.3:
            mitigation_strategies['market_risk'] = "Consider hedging strategies or diversification."
        if risk_factors.get('credit_risk') > 0.05:
            mitigation_strategies['credit_risk'] = "Monitor credit rating closely and consider credit default swaps."
        # ... Add more logic for other risk factors
        return mitigation_strategies
