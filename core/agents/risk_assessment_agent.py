#core/agents/risk_assessment_agent.py

import numpy as np
from scipy.stats import norm
import json
import datetime

class RiskAssessmentAgent:
    def __init__(self, knowledge_base_path="risk_rating_mapping.json"):
        """
        Initializes the Risk Assessment Agent.

        Args:
            knowledge_base_path (str): Path to the knowledge base file.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()
        self.debug_mode = self.knowledge_base.get("metadata", {}).get("debug_mode", False)

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

    def assess_risk(self, target_data, risk_type="investment", context=None):
        """
        Assesses the risk based on the provided target data and context.

        Args:
            target_data (dict): Data related to the target being assessed.
            risk_type (str): Type of risk assessment (e.g., "investment", "loan", "project").
            context (dict): Additional context for the risk assessment.

        Returns:
            dict: Risk assessment results.
        """
        if risk_type == "investment":
            return self.assess_investment_risk(target_data.get("company_name"), target_data.get("financial_data", {}), target_data.get("market_data", {}))
        elif risk_type == "loan":
            return self.assess_loan_risk(target_data.get("loan_details", {}), target_data.get("borrower_data", {}))
        elif risk_type == "project":
            return self.assess_project_risk(target_data.get("project_details", {}), context)
        else:
            return {"error": "Unknown risk type."}

    def assess_investment_risk(self, company_name, financial_data, market_data):
        """
        Assesses the risk associated with a potential investment.
        """
        print(f"Assessing investment risk for {company_name}...")
        risk_factors = {}

        # 1. Market Risk
        risk_factors["market_risk"] = self._calculate_market_risk(market_data)

        # 2. Credit Risk
        risk_factors["credit_risk"] = self._calculate_credit_risk(financial_data)

        # 3. Liquidity Risk
        risk_factors["liquidity_risk"] = self._calculate_liquidity_risk(market_data)

        # 4. Operational Risk
        risk_factors["operational_risk"] = self._assess_operational_risk(company_name)

        # 5. Geopolitical Risk
        risk_factors["geopolitical_risk"] = self._assess_geopolitical_risks(company_name)

        # 6. Industry-Specific Risk
        industry = financial_data.get("industry", "Unknown")
        risk_factors["industry_risk"] = self._assess_industry_risk(industry)

        # 7. Economic Risk
        risk_factors["economic_risk"] = self._assess_economic_risk()

        # 8. Volatility Risk
        risk_factors["volatility_risk"] = self._assess_volatility_risk()

        # 9. Currency Risk
        risk_factors["currency_risk"] = self._assess_currency_risk()

        # 10. Overall Risk Assessment
        overall_risk_score = self._calculate_overall_risk_score(risk_factors)
        print(f"Overall Risk Score: {overall_risk_score:.2f}")

        return {
            "overall_risk_score": overall_risk_score,
            "risk_factors": risk_factors
        }

    def assess_loan_risk(self, loan_details, borrower_data):
        """
        Assesses the risk associated with a loan.
        """
        print(f"Assessing loan risk...")
        risk_factors = {}

        # 1. Credit Risk
        risk_factors["credit_risk"] = self._calculate_credit_risk(borrower_data)

        # 2. Liquidity Risk
        risk_factors["liquidity_risk"] = self._assess_borrower_liquidity(borrower_data)

        # 3. Collateral Risk
        risk_factors["collateral_risk"] = self._assess_collateral_risk(loan_details)

        # 4. Economic Risk
        risk_factors["economic_risk"] = self._assess_economic_risk()

        # 5. Interest Rate Risk
        risk_factors["interest_rate_risk"] = self._assess_interest_rate_risk(loan_details)

        # 6. Overall Risk Assessment
        overall_risk_score = self._calculate_overall_risk_score(risk_factors)
        print(f"Overall Loan Risk Score: {overall_risk_score:.2f}")

        return {
            "overall_risk_score": overall_risk_score,
            "risk_factors": risk_factors
        }

    def assess_project_risk(self, project_details, context):
        """
        Assesses the risk associated with a project.
        """
        print(f"Assessing project risk...")
        risk_factors = {}

        # 1. Project Management Risk
        risk_factors["project_management_risk"] = self._assess_project_management_risk(project_details)

        # 2. Technical Risk
        risk_factors["technical_risk"] = self._assess_technical_risk(project_details)

        # 3. Market Risk
        risk_factors["market_risk"] = self._assess_project_market_risk(project_details, context)

        # 4. Financial Risk
        risk_factors["financial_risk"] = self._assess_project_financial_risk(project_details)

        # 5. Regulatory Risk
        risk_factors["regulatory_risk"] = self._assess_regulatory_risk(project_details)

        # 6. Overall Risk Assessment
        overall_risk_score = self._calculate_overall_risk_score(risk_factors)
        print(f"Overall Project Risk Score: {overall_risk_score:.2f}")

        return {
            "overall_risk_score": overall_risk_score,
            "risk_factors": risk_factors
        }

    def _calculate_market_risk(self, market_data):
        """Calculates market risk."""
        if "price_data" in market_data:
            prices = market_data["price_data"]
            returns = np.log(prices[1:] / prices[:-1])
            volatility = np.std(returns) * np.sqrt(252)
            print(f"Volatility: {volatility:.2f}")
            return volatility
        return 0.2  # Default value

    def _calculate_credit_risk(self, financial_data):
        """Calculates credit risk."""
        credit_rating = financial_data.get("credit_rating")
        if credit_rating:
            default_probability = self._estimate_default_probability(credit_rating)
            print(f"Default Probability: {default_probability:.2f}")
            return default_probability
        return 0.1  # Default value

    def _calculate_liquidity_risk(self, market_data):
        """Calculates liquidity risk."""
        trading_volume = market_data.get("trading_volume", 0)
        liquidity_score = self._assess_liquidity(trading_volume)
        print(f"Liquidity Score: {liquidity_score:.2f}")
        return liquidity_score

    def _estimate_default_probability(self, credit_rating):
        """Estimates the default probability based on credit rating."""
        credit_ratings = self.knowledge_base.get("credit_ratings", {})
        for rating_system in credit_ratings.values():
            if credit_rating in rating_system:
                return rating_system[credit_rating].get("default_probability", 0.3)
        return 0.3 # default value.

    def _assess_liquidity(self, trading_volume):
        """Assesses the liquidity risk based on trading volume."""
        if trading_volume > 1000000:
            return 0.1  # Low liquidity risk
        elif trading_volume > 100000:
            return 0.2  # Medium liquidity risk
        else:
            return 0.3  # High liquidity risk

    def _assess_operational_risk(self, company_name):
        """Assesses the operational risk."""
        company_size = self.knowledge_base.get("companies", {}).get(company_name, {}).get("size", "Medium")
        if company_size == "Large":
            return "Medium"
        else:
            return "Low"

    def _assess_geopolitical_risks(self, company_name):
        """Assesses the geopolitical risks."""
        company_location = self.knowledge_base.get("companies", {}).get(company_name, {}).get("location", "US")
        if company_location == "US":
            return ["Trade tensions with China"]
        else:
            return ["Political instability in emerging markets"]

    def _assess_industry_risk(self, industry):
        """Assesses the industry-specific risks."""
        if industry == "Technology":
            return "Medium"
        elif industry == "Financials":
            return "High"
        else:
            return "Low"

    def _calculate_overall_risk_score(self, risk_factors):
        """Calculates the overall risk score."""
        weights = self.knowledge_base.get("risk_weights", {})
        weighted_scores = []
        for factor, score in risk_factors.items():
            weight = weights.get(factor, 1) #default weight
            if isinstance(score, (int, float)):
                weighted_scores.append(score * weight)
            elif isinstance(score, str):
                if score == "Low":
                  weighted_scores.append(0.1*weight)
                elif score == "Medium":
                  weighted_scores.append(0.5*weight)
                elif score == "High":
                  weighted_scores.append(0.9*weight)
            elif isinstance(score, list):
                weighted_scores.append(0.5*weight) #place holder for complex risk assessments

        return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0

    def _assess_economic_risk(self):
        """Assesses economic risk based on global indices and economic data."""
        economic_data = self.knowledge_base.get("economic_data", {})
        if not economic_data: return 0.5
        usa_data = economic_data.get("USA", {})
        gdp_growth = usa_data.get("GDP_growth", 0)
        inflation = usa_data.get("inflation", 0)
        unemployment = usa_data.get("unemployment", 0)
        risk = (1 - gdp_growth) + inflation + unemployment
        return risk / 3

    def _assess_volatility_risk(self):
        """Assesses volatility risk based on VIX and VXN indices."""
        volatility_indices = self.knowledge_base.get("volatility_indices", {})
        if not volatility_indices: return 0.5
        vix = volatility_indices.get("VIX", {}).get("value", 20) / 40
        vxn = volatility_indices.get("VXN", {}).get("value", 25) / 40
        return (vix + vxn) / 2

    def _assess_currency_risk(self):
        """Assesses currency risk based on par values."""
        currency_par_values = self.knowledge_base.get("currency_par_values", {})
        if not currency_par_values: return 0.5
        usd_eur = abs(1 - currency_par_values.get("USD/EUR", 0.85))
        usd_jpy = abs(1 - currency_par_values.get("USD/JPY", 145) / 145 )
        return (usd_eur + usd_jpy) / 2

    def _assess_borrower_liquidity(self, borrower_data):
        """Assesses borrower liquidity."""
        liquidity_ratio = borrower_data.get("liquidity_ratio", 1)
        return 1 / liquidity_ratio if liquidity_ratio > 0 else 1

    def _assess_collateral_risk(self, loan_details):
        """Assesses collateral risk."""
        collateral_value = loan_details.get("collateral_value", 0)
        loan_amount = loan_details.get("loan_amount", 1)
        return 1 - (collateral_value / loan_amount) if loan_amount > 0 else 1

    def _assess_interest_rate_risk(self, loan_details):
        """Assesses interest rate risk."""
        interest_rate_type = loan_details.get("interest_rate_type", "fixed")
        if interest_rate_type == "variable":
            return 0.6
        else:
            return 0.2

    def _assess_project_management_risk(self, project_details):
        """Assesses project management risk."""
        project_manager_experience = project_details.get("project_manager_experience", "Medium")
        if project_manager_experience == "High":
            return 0.2
        elif project_manager_experience == "Medium":
            return 0.5
        else:
            return 0.8

    def _assess_technical_risk(self, project_details):
        """Assesses technical risk."""
        technology_maturity = project_details.get("technology_maturity", "Medium")
        if technology_maturity == "Mature":
            return 0.2
        elif technology_maturity == "Established":
            return 0.5
        else:
            return 0.8

    def _assess_project_market_risk(self, project_details, context):
        """Assesses project market risk."""
        market_demand = context.get("market_demand", "Medium")
        if market_demand == "High":
            return 0.2
        elif market_demand == "Medium":
            return 0.5
        else:
            return 0.8

    def _assess_project_financial_risk(self, project_details):
        """Assesses project financial risk."""
        project_budget = project_details.get("project_budget", 1000000)
        funding_secured = project_details.get("funding_secured", 500000)
        return 1 - (funding_secured / project_budget) if project_budget > 0 else 1

    def _assess_regulatory_risk(self, project_details):
        """Assesses regulatory risk."""
        regulatory_environment = project_details.get("regulatory_environment", "Medium")
        if regulatory_environment == "Favorable":
            return 0.2
        elif regulatory_environment == "Neutral":
            return 0.5
        else:
            return 0.8

{
  "RiskAssessmentAgent": {
    "description": "This class provides methods to assess various types of risks including investment, loan, and project risks.",
    "methods": {
      "__init__": {
        "description": "Initializes the Risk Assessment Agent.",
        "parameters": {
          "knowledge_base_path": {
            "type": "str",
            "default": "risk_rating_mapping.json",
            "description": "Path to the knowledge base file."
          }
        }
      },
      "_load_knowledge_base": {
        "description": "Loads the knowledge base from the JSON file.",
        "returns": {
          "type": "dict",
          "description": "The knowledge base data."
        }
      },
      "assess_risk": {
        "description": "Assesses the risk based on the provided target data and context.",
        "parameters": {
          "target_data": {
            "type": "dict",
            "description": "Data related to the target being assessed."
          },
          "risk_type": {
            "type": "str",
            "default": "investment",
            "description": "Type of risk assessment (e.g., 'investment', 'loan', 'project')."
          },
          "context": {
            "type": "dict",
            "description": "Additional context for the risk assessment."
          }
        },
        "returns": {
          "type": "dict",
          "description": "Risk assessment results."
        }
      },
      "assess_investment_risk": {
        "description": "Assesses the risk associated with a potential investment.",
        "parameters": {
          "company_name": {
            "type": "str",
            "description": "Name of the company."
          },
          "financial_data": {
            "type": "dict",
            "description": "Financial data of the company."
          },
          "market_data": {
            "type": "dict",
            "description": "Market data related to the investment."
          }
        },
        "returns": {
          "type": "dict",
          "description": "Risk assessment results for investment."
        }
      },
      "assess_loan_risk": {
        "description": "Assesses the risk associated with a loan.",
        "parameters": {
          "loan_details": {
            "type": "dict",
            "description": "Details of the loan."
          },
          "borrower_data": {
            "type": "dict",
            "description": "Data about the borrower."
          }
        },
        "returns": {
          "type": "dict",
          "description": "Risk assessment results for loan."
        }
      },
      "assess_project_risk": {
        "description": "Assesses the risk associated with a project.",
        "parameters": {
          "project_details": {
            "type": "dict",
            "description": "Details of the project."
          },
          "context": {
            "type": "dict",
            "description": "Additional context for the project."
          }
        },
        "returns": {
          "type": "dict",
          "description": "Risk assessment results for project."
        }
      },
      "_calculate_market_risk": {
        "description": "Calculates market risk.",
        "parameters": {
          "market_data": {
            "type": "dict",
            "description": "Market data."
          }
        },
        "returns": {
          "type": "float",
          "description": "Calculated market risk."
        }
      },
      "_calculate_credit_risk": {
        "description": "Calculates credit risk.",
        "parameters": {
          "financial_data": {
            "type": "dict",
            "description": "Financial data."
          }
        },
        "returns": {
          "type": "float",
          "description": "Calculated credit risk."
        }
      },
      "_calculate_liquidity_risk": {
        "description": "Calculates liquidity risk.",
        "parameters": {
          "market_data": {
            "type": "dict",
            "description": "Market data."
          }
        },
        "returns": {
          "type": "float",
          "description": "Calculated liquidity risk."
        }
      },
      "_estimate_default_probability": {
        "description": "Estimates the default probability based on credit rating.",
        "parameters": {
          "credit_rating": {
            "type": "str",
            "description": "Credit rating."
          }
        },
        "returns": {
          "type": "float",
          "description": "Estimated default probability."
        }
      },
      "_assess_liquidity": {
        "description": "Assesses the liquidity risk based on trading volume.",
        "parameters": {
          "trading_volume": {
            "type": "int",
            "description": "Trading volume."
          }
        },
        "returns": {
          "type": "float",
          "description": "Assessed liquidity risk."
        }
      },
      "_assess_operational_risk": {
        "description": "Assesses the operational risk.",
        "parameters": {
          "company_name": {
            "type": "str",
            "description": "Company name."
          }
        },
        "returns": {
          "type": "str",
          "description": "Assessed operational risk."
        }
      },
      "_assess_geopolitical_risks": {
        "description": "Assesses the geopolitical risks.",
        "parameters": {
          "company_name": {
            "type": "str",
            "description": "Company name."
          }
        },
        "returns": {
          "type": "list",
          "description": "Assessed geopolitical risks."
        }
      },
      "_assess_industry_risk": {
        "description": "Assesses the industry-specific risks.",
        "parameters": {
          "industry": {
            "type": "str",
            "description": "Industry."
          }
        },
        "returns": {
          "type": "str",
          "description": "Assessed industry risk."
        }
      },
      "_calculate_overall_risk_score": {
        "description": "Calculates the overall risk score.",
        "parameters": {
          "risk_factors": {
            "type": "dict",
            "description": "Dictionary of risk factors."
          }
        },
        "returns": {
          "type": "float",
          "description": "Calculated overall risk score."
        }
      },
      "_assess_economic_risk": {
        "description": "Assesses economic risk based on global indices and economic data.",
        "returns": {
          "type": "float",
          "description": "Assessed economic risk."
        }
      },
      "_assess_volatility_risk": {
        "description": "Assesses volatility risk based on VIX and VXN indices.",
        "returns": {
          "type": "float",
          "description": "Assessed volatility risk."
        }
      },
      "_assess_currency_risk": {
        "description": "Assesses currency risk based on par values.",
        "returns": {
          "type": "float",
          "description": "Assessed currency risk."
        }
      },
      "_assess_borrower_liquidity": {
        "description": "Assesses borrower liquidity.",
        "parameters": {
          "borrower_data": {
            "type": "dict",
            "description": "Data of the borrower."
          }
        },
        "returns": {
          "type": "float",
          "description": "Assessed borrower liquidity."
        }
      },
      "_assess_collateral_risk": {
        "description": "Assesses collateral risk.",
        "parameters": {
          "loan_details": {
            "type": "dict",
            "description": "Details of the loan."
          }
        },
        "returns": {
          "type": "float",
          "description": "Assessed collateral risk."
        }
      },
      "_assess_interest_rate_risk": {
        "description": "Assesses interest rate risk.",
        "parameters": {
          "loan_details": {
            "type": "dict",
            "description": "Details of the loan."
          }
        },
        "returns": {
          "type": "float",
          "description": "Assessed interest rate risk."
        }

},
      "_assess_project_management_risk": {
        "description": "Assesses project management risk.",
        "parameters": {
          "project_details": {
            "type": "dict",
            "description": "Details of the project."
          }
        },
        "returns": {
          "type": "float",
          "description": "Assessed project management risk."
        }
      },
      "_assess_technical_risk": {
        "description": "Assesses technical risk.",
        "parameters": {
          "project_details": {
            "type": "dict",
            "description": "Details of the project."
          }
        },
        "returns": {
          "type": "float",
          "description": "Assessed technical risk."
        }
      },
      "_assess_project_market_risk": {
        "description": "Assesses project market risk.",
        "parameters": {
          "project_details": {
            "type": "dict",
            "description": "Details of the project."
          },
          "context": {
            "type": "dict",
            "description": "Additional context for the project."
          }
        },
        "returns": {
          "type": "float",
          "description": "Assessed project market risk."
        }
      },
      "_assess_project_financial_risk": {
        "description": "Assesses project financial risk.",
        "parameters": {
          "project_details": {
            "type": "dict",
            "description": "Details of the project."
          }
        },
        "returns": {
          "type": "float",
          "description": "Assessed project financial risk."
        }
      },
      "_assess_regulatory_risk": {
        "description": "Assesses regulatory risk.",
        "parameters": {
          "project_details": {
            "type": "dict",
            "description": "Details of the project."
          }
        },
        "returns": {
          "type": "float",
          "description": "Assessed regulatory risk."
        }
      }
    }
  }
}
