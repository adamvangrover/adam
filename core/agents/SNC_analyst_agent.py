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

