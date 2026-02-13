"""
Sovereign Credit Pipeline
-------------------------
Implements the "Glass Box" pipeline for credit analysis using the Adam Sovereign Bundle.
This module orchestrates:
1. Data Ingestion (Mock EDGAR)
2. Quantitative Analysis (Quant Agent Logic)
3. Risk Assessment (Risk Officer Logic)
4. Artifact Generation (Spreads, Memos, Audit Logs)
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List

from core.pipelines.mock_edgar import EdgarSource

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SovereignPipeline")

class SovereignPipeline:
    def __init__(self, bundle_path: str, output_dir: str):
        """
        Initialize the pipeline with the path to the Sovereign Bundle.
        """
        self.bundle_path = bundle_path
        self.output_dir = output_dir
        self.edgar_source = EdgarSource()

        # Load Bundle Definitions
        self.manifest = self._load_yaml(os.path.join(bundle_path, "manifest.yaml"))
        self.quant_agent_def = self._load_yaml(os.path.join(bundle_path, "agents", "quant.yaml"))
        self.risk_agent_def = self._load_yaml(os.path.join(bundle_path, "agents", "risk_officer.yaml"))

        # Ensure Output Directory Exists
        os.makedirs(output_dir, exist_ok=True)

    def _load_yaml(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML {filepath}: {e}")
            return {}

    def run_pipeline(self, ticker: str):
        """
        Execute the full credit analysis pipeline for a given ticker.
        """
        logger.info(f"Starting Sovereign Pipeline for {ticker}...")

        # 1. Data Ingestion
        try:
            financial_history = self.edgar_source.get_financial_history(ticker)
            logger.info(f"Ingested Financial History for {ticker} ({len(financial_history['history'])} years)")
        except ValueError as e:
            logger.error(str(e))
            return

        # 2. Quant Agent Execution (Spreading)
        spread_data, quant_audit = self._run_quant_agent(financial_history)
        self._save_artifact(ticker, "spread", spread_data)

        # 3. Risk Officer Execution (Covenant & Critique)
        risk_report, risk_audit = self._run_risk_officer(financial_history, spread_data)
        self._save_artifact(ticker, "memo", risk_report) # Saving memo as JSON for structured access, can be rendered to MD

        # 4. Audit Aggregation
        full_audit_log = {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "quant_audit": quant_audit,
            "risk_audit": risk_audit,
            "pipeline_status": "SUCCESS"
        }
        self._save_artifact(ticker, "audit", full_audit_log)

        logger.info(f"Pipeline Completed for {ticker}. Artifacts saved to {self.output_dir}")

    def _run_quant_agent(self, financial_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Simulates the logic of the Quant Agent defined in agents/quant.yaml.
        """
        logger.info("Executing Quant Agent Logic...")

        history = financial_data.get("history", [])
        if not history:
            return {}, {}

        # Sort just in case, though mock is sorted
        history.sort(key=lambda x: x["fiscal_year"])

        current_financials = history[-1]
        company_name = financial_data.get("company_name")

        # Logic: Calculate Key Ratios (Current Year)
        ebitda = current_financials.get("ebitda", 0)
        total_debt = current_financials.get("total_debt", 0)
        interest_expense = current_financials.get("interest_expense", 0.01) # Avoid div by zero

        leverage_ratio = total_debt / ebitda if ebitda else 0
        interest_coverage = ebitda / interest_expense if interest_expense else 0

        # Logic: Growth Metrics
        # CAGR (3 Year if available)
        cagr_revenue = 0.0
        cagr_ebitda = 0.0
        yoy_revenue = 0.0
        yoy_ebitda = 0.0

        if len(history) >= 2:
            prev = history[-2]
            yoy_revenue = (current_financials["revenue"] - prev["revenue"]) / prev["revenue"]
            yoy_ebitda = (current_financials["ebitda"] - prev["ebitda"]) / prev["ebitda"]

        if len(history) >= 3:
            start = history[0]
            years = len(history) - 1
            cagr_revenue = (current_financials["revenue"] / start["revenue"]) ** (1/years) - 1
            cagr_ebitda = (current_financials["ebitda"] / start["ebitda"]) ** (1/years) - 1

        # Logic: Accounting Identity Check (Assets = Liab + Equity)
        assets = current_financials.get("total_assets", 0)
        liabilities = current_financials.get("total_liabilities", 0)
        equity = current_financials.get("total_equity", 0)
        identity_delta = assets - (liabilities + equity)
        identity_pass = abs(identity_delta) < 1.0 # Tolerance for rounding

        spread_data = {
            "ticker": company_name,
            "fiscal_year": current_financials.get("fiscal_year"),
            "metrics": {
                "Revenue": current_financials.get("revenue"),
                "EBITDA": ebitda,
                "Total Debt": total_debt,
                "Cash": current_financials.get("cash_equivalents"),
                "Net Debt": total_debt - current_financials.get("cash_equivalents", 0)
            },
            "growth_metrics": {
                "Revenue CAGR (3Y)": round(cagr_revenue * 100, 1),
                "EBITDA CAGR (3Y)": round(cagr_ebitda * 100, 1),
                "Revenue YoY": round(yoy_revenue * 100, 1),
                "EBITDA YoY": round(yoy_ebitda * 100, 1)
            },
            "ratios": {
                "Leverage (Debt/EBITDA)": round(leverage_ratio, 2),
                "Interest Coverage (EBITDA/Interest)": round(interest_coverage, 2)
            },
            "validation": {
                "identity_check": "PASS" if identity_pass else "FAIL",
                "identity_delta": identity_delta
            },
            "history": history # Include full history for charting
        }

        audit_entry = {
            "agent_id": self.quant_agent_def.get("id"),
            "action": "CALCULATE_SPREAD",
            "status": "SUCCESS",
            "details": f"Leverage: {leverage_ratio:.2f}x, Rev CAGR: {cagr_revenue:.1%}"
        }

        return spread_data, audit_entry

    def _run_risk_officer(self, financial_data: Dict[str, Any], spread_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Simulates the logic of the Risk Officer defined in agents/risk_officer.yaml.
        """
        logger.info("Executing Risk Officer Logic...")

        leverage = spread_data["ratios"]["Leverage (Debt/EBITDA)"]
        coverage = spread_data["ratios"]["Interest Coverage (EBITDA/Interest)"]
        rev_cagr = spread_data["growth_metrics"]["Revenue CAGR (3Y)"]
        ebitda_cagr = spread_data["growth_metrics"]["EBITDA CAGR (3Y)"]

        # Policy Check (Mock Policy: Leverage < 3.0x, Coverage > 4.0x)
        policy_violations = []
        warnings = []

        if leverage > 3.0:
            policy_violations.append(f"Leverage {leverage}x exceeds limit of 3.0x")
        if coverage < 4.0:
            policy_violations.append(f"Interest Coverage {coverage}x below minimum of 4.0x")

        # Growth Critique
        if rev_cagr < 0:
            warnings.append(f"Revenue is shrinking (CAGR {rev_cagr}%)")
        if ebitda_cagr < 0:
            warnings.append(f"EBITDA is shrinking (CAGR {ebitda_cagr}%)")

        # Recommendation
        if not policy_violations:
            recommendation = "APPROVE"
            rationale = "Borrower meets all standard covenants. Strong financial position."
            if warnings:
                rationale += " Note: " + "; ".join(warnings)
        else:
            recommendation = "REFER" # Refer to committee
            rationale = f"Policy Violations Detected: {', '.join(policy_violations)}"

        memo = {
            "title": f"Credit Memo: {spread_data.get('ticker')}",
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "recommendation": recommendation,
            "executive_summary": rationale,
            "financial_highlights": spread_data["metrics"],
            "growth_analysis": spread_data["growth_metrics"],
            "covenant_analysis": {
                "leverage_test": "FAIL" if leverage > 3.0 else "PASS",
                "coverage_test": "FAIL" if coverage < 4.0 else "PASS"
            }
        }

        audit_entry = {
            "agent_id": self.risk_agent_def.get("id"),
            "action": "POLICY_CHECK",
            "status": "WARNING" if policy_violations else "SUCCESS",
            "details": rationale
        }

        return memo, audit_entry

    def _save_artifact(self, ticker: str, artifact_type: str, data: Any):
        filename = f"{ticker}_{artifact_type}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
