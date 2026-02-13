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
            financials = self.edgar_source.get_annual_financials(ticker)
            logger.info(f"Ingested Financials for {ticker} (FY {financials['fiscal_year']})")
        except ValueError as e:
            logger.error(str(e))
            return

        # 2. Quant Agent Execution (Spreading)
        spread_data, quant_audit = self._run_quant_agent(financials)
        self._save_artifact(ticker, "spread", spread_data)

        # 3. Risk Officer Execution (Covenant & Critique)
        risk_report, risk_audit = self._run_risk_officer(financials, spread_data)
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

    def _run_quant_agent(self, financials: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Simulates the logic of the Quant Agent defined in agents/quant.yaml.
        """
        logger.info("Executing Quant Agent Logic...")

        # Logic: Calculate Key Ratios
        ebitda = financials.get("ebitda", 0)
        total_debt = financials.get("total_debt", 0)
        interest_expense = financials.get("interest_expense", 0.01) # Avoid div by zero

        leverage_ratio = total_debt / ebitda if ebitda else 0
        interest_coverage = ebitda / interest_expense if interest_expense else 0

        # Logic: Accounting Identity Check (Assets = Liab + Equity)
        assets = financials.get("total_assets", 0)
        liabilities = financials.get("total_liabilities", 0)
        equity = financials.get("total_equity", 0)
        identity_delta = assets - (liabilities + equity)
        identity_pass = abs(identity_delta) < 1.0 # Tolerance for rounding

        spread_data = {
            "ticker": financials.get("company_name"),
            "fiscal_year": financials.get("fiscal_year"),
            "metrics": {
                "Revenue": financials.get("revenue"),
                "EBITDA": ebitda,
                "Total Debt": total_debt,
                "Cash": financials.get("cash_equivalents"),
                "Net Debt": total_debt - financials.get("cash_equivalents", 0)
            },
            "ratios": {
                "Leverage (Debt/EBITDA)": round(leverage_ratio, 2),
                "Interest Coverage (EBITDA/Interest)": round(interest_coverage, 2)
            },
            "validation": {
                "identity_check": "PASS" if identity_pass else "FAIL",
                "identity_delta": identity_delta
            }
        }

        audit_entry = {
            "agent_id": self.quant_agent_def.get("id"),
            "action": "CALCULATE_RATIOS",
            "status": "SUCCESS",
            "details": f"Calculated Leverage: {leverage_ratio:.2f}x"
        }

        return spread_data, audit_entry

    def _run_risk_officer(self, financials: Dict[str, Any], spread_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Simulates the logic of the Risk Officer defined in agents/risk_officer.yaml.
        """
        logger.info("Executing Risk Officer Logic...")

        leverage = spread_data["ratios"]["Leverage (Debt/EBITDA)"]
        coverage = spread_data["ratios"]["Interest Coverage (EBITDA/Interest)"]

        # Policy Check (Mock Policy: Leverage < 3.0x, Coverage > 4.0x)
        policy_violations = []
        if leverage > 3.0:
            policy_violations.append(f"Leverage {leverage}x exceeds limit of 3.0x")
        if coverage < 4.0:
            policy_violations.append(f"Interest Coverage {coverage}x below minimum of 4.0x")

        # Recommendation
        if not policy_violations:
            recommendation = "APPROVE"
            rationale = "Borrower meets all standard covenants. Strong financial position."
        else:
            recommendation = "REFER" # Refer to committee
            rationale = f"Policy Violations Detected: {', '.join(policy_violations)}"

        memo = {
            "title": f"Credit Memo: {financials.get('company_name')}",
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "recommendation": recommendation,
            "executive_summary": rationale,
            "financial_highlights": spread_data["metrics"],
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
