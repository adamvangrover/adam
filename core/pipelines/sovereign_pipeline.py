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

    def _run_valuation_module(self, current_financials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates DCF, Probability of Default (Z-Score), and LGD.
        """
        logger.info("Running Valuation & Risk Module...")

        # --- 1. Deterministic DCF ---
        # Assumptions
        wacc = 0.09
        growth_rate = 0.03

        ebitda = current_financials.get("ebitda", 0)
        total_debt = current_financials.get("total_debt", 0)
        cash = current_financials.get("cash_equivalents", 0)
        total_liabilities = current_financials.get("total_liabilities", 0)

        # Simple FCF Proxy: EBITDA * 0.70 (Tax + Capex approx)
        base_fcf = ebitda * 0.70

        projected_fcf = []
        for i in range(1, 6):
            # Degrading growth: 5% -> 3%
            g = 0.05 - (0.005 * (i-1))
            fcf = base_fcf * ((1 + g) ** i)
            projected_fcf.append(fcf)

        # Terminal Value
        terminal_val = (projected_fcf[-1] * (1 + growth_rate)) / (wacc - growth_rate)

        # PV
        pv_fcf = sum([fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(projected_fcf)])
        pv_terminal = terminal_val / ((1 + wacc) ** 5)

        enterprise_value = pv_fcf + pv_terminal
        equity_value = enterprise_value - total_debt + cash

        # Mock Share Price (Equity / Arbitrary Share Count)
        # We'll just fake a share count to make price look "normal" (e.g., $100-$300 range)
        # Using a fixed divisor based on Equity Size to allow consistent updates
        mock_shares = equity_value / 250.0 if equity_value > 0 else 1.0
        share_price = equity_value / mock_shares

        # --- 2. Probability of Default (Altman Z-Score Proxy) ---
        # Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        # A = Working Capital / Assets (Proxy: (Assets - Liab) * 0.2 / Assets) - Very rough proxy
        # B = Retained Earnings / Assets (Proxy: Equity / Assets)
        # C = EBIT / Assets (Proxy: EBITDA * 0.8 / Assets)
        # D = Market Value Equity / Liabilities (Proxy: Equity / Liabilities)
        # E = Sales / Assets

        assets = current_financials.get("total_assets", 1)
        equity = current_financials.get("total_equity", 0)
        revenue = current_financials.get("revenue", 0)

        A = ((assets - total_liabilities) * 0.1) / assets
        B = equity / assets
        C = (ebitda * 0.8) / assets
        D = equity / total_liabilities if total_liabilities > 0 else 10.0
        E = revenue / assets

        z_score = (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)

        # Interpret Z-Score
        # > 3.0 Safe, 1.8-3.0 Grey, < 1.8 Distress
        if z_score > 2.99: pd_category = "Safe"
        elif z_score > 1.81: pd_category = "Grey Zone"
        else: pd_category = "Distress"

        # --- 3. Loss Given Default (LGD) ---
        # Based on Asset Coverage: Assets / Liabilities
        coverage = assets / total_liabilities if total_liabilities > 0 else 100.0
        if coverage > 2.0: lgd = 0.10 # High Recovery
        elif coverage > 1.0: lgd = 0.45 # Unsecured Standard
        else: lgd = 0.85 # Deeply Subordinated / Insolvency

        # --- 4. Forward Projections & Ratings ---
        # Projections (3 Years)
        current_rev = revenue
        current_ebitda = ebitda

        projections = []
        for i in range(1, 4):
            growth = 0.05 # 5% base growth assumption
            proj_rev = current_rev * ((1 + growth) ** i)
            proj_ebitda = current_ebitda * ((1 + growth) ** i)
            projections.append({
                "fiscal_year": current_financials.get("fiscal_year") + i,
                "revenue": round(proj_rev, 2),
                "ebitda": round(proj_ebitda, 2)
            })

        # Credit Rating Mapping (S&P Style) based on Z-Score
        if z_score > 3.0: credit_rating = "AAA"
        elif z_score > 2.8: credit_rating = "AA"
        elif z_score > 2.5: credit_rating = "A"
        elif z_score > 2.0: credit_rating = "BBB"
        elif z_score > 1.8: credit_rating = "BB"
        elif z_score > 1.2: credit_rating = "B"
        else: credit_rating = "CCC"

        # Price Targets (Bull / Base / Bear)
        # Base = Calculated Share Price
        base_target = share_price
        bull_target = base_target * 1.35
        bear_target = base_target * 0.65

        # Conviction Score (0-100)
        # Based on Z-Score stability and Growth
        # Z-Score factor (max 50) + Growth factor (max 50)
        z_factor = min(z_score / 3.0, 1.0) * 50
        g_factor = 30 # Base conviction
        if growth_rate > 0.05: g_factor += 20
        conviction = min(round(z_factor + g_factor), 99)

        # --- 5. Rationale Generation (Simulated LLM Output) ---
        risk_rationale = (
            f"The company exhibits a Z-Score of {z_score:.2f}, placing it in the {pd_category} category. "
            f"Asset coverage of {coverage:.2f}x suggests a Loss Given Default (LGD) of approx {lgd*100:.0f}%. "
            f"Credit Rating is assessed at {credit_rating}. "
            "Key drivers include strong EBITDA generation relative to debt service obligations."
        )
        if z_score < 1.8:
            risk_rationale += " IMMEDIATE ATTENTION REQUIRED: Solvency ratios are deteriorating."

        valuation_rationale = (
            f"Equity Price Target set at ${base_target:.2f} (Base Case) derived from a deterministic DCF model "
            f"assuming a {wacc*100:.1f}% WACC and {growth_rate*100:.1f}% terminal growth. "
            f"Upside scenario (Bull) at ${bull_target:.2f} assumes accelerated margin expansion. "
            f"Downside (Bear) at ${bear_target:.2f} reflects potential compression in free cash flow."
        )

        return {
            "dcf": {
                "enterprise_value": round(enterprise_value, 2),
                "equity_value": round(equity_value, 2),
                "share_price": round(share_price, 2),
                "wacc": wacc,
                "growth_rate": growth_rate,
                "base_fcf": base_fcf, # For client-side recalc
                "mock_shares": mock_shares
            },
            "risk_model": {
                "z_score": round(z_score, 2),
                "pd_category": pd_category,
                "lgd": round(lgd, 2),
                "asset_coverage": round(coverage, 2),
                "credit_rating": credit_rating,
                "rationale": risk_rationale
            },
            "forward_view": {
                "projections": projections,
                "price_targets": {
                    "bull": round(bull_target, 2),
                    "base": round(base_target, 2),
                    "bear": round(bear_target, 2)
                },
                "conviction_score": conviction,
                "rationale": valuation_rationale
            }
        }

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

        # Run Valuation Module
        valuation_data = self._run_valuation_module(current_financials)

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
            "valuation": valuation_data, # Added Valuation Data
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

    def _generate_scenarios(self, financials: Dict[str, Any], valuation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates probability-weighted scenarios (Bear, Base, Bull).
        """
        base_target = valuation["dcf"]["share_price"]
        current_rev = financials["metrics"]["Revenue"]

        # Bull Case (30% Prob)
        bull_target = base_target * 1.35
        bull_rev = current_rev * 1.15

        # Bear Case (20% Prob)
        bear_target = base_target * 0.65
        bear_rev = current_rev * 0.90

        return [
            {
                "case": "Bear",
                "probability": 0.20,
                "price_target": round(bear_target, 2),
                "revenue_outlook": round(bear_rev, 2),
                "description": "Recessionary environment, multiple compression, margin contraction."
            },
            {
                "case": "Base",
                "probability": 0.50,
                "price_target": round(base_target, 2),
                "revenue_outlook": round(current_rev * 1.05, 2), # 5% growth
                "description": "Steady state growth inline with consensus estimates."
            },
            {
                "case": "Bull",
                "probability": 0.30,
                "price_target": round(bull_target, 2),
                "revenue_outlook": round(bull_rev, 2),
                "description": "Accelerated adoption, margin expansion, multiple re-rating."
            }
        ]

    def _generate_swot(self, financials: Dict[str, Any], risk_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generates a mock SWOT analysis based on metrics.
        """
        strengths = ["Strong Market Position"]
        weaknesses = []
        opportunities = ["International Expansion", "AI Integration"]
        threats = ["Regulatory Headwinds"]

        if risk_data["z_score"] > 3.0:
            strengths.append("Robust Balance Sheet (High Z-Score)")
        else:
            weaknesses.append("Deteriorating Solvency Metrics")

        if risk_data["asset_coverage"] > 2.0:
            strengths.append("High Asset Coverage")

        if financials["growth_metrics"]["Revenue CAGR (3Y)"] > 10:
            strengths.append("High Revenue Growth")
        elif financials["growth_metrics"]["Revenue CAGR (3Y)"] < 0:
            weaknesses.append("Declining Revenue Trend")
            threats.append("Market Saturation")

        return {
            "Strengths": strengths,
            "Weaknesses": weaknesses,
            "Opportunities": opportunities,
            "Threats": threats
        }

    def _generate_cap_structure(self, financials: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates a mock Capital Structure table for LGD analysis.
        """
        total_debt = financials["metrics"]["Total Debt"]
        # Split debt: 50% Senior Secured, 30% Unsecured, 20% Subordinated
        senior = total_debt * 0.50
        unsecured = total_debt * 0.30
        sub = total_debt * 0.20

        return [
            {"tranche": "Senior Secured Revolver", "amount": round(senior, 2), "priority": 1, "recovery_est": 100},
            {"tranche": "Senior Unsecured Notes", "amount": round(unsecured, 2), "priority": 2, "recovery_est": 45},
            {"tranche": "Subordinated Debt", "amount": round(sub, 2), "priority": 3, "recovery_est": 5}
        ]

    def _generate_citations(self) -> List[Dict[str, str]]:
        return [
            {"source": "FY2023 10-K", "doc_id": "doc_10k_23", "relevance": "High"},
            {"source": "Q3 2024 Earnings Call Transcript", "doc_id": "doc_ec_q3_24", "relevance": "Medium"},
            {"source": "Moodys Credit Opinion", "doc_id": "doc_moodys_24", "relevance": "High"}
        ]

    def _save_artifact(self, ticker: str, artifact_type: str, data: Any):
        filename = f"{ticker}_{artifact_type}.json"
        filepath = os.path.join(self.output_dir, filename)

        # If saving spread, inject extra reporting data
        if artifact_type == "spread":
            # We need to construct the full report here or just augment the spread
            # Let's augment the spread to be the 'Master Artifact' for the dashboard
            # Note: In a real system, 'report' would be separate, but for the dashboard loader,
            # it's easier if everything is in one place or we save a separate _report.json

            # Let's save a separate _report.json that aggregates everything
            pass

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        # If this was the spread, generate the full report artifact now
        if artifact_type == "spread":
            report_data = {
                "ticker": data["ticker"],
                "scenarios": self._generate_scenarios(data, data["valuation"]),
                "swot": self._generate_swot(data, data["valuation"]["risk_model"]),
                "cap_structure": self._generate_cap_structure(data),
                "citations": self._generate_citations(),
                "executive_summary": f"Comprehensive credit analysis for {data['ticker']}. " + data["valuation"]["forward_view"]["rationale"]
            }
            report_path = os.path.join(self.output_dir, f"{ticker}_report.json")
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
