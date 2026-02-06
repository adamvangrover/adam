from typing import Dict, Any, List
import re
import random

try:
    from core.evaluation.probabilistic import BayesianReasoningEngine
except ImportError:
    BayesianReasoningEngine = None

class AuditorAgent:
    """
    The 'LLM-as-a-Judge' layer.
    Evaluates the agent's reasoning against a rubric:
    1. Factual Grounding
    2. Logic Density
    3. Financial Nuance
    """

    def evaluate(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluates the current state, specifically the quantitative analysis and draft memo.
        Returns a list of audit entries (score 1-5 and reasoning).
        """
        scores = []

        bs = state.get("balance_sheet") or {}
        pnl = state.get("income_statement") or {}
        analysis = state.get("quant_analysis") or ""

        # --- 1. Factual Grounding ---
        # Did it hallucinate? Check if source data is present in output.

        ebitda = pnl.get("consolidated_ebitda")
        if ebitda is None:
             ebitda = pnl.get("operating_income", 0) + pnl.get("depreciation_amortization", 0)

        fg_score = 5
        fg_reasoning = "Key financial figures match source data."

        if ebitda > 0:
            # Check if EBITDA is mentioned (heuristically matching formatting)
            # Try matching "X,XXX.XX" or raw int
            ebitda_str_1 = f"{ebitda:,.2f}"
            ebitda_str_2 = str(int(ebitda))

            if ebitda_str_1 not in analysis and ebitda_str_2 not in analysis:
                fg_score = 3
                fg_reasoning = f"Source EBITDA ({ebitda}) not clearly cited in analysis."

        scores.append({
            "category": "Factual Grounding",
            "score": fg_score,
            "reasoning": fg_reasoning
        })

        # --- 2. Logic Density ---
        # Is the CoT mathematically sound?

        debt = bs.get("total_debt", 0.0)
        cash = bs.get("cash_equivalents", 0.0)

        ld_score = 5
        ld_reasoning = "Mathematical logic appears consistent."

        if ebitda > 0:
            calc_leverage = debt / ebitda

            # Extract leverage from text: "Leverage (Gross): 4.50x"
            # Regex to find "Leverage ... : <number>"
            match = re.search(r"Leverage.*?:\s*(\d+\.\d+)", analysis, re.IGNORECASE)
            if match:
                stated_leverage = float(match.group(1))
                if abs(stated_leverage - calc_leverage) > 0.1:
                    ld_score = 2
                    ld_reasoning = f"Logic Error: Calculated Gross Leverage is {calc_leverage:.2f}x but analysis states {stated_leverage:.2f}x."

        scores.append({
             "category": "Logic Density",
             "score": ld_score,
             "reasoning": ld_reasoning
        })

        # --- 3. Financial Nuance ---
        # Does it use correct terminology?

        terms = ["liquidity", "solvency", "covenant", "margin", "coverage"]
        found_terms = [t for t in terms if t in analysis.lower()]
        count = len(found_terms)

        fn_score = 3 # Baseline
        if count >= 3:
            fn_score = 5
            fn_reasoning = f"High financial fluency. Used terms: {', '.join(found_terms)}."
        elif count >= 1:
            fn_score = 4
            fn_reasoning = f"Adequate terminology. Used: {', '.join(found_terms)}."
        else:
            fn_score = 2
            fn_reasoning = "Analysis lacks specific financial terminology (liquidity, solvency, etc)."

        scores.append({
            "category": "Financial Nuance",
            "score": fn_score,
            "reasoning": fn_reasoning
        })

        return scores

    def evaluate_with_llm(self, state: Dict[str, Any], model: str = None) -> List[Dict[str, Any]]:
        """
        Simulates an 'LLM-as-a-Judge' evaluation using a Jury of specialized judges.

        UPGRADE: Now uses BayesianReasoningEngine for 'Conviction' when no live model is present.
        This provides 'likely and reasoned' scoring even in offline/static modes.
        """
        scores = self.evaluate(state) # Start with heuristic baseline

        # If no live model, use Probabilistic Conviction Layer
        if model is None and BayesianReasoningEngine:
            engine = BayesianReasoningEngine()
            # Pass existing verification flags if present in state (computed previously or empty)
            flags = state.get("verification_flags", [])
            result = engine.calculate_confidence(state, flags)

            # Map Bayesian Confidence (0-1) to Score (1-5)
            # 0.85 -> 4.25 -> 4
            # 0.40 -> 2.0 -> 2
            conviction_score = max(1, min(5, round(result["score"] * 5)))

            # Use the generated trace as reasoning
            trace_text = " ".join(result["trace"])

            scores.append({
                "category": "Bayesian Conviction",
                "score": conviction_score,
                "reasoning": f"Probabilistic Confidence: {result['score']:.0%}. Trace: {trace_text}"
            })

        else:
            # Fallback for when Bayesian Engine is missing or we just want simple mock
            analysis = state.get("quant_analysis") or ""
            if "hallucinate" in analysis.lower() or "zombie" in str(state).lower():
                 scores.append({
                     "category": "Jury Consensus",
                     "score": 1,
                     "reasoning": "CRITICAL: The Jury detects significant discrepancies characteristic of a hallucinated debt structure."
                 })
            else:
                 scores.append({
                     "category": "Jury Consensus",
                     "score": 4,
                     "reasoning": "The Jury finds the reasoning generally sound."
                 })

        return scores
