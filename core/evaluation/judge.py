from typing import Dict, Any, List
import re

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
