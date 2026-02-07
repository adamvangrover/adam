from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import json
import logging
import re
import random
from core.llm_plugin import LLMPlugin

try:
    from core.evaluation.probabilistic import BayesianReasoningEngine
except ImportError:
    BayesianReasoningEngine = None

logger = logging.getLogger(__name__)

class HeuristicResult(BaseModel):
    """Container for deterministic check results."""
    category: str
    score: int
    reasoning: str
    flags: List[str] = Field(default_factory=list)

class EvaluationScore(BaseModel):
    """
    Rubric-based scoring for agent outputs, combining heuristic and LLM insights.
    Scale 1-5.
    """
    factual_grounding: int = Field(..., ge=1, le=5, description="Did it hallucinate facts or covenants? (Weighted by heuristic checks)")
    logic_density: int = Field(..., ge=1, le=5, description="Is the Chain of Thought mathematically and logically sound?")
    financial_nuance: int = Field(..., ge=1, le=5, description="Does it distinguish between key concepts like liquidity and solvency?")
    reasoning: str = Field(..., description="The Auditor's detailed explanation for the assigned scores.")
    overall_score: float = Field(..., description="The average of the component scores.")
    automated_flags: List[str] = Field(default_factory=list, description="Specific flags raised by the deterministic code audit.")

class AuditorAgent:
    """
    A specialized 'LLM-as-a-Judge' agent that performs a Hybrid Audit:
    1. Deterministic Heuristics: Checks math, specific data presence, and terminology via code/regex.
    2. Semantic Evaluation: Uses LLM to judge reasoning quality, incorporating heuristic findings.
    
    It evaluates other agents' reasoning objectively based on input data vs output text.
    """

    RUBRIC = """
    Evaluation Rubric:
    1. Factual Grounding (1-5):
       - 1: Major hallucinations (invented debt, wrong dates) or direct contradiction of source data.
       - 5: Perfect alignment with input data; all claims cited and verifiable.

    2. Logic Density (1-5):
       - 1: Jumps to conclusions, math errors (e.g. leverage calc), superficial analysis.
       - 5: Step-by-step derivation, mathematically sound, deep causal analysis.

    3. Financial Nuance (1-5):
       - 1: Confuses terms (e.g., EBITDA vs Revenue), misses obvious risks.
       - 5: Expert-level distinction (e.g., identifying structural subordination), understands context.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro", mock_mode: bool = False):
        self.model_name = model_name
        self.mock_mode = mock_mode
        if not mock_mode:
            self.llm = LLMPlugin(config={"provider": "gemini", "gemini_model_name": model_name})

    def evaluate(self, input_data: Dict[str, Any], agent_output: Union[Dict[str, Any], str]) -> EvaluationScore:
        """
        Evaluates the agent's output against the input data using a hybrid approach.
        
        Args:
            input_data: The source context (e.g. balance sheet, P&L).
            agent_output: The text or dictionary produced by the agent being audited.
        """
        logger.info(f"AuditorAgent evaluating output from model {self.model_name}...")

        # Normalize output to string for regex checks
        output_text = json.dumps(agent_output) if isinstance(agent_output, dict) else str(agent_output)

        # 1. Run Deterministic Heuristics
        heuristic_results = self._run_heuristics(input_data, output_text)
        
        # 2. Run LLM Evaluation (or Mock)
        if self.mock_mode:
            return self._mock_evaluate(input_data, agent_output, heuristic_results)

        try:
            return self._llm_evaluate(input_data, agent_output, heuristic_results)
        except Exception as e:
            logger.warning(f"LLM Evaluation failed: {e}. Falling back to mock/heuristic synthesis.")
            return self._mock_evaluate(input_data, agent_output, heuristic_results)

    def _run_heuristics(self, input_data: Dict[str, Any], output_text: str) -> List[HeuristicResult]:
        """
        Performs regex and logic checks on the content.
        """
        results = []
        
        # Extract financial statements if available
        bs = input_data.get("balance_sheet") or input_data.get("financials", {}).get("balance_sheet", {})
        pnl = input_data.get("income_statement") or input_data.get("financials", {}).get("income_statement", {})

        # --- Heuristic 1: Factual Grounding (Data Presence) ---
        ebitda = pnl.get("consolidated_ebitda")
        if ebitda is None:
            ebitda = pnl.get("operating_income", 0) + pnl.get("depreciation_amortization", 0)

        fg_flags = []
        fg_score = 5
        if ebitda and ebitda > 0:
            # Check formatted and raw versions
            ebitda_str_1 = f"{ebitda:,.2f}"
            ebitda_str_2 = str(int(ebitda))
            if ebitda_str_1 not in output_text and ebitda_str_2 not in output_text:
                fg_score = 3
                fg_flags.append(f"Source EBITDA ({ebitda}) not clearly cited in analysis.")

        results.append(HeuristicResult(
            category="Factual Grounding",
            score=fg_score,
            reasoning="Automated data citation check.",
            flags=fg_flags
        ))

        # --- Heuristic 2: Logic Density (Math Check) ---
        debt = bs.get("total_debt", 0.0)
        ld_flags = []
        ld_score = 5
        
        if ebitda and ebitda > 0:
            calc_leverage = debt / ebitda
            # Regex to find "Leverage ... : <number>" or similar patterns
            match = re.search(r"Leverage.*?:\s*(\d+\.\d+)", output_text, re.IGNORECASE)
            if match:
                stated_leverage = float(match.group(1))
                if abs(stated_leverage - calc_leverage) > 0.1:
                    ld_score = 2
                    ld_flags.append(f"Math Error: Calculated Gross Leverage is {calc_leverage:.2f}x but analysis states {stated_leverage:.2f}x.")

        results.append(HeuristicResult(
            category="Logic Density",
            score=ld_score,
            reasoning="Automated math consistency check.",
            flags=ld_flags
        ))

        # --- Heuristic 3: Financial Nuance (Terminology) ---
        terms = ["liquidity", "solvency", "covenant", "margin", "coverage", "subordination", "tranche"]
        found_terms = [t for t in terms if t in output_text.lower()]
        count = len(found_terms)
        
        fn_score = 3
        fn_flags = []
        if count >= 3:
            fn_score = 5
        elif count < 1:
            fn_score = 2
            fn_flags.append("Analysis lacks specific financial terminology.")

        results.append(HeuristicResult(
            category="Financial Nuance",
            score=fn_score,
            reasoning=f"Found terms: {', '.join(found_terms)}",
            flags=fn_flags
        ))

        return results

    def _llm_evaluate(self, input_data: Dict[str, Any], agent_output: Any, heuristics: List[HeuristicResult]) -> EvaluationScore:
        """
        Generates the final evaluation using the LLM, context-aware of the heuristic failures.
        """
        # Prepare heuristic summary for the prompt
        heuristic_summary = "\n".join([
            f"- {h.category} (Automated Check): Score {h.score}/5. {h.reasoning} Flags: {h.flags}"
            for h in heuristics
        ])

        prompt = (
            f"You are an expert Credit Auditor. Evaluate the following analysis based on the provided input data.\n"
            f"An automated code-based audit has already been performed. You must incorporate its findings into your final scoring.\n\n"
            f"AUTOMATED AUDIT FINDINGS:\n{heuristic_summary}\n\n"
            f"--- RUBRIC ---\n{self.RUBRIC}\n\n"
            f"--- INPUT DATA ---\n{json.dumps(input_data, default=str)}\n\n"
            f"--- AGENT OUTPUT ---\n{json.dumps(agent_output, default=str)}\n\n"
            f"Instructions:\n"
            f"1. If the Automated Audit found Math Errors or Hallucinations, your score for that category MUST NOT exceed 2.\n"
            f"2. Provide a structured evaluation explaining the final scores."
        )

        score, _ = self.llm.generate_structured(prompt, EvaluationScore)
        
        # Merge flags into the final object for visibility
        all_flags = [flag for h in heuristics for flag in h.flags]
        score.automated_flags = all_flags
        
        return score

    def _mock_evaluate(self, input_data: Dict[str, Any], agent_output: Any, heuristics: List[HeuristicResult]) -> EvaluationScore:
        """
        Fallback evaluation combining heuristic scores with simple logic and optional Bayesian reasoning.
        """
        # Calculate average from heuristics
        total_score = sum(h.score for h in heuristics)
        avg_score = total_score / len(heuristics)
        
        all_flags = [flag for h in heuristics for flag in h.flags]
        
        reasoning = "Automated audit completed. "
        if all_flags:
            reasoning += "Issues found in: " + ", ".join([h.category for h in heuristics if h.flags]) + ". "
        else:
            reasoning += "No automated flags raised. Output appears technically sound. "

        # UPGRADE: Use BayesianReasoningEngine for 'Conviction' if available
        if BayesianReasoningEngine:
            try:
                engine = BayesianReasoningEngine()
                # Use heuristics flags as the signals for Bayesian engine
                result = engine.calculate_confidence({"heuristics": [h.dict() for h in heuristics]}, all_flags)
                
                # Adjust reasoning with Bayesian Trace
                trace_text = " ".join(result.get("trace", []))
                reasoning += f" [Bayesian Confidence: {result.get('score', 0):.0%}. {trace_text}]"
            except Exception as e:
                logger.warning(f"Bayesian engine failed in mock evaluation: {e}")

        return EvaluationScore(
            factual_grounding=heuristics[0].score,
            logic_density=heuristics[1].score,
            financial_nuance=heuristics[2].score,
            reasoning=reasoning,
            overall_score=round(avg_score, 2),
            automated_flags=all_flags
        )