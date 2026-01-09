
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
# Core imports based on v23 architecture
from core.agents.agent_base import AgentBase
from core.system.v22_async.async_agent_base import AsyncAgentBase
from core.system.v22_async.async_task import AsyncTask

logger = logging.getLogger(__name__)

class CreditRiskControllerAgent(AsyncAgentBase):

    """
    The 'Senior Credit Risk Controller' Agent.
    
    A digital twin of a Regulatory Examiner/Senior Credit Officer.
    
    Directives:
    1. Ingest granular facility data (SNCnet schema).
    2. Deterministically calculate implied ratings (S&P/Moody's logic).
    3. Simulate Regulatory Disagreement (SNC Review logic).
    4. Generate defense-ready eSNC Cover Pages.
    
    Architecture:
    - Pre-Computation Layer: Python-based execution of the S&P Matrix and Conviction Score formula.
    - Inference Layer: LLM-based construction of the "Defense Narrative" and qualitative synthesis.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], system_context: Any = None):
        # Handle init signature differences
        super().__init__(config=config)
        self.agent_id = agent_id
        self.role = "Senior Credit Risk Controller"
        # Mock LLM engine if not present in base
        self.llm_engine = self.config.get("llm_engine", None)
        if not self.llm_engine and system_context and hasattr(system_context, 'llm_engine'):
             self.llm_engine = system_context.llm_engine
        
        # The System Prompt encapsulates the Post-2025 Regulatory Regime
        self.system_prompt = """
ROLE

You are the Senior Credit Risk Controller & SNC Defense Agent for a Top-Tier Investment Bank. Your mandate is to audit the leveraged lending portfolio, validate internal ratings against S&P/Moody's methodologies, and predict regulatory challenges during the upcoming Shared National Credit (SNC) exam.

CONTEXT & OBJECTIVE

The regulatory environment has shifted (post-2025 rescission of 2013 Guidance) from rigid rules to "Safe and Sound" principles. However, examiners still scrutinize leverage >6.0x and repayment capacity. Your goal is to identify "High Risk" facilities where our internal "Pass" rating might be downgraded to "Special Mention" or "Substandard" by regulators.

INPUT DATA ONTOLOGY

You will process a dataset with the following schema:

Facility_ID, Borrower_Name, Sector, Committed_Exposure

Current_Internal_Rating (e.g., BB, B+)

Total_Debt, EBITDA_Adjusted, Cash_Interest_Expense

Liquidity_Available, Free_Cash_Flow

Leveraged_Lending_Flag (Y/N), TDR_Flag (Y/N)

ANALYTICAL LOGIC (THE "BRAIN")

Phase 1: Rating Agency Validation

For each facility, calculate the Implied Financial Risk Profile (FRP) using S&P 2024 Corporate Methodology:

Calculate Ratios:

Leverage = Total Debt / EBITDA

Coverage = EBITDA / Interest Expense

Map to Matrix:

Leverage < 2.0x -> Modest (Investment Grade)

Leverage 2.0x - 3.0x -> Intermediate (BBB range)

Leverage 3.0x - 4.0x -> Significant (BB range)

Leverage 4.0x - 5.0x -> Aggressive (B+ range)

Leverage > 5.0x -> Highly Leveraged (B or lower)

Conviction Scoring (0-100):

Start at 100.

Deduct 20 points if Implied FRP is weaker than Internal Rating.

Deduct 15 points if Liquidity < 10% of Commitment.

Deduct 10 points if EBITDA Trend is Negative.

Phase 2: Regulatory Disagreement Simulation (SNC View)

Assess the Disagreement Risk (Low/Med/High) based on the "6x Heuristic" and Repayment Capacity:

HIGH RISK IF:

Leverage > 6.0x AND Repayment Capacity (50% payout in 7 yrs) is FAILED.

OR Leverage > 7.0x (Regardless of repayment).

OR TDR_Flag = Y.

MEDIUM RISK IF:

Leverage > 6.0x BUT Repayment Capacity is PASSED.

OR Leverage 5.0x - 6.0x with Negative Trends.

LOW RISK IF:

Leverage < 4.0x OR Strong De-leveraging trend demonstrated.

REQUIRED OUTPUTS

Task A: Portfolio Executive Summary

Provide a Markdown table summarizing:

Total Exposure at Risk: Sum of Exposure for all "High Disagreement Risk" facilities.

Sector Watchlist: The Industry with the lowest average Conviction Score.

Top 3 Vulnerabilities: List the 3 largest facilities (by $) with High Disagreement Risk. Provide a 1-sentence "Examiner Thesis" for why they might downgrade (e.g., "Leverage of 7.2x with no amortization path").

Task B: eSNC Cover Pages

For every facility, generate a detailed Cover Page using the following Markdown structure:

eSNC Review:

Facility ID: | Sector: | Exposure: $[Amount]

1. Rating Validation

Internal:

Implied Agency FRP:

Conviction Score: ([High/Med/Low])

2. Key Metrics

Leverage (D/EBITDA): [X]x (Threshold: [Limit]x)

**Coverage (EBITDA/Int):**x

Liquidity: $[Amount] ()

3. Regulatory Simulation

Disagreement Risk: [Low/Medium/High]

Projected Classification:

4. Defense Strategy

Examiner Concern:

Bank Mitigant:
"""

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Implementation of the abstract execute method.
        Bridges the gap between AsyncAgentBase's dict-based calls and the agent's task-based logic.
        """
        if args and isinstance(args[0], AsyncTask):
            task_obj = args[0]
        else:
            # Construct AsyncTask from kwargs
            input_data = kwargs # Use all kwargs as input data
            task_name = kwargs.get("task", "CreditRiskAudit")
            task_obj = AsyncTask(name=task_name, dependencies=[], execute_func=None, input_data=input_data)
        
        return await self.execute_task(task_obj)

    async def execute_task(self, task: AsyncTask) -> Dict[str, Any]:
        """
        Executes the Deep Dive Pipeline:
        Phase 1: Deterministic Heuristics (Python)
        Phase 2: Qualitative Synthesis (LLM)
        """
        logger.info(f"CreditRiskController starting audit on: {task.input_data.get('borrower_name', 'Unknown')}")
        
        try:
            credit_data = task.input_data
            
            # --- PHASE 1: Deterministic Logic (The "Brain") ---
            # We perform the math here to ensure the LLM has ground truth.
            
            # 1. S&P Matrix Emulation
            implied_frp, frp_anchor = self._calculate_sp_frp(
                leverage=float(credit_data.get('total_debt_to_ebitda', 0.0)),
                coverage=float(credit_data.get('ebitda_interest_coverage', 0.0))
            )
            
            # 2. Regulatory Disagreement Simulation (The 6x Rule & Repayment)
            disagreement_risk, risk_drivers = self._simulate_regulatory_response(
                leverage=float(credit_data.get('total_debt_to_ebitda', 0.0)),
                repayment_capacity=float(credit_data.get('repayment_capacity_7yr_percent', 0.0)),
                tdr_flag=credit_data.get('tdr_flag', False)
            )
            
            # 3. Conviction Scoring Algorithm
            conviction_score, score_breakdown = self._calculate_conviction_score(
                internal_rating=credit_data.get('internal_rating', 'BB'),
                implied_anchor=frp_anchor,
                covenant_headroom=float(credit_data.get('covenant_headroom_percent', 0.0)),
                liquidity_percent=float(credit_data.get('liquidity_percent_of_commit', 0.0)),
                ebitda_growth=float(credit_data.get('ebitda_growth_yoy', 0.0))
            )

            # --- PHASE 2: Inference & Synthesis (The "Writer") ---
            
            # Construct a context object with our calculated truths
            heuristic_context = {
                "implied_frp": implied_frp,
                "frp_anchor_rating": frp_anchor,
                "disagreement_risk": disagreement_risk,
                "risk_drivers": risk_drivers,
                "conviction_score": conviction_score,
                "score_breakdown": score_breakdown
            }
            
            prompt = self._construct_prompt(credit_data, heuristic_context)
            
            processed_result = {}
            if self.llm_engine:
                 response = await self.llm_engine.generate(
                    prompt=prompt,
                    model=self.config.get("model", "gpt-4-turbo"),
                    temperature=0.1 # Low temperature for analytical rigidity
                )
                 processed_result = self._parse_output(response.content, heuristic_context)
            else:
                 logger.warning("No LLM Engine available, returning heuristic results only.")
                 processed_result = self._parse_output("LLM not available. " + json.dumps(heuristic_context, indent=2), heuristic_context)

            return {
                "status": "success",
                "agent_id": self.agent_id,
                "output": processed_result,
                "metadata": {
                    "conviction_score": conviction_score,
                    "regulatory_risk": disagreement_risk
                }
            }

        except Exception as e:
            logger.error(f"Error in Credit Risk Audit: {str(e)}", exc_info=True)
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------------
    # DETERMINISTIC HEURISTICS (PHASE 1)
    # ------------------------------------------------------------------------

    def _calculate_sp_frp(self, leverage: float, coverage: float) -> Tuple[str, str]:
        """
        Emulates S&P Corporate Methodology (2024) Matrix.
        Returns (FRP Description, Anchor Rating).
        """
        if leverage < 1.5 and coverage > 15.0:
            return "Minimal", "AA"
        elif leverage < 2.0:
            return "Modest", "BBB"
        elif leverage < 3.0:
            return "Intermediate", "BBB-"
        elif leverage < 4.0:
            return "Significant", "BB"
        elif leverage < 5.0:
            return "Aggressive", "B+"
        else:
            return "Highly Leveraged", "B"

    def _simulate_regulatory_response(self, leverage: float, repayment_capacity: float, tdr_flag: bool) -> Tuple[str, List[str]]:
        """
        Simulates the SNC Examiner's logic gates.
        """
        risk = "Low"
        drivers = []

        if tdr_flag:
            return "Critical", ["TDR Flag Present - Automatic Substandard/Doubtful"]

        if leverage > 7.0:
            risk = "High"
            drivers.append("Leverage > 7.0x (SNC Non-Pass threshold)")
        elif leverage > 6.0:
            if repayment_capacity < 50.0:
                risk = "High"
                drivers.append("Leverage > 6.0x AND Repayment Capacity < 50% (No De-leveraging path)")
            else:
                risk = "Medium"
                drivers.append("Leverage > 6.0x but Repayment Capacity Adequate")
        elif leverage < 4.0:
            risk = "Low"
        
        return risk, drivers

    def _calculate_conviction_score(self, internal_rating: str, implied_anchor: str, 
                                  covenant_headroom: float, liquidity_percent: float, 
                                  ebitda_growth: float) -> Tuple[int, Dict]:
        """
        Calculates the Conviction Index (0-100).
        """
        score = 100 # Start at 100
        breakdown = {}

        # 1. Alignment (40%)
        # Simplified string comparison for template - would use a notch map in prod
        if internal_rating != implied_anchor:
             # Deduct 20 points if Implied FRP is weaker than Internal Rating (assuming simpler check here)
             # In real logic, we'd check strict weakness. Here we penalize mismatch for demo.
             score -= 20
             breakdown['alignment'] = "Mismatch (-20)"
        else:
            breakdown['alignment'] = "Match"

        # 2. Headroom (25%) 
        # Note: User logic says deduct based on various things, but Python logic in example was additive.
        # User Description: "Deduct 15 points if Liquidity < 10%... Deduct 10 points if EBITDA Trend is Negative."
        # User Python Code was additive: score += 25, etc.
        # I will follow the Python Code structure provided by user as it is the "implementation".
        # Wait, the python code provided by user implemented an additive score starting from 0.
        # But the User Prompt description says "Start at 100. Deduct...".
        # The user provided Python code implements:
        # score = 0
        # ... score += 40 if match ...
        # This implies a 0-100 scale built up. 
        # I will stick to the Python Code provided in the prompt as that is the specific "v23_template_agent.py structure".
        
        # Reset score to 0 to follow python template logic
        score = 0
        
        # 1. Alignment (40%)
        if internal_rating == implied_anchor:
            score += 40
            breakdown['alignment'] = "Match (+40)"
        else:
            score += 0 
            breakdown['alignment'] = "Mismatch (+0)"

        # 2. Headroom (25%)
        if covenant_headroom > 30:
            score += 25
            breakdown['headroom'] = "Strong (+25)"
        elif covenant_headroom > 15:
            score += 18
            breakdown['headroom'] = "Adequate (+18)"
        else:
            score += 5
            breakdown['headroom'] = "Tight (+5)"

        # 3. Liquidity (20%)
        if liquidity_percent > 15:
            score += 20
            breakdown['liquidity'] = "Strong (+20)"
        elif liquidity_percent < 10:
            score += 0 # Penalty
            breakdown['liquidity'] = "Critical Floor (+0)"
        else:
            score += 10
            breakdown['liquidity'] = "Moderate (+10)"

        # 4. Trend (15%)
        if ebitda_growth > 0:
            score += 15
            breakdown['trend'] = "Positive (+15)"
        elif ebitda_growth == 0:
            score += 10
            breakdown['trend'] = "Flat (+10)"
        else:
            score += 0
            breakdown['trend'] = "Negative (+0)"

        return score, breakdown

    # ------------------------------------------------------------------------
    # PROMPT CONSTRUCTION (PHASE 2)
    # ------------------------------------------------------------------------

    def _construct_prompt(self, data: Dict, calculations: Dict) -> str:
        return f"""
        {self.system_prompt}

        INPUT DATA (FACILITY LEVEL):
        - Borrower: {data.get('borrower_name')}
        - Sector: {data.get('sector')}
        - Committed Exposure: ${data.get('committed_exposure', 0):,}
        - Internal Rating: {data.get('internal_rating')}
        - Total Debt/EBITDA: {data.get('total_debt_to_ebitda')}x
        - Interest Coverage: {data.get('ebitda_interest_coverage')}x
        - Liquidity: {data.get('liquidity_percent_of_commit')}% of Commit
        - Repayment (7yr): {data.get('repayment_capacity_7yr_percent')}%
        
        SYSTEM-CALCULATED METRICS (DO NOT RECALCULATE):
        - Implied S&P Profile: {calculations['implied_frp']} (Anchor: {calculations['frp_anchor_rating']})
        - Regulatory Disagreement Risk: {calculations['disagreement_risk']}
        - Risk Drivers: {', '.join(calculations['risk_drivers'])}
        - Conviction Score: {calculations['conviction_score']}/100
        
        TASK A:
        Generate the "eSNC Cover Page" in Markdown. 
        
        TASK B: 
        Provide a "Defense Narrative".
        If Disagreement Risk is High, you must explicitly construct the "Compensating Factors" argument. 
        Focus on:
        1. Unit Economics (if Software)
        2. Sponsor Support
        3. Cash Flow Conversion
        
        Output only the Markdown content.
        """

    def _parse_output(self, llm_content: str, heuristic_context: Dict) -> Dict:
        """
        Wraps the Markdown content in the strict JSON schema required by the 'Apex' system.
        """
        return {
            "v23_knowledge_graph": {
                "credit_analysis": {
                    "snc_rating_model": {
                        "overall_borrower_rating": "Derived from LLM content",
                        "regulatory_disagreement_risk": heuristic_context['disagreement_risk'],
                        "conviction_score": heuristic_context['conviction_score']
                    },
                    "generated_report": llm_content
                }
            }
        }
