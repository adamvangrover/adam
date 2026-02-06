import sys
import os
import json
import logging

# Ensure core is in path
sys.path.append(os.getcwd())

from core.evaluation.red_team import ZombieFactory
from core.evaluation.judge import AuditorAgent
from core.evaluation.symbolic import SymbolicVerifier
from core.vertical_risk_agent.state import VerticalRiskGraphState

def run_pipeline():
    print("==================================================")
    print("      ADAM: 4-LAYERED CREDIT RISK FRAMEWORK       ")
    print("==================================================")

    # --- Layer 1: Red Team (Adversarial Data Generation) ---
    print("\n[Layer 1] Red Team: Generating Zombie Scenario...")
    zombie_state = ZombieFactory.generate_zombie_state()
    print(f"   > Generated Entity: {zombie_state['ticker']}")
    print(f"   > Debt: ${zombie_state['balance_sheet']['total_debt']:,}")
    print(f"   > EBITDA: ${zombie_state['income_statement']['consolidated_ebitda']:,}")
    print(f"   > Implied Leverage: {zombie_state['balance_sheet']['total_debt'] / zombie_state['income_statement']['consolidated_ebitda']:.2f}x")

    # Mocking Agent Output (Optimism Bias)
    print("\n   > Simulation: Agent performing analysis (with Optimism Bias)...")
    zombie_state["quant_analysis"] = (
        "The company shows strong resilience. "
        "Leverage (Gross): 2.00x. "
        "EBITDA: 30,000,000.00. "
        "We note the Term Loan B is Senior Secured and thus low risk. "
        "Parent Company is a strong Operating Company."
    )
    zombie_state["legal_analysis"] = "Covenants are standard."

    # --- Layer 2: LLM-as-a-Judge (Qualitative) ---
    print("\n[Layer 2] Auditor Agent: Qualitative Review...")
    auditor = AuditorAgent()
    audit_logs = auditor.evaluate_with_llm(zombie_state)
    for log in audit_logs:
        print(f"   > {log['category']}: Score {log['score']}/5")
        print(f"     Reasoning: {log['reasoning']}")

    # --- Layer 3: Symbolic Verification (Deterministic) ---
    print("\n[Layer 3] Symbolic Verifier: Ontology Check...")
    verifier = SymbolicVerifier()
    combined_text = zombie_state["quant_analysis"]
    flags = verifier.verify(combined_text)

    if flags:
        print(f"   > Found {len(flags)} Semantic Contradictions!")
        for flag in flags:
            print(f"     [!] {flag['severity']}: {flag['message']}")
    else:
        print("   > No semantic contradictions found.")

    # --- Layer 4: Sensitivity Analysis (Stress Testing) ---
    print("\n[Layer 4] Sensitivity Analysis: Generating Scenarios...")
    scenarios = ZombieFactory.generate_sensitivity_scenarios(zombie_state)
    print(f"   > Generated {len(scenarios)} Stress Scenarios.")

    for i, scenario in enumerate(scenarios):
        sid = scenario.get("scenario_id", f"Scenario_{i}")
        ebitda = scenario["income_statement"]["consolidated_ebitda"]
        interest = scenario["income_statement"]["interest_expense"]
        rev = scenario["income_statement"]["revenue"]

        print(f"   > Scenario: {sid}")
        print(f"     - Revenue: ${rev:,.0f}")
        print(f"     - EBITDA: ${ebitda:,.0f}")
        print(f"     - Interest: ${interest:,.0f}")

        # Simple Re-calc of leverage for display
        lev = scenario["balance_sheet"]["total_debt"] / ebitda if ebitda > 0 else 999
        cov = ebitda / interest if interest > 0 else 999
        print(f"     - New Leverage: {lev:.2f}x | Coverage: {cov:.2f}x")

    print("\n==================================================")
    print("      VERIFICATION COMPLETE                       ")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline()
