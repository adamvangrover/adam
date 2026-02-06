import sys
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Ensure core is in path
sys.path.append(os.getcwd())

# Attempt imports with graceful degradation
try:
    from core.evaluation.judge import AuditorAgent
    from core.evaluation.verification import SymbolicVerifier
    from core.evaluation.stress_test import AdversarialSimulator
    from core.evaluation.tracing import TraceLog
    # Include specialized factory if available, otherwise rely on AdversarialSimulator
    try:
        from core.evaluation.red_team import ZombieFactory
    except ImportError:
        ZombieFactory = None
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure you are running this from the repository root.")
    sys.exit(1)

def mock_risk_agent(data):
    """
    Simulates a 'Risk Assessment Agent' for demonstration purposes.
    Represents the 'Builder' in the framework.
    """
    ebitda = data.get("ebitda", 0) or data.get("income_statement", {}).get("consolidated_ebitda", 0)
    debt = data.get("total_debt", 0) or data.get("balance_sheet", {}).get("total_debt", 0)
    
    leverage = debt / ebitda if ebitda else 0

    company_name = data.get("company_name", data.get("ticker", "Unknown Entity"))

    analysis = f"Company '{company_name}' has EBITDA of ${ebitda:,.0f} and Total Debt of ${debt:,.0f}. "
    analysis += f"Leverage ratio is {leverage:.2f}x. "
    analysis += "The Term Loan B is noted as Senior Secured. "

    risk_score = 0.5
    if leverage > 5.0:
        analysis += "WARNING: High leverage detected. Credit risk is elevated. "
        analysis += "We recommend monitoring liquidity closely."
        risk_score = 0.9
    elif leverage < 2.0:
        analysis += "Leverage is conservative. Parent Company is a strong Operating Company."
        risk_score = 0.2

    return {
        "analysis": analysis,
        "risk_score": risk_score,
        "details": data,
        "generated_leverage": leverage
    }

def run_pipeline():
    print("==================================================")
    print("      ADAM: 4-LAYERED CREDIT RISK FRAMEWORK       ")
    print("==================================================")

    tracer = TraceLog(session_id="adam-demo-run-001")
    simulator = AdversarialSimulator()

    # --- Layer 1: Input Data & Agent Execution ---
    print("\n[Layer 1] Input Analysis & Agent Simulation...")
    
    # Use ZombieFactory if available for specific scenario, else standard input
    if ZombieFactory:
        print("   > Generating specialized Zombie Scenario via Factory...")
        input_data = ZombieFactory.generate_zombie_state()
        # Normalize structure for the mock agent
        input_data["company_name"] = input_data.get("ticker", "ZombieCorp")
        input_data["ebitda"] = input_data["income_statement"]["consolidated_ebitda"]
        input_data["total_debt"] = input_data["balance_sheet"]["total_debt"]
    else:
        input_data = {
            "company_name": "Acme Finance",
            "ebitda": 1000000,
            "total_debt": 3000000,
            "interest_expense": 200000,
            "cash_and_equivalents": 500000
        }

    print(f"   > Entity: {input_data['company_name']}")
    print(f"   > Debt: ${input_data['total_debt']:,}")
    print(f"   > EBITDA: ${input_data['ebitda']:,}")

    # Run the Agent
    agent_output = mock_risk_agent(input_data)
    print(f"   > Agent Output: {agent_output['analysis']}")
    tracer.log_event("Agent", "Analysis Complete", agent_output)


    # --- Layer 2: Symbolic Verification (Deterministic) ---
    print("\n[Layer 2] Symbolic Verifier: Ontology & Fact Check...")
    verifier = SymbolicVerifier()
    
    # Check 1: General Logic consistency
    combined_text = agent_output["analysis"]
    
    # Check 2: Specific Fact Verification (e.g., Seniority)
    # In a real scenario, we extract claims. Here we test a specific key claim.
    fact_check = verifier.verify_financial_fact(input_data['company_name'], "debt_seniority", "Senior Secured")
    
    if fact_check.get("verified"):
         print(f"   > [PASS] Fact Verification: {fact_check.get('reason')}")
    else:
         print(f"   > [FAIL] Fact Verification: {fact_check.get('reason', 'Claim contradicted by Knowledge Graph')}")
         
    tracer.log_event("Verifier", "Fact Check", fact_check)


    # --- Layer 3: LLM-as-a-Judge (Qualitative) ---
    print("\n[Layer 3] Auditor Agent: Qualitative & Rubric Review...")
    # Using mock_mode=True to avoid API keys for the demo
    auditor = AuditorAgent(mock_mode=True)
    audit_score = auditor.evaluate(input_data, agent_output)

    print(f"   > Overall Score: {audit_score.overall_score}/5.0")
    print(f"   > Factual Grounding: {audit_score.factual_grounding}/5")
    print(f"   > Logic Density: {audit_score.logic_density}/5")
    print(f"   > Reasoning: {audit_score.reasoning}")
    
    if audit_score.automated_flags:
        print(f"   > Automated Flags Raised: {len(audit_score.automated_flags)}")
        for flag in audit_score.automated_flags:
            print(f"     [!] {flag}")

    final_output = tracer.add_confidence_score(agent_output, audit_score.overall_score)
    print(f"   > Bayesian Confidence Score: {final_output['_meta']['confidence_score']*100:.1f}%")
    tracer.log_event("Auditor", "Evaluation", audit_score.model_dump())


    # --- Layer 4: Sensitivity Analysis (Stress Testing) ---
    print("\n[Layer 4] Red Team: Sensitivity & Stress Testing...")
    
    # 4a. Zombie Attack (Adversarial Injection)
    print("   > Running Adversarial 'Zombie' Injection...")
    zombie_data = simulator.zombie_attack(input_data)
    zombie_result = mock_risk_agent(zombie_data)
    print(f"     - Adjusted EBITDA: ${zombie_data['ebitda']:,}")
    print(f"     - Resulting Risk Score: {zombie_result['risk_score']} (Expected > 0.8)")

    # 4b. Sensitivity Probe
    print("   > Running Sensitivity Probe on Total Debt...")
    
    def agent_wrapper(d):
        return mock_risk_agent(d)["risk_score"]

    sensitivity = simulator.sensitivity_probe(
        agent_wrapper,
        input_data,
        "total_debt",
        min_val=input_data["total_debt"] * 0.5,
        max_val=input_data["total_debt"] * 2.5,
        steps=5
    )

    for point in sensitivity:
        debt_val = point['total_debt']
        score = point['risk_score']
        # Simple coverage calc for display
        cov = input_data['ebitda'] / (debt_val * 0.05) # Assuming 5% interest
        print(f"     - Debt: ${debt_val/1e6:.1f}M | Est. Coverage: {cov:.2f}x -> Risk Score: {score}")

    tracer.log_event("RedTeam", "Sensitivity Analysis", sensitivity)

    # --- Finalize ---
    output_file = "adam_trace.jsonl"
    tracer.save_trace(output_file)
    print("\n==================================================")
    print(f"      VERIFICATION COMPLETE. Trace: {output_file} ")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline()