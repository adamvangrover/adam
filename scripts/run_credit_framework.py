import sys
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add root to path so we can import core modules
sys.path.append(os.getcwd())

try:
    from core.evaluation.judge import AuditorAgent
    from core.evaluation.verification import SymbolicVerifier
    from core.evaluation.stress_test import AdversarialSimulator
    from core.evaluation.tracing import TraceLog
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def mock_risk_agent(data):
    """
    Simulates a 'Risk Assessment Agent' for demonstration purposes.
    """
    ebitda = data.get("ebitda", 0)
    debt = data.get("total_debt", 0)
    leverage = debt / ebitda if ebitda else 0

    analysis = f"Company '{data.get('company_name')}' has EBITDA of ${ebitda:,.0f} and Total Debt of ${debt:,.0f}. "
    analysis += f"Leverage ratio is {leverage:.2f}x. "

    risk_score = 0.5
    if leverage > 5.0:
        analysis += "WARNING: High leverage detected. Credit risk is elevated. "
        risk_score = 0.9
    elif leverage < 2.0:
        analysis += "Leverage is conservative. "
        risk_score = 0.2

    # Simulate a specific claim that the Symbolic Verifier can check
    # In a real scenario, this would come from the LLM's generation
    if data.get("company_name") == "Acme Finance":
        analysis += "Note: Acme Finance debt is strictly Senior Secured."

    return {
        "analysis": analysis,
        "risk_score": risk_score,
        "details": data
    }

def main():
    print("==================================================")
    print("   ADAM Financial Evaluation Framework Demo")
    print("==================================================")

    tracer = TraceLog(session_id="demo-run-001")

    # --- Step 0: Input Data ---
    input_data = {
        "company_name": "Acme Finance",
        "ebitda": 1000000,
        "total_debt": 3000000,
        "interest_expense": 200000,
        "cash_and_equivalents": 500000
    }
    print(f"\n[Input] Loaded data for {input_data['company_name']}")
    tracer.log_event("Input", "Data Received", input_data)

    # --- Step 1: Run Agent (The Builder) ---
    print("\n[Agent] Running Risk Analysis...")
    agent_output = mock_risk_agent(input_data)
    print(f" > Output: {agent_output['analysis']}")
    print(f" > Risk Score: {agent_output['risk_score']}")
    tracer.log_event("Agent", "Analysis Complete", agent_output)

    # --- Step 2: Symbolic Verification (The Guardrail) ---
    print("\n[Layer 2] Running Symbolic Verification (Unified Knowledge Graph)...")
    verifier = SymbolicVerifier() # Connects to live UKG

    # We explicitly check the claim about "Acme Finance" having "Senior Secured" debt
    # In a full implementation, an extraction model would identify this claim first.
    verification_results = verifier.verify_financial_fact("Acme Finance", "debt_seniority", "Senior Secured")

    # Let's check what the verification result is.
    # (Hint: The mock KG says Acme Finance is 'Subordinated', so this should FAIL)
    print(f" > Verified Claim 'Debt is Senior Secured': {verification_results['verified']}")
    print(f" > Reason: {verification_results.get('reason')}")

    if not verification_results["verified"]:
        print(" > ALERT: Symbolic Verification failed! Flagging for review.")

    tracer.log_event("Verifier", "Fact Check", verification_results)

    # --- Step 3: LLM-as-a-Judge (The Auditor) ---
    print("\n[Layer 1] Running Auditor Agent (Mock Mode for Demo)...")
    # Note: We use mock_mode=True for the demo script to avoid requiring API keys
    # In production, set mock_mode=False to use real Gemini/OpenAI
    auditor = AuditorAgent(mock_mode=True)
    audit_score = auditor.evaluate(input_data, agent_output)

    print(f" > Auditor Score: {audit_score.overall_score}/5.0")
    print(f" > Reasoning: {audit_score.reasoning}")

    # Add confidence score to output
    final_output = tracer.add_confidence_score(agent_output, audit_score.overall_score)
    print(f" > Calculated Confidence: {final_output['_meta']['confidence_score']*100:.1f}%")

    tracer.log_event("Auditor", "Evaluation", audit_score.model_dump())

    # --- Step 4: Stress Testing (The Red Team) ---
    print("\n[Layer 3] Running Adversarial Stress Test...")
    simulator = AdversarialSimulator()

    # 4a. Zombie Attack
    print(" > Injecting 'Zombie' parameters (Low EBITDA, High Debt)...")
    zombie_data = simulator.zombie_attack(input_data)
    zombie_output = mock_risk_agent(zombie_data)
    print(f" > Zombie Scenario Risk Score: {zombie_output['risk_score']} (Should be > 0.8)")
    tracer.log_event("RedTeam", "Zombie Attack", zombie_output)

    # 4b. Sensitivity Probe
    print(" > Running Sensitivity Probe on Total Debt ($3M to $10M)...")

    def agent_wrapper(d):
        return mock_risk_agent(d)["risk_score"]

    sensitivity = simulator.sensitivity_probe(
        agent_wrapper,
        input_data,
        "total_debt",
        3000000,
        10000000,
        5
    )
    for point in sensitivity:
        print(f"   - Debt: ${point['total_debt']/1e6:.1f}M -> Risk Score: {point['risk_score']}")
    tracer.log_event("RedTeam", "Sensitivity Analysis", sensitivity)

    # --- Step 5: Finalize ---
    output_file = "demo_trace.jsonl"
    tracer.save_trace(output_file)
    print(f"\n[System] Full execution trace saved to {output_file}")

if __name__ == "__main__":
    main()
