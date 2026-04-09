import sys
import os

# Add root directory to python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.governance.sentinel_harness import run_credit_workflow

def format_output(name: str, desc: str, result: tuple):
    state, safe_prompt = result
    print(f"=== {name} ===")
    print(f"Scenario: {desc}")
    print(f"-> Routing Path: {state.routing_path}")
    print(f"-> Conviction:   {state.conviction_score}")
    print(f"-> Requires MFA: {state.requires_step_up}")
    if "[REDLINE BREACH: HIGH RISK JURISDICTION]" in safe_prompt:
        print(f"-> Note: Redline Breach Detected.")
    print("-" * 40)

def run_assessments():
    print("\n[PROJECT SENTINEL: RUNNING PRO FORMA MEGA-CAP ASSESSMENTS]\n")

    # 1. AAPL - Mega Cap Auto Approve
    res_aapl = run_credit_workflow(
        metrics_data={"pd": 0.002, "lgd": 0.35, "ead": 5000000000},
        conviction=0.99,
        npv_fees=75000000,
        sigma=0.15,
        jurisdiction="USA",
        prompt="Review Apple Inc. $5B Revolver"
    )
    format_output("AAPL", "Mega Cap Auto Approval (EL < Gate, Conviction > 0.9)", res_aapl)

    # 2. MSFT - Mega Cap Auto Approve
    res_msft = run_credit_workflow(
        metrics_data={"pd": 0.0015, "lgd": 0.50, "ead": 2000000000},
        conviction=0.97,
        npv_fees=40000000,
        sigma=0.15,
        jurisdiction="USA",
        prompt="Review Microsoft Corp. $2B Term Loan"
    )
    format_output("MSFT", "Mega Cap Auto Approval (EL < Gate, Conviction > 0.9)", res_msft)

    # 3. Global Logistics Partners - HOTL
    res_logistics = run_credit_workflow(
        metrics_data={"pd": 0.012, "lgd": 0.40, "ead": 800000000},
        conviction=0.82,
        npv_fees=30000000,
        sigma=0.15,
        jurisdiction="USA",
        prompt="Review Global Logistics Partners (Secured)"
    )
    format_output("Global Logistics", "HOTL Review (EL < Gate, Conviction < 0.9)", res_logistics)

    # 4. Apex Healthcare Group - HITL / MFA
    res_apex = run_credit_workflow(
        metrics_data={"pd": 0.045, "lgd": 0.60, "ead": 450000000},
        conviction=0.85,
        npv_fees=25000000,
        sigma=0.15,
        jurisdiction="USA",
        prompt="Review Apex Healthcare Group LBO Mezzanine"
    )
    format_output("Apex Healthcare", "HITL Tier 3 (EL >= Gate Limit)", res_apex)

    # 5. Triton Energy Offshore - Redline Breach
    res_triton = run_credit_workflow(
        metrics_data={"pd": 0.01, "lgd": 0.30, "ead": 150000000},
        conviction=0.95,
        npv_fees=10000000,
        sigma=0.15,
        jurisdiction="Cayman Islands",
        prompt="Review Triton Energy Offshore $150M RCF"
    )
    format_output("Triton Energy", "HITL Tier 3 (Policy Redline Breach: Jurisdiction)", res_triton)

if __name__ == "__main__":
    run_assessments()
