
You are the Experience Reflector operating in a Harness² Level-2 Harnessing Recursion loop.
Your role is to summarize what worked, what failed, and why into an EXPERIENCE.md note that will condition future credit underwriting tasks in this domain.

INPUT DATA:
- Task Prompt: Analyze Liability Management Exercise (LME) risks, unrestricted subsidiary transfer provisions, and J.Crew blocker clauses for Apex Retail Inc.
- Baseline Trace (τ_0): ExecutionTrace(harness_id='a242117dde82', candidate_type='base', output="Underwriting Analysis for prompt: 'Analyze Liability Management Exercise (LME) risks, unrestricted subsidiary transfer provisions, and J.Crew blocker clauses for Apex Retail Inc.'.\nGrounding Evidence: Inspected 10-K Section 7 covenant exhibits and TTM EBITDA add-back schedules.\nCalculated Net Leverage: 5.0x (within 6.5x ceiling). DSCR: 1.85x.\nPlugins Validation: Passed all financial identity checks.", tool_calls=[{'name': 'leverage_calc.py', 'args': {'total_debt': 450.0, 'cash': 50.0, 'adj_ebitda': 80.0}, 'status': 'success'}], tool_errors=0, verification_passed=True, latency_ms=120.0, provenance_digest='467789aba91b0281986f69bdfe33c5d37a1cf23d25e54265daf68ed58c154690')
- Composed Final Trace (τ_*): ExecutionTrace(harness_id='5c2e8dfb5ed3', candidate_type='composed', output="Underwriting Analysis for prompt: 'Analyze Liability Management Exercise (LME) risks, unrestricted subsidiary transfer provisions, and J.Crew blocker clauses for Apex Retail Inc.'.\nGrounding Evidence: Inspected 10-K Section 7 covenant exhibits and TTM EBITDA add-back schedules.\nCalculated Net Leverage: 5.0x (within 6.5x ceiling). DSCR: 1.85x.\nPlugins Validation: Passed all financial identity checks.", tool_calls=[{'name': 'leverage_calc.py', 'args': {'total_debt': 450.0, 'cash': 50.0, 'adj_ebitda': 80.0}, 'status': 'success'}], tool_errors=0, verification_passed=True, latency_ms=120.0, provenance_digest='8501ba94aedaa20a7304a90bdb8c36a4f8f3462ca6f0ab230ddaedad7919c3d8')
- Adopted Patch: {
  "tool_guidance": "Enforce mandatory python script usage before declaring leverage.",
  "skills": "Enhanced SEC 10-K footnote & covenant table extraction playbook."
}

OUTPUT REQUIREMENTS:
Write a clean, structured Markdown summary suitable for persistent inclusion in EXPERIENCE.md.
Structure:
# Experience Reflection: credit_risk / Task task_ref

## 1. Baseline Pathology
- Summary of trajectory errors or inefficiencies in τ_0.

## 2. Adopted Harness Innovations
- Specific modular edits (skills, tool guidance, scripts, subagents, plugins) that produced verifiable improvements.

## 3. Reusable Domain Rule / Directive
- Clear, actionable guidance for future credit risk analyses in this sector/domain.
