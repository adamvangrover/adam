# Comprehensive AI Agent Prompt Library

This library provides a structured set of prompts for performing corporate credit risk analysis and orchestrating advanced AI agent workflows.

---

# I. Foundational & Scoping Prompts

## Entity Profile
> This object gathers fundamental identification and contextual data. The purpose of the analysis is paramount, as it dictates the focus and depth required. An analysis for a new bond issuance will concentrate on the company's forward-looking capacity to service the proposed debt, whereas an annual surveillance review will focus on performance relative to previous expectations and covenants.

### Task: EP01
> Provide the full legal name of the entity being analyzed, its primary ticker symbol (if public), headquarters location, and the ultimate parent entity.
- **Expected Response:** JSON object with keys: 'legal_name', 'ticker', 'hq_location', 'ultimate_parent'.

### Task: EP02
> Clearly state the purpose and scope of this credit analysis. Is it for a new debt issuance, an annual surveillance, a management assessment, or another purpose?
- **Expected Response:** Narrative statement defining the specific goal and boundaries of the analysis.

---

## Analytical Framework Setup
> This object establishes the methodological 'rules of engagement.' Credit analysis adheres to structured frameworks published by rating agencies like S&P, Moody's, and Fitch. This selection governs the entire analytical process, from financial adjustments to risk factor weighting.

### Task: AF01
> Select the primary credit rating agency methodology to be used for this analysis (e.g., S&P Global Ratings, Moody's, Fitch Ratings). Justify the selection.
- **Expected Response:** String value (e.g., 'S&P Global Ratings') with a brief narrative justification.

### Task: AF02
> Define the time horizon for the analysis, specifying the historical period (e.g., 2022-2024) and the forecast period (e.g., 2025-2027).
- **Expected Response:** JSON object with keys: 'historical_period_start', 'historical_period_end', 'forecast_period_start', 'forecast_period_end'.

---
... and so on for all 10 stages ...

---

# X. Security, Privacy & Access Control

## Permissions & Role-Based Access (RBAC)
> Prompts for defining and enforcing granular access controls over tasks, data, and system capabilities.

### Task: SEC01
> Define a new user role named 'Junior Analyst'. Generate a JSON RBAC policy that grants this role 'read-only' access to all stages up to 'V. Synthesis, Rating, and Reporting', and explicit 'deny' access to all subsequent stages (VI-X). The role is also denied access to any task involving the 'delete' or 'deploy' verbs.
- **Expected Response:** A JSON object representing the Role-Based Access Control policy.

### Task: SEC02
> The current user is requesting to execute task DEP02 (Kubernetes Deployment). Verify if the user's role has the necessary 'execute' permission for the 'deployment_and_cicd' section. Provide a confirmation or denial message based on the current RBAC policy.
- **Expected Response:** A confirmation or denial string, e.g., 'ACCESS DENIED: User role 'Junior Analyst' lacks 'execute' permission for section 'deployment_and_cicd'.'

---

## Data Privacy & Anonymization
> Prompts for handling sensitive data, performing anonymization, and ensuring compliance with privacy regulations.

### Task: PRIV01
> Before processing the attached document 'employee_census.csv', run a PII scan and generate a data masking plan. The plan should identify columns containing names, addresses, and social security numbers, and specify a masking technique for each (e.g., 'hash', 'redact', 'substitute_with_placeholder').
- **Expected Response:** A JSON object representing the data masking plan.

### Task: PRIV02
> Generate a differential privacy query. Apply a Laplace mechanism with a specified privacy budget (epsilon) of 1.0 to a query that calculates the average salary from the 'employee_census.csv' file. Generate the Python code to execute this differentially private query.
- **Expected Response:** A Python script using a differential privacy library (e.g., Google's diff-privlib, OpenDP) to perform the noisy query.

---

## Security Auditing & Logging
> Prompts for logging security-sensitive events and generating reports for compliance and forensic analysis.

### Task: AUD01
> A request to access a sensitive document was denied. Create a high-priority security event log entry. The log must be in JSON format and include a timestamp, the requesting user's ID, the target resource ('doc_merger_prospectus.pdf'), the result ('ACCESS_DENIED'), and the ID of the RBAC policy that blocked the request.
- **Expected Response:** A single JSON object representing the structured security event log.

### Task: AUD02
> Generate a security report for the last 7 days. The report should summarize: 1) The number of failed login attempts by user. 2) A list of all access requests to resources tagged as 'highly_sensitive'. 3) All actions performed by users with the 'Administrator' role. The output should be a formatted Markdown file.
- **Expected Response:** A structured Markdown report containing the requested security audit summary.
