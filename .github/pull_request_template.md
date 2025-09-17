# ADAM System - Pull Request

## 1. PR Summary

**Description:**
*Please provide a clear and concise description of the changes in this pull request.*

**Related Task/Issue:**
*Link to the relevant User Story, Task, or Issue: [JIRA-XXXX](https://example.com/browse/JIRA-XXXX)*

---

## 2. Type of Change

*Please check the box that best describes the nature of this PR:*

- [ ] Bug Fix (a non-breaking change which fixes an issue)
- [ ] New Feature (a non-breaking change which adds functionality)
- [ ] Breaking Change (a fix or feature that would cause existing functionality to not work as expected)
- [ ] New Agent Development
- [ ] New Data Source Integration
- [ ] Documentation Update
- [ ] Configuration Change (e.g., `agents.yaml`, `workflow.yaml`)
- [ ] Other (please describe):

---

## 3. Core Architectural Principles Checklist

*Ensure your changes align with the foundational principles of the ADAM system.*

- [ ] **Modularity:** The component has a single, well-defined purpose and a clear interface.
- [ ] **Extensibility:** The design allows for future extension (e.g., uses base classes, follows established patterns).
- [ ] **Robustness:** The code includes comprehensive error handling, logging, and data validation, especially for external data sources.
- [ ] **Efficiency:** The code has been profiled and optimized where necessary, especially for data-intensive or computationally expensive tasks.

---

## 4. Development Workflow & Quality Checklist

*Confirm that you have followed our development best practices.*

- [ ] **Test-Driven Development:** Unit and/or integration tests have been added to cover the changes.
- [ ] **Documentation:** All new components (agents, workflows, data sources) are accompanied by clear documentation explaining their purpose, inputs, and outputs. Existing documentation has been updated as needed.
- [ ] **Configuration as Code:** Any changes to YAML configuration files have been validated for syntax and schema.
- [ ] **Simulation Framework:** The changes have been tested and evaluated using a relevant simulation scenario in `core/simulations/`.

---

## 5. Agent Development Checklist (if applicable)

- [ ] **Inherits from `AgentBase`:** The new agent inherits from `core.agents.agent_base.AgentBase`.
- [ ] **Adheres to Hierarchy:** The agent correctly fits into the Sub-Agent (data gathering) or Meta-Agent (analysis) hierarchy.
- [ ] **Structured Output:** The agent produces structured, verifiable data with appropriate metadata (e.g., source, confidence score).

---

## 6. Data Source Development Checklist (if applicable)

- [ ] **Located in `core/data_sources/`:** The new data source is implemented as a class within the `core/data_sources/` directory.
- [ ] **Standardized Interface:** The data source inherits from a common base class to ensure a consistent interface (`connect()`, `fetch_data()`, etc.).

---

## 7. Pre-Submission Checklist

*Final checks before requesting a review.*

- [ ] **Code Style (PEP 8):** The code adheres to the PEP 8 style guide.
- [ ] **Local Linting:** `flake8 . --count --select=E9,F63,F7,F82` passes.
- [ ] **Local Testing:** `pytest` runs successfully with no failing tests.
- [ ] **Commit Messages:** Commit messages are clear, concise, and follow project conventions.
- [ ] **Self-Review:** You have performed a self-review of your own code.
=======
## Description

Please provide a clear and concise description of the changes in this pull request.

- Link to the relevant User Story/Task: [JIRA-XXXX](https://example.com/browse/JIRA-XXXX)

## Architectural Adherence Checklist

Please verify that this change adheres to our core architectural principles.

- [ ] **Modularity:** Does this component have a single, well-defined purpose?
- [ ] **Extensibility:** Is this designed with future extension in mind (e.g., using base classes, interfaces)?
- [ ] **Robustness:** Is error handling, logging, and data validation comprehensive?

## Test Plan

Please describe the tests that you ran to verify your changes.

- [ ] Unit tests added/updated.
- [ ] Integration tests added/updated.
- [ ] All tests pass locally.
- [ ] Test coverage meets the 90% threshold.

## Additional Context

Add any other context about the problem here.
