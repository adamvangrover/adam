# Adam Operational Prompt Library (AOPL-v2.0)

The `prompt_library/AOPL-v2.0/` directory houses the core instructional prompt frameworks for the agent swarm. It acts as the "genetic code" for the agents' cognitive models.

## Structure
- **Sovereign Swarm Architecture**: Prompts defining the Orchestrators, Adversarial Red Team, and Hardened Shield agents are scaffolded as standalone Markdown files under `swarm/`. They are strictly formatted with explicit 'Role (Persona)' and 'Task' sections.
- **Domain-Specific Analysis**: Analytical prompts, such as market or credit analysis agents, are categorized into specific subdirectories like `professional_outcomes/`.

## Dynamic Search Hierarchy
When agents (especially search agents) perform live queries, the prompts dictate a strict graceful fallback strategy known as the Dynamic Search Hierarchy:
1. **Primary**: High-fidelity, direct sources (e.g., Live EDGAR, direct dockets).
2. **Secondary/Fallback**: If primary sources fail or are blocked, the agent must silently and gracefully degrade to trailing market proxies or open-web financial press.
- **Rule**: Agents must *never* hallucinate data if primary sources fail; they must explicitly report the fallback or the failure.
