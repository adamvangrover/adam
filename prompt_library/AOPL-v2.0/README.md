# Adam Operational Prompt Library (AOPL-v2.0)

This directory contains the Adam Operational Prompt Library (AOPL), establishing standard, domain-specific instructions for the LLM agents across the system.

## Organization
Domain-specific analytical prompts, such as market or credit analysis agents, belong in subdirectories like `professional_outcomes/`. Prompts for core architecture and agent operations reside in the `swarm/` directory or root.

## Dynamic Search Hierarchy
When building search agents, prompts must mandate a dynamic search hierarchy with explicit graceful fallbacks. This ensures that the multi-agent system degrades gracefully instead of failing entirely or hallucinating when external services are unavailable.

*   **Primary Sources:** Live integrations (e.g., `yfinance`, `duckduckgo_search`, `neo4j`, `sqlalchemy`).
*   **Secondary Proxies:** If primary sources fail, agents gracefully fallback to secondary public proxies (e.g., trailing market proxies for EDGAR delays, open-web financial press for blocked Dockets).
*   **Mock Fallbacks:** If secondary proxies fail, the agent should utilize internal static 'MockLLM' fallbacks.

**Rule:** Agents are explicitly instructed never to hallucinate data if primary sources fail. They must either find proxy data or state that the data is unavailable.
