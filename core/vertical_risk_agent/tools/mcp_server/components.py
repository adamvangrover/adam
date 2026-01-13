# Component Registry for Adam MCP Server
# Defines available modules for dynamic loading

COMPONENTS = {
    # Industry Specialists
    "technology": "core.agents.industry_specialists.technology",
    "financials": "core.agents.industry_specialists.financials",
    "energy": "core.agents.industry_specialists.energy",
    "healthcare": "core.agents.industry_specialists.healthcare",
    "industrials": "core.agents.industry_specialists.industrials",
    "materials": "core.agents.industry_specialists.materials",
    "utilities": "core.agents.industry_specialists.utilities",
    "consumer_discretionary": "core.agents.industry_specialists.consumer_discretionary",
    "consumer_staples": "core.agents.industry_specialists.consumer_staples",
    "real_estate": "core.agents.industry_specialists.real_estate",
    "telecom": "core.agents.industry_specialists.telecommunication_services",

    # Developer Swarm
    "coder": "core.agents.developer_swarm.coder_agent",
    "reviewer": "core.agents.developer_swarm.reviewer_agent",
    "test_agent": "core.agents.developer_swarm.test_agent",
    "planner": "core.agents.developer_swarm.planner_agent",

    # Core Engines
    "seal_loop": "core.v23_graph_engine.autonomous_self_improvement",
    "meta_orchestrator": "core.engine.meta_orchestrator",

    # Risk Engines
    "generative_risk": "core.vertical_risk_agent.generative_risk",
    "quantum_risk": "core.v22_quantum_pipeline.qmc_engine",

    # Meta Agents
    "crisis_simulation": "core.agents.meta_agents.crisis_simulation_agent",
    "sentiment_analysis": "core.agents.meta_agents.sentiment_analysis_meta_agent",
    "portfolio_monitoring": "core.agents.meta_agents.portfolio_monitoring_ews_agent"
}

def get_component_module(name: str):
    return COMPONENTS.get(name)
