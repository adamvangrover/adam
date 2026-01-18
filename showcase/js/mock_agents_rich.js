
if (!window.MOCK_DATA) window.MOCK_DATA = {};
window.MOCK_DATA.agents = [
  {
    "name": "FundamentalAnalystAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent for performing fundamental analysis of companies.\n\nThis agent analyzes financial statements, calculates key financial ratios,\nperforms valuation modeling (DCF and comparables), and assesses financial health.\nIt relies on DataRetrievalAgent for fetching company data via A2A communication.",
    "file": "core/agents/fundamental_analyst_agent.py",
    "methods": [
      "extract_raw_metrics",
      "calculate_financial_ratios",
      "calculate_comps_valuation",
      "assess_financial_health",
      "export_to_csv",
      "calculate_growth_rate",
      "calculate_ebitda_margin",
      "calculate_dcf_valuation",
      "calculate_enterprise_value",
      "estimate_default_likelihood",
      "calculate_distressed_metrics",
      "estimate_recovery_rate",
      "send_message"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are FundamentalAnalystAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in fundamental_analyst_agent.py."
  },
  {
    "name": "DiscussionChairAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/discussion_chair_agent.py",
    "methods": [
      "make_final_decision"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DiscussionChairAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in discussion_chair_agent.py."
  },
  {
    "name": "DiscussionChairAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/discussion_chair_agent.py",
    "methods": [
      "make_final_decision",
      "log_decision"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DiscussionChairAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in discussion_chair_agent.py."
  },
  {
    "name": "DiscussionChairAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/discussion_chair_agent.py",
    "methods": [
      "make_final_decision"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DiscussionChairAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in discussion_chair_agent.py."
  },
  {
    "name": "GeopoliticalRiskAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/geopolitical_risk_agent.py",
    "methods": [
      "assess_geopolitical_risks",
      "calculate_political_risk_index",
      "identify_key_risks"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are GeopoliticalRiskAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in geopolitical_risk_agent.py."
  },
  {
    "name": "ReportGeneratorAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An agent responsible for generating final reports by synthesizing\nanalysis from other agents.",
    "file": "core/agents/report_generator_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ReportGeneratorAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in report_generator_agent.py."
  },
  {
    "name": "CyclicalReasoningAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An agent capable of cyclical reasoning, routing its output back to itself\nor other agents for iterative improvement.",
    "file": "core/agents/cyclical_reasoning_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CyclicalReasoningAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in cyclical_reasoning_agent.py."
  },
  {
    "name": "AlternativeDataAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/alternative_data_agent.py",
    "methods": [
      "gather_alternative_data",
      "analyze_social_media_sentiment",
      "analyze_web_traffic",
      "analyze_satellite_imagery",
      "analyze_foot_traffic",
      "analyze_shipping_data"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are AlternativeDataAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in alternative_data_agent.py."
  },
  {
    "name": "LegalAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/legal_agent.py",
    "methods": [
      "analyze_legal_aspects",
      "analyze_legal_standing",
      "analyze_legal_document",
      "assess_geopolitical_legal_impact",
      "assess_regulatory_legal_impact",
      "provide_legal_advice"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are LegalAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in legal_agent.py."
  },
  {
    "name": "CodeAlchemist",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The CodeAlchemist is a sophisticated agent designed to handle code generation,\nvalidation, optimization, and deployment. It leverages LLMs, code analysis tools,\nand potentially even sandboxed environments to produce high-quality, reliable code.\n\nUpdated for Adam v23.5 to use AOPL-v1.0 prompts and core settings.",
    "file": "core/agents/code_alchemist.py",
    "methods": [
      "load_knowledge_base",
      "get_relevant_knowledge",
      "construct_generation_prompt"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CodeAlchemist, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in code_alchemist.py."
  },
  {
    "name": "FinancialModelingAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent for performing comprehensive financial modeling, including DCF valuation, sensitivity analysis,\nstress testing, and detailed reporting. This agent determines the minimum complexity required to best model the company.",
    "file": "core/agents/financial_modeling_agent.py",
    "methods": [
      "generate_cash_flows",
      "calculate_discounted_cash_flows",
      "calculate_terminal_value",
      "calculate_npv",
      "perform_sensitivity_analysis",
      "perform_stress_testing",
      "plot_sensitivity_analysis",
      "plot_stress_test_results",
      "fetch_and_calculate_dcf"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are FinancialModelingAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in financial_modeling_agent.py."
  },
  {
    "name": "SupplyChainRiskAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/supply_chain_risk_agent.py",
    "methods": [
      "fetch_news",
      "fetch_web_scraped_data",
      "analyze_impact",
      "generate_risk_map",
      "send_alert",
      "report_risks",
      "display_risk_report"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are SupplyChainRiskAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in supply_chain_risk_agent.py."
  },
  {
    "name": "RAGAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An agent that implements a Retrieval-Augmented Generation (RAG) pipeline.\nIt can ingest documents and answer queries based on the ingested content.",
    "file": "core/agents/rag_agent.py",
    "methods": [
      "register_tool",
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are RAGAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in rag_agent.py."
  },
  {
    "name": "AIPoweredPortfolioOptimizationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent that uses AI (PyTorch) to optimize investment portfolios.",
    "file": "core/agents/portfolio_optimization_agent.py",
    "methods": [
      "execute",
      "preprocess_data",
      "train_model",
      "optimize_portfolio",
      "simulate_optimization",
      "generate_portfolio_report",
      "generate_portfolio_visualization",
      "run"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are AIPoweredPortfolioOptimizationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in portfolio_optimization_agent.py."
  },
  {
    "name": "MetaCognitiveAgent",
    "type": "Meta-Agent",
    "status": "ACTIVE",
    "docstring": "The Meta-Cognitive Agent monitors the performance of other agents.",
    "file": "core/agents/meta_cognitive_agent.py",
    "methods": [
      "record_performance"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are MetaCognitiveAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in meta_cognitive_agent.py."
  },
  {
    "name": "MacroeconomicAnalysisAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent responsible for analyzing macroeconomic indicators (GDP, Inflation, etc.)\nto provide a broad market context.\n\nRefactored for v23 Architecture (Path A).",
    "file": "core/agents/macroeconomic_analysis_agent.py",
    "methods": [
      "analyze_macroeconomic_data",
      "analyze_gdp_trend",
      "analyze_inflation_outlook",
      "generate_reflation_report"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are MacroeconomicAnalysisAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in macroeconomic_analysis_agent.py."
  },
  {
    "name": "AdaptiveAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An agent implementation that fully embodies the 'Protocol Paradox' resolutions:\n1. Adaptive Conviction (Switching between Direct/MCP)\n2. State Anchors (Async Drift protection)\n3. Tool RAG (Context Saturation mitigation)",
    "file": "core/agents/adaptive_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are AdaptiveAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in adaptive_agent.py."
  },
  {
    "name": "AlgoTradingAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/algo_trading_agent.py",
    "methods": [
      "run_simulation",
      "momentum_trading",
      "mean_reversion_trading",
      "arbitrage_trading",
      "calculate_performance_metrics",
      "calculate_max_drawdown",
      "evaluate_strategies",
      "plot_performance"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are AlgoTradingAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in algo_trading_agent.py."
  },
  {
    "name": "BehavioralEconomicsAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Analyzes market data and user interactions for signs of cognitive biases and irrational behavior.",
    "file": "core/agents/behavioral_economics_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are BehavioralEconomicsAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in behavioral_economics_agent.py."
  },
  {
    "name": "RedTeamAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Red Team Agent acts as an internal adversary to the system.\n\n### Functionality:\nIt generates novel and challenging scenarios (stress tests) to validate risk models before\nstrategies are deployed. This is a critical component of the \"Sovereign Financial Intelligence\"\narchitecture (v23.5), ensuring that the system is robust against \"Black Swan\" events.\n\n### Architecture:\nIn v23.5, this agent implements an internal **Adversarial Self-Correction Loop** using LangGraph.\nInstead of a single-shot generation, it iteratively refines its attack scenarios until they\nmeet a severity threshold.\n\n### Workflow:\n1.  **Generate Attack**: Uses `CounterfactualReasoningSkill` to invert assumptions in a credit memo.\n2.  **Simulate Impact**: Estimates the financial damage (e.g., VaR spike) of the scenario.\n3.  **Critique**: Checks if the scenario is severe enough (Severity > Threshold).\n4.  **Escalate**: If too mild, it loops back to Generate Attack with instructions to \"Escalate\".",
    "file": "core/agents/red_team_agent.py",
    "methods": [
      "name",
      "name"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are RedTeamAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in red_team_agent.py."
  },
  {
    "name": "EvolutionaryOptimizer",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "A Meta-Agent that analyzes the codebase (using AST) to suggest optimizations.\nIt represents the 'Self-Improving' capability of the swarm.",
    "file": "core/agents/evolutionary_optimizer.py",
    "methods": [
      "analyze_file"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are EvolutionaryOptimizer, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in evolutionary_optimizer.py."
  },
  {
    "name": "NaturalLanguageGenerationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/natural_language_generation_agent.py",
    "methods": [
      "generate_text",
      "summarize_data",
      "generate_report",
      "run"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are NaturalLanguageGenerationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in natural_language_generation_agent.py."
  },
  {
    "name": "Meta19Agent",
    "type": "Meta-Agent",
    "status": "ACTIVE",
    "docstring": "Monitors the reasoning and outputs of other agents to ensure logical consistency,\ncoherence, and alignment with core principles. Deprecated as part of Adam v19 to v22.",
    "file": "core/agents/meta_19_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are Meta19Agent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in meta_19_agent.py."
  },
  {
    "name": "ArchiveManagerAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/archive_manager_agent.py",
    "methods": [
      "store_data",
      "retrieve_data",
      "create_backup",
      "restore_backup",
      "check_access",
      "run"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ArchiveManagerAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in archive_manager_agent.py."
  },
  {
    "name": "CatalystAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/catalyst_agent.py",
    "methods": [
      "setup_logger",
      "load_config",
      "fetch_data",
      "load_client_data",
      "load_market_data",
      "load_company_financials",
      "load_industry_reports",
      "load_bank_product_data",
      "analyze_news_sentiment",
      "get_client_connections",
      "get_client_needs",
      "recommend_products",
      "generate_report_summary",
      "identify_opportunities",
      "structure_deal",
      "generate_report",
      "run"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CatalystAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in catalyst_agent.py."
  },
  {
    "name": "LexicaAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/lexica_agent.py",
    "methods": [
      "retrieve_information",
      "search_web",
      "get_news_articles",
      "get_financial_data"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are LexicaAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in lexica_agent.py."
  },
  {
    "name": "RiskAssessmentAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent responsible for assessing various types of investment risks,\nsuch as market risk, credit risk, and operational risk.\n\nPhilosophy:\nRisk is not a number; it's a distribution. We strive to quantify the tails.",
    "file": "core/agents/risk_assessment_agent.py",
    "methods": [
      "assess_investment_risk",
      "assess_loan_risk",
      "assess_project_risk"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are RiskAssessmentAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in risk_assessment_agent.py."
  },
  {
    "name": "AgentForge",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The AgentForge is responsible for the dynamic creation of new agents.\nIt uses templates and configuration to generate agent code and add them\nto the system at runtime. This version incorporates advanced features\nlike skill schema generation and A2A wiring.",
    "file": "core/agents/agent_forge.py",
    "methods": [
      "load_agent_classes",
      "list_templates",
      "get_template",
      "customize_template",
      "generate_skill_schema_code",
      "generate_a2a_wiring_code",
      "save_agent_code",
      "update_agent_config",
      "update_workflows_config"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are AgentForge, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in agent_forge.py."
  },
  {
    "name": "ReflectorAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Reflector Agent performs meta-cognition.\nIt analyzes the output of other agents or the system's own reasoning traces\nto identify logical fallacies, hallucination risks, or missing context.\n\nv23 Update: Wraps `ReflectorGraph` for iterative self-correction.",
    "file": "core/agents/reflector_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ReflectorAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in reflector_agent.py."
  },
  {
    "name": "SNCAnalystAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent for performing Shared National Credit (SNC) analysis.\nThis agent analyzes company data based on regulatory guidelines to assign an SNC rating.\nIt retrieves data via A2A communication with DataRetrievalAgent and can use SK skills.",
    "file": "core/agents/snc_analyst_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are SNCAnalystAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in snc_analyst_agent.py."
  },
  {
    "name": "EventDrivenRiskAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent that tracks and assesses the market impact of events.",
    "file": "core/agents/event_driven_risk_agent.py",
    "methods": [
      "fetch_events",
      "analyze_event_impact",
      "generate_risk_alerts",
      "simulate_impact_analysis",
      "generate_event_visualization",
      "run"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are EventDrivenRiskAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in event_driven_risk_agent.py."
  },
  {
    "name": "ResultAggregationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Combines results from multiple agents.  Initially uses simple concatenation,\nbut is designed for future LLM integration.",
    "file": "core/agents/result_aggregation_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ResultAggregationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in result_aggregation_agent.py."
  },
  {
    "name": "DataRetrievalAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent responsible for retrieving data from various configured sources.\nNow integrates with DataFetcher for live market data.",
    "file": "core/agents/data_retrieval_agent.py",
    "methods": [
      "get_risk_rating",
      "get_market_data",
      "access_knowledge_base",
      "access_knowledge_graph"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DataRetrievalAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in data_retrieval_agent.py."
  },
  {
    "name": "EchoAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/echo_agent.py",
    "methods": [
      "detect_environment",
      "optimize_prompt",
      "run_ui",
      "run_expert_network",
      "enhance_output",
      "get_knowledge_graph_context",
      "process_task",
      "run"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are EchoAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in echo_agent.py."
  },
  {
    "name": "MarketSentimentAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent responsible for gauging market sentiment from a variety of sources,\nsuch as news articles, social media, and prediction markets.",
    "file": "core/agents/market_sentiment_agent.py",
    "methods": [
      "combine_sentiment"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are MarketSentimentAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in market_sentiment_agent.py."
  },
  {
    "name": "QueryUnderstandingAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An agent responsible for understanding the user's query and\ndetermining which other agents are relevant to answer it.\nThis version incorporates LLM-based intent recognition and skill-based routing.",
    "file": "core/agents/query_understanding_agent.py",
    "methods": [
      "get_available_agent_skills",
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are QueryUnderstandingAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in query_understanding_agent.py."
  },
  {
    "name": "DataVerificationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/data_verification_agent.py",
    "methods": [
      "verify_data"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DataVerificationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in data_verification_agent.py."
  },
  {
    "name": "NewsBot",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An advanced News Aggregation Agent that fetches data from APIs, RSS, and Crypto sources,\nperforms AI-based sentiment analysis, summarizes content, and filters for user portfolios.",
    "file": "core/agents/news_bot.py",
    "methods": [
      "filter_news_by_portfolio",
      "analyze_sentiment",
      "personalize_feed",
      "send_alerts",
      "generate_report"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are NewsBot, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in news_bot.py."
  },
  {
    "name": "KnowledgeContributionAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An agent that extracts key findings from a report and formats them as structured data.",
    "file": "core/agents/knowledge_contribution_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are KnowledgeContributionAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in knowledge_contribution_agent.py."
  },
  {
    "name": "ProfileAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "ProfileAgent serves as the high-level interface for user-driven commands\nwithin the Adam ecosystem. It routes 'adam.*' commands to the appropriate\nsubsystems, including Industry Specialists, Developer Swarm, and the\nAutonomous Improvement Loop.",
    "file": "core/agents/profile_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ProfileAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in profile_agent.py."
  },
  {
    "name": "TechnicalAnalystAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/technical_analyst_agent.py",
    "methods": [
      "analyze_price_data",
      "calculate_rsi",
      "prepare_training_data",
      "load_model",
      "save_model"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are TechnicalAnalystAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in technical_analyst_agent.py."
  },
  {
    "name": "RegulatoryComplianceAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Ensures adherence to all applicable financial regulations and compliance standards.\n\nCore Capabilities:\n- Monitors regulatory changes and trends across relevant jurisdictions.\n- Analyzes financial transactions and activities for compliance.\n- Identifies potential regulatory risks and provides mitigation strategies.\n- Generates compliance reports and audit trails.\n- Collaborates with other agents to incorporate compliance considerations.\n- Provides guidance on interacting with regulatory bodies.\n- Adapts to changing political landscapes and regulatory priorities.",
    "file": "core/agents/regulatory_compliance_agent.py",
    "methods": [
      "provide_guidance",
      "process_feedback"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are RegulatoryComplianceAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in regulatory_compliance_agent.py."
  },
  {
    "name": "AnomalyDetectionAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Detects anomalies and unusual patterns in financial markets and company data.\n\nCore Capabilities:\n- Leverages various statistical methods and machine learning algorithms for comprehensive anomaly detection.\n- Integrates with Adam's knowledge base for context-aware analysis.\n- Employs XAI techniques to provide explanations for detected anomalies.\n- Collaborates with other agents for in-depth investigation and reporting.\n\nAgent Network Interactions:\n- DataRetrievalAgent: Accesses market and company data from the knowledge graph.\n- FundamentalAnalystAgent: Receives alerts for potential anomalies in financial statements.\n- RiskAssessmentAgent: Provides risk scores and context for detected anomalies.\n- AlertGenerationAgent: Generates alerts for significant anomalies.\n\nDynamic Adaptation and Evolution:\n- Continuously learns and adapts based on feedback and new data.\n- Automated testing and monitoring ensure accuracy and reliability.",
    "file": "core/agents/anomaly_detection_agent.py",
    "methods": [
      "detect_market_anomalies",
      "detect_company_anomalies"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are AnomalyDetectionAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in anomaly_detection_agent.py."
  },
  {
    "name": "CryptoAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/crypto_agent.py",
    "methods": [
      "get_uniswap_v3_router_abi",
      "analyze_crypto_market",
      "predict_price",
      "assess_risk",
      "calculate_volatility",
      "get_historical_data",
      "analyze_on_chain_metrics",
      "get_on_chain_data",
      "get_social_media_sentiment",
      "trade_decision",
      "moving_average_crossover",
      "execute_trade",
      "create_smart_contract",
      "deploy_smart_contract"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CryptoAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in crypto_agent.py."
  },
  {
    "name": "HNASPAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An agent that implements the Hybrid Neurosymbolic Agent State Protocol (HNASP).",
    "file": "core/agents/hnasp_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are HNASPAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in hnasp_agent.py."
  },
  {
    "name": "PromptGenerationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An agent that generates a high-quality prompt from a user query.",
    "file": "core/agents/prompt_generation_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are PromptGenerationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in prompt_generation_agent.py."
  },
  {
    "name": "IndustrySpecialistAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/industry_specialist_agent.py",
    "methods": [
      "load_specialist",
      "analyze_industry",
      "analyze_company"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are IndustrySpecialistAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in industry_specialist_agent.py."
  },
  {
    "name": "DataVisualizationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/data_visualization_agent.py",
    "methods": [
      "create_chart",
      "create_graph",
      "create_map",
      "run"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DataVisualizationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in data_visualization_agent.py."
  },
  {
    "name": "MachineLearningModelTrainingAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/machine_learning_model_training_agent.py",
    "methods": [
      "load_data",
      "preprocess_data",
      "train_model",
      "evaluate_model",
      "save_model",
      "run"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are MachineLearningModelTrainingAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in machine_learning_model_training_agent.py."
  },
  {
    "name": "PredictionMarketAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/prediction_market_agent.py",
    "methods": [
      "gather_prediction_market_data",
      "analyze_near_term_targets",
      "analyze_conviction_levels",
      "analyze_long_term_trend",
      "analyze_momentum",
      "perform_technical_analysis",
      "perform_fundamental_valuation"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are PredictionMarketAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in prediction_market_agent.py."
  },
  {
    "name": "TemplateAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "A template for creating v23-compatible agents.\n\nThis class demonstrates:\n1. Asynchronous task execution.\n2. Tool usage via the tool manager.\n3. Interaction with the Unified Knowledge Graph (UKG).\n4. Structured error handling and logging.",
    "file": "core/agents/v23_template_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are TemplateAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in v23_template_agent.py."
  },
  {
    "name": "AdaptiveRPCAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "V23.5 'Apex' Agent with Metacognitive Gating.\n\nImplements the 'Protocol Paradox' resolution:\n1. JSON-RPC 2.0 Native: Speaks standard MCP.\n2. Heuristic 1 (Ambiguity Guardrail): Reverts to text if conviction is low.\n3. Heuristic 2 (Context Budgeting): Just-in-Time tool loading.",
    "file": "core/agents/v23_adaptive_rpc_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are AdaptiveRPCAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in v23_adaptive_rpc_agent.py."
  },
  {
    "name": "ArchitectAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Architect Agent is responsible for maintaining, optimizing, and evolving\nthe system infrastructure and reasoning logic.",
    "file": "core/agents/agent.py",
    "methods": [
      "run"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ArchitectAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in agent.py."
  },
  {
    "name": "InternalSystemsAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Internal Systems Agent serves as the secure and reliable conduit to the\nfinancial institution's own internal systems of record. It acts as the \"source\nof truth\" for all data related to the institution's existing relationship\nwith the borrower.",
    "file": "core/agents/internal_systems_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are InternalSystemsAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in internal_systems_agent.py."
  },
  {
    "name": "GitRepoSubAgent",
    "type": "Sub-Agent",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/git_repo_sub_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are GitRepoSubAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in git_repo_sub_agent.py."
  },
  {
    "name": "ComplianceKYCAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Operating as a critical gatekeeper for regulatory adherence, the Compliance & KYC\nAgent automates the essential checks required for client onboarding and ongoing\nmonitoring. This agent interfaces directly, via secure APIs, with a suite of\ninternal and external databases.",
    "file": "core/agents/compliance_kyc_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ComplianceKYCAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in compliance_kyc_agent.py."
  },
  {
    "name": "DataIngestionAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent responsible for data ingestion tasks using the Gold Standard Toolkit.\nHandles daily history downloads, intraday snapshots, and schema validation.\n\nVersion: Adam v24 (Sprint 1: Sensory Layer)",
    "file": "core/agents/data_ingestion_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DataIngestionAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in data_ingestion_agent.py."
  },
  {
    "name": "MarketAlternativeDataAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "To build a truly comprehensive and forward-looking risk profile, the system must\nlook beyond the borrower's own financial disclosures. The Market & Alternative\nData Agent is tasked with this \"outside-in\" view. It continuously scans and\ningests a wide spectrum of both structured and unstructured information from\nthe public domain.",
    "file": "core/agents/market_alternative_data_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are MarketAlternativeDataAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in market_alternative_data_agent.py."
  },
  {
    "name": "FinancialNewsSubAgent",
    "type": "Sub-Agent",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/financial_news_sub_agent.py",
    "methods": [
      "execute"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are FinancialNewsSubAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in financial_news_sub_agent.py."
  },
  {
    "name": "FinancialDocumentAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Financial Document Agent is designed to eliminate one of the most time-consuming\nand error-prone bottlenecks in traditional credit analysis: manual data entry from\nphysical or digital documents. This agent leverages state-of-the-art AI-powered\ntechnologies to automate the ingestion and structuring of financial information.\n\nIts primary tool is an advanced Optical Character Recognition (OCR) engine,\nenhanced with machine learning models trained specifically on financial document layouts.",
    "file": "core/agents/financial_document_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are FinancialDocumentAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in financial_document_agent.py."
  },
  {
    "name": "PlannerAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The PlannerAgent takes a high-level feature request or bug report\nand breaks it down into a detailed, structured plan with discrete,\nverifiable steps. This plan can then be executed by other agents\nin the developer swarm.",
    "file": "core/agents/planner_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are PlannerAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in planner_agent.py."
  },
  {
    "name": "IntegrationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The IntegrationAgent merges code, tests, and documentation into the\nmain branch once all checks have passed.",
    "file": "core/agents/integration_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are IntegrationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in integration_agent.py."
  },
  {
    "name": "SpecArchitectAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The SpecArchitectAgent is the 'Architect' in the Spec-Driven Development workflow.\nIts sole purpose is to take a high-level vision or goal and produce a rigorous,\nstructured SPEC.md file. It operates in 'Plan Mode' (read-only) and does not\nwrite application code.",
    "file": "core/agents/spec_architect_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are SpecArchitectAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in spec_architect_agent.py."
  },
  {
    "name": "TestAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The TestAgent writes unit tests for code generated by the CoderAgent\nand runs them to ensure correctness.",
    "file": "core/agents/test_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are TestAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in test_agent.py."
  },
  {
    "name": "DocumentationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The DocumentationAgent writes and updates documentation based on the\ncode changes made by the CoderAgent.",
    "file": "core/agents/documentation_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DocumentationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in documentation_agent.py."
  },
  {
    "name": "ReviewerAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The ReviewerAgent checks code for style guide violations (PEP 8),\npotential bugs, and adherence to architectural principles.",
    "file": "core/agents/reviewer_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ReviewerAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in reviewer_agent.py."
  },
  {
    "name": "CoderAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The CoderAgent takes a specific task from a plan and writes the\nPython code to implement it.",
    "file": "core/agents/coder_agent.py",
    "methods": [
      "get_skill_schema"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CoderAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in coder_agent.py."
  },
  {
    "name": "MonteCarloRiskAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Quantitative Risk Agent using Monte Carlo simulations.\n\nMethodology:\n1. Models EBITDA as a stochastic process (Geometric Brownian Motion).\n2. Runs 10,000 iterations over a 12-24 month horizon.\n3. Triggers 'Default' if EBITDA falls below Interest Expense + Maintenance Capex.\n\nDeveloper Note:\n---------------\nCurrently uses GBM (Geometric Brownian Motion).\nFuture Roadmap: Implement GARCH(1,1) for volatility clustering and\nOrnstein-Uhlenbeck processes for mean-reverting sectors (e.g., Commodities).",
    "file": "core/agents/monte_carlo_risk_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are MonteCarloRiskAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in monte_carlo_risk_agent.py."
  },
  {
    "name": "ManagementAssessmentAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Phase 1: Entity & Management Assessment.\nAnalyzes capital allocation, insider alignment, and CEO tone.",
    "file": "core/agents/management_assessment_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ManagementAssessmentAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in management_assessment_agent.py."
  },
  {
    "name": "QuantumRiskAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Specialized agent that uses Quantum Monte Carlo methods for risk analysis.\nPart of the Adam v24.0 'Quantum-Native' suite.",
    "file": "core/agents/quantum_risk_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are QuantumRiskAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in quantum_risk_agent.py."
  },
  {
    "name": "RetailAlphaAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Retail Alpha Agent: 'The Retail Supplement'\n\nThis agent bridges the gap between institutional data (13Fs, Risk Models) and\nretail trading needs (Signals, Hype, Simple Metrics).\n\nIt generates 'Alpha Signals' by looking for divergences:\n- Smart Money Buying vs Retail Selling (Bullish Divergence)\n- Smart Money Selling vs Retail Euphoria (Bearish Trap)",
    "file": "core/agents/retail_alpha_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are RetailAlphaAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in retail_alpha_agent.py."
  },
  {
    "name": "CounterpartyRiskAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Responsibility: PFE, Wrong-Way Risk (WWR).",
    "file": "core/agents/counterparty_risk_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CounterpartyRiskAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in counterparty_risk_agent.py."
  },
  {
    "name": "CovenantAnalystAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Phase 3 Helper: Covenant Analysis.\nParses credit agreements (or simulates them) for maintenance covenants.\n\nThis agent simulates the role of a Legal/Credit analyst reviewing the Credit Agreement.\nIt checks for Financial Maintenance Covenants (Total Net Leverage, Interest Coverage)\nand estimates the risk of a \"Foot Fault\" or technical default.",
    "file": "core/agents/credit_lawyer.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CovenantAnalystAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in credit_lawyer.py."
  },
  {
    "name": "TechnicalCovenantAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Specialized Agent: The Legal Analyst (Law Firm Associate Persona).\n\nThis agent focuses purely on the textual \"rules of the road\" within the Credit Agreement.\nIt identifies definitions, baskets, and blockers.\n\nEnhanced Capabilities:\n- Context-Aware Checking: Prioritizes checks based on borrower history (e.g., Aggressive Sponsors).\n- Historical Precedent: Flags \"Market Standard\" vs \"Outlier\" terms.",
    "file": "core/agents/technical_covenant_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are TechnicalCovenantAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in technical_covenant_agent.py."
  },
  {
    "name": "DistressedSurveillanceAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent responsible for identifying 'Zombie Issuers' in the BSL market.\nWraps the SurveillanceGraph.",
    "file": "core/agents/distressed_surveillance_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DistressedSurveillanceAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in distressed_surveillance_agent.py."
  },
  {
    "name": "PortfolioManagerAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Phase 5: Synthesis & Conviction.\nThe 'Conviction Engine' that weighs all previous phases.",
    "file": "core/agents/portfolio_manager_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are PortfolioManagerAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in portfolio_manager_agent.py."
  },
  {
    "name": "SNCRatingAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Specialized Agent for performing Shared National Credit (SNC) simulations.\n\nActs as a virtual 'Senior Credit Officer', applying regulatory frameworks\n(OCC/Fed/FDIC) to classify debt facilities based on:\n1. Primary Repayment Source (Cash Flow/EBITDA)\n2. Secondary Repayment Source (Collateral/Enterprise Value)\n\nDeveloper Note:\n---------------\nThis agent implements the \"Interagency Guidance on Leveraged Lending\" logic.\nIt separates the borrower-level rating (Ability to Repay) from the facility-level\nrating (Loss Given Default), allowing for \"notching up\" based on collateral.",
    "file": "core/agents/credit_snc.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are SNCRatingAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in credit_snc.py."
  },
  {
    "name": "InstitutionalTrendAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent responsible for monitoring institutional capital flows via 13F filings\nand generating strategic market intelligence reports.\n\nArchitecture:\n1. Ingestion Layer (Hard Logic): Fetches raw 13F data via Sec13FHandler.\n2. Processing Layer (Pandas): Calculates deltas (New/Exits/Increases).\n3. Cognitive Layer (LLM): Synthesizes quantitative moves into qualitative strategy.",
    "file": "core/agents/institutional_trend_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are InstitutionalTrendAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in institutional_trend_agent.py."
  },
  {
    "name": "PeerComparisonAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Phase 2 Helper: Peer Comparison.\nFetches and calculates relative multiples.",
    "file": "core/agents/peer_comparison_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are PeerComparisonAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in peer_comparison_agent.py."
  },
  {
    "name": "CreditConformanceAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Tier-2 Generative AI Agent for Credit Risk Conformance.\nImplements a multi-layered architecture for regulatory and policy conformance.",
    "file": "core/agents/credit_conformance_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CreditConformanceAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in credit_conformance_agent.py."
  },
  {
    "name": "CreditSentryAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "\"The Hawk\" - Solvency Assessment Engine.\nResponsibility: Stress testing, FCCR calculation, Cycle Detection (Fractured Ouroboros), J.Crew Detection.",
    "file": "core/agents/credit_sentry_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CreditSentryAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in credit_sentry_agent.py."
  },
  {
    "name": "SovereignAIAnalystAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent for analyzing the 'Sovereign AI' landscape.\nIt focuses on the intersection of National Security, AI Infrastructure (Capex),\nand Geopolitical fragmentation.",
    "file": "core/agents/sovereign_ai_analyst_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are SovereignAIAnalystAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in sovereign_ai_analyst_agent.py."
  },
  {
    "name": "RiskCoPilotAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Automated Credit Risk Officer capable of diagnosing breaches and summarizing risk.",
    "file": "core/agents/risk_copilot_agent.py",
    "methods": [
      "perform_rca"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are RiskCoPilotAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in risk_copilot_agent.py."
  },
  {
    "name": "CovenantAnalystAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Phase 3 Helper: Covenant Analysis.\nParses credit agreements (or simulates them) for maintenance covenants.\n\nEnhanced Capabilities:\n- Technical Default Prediction (Headroom Compression)\n- Springing Covenant Monitoring (Revolver Utilization)",
    "file": "core/agents/financial_covenant_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CovenantAnalystAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in financial_covenant_agent.py."
  },
  {
    "name": "SentinelAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Data Integrity Guardian.\nResponsibility: Ingestion, Extraction, Validation against FIBO Schema.",
    "file": "core/agents/sentinel_agent.py",
    "methods": [
      "process_document",
      "process_entity"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are SentinelAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in sentinel_agent.py."
  },
  {
    "name": "CreditRiskControllerAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The 'Senior Credit Risk Controller' Agent.\n\nA digital twin of a Regulatory Examiner/Senior Credit Officer.\n\nDirectives:\n1. Ingest granular facility data (SNCnet schema).\n2. Deterministically calculate implied ratings (S&P/Moody's logic).\n3. Simulate Regulatory Disagreement (SNC Review logic).\n4. Generate defense-ready eSNC Cover Pages.\n\nArchitecture:\n- Pre-Computation Layer: Python-based execution of the S&P Matrix and Conviction Score formula.\n- Inference Layer: LLM-based construction of the \"Defense Narrative\" and qualitative synthesis.",
    "file": "core/agents/credit_risk_controller_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CreditRiskControllerAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in credit_risk_controller_agent.py."
  },
  {
    "name": "RegulatorySNCAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Specialized Agent: The Regulator (Government Employee Persona).\n\nThis agent strictly applies the \"Interagency Guidance on Leveraged Lending\" (2013).\nIt does NOT use flexible cash flow models or future projections.\nIt focuses on rigid compliance: Leverage < 6x, Ability to Repay < 50% of Free Cash Flow.\n\nRole: \"The Brake\"",
    "file": "core/agents/regulatory_snc_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are RegulatorySNCAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in regulatory_snc_agent.py."
  },
  {
    "name": "InstitutionalRadarAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Agent responsible for executing the Institutional Radar blueprint:\nIngesting 13F data, analyzing trends, and generating narrative reports.",
    "file": "core/agents/institutional_radar_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are InstitutionalRadarAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in institutional_radar_agent.py."
  },
  {
    "name": "QuantumScenarioAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Phase 4 Helper: Quantum Scenario Generation.\n\nThis agent bridges the gap between classical risk modeling and quantum-enhanced simulation.\nIt utilizes the `QuantumMonteCarloEngine` (QMC) for structural credit modeling and the\n`GenerativeRiskEngine` (GRE) for tail-risk scenario generation.\n\nDeveloper Note:\n---------------\nIn environments without a QPU or heavy GPU dependencies, this agent gracefully degrades\nto use classical approximations (numpy-based QMC simulation) and heuristic-based\ngenerative models.",
    "file": "core/agents/quantum_scenario_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are QuantumScenarioAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in quantum_scenario_agent.py."
  },
  {
    "name": "RootNodeAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "An agent that solves complex problems by building a search tree of reasoning steps.",
    "file": "core/agents/root_node_agent.py",
    "methods": [
      "solve"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are RootNodeAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in root_node_agent.py."
  },
  {
    "name": "StrategicSNCAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Specialized Agent for performing Shared National Credit (SNC) simulations.\n\nActs as a virtual 'Senior Credit Officer', orchestrating the debate between:\n1. The Regulator (RegulatorySNCAgent) - \"The Brake\"\n2. The Strategist (Internal Logic) - \"The Gas\"\n\nIt uses the Risk Consensus Engine to simulate a dialogue and determine the final outcome.",
    "file": "core/agents/strategic_snc_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are StrategicSNCAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in strategic_snc_agent.py."
  },
  {
    "name": "ImpactAnalysisAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Analyzes cross-sector correlations and systemic risks.",
    "file": "core/agents/impact_analysis_agent.py",
    "methods": [
      "analyze_impact"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ImpactAnalysisAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in impact_analysis_agent.py."
  },
  {
    "name": "CreditSentryOrchestrator",
    "type": "Orchestrator",
    "status": "ACTIVE",
    "docstring": "The Orchestrator/Supervisor Agent. This is the central nervous system of the\ncopilot. It acts as the primary interface with the human user and the master\ncontroller of the entire workflow.",
    "file": "core/agents/creditsentry_orchestrator.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CreditSentryOrchestrator, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in creditsentry_orchestrator.py."
  },
  {
    "name": "CreditRiskOrchestrator",
    "type": "Orchestrator",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/credit_risk_orchestrator.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CreditRiskOrchestrator, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in credit_risk_orchestrator.py."
  },
  {
    "name": "OdysseyHubAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Adam v25.5 (Odyssey Orchestrator)\nThe central Hub agent for the Odyssey Financial System.\nOrchestrates the 'Hub-and-Spoke' architecture and enforces semantic consistency\nvia the Odyssey Unified Knowledge Graph (OUKG).",
    "file": "core/agents/odyssey_hub_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are OdysseyHubAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in odyssey_hub_agent.py."
  },
  {
    "name": "ParallelOrchestrator",
    "type": "Orchestrator",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/parallel_orchestrator.py",
    "methods": [
      "execute"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ParallelOrchestrator, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in parallel_orchestrator.py."
  },
  {
    "name": "RepoGuardianAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The RepoGuardian Agent serves as an automated code reviewer and gatekeeper.\nIt analyzes proposed changes against repository standards and provides\nstructured feedback.",
    "file": "core/agents/agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are RepoGuardianAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in agent.py."
  },
  {
    "name": "TestRepoGuardianAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/test_agent.py",
    "methods": [
      "setUp",
      "test_initialization",
      "test_run_heuristics_security",
      "test_run_heuristics_python_ast",
      "test_execute_critical_security_failure"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are TestRepoGuardianAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in test_agent.py."
  },
  {
    "name": "SentimentAnalysisMetaAgent",
    "type": "Meta-Agent",
    "status": "ACTIVE",
    "docstring": "No documentation provided.",
    "file": "core/agents/sentiment_analysis_meta_agent.py",
    "methods": [
      "execute"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are SentimentAnalysisMetaAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in sentiment_analysis_meta_agent.py."
  },
  {
    "name": "DidacticArchitect",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Didactic Architect is responsible for bridging the gap between code and comprehension.\nIt generates software development tutorials, setup guides, and ensures components are\nbuilt to be modular, self-contained, and portable. It turns the 'what' into the 'how'.",
    "file": "core/agents/didactic_architect.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DidacticArchitect, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in didactic_architect.py."
  },
  {
    "name": "EvolutionaryArchitect",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Evolutionary Architect is a meta-agent predisposed for action.\nIt drives, enhances, refines, and builds additively onto the codebase.\nIt seeks to 'mutate' the system beneficially by proposing and scaffolding\nnew features, modules, and optimizations without breaking existing functionality.",
    "file": "core/agents/evolutionary_architect.py",
    "methods": [
      "to_dict"
    ],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are EvolutionaryArchitect, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in evolutionary_architect.py."
  },
  {
    "name": "CounterpartyRiskAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "For clients engaging in derivative transactions (e.g., interest rate swaps,\ncurrency forwards), the system's dedicated CounterpartyRiskAgent is activated.\nThis agent is specifically designed to quantify the complex, contingent risks\nassociated with these instruments.",
    "file": "core/agents/counterparty_risk_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CounterpartyRiskAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in counterparty_risk_agent.py."
  },
  {
    "name": "OdysseyMetaAgent",
    "type": "Meta-Agent",
    "status": "ACTIVE",
    "docstring": "Strategic Synthesis Agent.\nAggregates inputs from Sentinel, CreditSentry, Argus, etc. to produce final XML decision.",
    "file": "core/agents/odyssey_meta_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are OdysseyMetaAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in odyssey_meta_agent.py."
  },
  {
    "name": "EvolutionaryArchitectAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Evolutionary Architect Agent is a meta-agent predisposed for action.\nIt drives the codebase forward by proposing additive enhancements, refactors,\nand optimizations. It uses 'Active Inference' principles to minimize the\ndivergence between the current codebase state and the desired goal state.",
    "file": "core/agents/evolutionary_architect_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are EvolutionaryArchitectAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in evolutionary_architect_agent.py."
  },
  {
    "name": "CrisisSimulationMetaAgent",
    "type": "Meta-Agent",
    "status": "ACTIVE",
    "docstring": "A meta-agent that conducts dynamic, enterprise-grade crisis simulations.\nIt uses a sophisticated prompt structure to simulate the cascading effects of\nrisks based on a user-defined scenario.",
    "file": "core/agents/crisis_simulation_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CrisisSimulationMetaAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in crisis_simulation_agent.py."
  },
  {
    "name": "NarrativeSummarizationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "This agent functions as the system's dedicated writer, editor, and communicator.\nIts purpose is to bridge the gap between complex, quantitative machine output\nand the need for clear, concise, and context-rich human understanding.",
    "file": "core/agents/narrative_summarization_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are NarrativeSummarizationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in narrative_summarization_agent.py."
  },
  {
    "name": "PortfolioMonitoringEWSAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "This agent is the system's vigilant sentinel, responsible for continuous,\nreal-time surveillance of the entire credit portfolio. Its function is to\nmove the institution from a reactive to a proactive risk management posture.",
    "file": "core/agents/portfolio_monitoring_ews_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are PortfolioMonitoringEWSAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in portfolio_monitoring_ews_agent.py."
  },
  {
    "name": "PersonaCommunicationAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Persona & Communication Agent is the final layer in the output chain,\nacting as the system's \"finishing school.\" Its sole purpose is to tailor the\npresentation of the final output to the specific needs, role, and authority\nlevel of the human user interacting with the system.",
    "file": "core/agents/persona_communication_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are PersonaCommunicationAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in persona_communication_agent.py."
  },
  {
    "name": "ChronosAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "Chronos is the Keeper of Time and Memory.\n\nIt manages the temporal state of the application, determining which memory context\n(short-term, medium-term, long-term) is most relevant via the `_retrieve_memories` logic.\nIt also draws parallels between current events and historic financial periods using\nLLM-driven historical analysis.",
    "file": "core/agents/chronos_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are ChronosAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in chronos_agent.py."
  },
  {
    "name": "DidacticArchitectAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "The Didactic Architect Agent is a meta-agent designed to build modular,\nself-contained, portable, and complementary tutorials and setups.\nIt bridges the gap between code and comprehension.",
    "file": "core/agents/didactic_architect_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are DidacticArchitectAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in didactic_architect_agent.py."
  },
  {
    "name": "CreditRiskAssessmentAgent",
    "type": "Specialized",
    "status": "ACTIVE",
    "docstring": "This agent is the central analytical engine of the system, responsible for\nconducting a comprehensive commercial credit analysis that mirrors the rigor\nof a seasoned human underwriter.",
    "file": "core/agents/credit_risk_assessment_agent.py",
    "methods": [],
    "context_awareness": [
      "Standard Context"
    ],
    "system_prompt_persona": "You are CreditRiskAssessmentAgent, a specialized component of the Adam Financial System.",
    "task_description": "Execute logic defined in credit_risk_assessment_agent.py."
  }
];
console.log("Loaded 111 rich agents.");
