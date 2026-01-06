import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from core.agents.agent_base import AgentBase
from core.llm.base_llm_engine import BaseLLMEngine
from core.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

class Chronos(AgentBase):
    """
    Chronos is the Keeper of Time and Memory.
    It manages the temporal state of the application, determining which memory context
    (short-term, medium-term, long-term) is most relevant.
    It also draws parallels between current events and historic financial periods.
    """

    SYSTEM_PROMPT = """
    You are Chronos, the Temporal State Manager and Historian of the Adam system.

    Your Prime Directives:
    1. **Context Window Optimization**: Analyze incoming queries to determine the optimal memory retrieval strategy.
       - Short-term: Immediate conversation (last 1 hour).
       - Medium-term: Project context (last 1 week).
       - Long-term: Archival knowledge (Vector Database).
    2. **Historical Parallelism**: Constantly scan current market/system data and find statistically or narratively similar periods in history (e.g., "This resembles the 2008 liquidity crunch" or "The 1999 Dot-com euphoria").
    3. **Temporal Coherence**: Ensure the system's output maintains a consistent timeline.

    You do not just retrieve data; you curate the *timeframe* of data that matters.
    """

    def __init__(self,
                 llm_engine: BaseLLMEngine,
                 tools: List[BaseTool] = [],
                 memory_engine: Any = None,
                 config: Dict[str, Any] = {}):
        super().__init__(config=config)
        self.llm_engine = llm_engine
        self.tools = tools
        self.memory_engine = memory_engine
        self.role = "Temporal State & Historic Context Manager"
        self.goal = "To optimize working memory usage and provide deep historical context for current events."
        self.current_era_analogy = None

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Determines memory strategy and historical context.
        """
        context = kwargs if kwargs else (args[0] if args and isinstance(args[0], dict) else {})

        logger.info("Chronos calculating temporal state...")

        query = context.get("user_query", "")
        market_data = context.get("market_snapshot", {})

        # 1. Determine Memory Strategy
        memory_strategy = await self._determine_memory_strategy(query)

        # 2. Find Historical Analogy (if market data exists)
        analogy = "No market data available for analogy."
        if market_data:
            analogy = await self._find_historical_analogy(market_data)
            self.current_era_analogy = analogy

        return {
            "response": f"Temporal State Set. Strategy: {memory_strategy}. Analogy: {analogy}",
            "temporal_context": {
                "strategy": memory_strategy,
                "historical_analogy": analogy,
                "timestamp": datetime.now().isoformat()
            }
        }

    async def _determine_memory_strategy(self, query: str) -> str:
        """
        Uses LLM to classify the query's temporal need.
        """
        prompt = f"""
        Analyze the following query and determine the best memory retrieval strategy:
        Query: "{query}"

        Options:
        - "immediate": Requires only the last few messages.
        - "project": Requires context from current active files/session.
        - "archival": Requires deep search into long-term vector storage.
        - "comprehensive": Requires all layers.

        Return only the option name.
        """
        response = await self.llm_engine.generate_response(f"{self.SYSTEM_PROMPT}\n\n{prompt}")
        return response.strip().lower()

    async def _find_historical_analogy(self, market_data: Dict[str, Any]) -> str:
        """
        Compares current metrics to encoded historical knowledge.
        """
        data_str = json.dumps(market_data)
        prompt = f"""
        Given the following current market metrics:
        {data_str}

        Compare this environment to a specific period in financial history (e.g., 1929, 1970s stagflation, 2000 tech bubble, 2008 GFC, 2020 Covid crash).
        Provide the period, the similarity score (0-100), and a one-sentence justification.
        """
        return await self.llm_engine.generate_response(f"{self.SYSTEM_PROMPT}\n\n{prompt}")

    def get_context_window_params(self, strategy: str) -> Dict[str, int]:
        """
        Returns token limits/message counts based on strategy.
        """
        strategies = {
            "immediate": {"messages": 10, "vectors": 0},
            "project": {"messages": 50, "vectors": 5},
            "archival": {"messages": 5, "vectors": 20},
            "comprehensive": {"messages": 30, "vectors": 15}
        }
        return strategies.get(strategy, strategies["project"])
