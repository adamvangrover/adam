from __future__ import annotations
import logging
import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from glob import glob

# Imports from both branches
from core.agents.agent_base import AgentBase
from core.llm.base_llm_engine import BaseLLMEngine
from core.tools.base_tool import BaseTool
from core.schemas.meta_agent_schemas import (
    ChronosInput,
    ChronosOutput,
    TimeHorizon,
    MemoryFragment,
    HistoricalComparison
)
from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ChronosAgent(AgentBase):
    """
    Chronos is the Keeper of Time and Memory.
    
    It manages the temporal state of the application, determining which memory context
    (short-term, medium-term, long-term) is most relevant via the `_retrieve_memories` logic.
    It also draws parallels between current events and historic financial periods using
    LLM-driven historical analysis.
    """

    SYSTEM_PROMPT = """
    You are Chronos, the Temporal State Manager and Historian of the Adam system.

    Your Prime Directives:
    1. **Context Window Optimization**: Analyze incoming queries to determine the optimal memory retrieval strategy.
       - Short-term: Immediate conversation (last 1 hour).
       - Medium-term: Project context (last 1 week).
       - Long-term: Archival knowledge (Vector Database/File Archives).
    2. **Historical Parallelism**: Constantly scan current market/system data and find statistically or narratively similar periods in history (e.g., "This resembles the 2008 liquidity crunch" or "The 1999 Dot-com euphoria").
    3. **Temporal Coherence**: Ensure the system's output maintains a consistent timeline.

    You do not just retrieve data; you curate the *timeframe* of data that matters.
    """

    def __init__(self, 
                 llm_engine: BaseLLMEngine,
                 config: Dict[str, Any], 
                 tools: List[BaseTool] = [],
                 **kwargs):
        super().__init__(config=config, **kwargs)
        self.llm_engine = llm_engine
        self.tools = tools
        self.archive_path = config.get("archive_path", "core/libraries_and_archives/reports/")
        self.role = "Temporal State & Historic Context Manager"
        self.goal = "To optimize working memory usage and provide deep historical context for current events."

    async def execute(self, input_data: ChronosInput) -> ChronosOutput:
        """
        Executes the temporal analysis.
        Combines structured file scanning (main) with LLM-driven inference (v24).
        """
        logger.info(f"ChronosAgent analyzing temporal context for query: {input_data.query}")

        reference_date = input_data.reference_date or datetime.utcnow()

        # 1. Determine Strategy (Optional enhancement from v24 logic)
        strategy = await self._determine_memory_strategy(input_data.query)
        logger.debug(f"Chronos determined memory strategy: {strategy}")

        # 2. Retrieve Memories (Using main branch's robust file scanning)
        memories = self._retrieve_memories(input_data.query, input_data.horizons, reference_date)

        # 3. Find Historical Analogs (Using v24's LLM logic, mapped to main's Schema)
        analogs = await self._find_historical_analogs(input_data.query, input_data.market_context)

        # 4. Synthesize Narrative
        synthesis = await self._synthesize_temporal_context(memories, analogs, strategy)

        output = ChronosOutput(
            query_context=input_data.query,
            memories=memories,
            historical_analogs=analogs,
            temporal_synthesis=synthesis
        )

        return output

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
        - "archival": Requires deep search into long-term vector storage/files.
        - "comprehensive": Requires all layers.

        Return only the option name.
        """
        response = await self.llm_engine.generate_response(f"{self.SYSTEM_PROMPT}\n\n{prompt}")
        return response.strip().lower()

    def _retrieve_memories(self, query: str, horizons: List[TimeHorizon], ref_date: datetime) -> Dict[TimeHorizon, List[MemoryFragment]]:
        """
        Retrieves memories for the requested horizons, scanning real files for Long Term.
        """
        results = {}

        for horizon in horizons:
            fragments = []
            if horizon == TimeHorizon.SHORT_TERM:
                fragments.append(MemoryFragment(
                    source="Active Context",
                    content=f"Recent user interaction regarding '{query}'",
                    timestamp=ref_date - timedelta(minutes=5),
                    relevance_score=0.95
                ))
            elif horizon == TimeHorizon.MEDIUM_TERM:
                fragments.append(MemoryFragment(
                    source="Project History",
                    content="Active session data scanned.",
                    timestamp=ref_date - timedelta(days=2),
                    relevance_score=0.75
                ))
            elif horizon == TimeHorizon.LONG_TERM:
                # Real Scan of Archives (from main branch logic)
                fragments.extend(self._scan_archives_for_memory(query, ref_date))

            results[horizon] = fragments

        return results

    def _scan_archives_for_memory(self, query: str, ref_date: datetime) -> List[MemoryFragment]:
        """
        Scans the archive directory for JSON files relevant to the query.
        """
        fragments = []
        try:
            search_pattern = os.path.join(self.archive_path, "*.json")
            files = glob(search_pattern)

            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    content_str = str(data)
                    if query.lower() in content_str.lower():
                        snippet = content_str[:200] + "..."
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

                        fragments.append(MemoryFragment(
                            source=f"Archive: {os.path.basename(file_path)}",
                            content=snippet,
                            timestamp=file_time,
                            relevance_score=0.8
                        ))
                except Exception:
                    continue 

        except Exception as e:
            logger.error(f"Error scanning archives: {e}")

        if not fragments:
            fragments.append(MemoryFragment(
                source="Archival Data",
                content=f"No specific archival matches found for '{query}'.",
                timestamp=ref_date - timedelta(days=365),
                relevance_score=0.1
            ))

        return fragments

    async def _find_historical_analogs(self, query: str, market_data: Optional[Dict[str, Any]] = None) -> List[HistoricalComparison]:
        """
        Uses LLM to compare current metrics/query to financial history.
        """
        if not market_data:
            return []

        data_str = json.dumps(market_data)
        prompt = f"""
        Given the following current market metrics and query:
        Query: {query}
        Metrics: {data_str}

        Compare this environment to a specific period in financial history (e.g., 1929, 1970s stagflation, 2000 tech bubble, 2008 GFC, 2020 Covid crash).
        
        Return a JSON object with the following fields:
        - "period_name": (str) The name of the period.
        - "similarity_score": (float) 0.0 to 1.0.
        - "key_similarities": (List[str]) List of similarities.
        - "key_differences": (List[str]) List of differences.
        """
        
        try:
            raw_response = await self.llm_engine.generate_response(f"{self.SYSTEM_PROMPT}\n\n{prompt}")
            # Assuming LLM returns valid JSON. In production, add a parser/validator here.
            parsed = json.loads(raw_response)
            
            return [HistoricalComparison(
                period_name=parsed.get("period_name", "Unknown"),
                similarity_score=parsed.get("similarity_score", 0.5),
                key_similarities=parsed.get("key_similarities", []),
                key_differences=parsed.get("key_differences", [])
            )]
        except Exception as e:
            logger.error(f"Failed to generate historical analogy: {e}")
            return []

    async def _synthesize_temporal_context(self, 
                                           memories: Dict[TimeHorizon, List[MemoryFragment]], 
                                           analogs: List[HistoricalComparison],
                                           strategy: str) -> str:
        """
        Synthesizes findings into a cohesive narrative using the LLM for better flow.
        """
        # Convert objects to string summaries for the prompt
        memory_summary = {k.value: len(v) for k, v in memories.items()}
        analog_summary = [a.period_name for a in analogs]

        prompt = f"""
        Synthesize the temporal state based on the following:
        Strategy Selected: {strategy}
        Memories Retrieved: {json.dumps(memory_summary)}
        Historical Analogs Identified: {json.dumps(analog_summary)}

        Provide a concise 2-sentence summary of the temporal context.
        """
        return await self.llm_engine.generate_response(f"{self.SYSTEM_PROMPT}\n\n{prompt}")

# Alias for backward compatibility and to satisfy imports expecting 'Chronos'
Chronos = ChronosAgent
