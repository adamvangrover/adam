from __future__ import annotations
from typing import Any, Dict, List
import logging
import os
import json
from datetime import datetime, timedelta
from glob import glob
from core.agents.agent_base import AgentBase
from core.schemas.meta_agent_schemas import (
    ChronosInput,
    ChronosOutput,
    TimeHorizon,
    MemoryFragment,
    HistoricalComparison
)

class ChronosAgent(AgentBase):
    """
    The Chronos Agent manages temporal state, determining the best working memory
    references for the current context across short, medium, and long terms.
    It also finds comparisons to historic periods by scanning archival data.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.archive_path = config.get("archive_path", "core/libraries_and_archives/reports/")

    async def execute(self, input_data: ChronosInput) -> ChronosOutput:
        """
        Executes the temporal analysis.
        """
        logging.info(f"ChronosAgent analyzing temporal context for query: {input_data.query}")

        reference_date = input_data.reference_date or datetime.utcnow()

        # 1. Retrieve Memories (Real Scan)
        memories = self._retrieve_memories(input_data.query, input_data.horizons, reference_date)

        # 2. Find Historical Analogs (Mock + Heuristic)
        analogs = self._find_historical_analogs(input_data.query)

        # 3. Synthesize
        synthesis = self._synthesize_temporal_context(memories, analogs)

        output = ChronosOutput(
            query_context=input_data.query,
            memories=memories,
            historical_analogs=analogs,
            temporal_synthesis=synthesis
        )

        return output

    def _retrieve_memories(self, query: str, horizons: List[TimeHorizon], ref_date: datetime) -> Dict[TimeHorizon, List[MemoryFragment]]:
        """
        Retrieves memories for the requested horizons, scanning real files for Long Term.
        """
        results = {}

        for horizon in horizons:
            fragments = []
            if horizon == TimeHorizon.SHORT_TERM:
                # Use current query context
                fragments.append(MemoryFragment(
                    source="Active Context",
                    content=f"Recent user interaction regarding '{query}'",
                    timestamp=ref_date - timedelta(minutes=5),
                    relevance_score=0.95
                ))
            elif horizon == TimeHorizon.MEDIUM_TERM:
                # Mock Project History (could be Git log in future)
                fragments.append(MemoryFragment(
                    source="Project History",
                    content=f"Related task executed recently.",
                    timestamp=ref_date - timedelta(days=2),
                    relevance_score=0.75
                ))
            elif horizon == TimeHorizon.LONG_TERM:
                # Real Scan of Archives
                fragments.extend(self._scan_archives_for_memory(query, ref_date))

            results[horizon] = fragments

        return results

    def _scan_archives_for_memory(self, query: str, ref_date: datetime) -> List[MemoryFragment]:
        """
        Scans the `core/libraries_and_archives/reports/` directory for JSON files
        that might be relevant to the query.
        """
        fragments = []
        try:
            # Find all json files
            search_pattern = os.path.join(self.archive_path, "*.json")
            files = glob(search_pattern)

            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Check relevance (Naïve string matching)
                    content_str = str(data)
                    if query.lower() in content_str.lower():
                        # Extract a snippet
                        snippet = content_str[:200] + "..."

                        # Try to find a date
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

                        fragments.append(MemoryFragment(
                            source=f"Archive: {os.path.basename(file_path)}",
                            content=snippet,
                            timestamp=file_time,
                            relevance_score=0.8 # Fixed score for match
                        ))
                except Exception:
                    continue # Skip file if read error

        except Exception as e:
            logging.error(f"Error scanning archives: {e}")

        # Fallback if no files found
        if not fragments:
             fragments.append(MemoryFragment(
                    source="Archival Data",
                    content=f"No specific archival matches found for '{query}'.",
                    timestamp=ref_date - timedelta(days=365),
                    relevance_score=0.1
                ))

        return fragments

    def _find_historical_analogs(self, query: str) -> List[HistoricalComparison]:
        """
        Finds historical comparisons (e.g., market conditions, code evolution phases).
        """
        # Mock logic - in reality would query a Knowledge Graph or Vector DB
        return [
            HistoricalComparison(
                period_name="The Great Refactor of '22",
                similarity_score=0.88,
                key_similarities=["Massive code restructuring", "Adoption of async patterns"],
                key_differences=["Focused on backend", "Less AI integration"]
            )
        ]

    def _synthesize_temporal_context(self, memories: Dict[TimeHorizon, List[MemoryFragment]], analogs: List[HistoricalComparison]) -> str:
        """
        Synthesizes the findings into a cohesive narrative.
        """
        synthesis = "Temporal analysis indicates recurring patterns. "

        # Add details from Short Term
        st = memories.get(TimeHorizon.SHORT_TERM, [])
        if st:
            synthesis += "Immediate context focuses on active user interaction. "

        # Add details from Long Term
        lt = memories.get(TimeHorizon.LONG_TERM, [])
        found_real_archives = any("Archive:" in m.source for m in lt)
        if found_real_archives:
            synthesis += f"Found {len(lt)} relevant archival documents from previous periods. "
        else:
             synthesis += "Long-term archival search yielded limited specific matches. "

        synthesis += "Recommended action: Proceed with caution, referencing historical analogs like 'The Great Refactor of '22'."
        return synthesis
