import re
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from core.prompting.plugins.skeleton_inject_plugins import SkeletonPlugin, SynthesisPlugin, CritiquePlugin
from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

class DataFetcher:
    """Base class/interface for data fetching."""
    def fetch(self, keys: List[str]) -> Dict[str, Any]:
        raise NotImplementedError

class MockDataFetcher(DataFetcher):
    def fetch(self, keys: List[str]) -> Dict[str, Any]:
        full_mock_db = {
            "REVENUE_CURRENT": "$4.5B",
            "REVENUE_YOY_VAR": "-12%",
            "REVENUE_DIRECTION": "deteriorated significantly",
            "EBITDA_MARGIN": "14.2%",
            "MARGIN_DIRECTION": "contracted",
            "NET_LEVERAGE": "4.8x",
            "COVENANT_STATUS": "Compliant",
            "LIQUIDITY_AVAIL": "$350M"
        }
        return {k: full_mock_db.get(k, "N/A") for k in keys}

class JSONFileFetcher(DataFetcher):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._data = self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            logger.warning(f"JSON Data file not found: {self.filepath}")
            return {}
        with open(self.filepath, 'r') as f:
            return json.load(f)

    def fetch(self, keys: List[str]) -> Dict[str, Any]:
        # Graceful handling: Return "N/A" for missing keys
        return {k: self._data.get(k, "N/A") for k in keys}

class WorkflowResult(BaseModel):
    final_text: str
    audit_trace: Dict[str, Any]
    critique: Optional[Dict[str, Any]] = None

class SkeletonInjectWorkflow:
    def __init__(self, llm_client, data_fetcher: Optional[DataFetcher] = None, tone: str = "Institutional"):
        """
        Args:
            llm_client: LLM interface.
            data_fetcher: Strategy for fetching data.
            tone: Tone configuration (e.g. "Bearish", "Neutral") injected into system prompts.
        """
        self.llm_client = llm_client
        self.data_fetcher = data_fetcher or MockDataFetcher()
        self.tone = tone
        self.workflow_id = str(uuid.uuid4())

        # Plugins
        self.skeleton_plugin = SkeletonPlugin()
        self.synthesis_plugin = SynthesisPlugin()
        self.critique_plugin = CritiquePlugin()

    def run(self, context: str, skip_critique: bool = False) -> WorkflowResult:
        logger.info(f"Starting Skeleton & Inject Workflow [{self.workflow_id}]")
        audit_trace = {
            "workflow_id": self.workflow_id,
            "timestamp": datetime.utcnow().isoformat(),
            "tone": self.tone
        }

        # Phase 1
        logger.info("Phase 1: Generating Skeleton")
        skeleton_messages = self.skeleton_plugin.render_messages({
            "context": context,
            "tone": self.tone
        })

        raw_skeleton = self.llm_client.generate(skeleton_messages)
        skeleton_output = self.skeleton_plugin.parse_response(raw_skeleton)
        skeleton_text = skeleton_output.skeleton_text
        audit_trace["phase_1_output"] = skeleton_text

        # Middleware
        placeholders = self._extract_placeholders(skeleton_text)
        logger.info(f"Extracted {len(placeholders)} placeholders")

        truth_data = self.data_fetcher.fetch(placeholders)
        audit_trace["injected_data"] = truth_data

        # Phase 2
        logger.info("Phase 2: Synthesis")
        synthesis_messages = self.synthesis_plugin.render_messages({
            "draft_text": skeleton_text,
            "truth_data": truth_data,
            "tone": self.tone
        })

        raw_final = self.llm_client.generate(synthesis_messages)
        final_output = self.synthesis_plugin.parse_response(raw_final)
        audit_trace["phase_2_output"] = final_output.final_text

        critique_data = None
        if not skip_critique:
            logger.info("Phase 3: Critique & Sign-off")
            critique_messages = self.critique_plugin.render_messages({
                "final_text": final_output.final_text
            })
            raw_critique = self.llm_client.generate(critique_messages)
            critique_output = self.critique_plugin.parse_response(raw_critique)
            critique_data = critique_output.model_dump()
            audit_trace["critique"] = critique_data

        logger.info("Workflow Complete")
        return WorkflowResult(
            final_text=final_output.final_text,
            audit_trace=audit_trace,
            critique=critique_data
        )

    def _extract_placeholders(self, text: str) -> list[str]:
        pattern = r"\{\{\s*([A-Z0-9_]+)\s*\}\}"
        matches = re.findall(pattern, text)
        return list(set(matches))
