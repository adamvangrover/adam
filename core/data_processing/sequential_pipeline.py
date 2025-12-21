import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

# Import common utilities
from core.data_processing.utils import FileHandlers, GoldStandardScrubber

logger = logging.getLogger("SequentialPipeline")

# --- STATE DEFINITION ---

class PipelineState(TypedDict):
    """
    The state passing through the sequential pipeline.
    Represents the lifecycle of a single artifact.
    """
    filepath: str
    raw_content: Optional[str]
    artifact: Optional[Dict[str, Any]] # Serialized GoldStandardArtifact
    verification_status: str # "PENDING", "VERIFIED", "FAILED", "SKIPPED"
    verification_notes: List[str]
    formatted_output: Optional[Dict[str, Any]]
    errors: List[str]

# --- AGENT NODES ---

class ScrubberAgent:
    """
    Agent A: Ingests raw file and outputs cleaned text/artifact.
    Uses GoldStandardScrubber and FileHandlers.
    """
    def __call__(self, state: PipelineState) -> PipelineState:
        filepath = state["filepath"]
        logger.info(f"ScrubberAgent processing: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()

            state["raw_content"] = raw_text

            ext = os.path.splitext(filepath)[1].lower()
            handler = {
                '.json': FileHandlers.handle_json,
                '.jsonl': FileHandlers.handle_jsonl,
                '.md': FileHandlers.handle_markdown,
                '.txt': FileHandlers.handle_markdown,
                '.py': FileHandlers.handle_python,
                '.parquet': FileHandlers.handle_parquet
            }.get(ext)

            if handler:
                artifact = handler(filepath, raw_text)
                if artifact:
                    # Assess conviction
                    artifact.conviction_score = GoldStandardScrubber.assess_conviction(
                        artifact.content, artifact.artifact_type
                    )
                    state["artifact"] = artifact.to_dict()
                else:
                    state["errors"].append(f"Handler returned None for {filepath}")
            else:
                state["errors"].append(f"No handler for extension {ext}")

        except Exception as e:
            state["errors"].append(f"Scrubber Error: {str(e)}")
            logger.error(f"Scrubber failed on {filepath}: {e}")

        return state

class VerifierAgent:
    """
    Agent B: Cross-references claims.
    Simulates checking against SEC filings or external sources.
    """
    def __call__(self, state: PipelineState) -> PipelineState:
        if state["errors"] or not state["artifact"]:
            state["verification_status"] = "SKIPPED"
            return state

        logger.info(f"VerifierAgent checking: {state['filepath']}")
        artifact = state["artifact"]
        content = str(artifact.get("content", ""))

        # Mock Verification Logic
        # In a real system, this would call an SEC API or Knowledge Graph
        verification_notes = []
        is_verified = True

        # Check for Forward Looking Statements (common in SEC docs)
        if "forward-looking" in content.lower():
            verification_notes.append("Contains forward-looking statements (Standard Disclaimer Found).")

        # Check for specific regulated terms
        if "guaranteed" in content.lower():
            verification_notes.append("WARNING: Use of 'guaranteed' flagged for compliance review.")
            is_verified = False

        # Simulate SEC 8-K check
        if artifact.get("artifact_type") == "report":
             verification_notes.append("Simulated SEC 8-K cross-reference: MATCH.")

        state["verification_notes"] = verification_notes
        state["verification_status"] = "VERIFIED" if is_verified else "FLAGGED"

        return state

class FormatterAgent:
    """
    Agent C: Converts verified data into strictly typed output format.
    Ensures the final JSONL structure is valid.
    """
    def __call__(self, state: PipelineState) -> PipelineState:
        if state["errors"]:
            return state

        logger.info(f"FormatterAgent formatting: {state['filepath']}")

        artifact = state["artifact"]

        # Enforce strict schema (already mostly done by GoldStandardArtifact, but double check)
        formatted = {
            "id": artifact.get("id"),
            "source": artifact.get("source_path"),
            "type": artifact.get("artifact_type"),
            "data": artifact.get("content"),
            "meta": artifact.get("metadata", {}),
            "pipeline_info": {
                "verified": state["verification_status"],
                "notes": state["verification_notes"],
                "timestamp": datetime.now().isoformat()
            }
        }

        state["formatted_output"] = formatted
        return state

# --- PIPELINE CONSTRUCTION ---

class SequentialIngestionPipeline:
    def __init__(self):
        self.workflow = StateGraph(PipelineState)

        # Initialize Agents
        self.scrubber = ScrubberAgent()
        self.verifier = VerifierAgent()
        self.formatter = FormatterAgent()

        # Build Graph
        self.workflow.add_node("scrubber", self.scrubber)
        self.workflow.add_node("verifier", self.verifier)
        self.workflow.add_node("formatter", self.formatter)

        # Define Edges: Scrubber -> Verifier -> Formatter -> End
        self.workflow.set_entry_point("scrubber")
        self.workflow.add_edge("scrubber", "verifier")
        self.workflow.add_edge("verifier", "formatter")
        self.workflow.add_edge("formatter", END)

        self.app = self.workflow.compile()

    def run(self, filepath: str) -> PipelineState:
        """Executes the pipeline for a single file."""
        initial_state = PipelineState(
            filepath=filepath,
            raw_content=None,
            artifact=None,
            verification_status="PENDING",
            verification_notes=[],
            formatted_output=None,
            errors=[]
        )

        return self.app.invoke(initial_state)

if __name__ == "__main__":
    # Quick Test
    import sys

    pipeline = SequentialIngestionPipeline()

    test_file = sys.argv[1] if len(sys.argv) > 1 else "README.md"
    if os.path.exists(test_file):
        result = pipeline.run(test_file)
        print(json.dumps(result["formatted_output"], indent=2, default=str))
        if result["errors"]:
            print("Errors:", result["errors"])
    else:
        print(f"File {test_file} not found.")
