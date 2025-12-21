from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Optional, List
import json
import os

class PipelineState(TypedDict):
    file_path: str
    pipeline_status: str
    verification_status: Optional[str]
    formatted_artifact: Optional[dict]
    error_message: Optional[str]
    verification_errors: Optional[List[str]]

def load_file(state: PipelineState):
    file_path = state["file_path"]
    if not file_path.endswith(".json"):
        return {
            "pipeline_status": "failed",
            "error_message": "Unsupported file extension"
        }
    try:
        with open(file_path, "r") as f:
            content = json.load(f)
        return {
            "formatted_artifact": content
        }
    except Exception as e:
        return {
            "pipeline_status": "failed",
            "error_message": str(e)
        }

def verify_content(state: PipelineState):
    if state.get("pipeline_status") == "failed":
        return {}

    artifact = state.get("formatted_artifact", {})
    if "title" not in artifact:
        return {
            "pipeline_status": "failed",
            "verification_status": "rejected",
            "verification_errors": ["Report missing 'title' field"]
        }
    return {
        "pipeline_status": "success",
        "verification_status": "verified"
    }

def create_sequential_data_pipeline():
    workflow = StateGraph(PipelineState)
    workflow.add_node("loader", load_file)
    workflow.add_node("verifier", verify_content)

    workflow.set_entry_point("loader")
    workflow.add_edge("loader", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile()
