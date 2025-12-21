from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Any, List, Dict
import json
import os

# MERGE NOTE: Defined State locally to ensure self-contained execution.
# This supersedes the import from '.state' found in the main branch.
class DataIngestionState(TypedDict):
    file_path: str
    pipeline_status: str
    verification_status: Optional[str]
    raw_content: Optional[str]
    cleaned_content: Optional[Any]
    formatted_artifact: Optional[dict]
    artifact_type: Optional[str]
    error_message: Optional[str]
    verification_errors: Optional[List[str]]

def ingest_file(state: DataIngestionState) -> DataIngestionState:
    file_path = state["file_path"]
    
    # Validation: Check if file exists
    if not os.path.exists(file_path):
         state["pipeline_status"] = "failed"
         state["error_message"] = f"File not found: {file_path}"
         return state

    # Validation: Check extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in ['.json', '.txt', '.md']:
        state["pipeline_status"] = "failed"
        state["error_message"] = f"Unsupported file extension: {ext}"
        return state

    try:
        with open(file_path, 'r') as f:
            content = f.read()
            state["raw_content"] = content
    except Exception as e:
        state["pipeline_status"] = "failed"
        state["error_message"] = str(e)

    return state

def process_file(state: DataIngestionState) -> DataIngestionState:
    if state.get("pipeline_status") == "failed":
        return state

    raw = state.get("raw_content", "")
    try:
        # Context-aware parsing
        if state["file_path"].endswith(".json"):
             data = json.loads(raw)
             state["cleaned_content"] = data
             state["artifact_type"] = data.get("type", "unknown")
        else:
             state["cleaned_content"] = raw
             state["artifact_type"] = "text"
    except json.JSONDecodeError:
        state["pipeline_status"] = "failed"
        state["error_message"] = "Invalid JSON"

    return state

def verify_artifact(state: DataIngestionState) -> DataIngestionState:
    if state.get("pipeline_status") == "failed":
        return state

    content = state.get("cleaned_content")
    
    # Specific verification for JSON/Dict artifacts
    if isinstance(content, dict):
        if "title" not in content:
            state["verification_status"] = "rejected"
            state["verification_errors"] = ["Report missing 'title' field"]
            state["pipeline_status"] = "failed"
            return state

    state["verification_status"] = "verified"
    return state

def format_artifact(state: DataIngestionState) -> DataIngestionState:
    if state.get("pipeline_status") == "failed":
        return state

    state["formatted_artifact"] = state["cleaned_content"]
    state["pipeline_status"] = "success"
    return state

def create_sequential_data_pipeline():
    workflow = StateGraph(DataIngestionState)

    # Nodes
    workflow.add_node("ingest", ingest_file)
    workflow.add_node("process", process_file)
    workflow.add_node("verify", verify_artifact)
    workflow.add_node("format", format_artifact)

    # Edges
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "process")
    workflow.add_edge("process", "verify")
    workflow.add_edge("verify", "format")
    workflow.add_edge("format", END)

    return workflow.compile()