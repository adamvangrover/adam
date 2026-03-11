from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field

class BlockerRef(BaseModel):
    id: Optional[str] = None
    identifier: Optional[str] = None
    state: Optional[str] = None

class Issue(BaseModel):
    id: str
    identifier: str
    title: str
    description: Optional[str] = None
    priority: Optional[int] = None
    state: str
    branch_name: Optional[str] = None
    url: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    blocked_by: List[BlockerRef] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class WorkflowDefinition(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)
    prompt_template: str

class Workspace(BaseModel):
    path: str
    workspace_key: str
    created_now: bool

class LiveSession(BaseModel):
    session_id: str
    thread_id: str
    turn_id: str
    codex_app_server_pid: Optional[str] = None
    last_codex_event: Optional[str] = None
    last_codex_timestamp: Optional[datetime] = None
    last_codex_message: Optional[Dict[str, Any]] = None
    codex_input_tokens: int = 0
    codex_output_tokens: int = 0
    codex_total_tokens: int = 0
    last_reported_input_tokens: int = 0
    last_reported_output_tokens: int = 0
    last_reported_total_tokens: int = 0
    turn_count: int = 0

class RunAttempt(BaseModel):
    issue_id: str
    issue_identifier: str
    attempt: Optional[int] = None
    workspace_path: str
    started_at: datetime
    status: str
    error: Optional[str] = None

    # Keeping worker/session refs inside running entry for the orchestrator
    worker_handle: Optional[Any] = None
    monitor_handle: Optional[Any] = None
    issue: Optional[Issue] = None
    session: Optional[LiveSession] = None

class RetryEntry(BaseModel):
    issue_id: str
    identifier: str
    attempt: int
    due_at_ms: float
    timer_handle: Optional[Any] = None
    error: Optional[str] = None

class CodexTotals(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds_running: float = 0.0

class OrchestratorRuntimeState(BaseModel):
    poll_interval_ms: int = 30000
    max_concurrent_agents: int = 10
    running: Dict[str, RunAttempt] = Field(default_factory=dict)
    claimed: Set[str] = Field(default_factory=set)
    retry_attempts: Dict[str, RetryEntry] = Field(default_factory=dict)
    completed: Set[str] = Field(default_factory=set)
    codex_totals: CodexTotals = Field(default_factory=CodexTotals)
    codex_rate_limits: Optional[Dict[str, Any]] = None
