from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum

class ChangeType(str, Enum):
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"

class FileDiff(BaseModel):
    filepath: str = Field(..., description="Path to the file relative to repo root")
    change_type: ChangeType = Field(..., description="Type of change")
    diff_content: str = Field(..., description="The unified diff string or file content")
    old_content: Optional[str] = Field(None, description="Previous content if available (for context)")
    new_content: Optional[str] = Field(None, description="New content if available")

class PullRequest(BaseModel):
    pr_id: str = Field(..., description="Unique identifier for the PR/ChangeSet")
    author: str = Field(..., description="Author of the change")
    title: str = Field(..., description="Title of the change")
    description: str = Field(..., description="Detailed description of the change")
    files: List[FileDiff] = Field(..., description="List of file changes")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ReviewFocus(str, Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    FUNCTIONALITY = "functionality"
    DOCUMENTATION = "documentation"
    BACKWARD_COMPATIBILITY = "backward_compatibility"
    ALL = "all"

class CodeReviewParams(BaseModel):
    focus_areas: List[ReviewFocus] = Field(default=[ReviewFocus.ALL], description="Areas to focus the review on")
    strictness: int = Field(default=5, ge=1, le=10, description="Strictness level (1-10)")
    require_tests: bool = Field(default=True, description="Whether to require tests for functional changes")

class ReviewComment(BaseModel):
    filepath: str = Field(..., description="File path related to the comment")
    line_number: Optional[int] = Field(None, description="Line number start")
    end_line_number: Optional[int] = Field(None, description="Line number end")
    severity: Literal["info", "suggestion", "warning", "critical"] = Field(..., description="Severity of the comment")
    message: str = Field(..., description="The comment content")
    suggestion: Optional[str] = Field(None, description="Suggested code fix if applicable")

class ReviewDecisionStatus(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"

class AnalysisResult(BaseModel):
    """Result of static code analysis."""
    missing_docstrings: List[str] = []
    missing_type_hints: List[str] = []
    dangerous_functions: List[str] = []
    security_findings: List[Dict[str, str]] = []
    model_config = ConfigDict(extra="allow")

class ReviewDecision(BaseModel):
    pr_id: str = Field(..., description="ID of the PR being reviewed")
    status: ReviewDecisionStatus = Field(..., description="Final decision")
    summary: str = Field(..., description="High-level summary of the review")
    comments: List[ReviewComment] = Field(default_factory=list, description="Detailed comments")
    score: int = Field(..., ge=0, le=100, description="Quality score (0-100)")
    automated_fixes: Optional[List[Dict[str, str]]] = Field(None, description="List of automated patches if generated")
    analysis_results: Optional[Dict[str, AnalysisResult]] = Field(None, description="Raw analysis data per file")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RepoState(BaseModel):
    """Snapshot of repo metrics for comparison."""
    file_count: int
    test_coverage_percent: Optional[float] = None
    lint_score: Optional[float] = None
    open_issues: int = 0
    security_vulnerabilities: int = 0
