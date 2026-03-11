from typing import Dict, Any, List, Optional
import os
import subprocess
import shutil
import tempfile
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class AgentInput(BaseModel):
    query: str = Field(..., description="The GitHub repository URL to analyze.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context including 'branch' or 'since_days'.")
    tools: List[str] = Field(default_factory=list, description="List of allowed tool names (unused here).")

class AgentOutput(BaseModel):
    answer: str = Field(..., description="The synthesized analysis of the repository.")
    sources: List[str] = Field(default_factory=list, description="List of sources (e.g. repo URL).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the Alpha signal.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw metrics: commit_count, unique_authors, etc.")

class GitHubAlphaAgent(AgentBase):
    """
    Protocol: ARCHITECT_INFINITE - Day 11
    Role: Analyze GitHub repositories for 'Developer Alpha' - a leading indicator of project health.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.temp_dir = tempfile.gettempdir()

    async def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyzes a GitHub repository for activity metrics.

        Args:
            query (str): The GitHub URL.
            context (dict): Optional parameters like 'branch' (default: main/master) and 'days' (default: 30).

        Returns:
            dict: AgentOutput schema.
        """
        context = context or {}
        repo_url = query.strip()
        days = context.get("days", 30)
        branch = context.get("branch", None) # Let git decide default if None

        logger.info(f"GitHubAlphaAgent analyzing: {repo_url} for past {days} days.")

        # Create a unique temp directory for this clone
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        # Use a safe timestamp
        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        clone_path = os.path.join(self.temp_dir, f"alpha_agent_{repo_name}_{ts}")

        try:
            # 1. Clone the repo
            self._clone_repo(repo_url, clone_path)

            # 2. Extract Metrics
            commit_count = self._get_commit_count(clone_path, days)
            unique_authors = self._get_unique_authors(clone_path, days)
            last_commit_date = self._get_last_commit_date(clone_path)

            # 3. Calculate Alpha Score (0.0 to 1.0)
            # Heuristic:
            # - High commits (> 50/month) + High authors (> 5) = High Alpha (0.8-0.9)
            # - Low commits (< 5/month) = Low Alpha (0.1-0.3)
            # - Stale (> 30 days since last commit) = Dead (0.0)

            alpha_score = 0.5
            signal = "NEUTRAL"

            days_since_last = (datetime.now() - last_commit_date).days

            if days_since_last > 90:
                alpha_score = 0.1
                signal = "DEAD"
            elif days_since_last > 30:
                alpha_score = 0.3
                signal = "STAGNANT"
            else:
                # Active
                # Normalize to max 1.0
                activity_score = min(commit_count / 100, 1.0) * 0.5 # Max 0.5 from volume (100 commits)
                diversity_score = min(unique_authors / 10, 1.0) * 0.5 # Max 0.5 from team size (10 authors)
                alpha_score = activity_score + diversity_score

                # Boost for recent activity
                if days_since_last < 2:
                    alpha_score = min(alpha_score * 1.1, 1.0)

                if alpha_score > 0.8:
                    signal = "HIGH_MOMENTUM"
                elif alpha_score > 0.5:
                    signal = "ACTIVE"
                else:
                    signal = "LOW_ACTIVITY"

            # 4. Synthesize Answer
            answer = (
                f"Repository Analysis for {repo_name}:\n"
                f"- Signal: {signal}\n"
                f"- Alpha Score: {alpha_score:.2f}\n"
                f"- Commits (Last {days} days): {commit_count}\n"
                f"- Active Developers: {unique_authors}\n"
                f"- Last Commit: {last_commit_date.strftime('%Y-%m-%d')} ({days_since_last} days ago)"
            )

            return AgentOutput(
                answer=answer,
                sources=[repo_url],
                confidence=alpha_score,
                metadata={
                    "commit_count": commit_count,
                    "unique_authors": unique_authors,
                    "days_analyzed": days,
                    "last_commit": last_commit_date.isoformat(),
                    "signal": signal
                }
            ).model_dump()

        except Exception as e:
            logger.error(f"Error analyzing repo {repo_url}: {str(e)}")
            return AgentOutput(
                answer=f"Failed to analyze repository: {str(e)}",
                sources=[repo_url],
                confidence=0.0,
                metadata={"error": str(e)}
            ).model_dump()
        finally:
            # Cleanup
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path, ignore_errors=True)

    def _clone_repo(self, url: str, path: str):
        # Shallow clone for speed, but deep enough for history?
        # If we need 'last 30 days', we need history.
        # --filter=blob:none downloads commits but not file contents, which is much faster for log analysis.
        cmd = ["git", "clone", "--filter=blob:none", "--", url, path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _get_commit_count(self, path: str, days: int) -> int:
        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        cmd = ["git", "rev-list", "--count", "--since", since_date, "HEAD"]
        result = subprocess.run(cmd, cwd=path, capture_output=True, text=True, check=True)
        return int(result.stdout.strip() or 0)

    def _get_unique_authors(self, path: str, days: int) -> int:
        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        # git log --since=... --format='%aN'
        cmd_log = ["git", "log", f"--since={since_date}", "--format=%aN"]
        result = subprocess.run(cmd_log, cwd=path, capture_output=True, text=True, check=True)
        authors = set(line.strip() for line in result.stdout.splitlines() if line.strip())
        return len(authors)

    def _get_last_commit_date(self, path: str) -> datetime:
        # git log -1 --format=%cd --date=iso
        cmd = ["git", "log", "-1", "--format=%cd", "--date=iso"]
        result = subprocess.run(cmd, cwd=path, capture_output=True, text=True, check=True)
        date_str = result.stdout.strip()
        try:
            # Ensure naive datetime for comparison with datetime.now()
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except ValueError:
            # Fallback for formats with space
            # 2023-10-25 14:30:00 +0000 -> 2023-10-25T14:30:00+00:00
            # Crude parser for ISO-like git date
            if " " in date_str:
                parts = date_str.split(" ")
                if len(parts) >= 2:
                    # Just take date and time, ignore timezone for simplicity or assume UTC
                    return datetime.fromisoformat(f"{parts[0]}T{parts[1]}")
            return datetime.now() # Fallback
