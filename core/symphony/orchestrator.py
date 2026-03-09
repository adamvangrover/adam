import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from core.symphony.models import Issue, OrchestratorRuntimeState, RunAttempt, RetryEntry
from core.symphony.config import SymphonyConfig, validate_config, load_workflow
from core.symphony.tracker import TrackerClient, TrackerError, create_tracker_client
from core.symphony.workspace import WorkspaceManager
from core.symphony.agent_runner import AgentRunner, AgentError

logger = logging.getLogger(__name__)

class SymphonyOrchestrator:
    def __init__(self, workflow_path: str):
        self.workflow_path = workflow_path
        self.workflow = None
        self.config: Optional[SymphonyConfig] = None
        self.tracker: Optional[TrackerClient] = None
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.agent_runner: Optional[AgentRunner] = None
        self.state = OrchestratorRuntimeState()
        self._running = False
        self._loop = None

    def reload_config(self) -> bool:
        try:
            workflow = load_workflow(self.workflow_path)
            config = SymphonyConfig(workflow.config)

            err = validate_config(config)
            if err:
                logger.error(f"Config validation failed: {err.code} - {err.message}")
                return False

            self.workflow = workflow
            self.config = config
            self.tracker = create_tracker_client(self.config)
            self.workspace_manager = WorkspaceManager(self.config)
            self.agent_runner = AgentRunner(self.config, self.workspace_manager)

            self.state.poll_interval_ms = self.config.polling_interval_ms
            self.state.max_concurrent_agents = self.config.max_concurrent_agents
            return True
        except Exception as e:
            logger.error(f"Failed to load or validate workflow config: {e}")
            return False

    async def start(self):
        if not self.reload_config():
            logger.critical("Failed startup validation.")
            return

        self._loop = asyncio.get_running_loop()
        self._running = True

        # Startup terminal workspace cleanup
        try:
            logger.info("Performing startup terminal workspace cleanup...")
            terminal_states = self.config.tracker_terminal_states
            terminal_issues = self.tracker.fetch_issues_by_states(terminal_states)
            for issue in terminal_issues:
                self.workspace_manager.cleanup_workspace(issue.identifier)
        except Exception as e:
            logger.warning(f"Startup cleanup failed, continuing: {e}")

        # Start poll loop
        asyncio.create_task(self._poll_loop())

    async def _poll_loop(self):
        while self._running:
            try:
                await self._on_tick()
            except Exception as e:
                logger.error(f"Error in poll tick: {e}")

            await asyncio.sleep(self.state.poll_interval_ms / 1000.0)

    async def _on_tick(self):
        # Hot reload
        self.reload_config()

        await self._reconcile_running_issues()

        err = validate_config(self.config)
        if err:
            logger.error(f"Skipping dispatch due to invalid config: {err.message}")
            return

        try:
            candidates = await asyncio.to_thread(self.tracker.fetch_candidate_issues)
        except TrackerError as e:
            logger.error(f"Failed to fetch candidates: {e}")
            return

        # Sort candidates
        def sort_key(i: Issue):
            prio = i.priority if i.priority is not None else 999
            created = i.created_at.timestamp() if i.created_at else float('inf')
            return (prio, created, i.identifier)

        candidates.sort(key=sort_key)

        for issue in candidates:
            if not self._has_available_slots(issue):
                break
            if self._should_dispatch(issue):
                await self._dispatch_issue(issue, attempt=None)

    def _has_available_slots(self, issue: Issue) -> bool:
        if len(self.state.running) >= self.state.max_concurrent_agents:
            return False

        state_limits = self.config.max_concurrent_agents_by_state
        state_key = issue.state.lower().strip()
        if state_key in state_limits:
            current_in_state = sum(1 for r in self.state.running.values() if r.issue and r.issue.state.lower().strip() == state_key)
            if current_in_state >= state_limits[state_key]:
                return False

        return True

    def _should_dispatch(self, issue: Issue) -> bool:
        if issue.id in self.state.running:
            return False
        if issue.id in self.state.claimed:
            return False
        if issue.state.lower().strip() == 'todo':
            for b in issue.blocked_by:
                if b.state and b.state.lower().strip() not in [s.lower().strip() for s in self.config.tracker_terminal_states]:
                    return False
        return True

    async def _dispatch_issue(self, issue: Issue, attempt: Optional[int]):
        logger.info(f"Dispatching issue {issue.identifier} (attempt {attempt})")

        self.state.claimed.add(issue.id)
        if issue.id in self.state.retry_attempts:
            timer = self.state.retry_attempts[issue.id].timer_handle
            if timer:
                timer.cancel()
            del self.state.retry_attempts[issue.id]

        run_attempt = RunAttempt(
            issue_id=issue.id,
            issue_identifier=issue.identifier,
            attempt=attempt,
            workspace_path="",
            started_at=datetime.now(timezone.utc),
            status="PreparingWorkspace",
            issue=issue
        )
        self.state.running[issue.id] = run_attempt

        # Launch worker
        task = asyncio.create_task(self._worker_task(issue, attempt))
        run_attempt.worker_handle = task

    async def _worker_task(self, issue: Issue, attempt: Optional[int]):
        run_attempt = self.state.running.get(issue.id)
        if not run_attempt:
            return

        def on_event(event: Dict[str, Any]):
            logger.info(f"event={event.get('event')} session_id={event.get('session_id')}")

        def fetch_issue_cb(issue_id: str) -> Optional[Issue]:
            try:
                refreshed = self.tracker.fetch_issue_states_by_ids([issue_id])
                return refreshed[0] if refreshed else None
            except Exception as e:
                logger.error(f"Failed to refresh issue in worker loop: {e}")
                return None

        def session_cb(session: Any):
            if run_attempt:
                run_attempt.session = session

        reason = "normal"
        error_msg = None

        try:
            await self.agent_runner.run(
                self.workflow,
                issue,
                attempt,
                on_event,
                fetch_issue_cb,
                session_cb
            )
        except Exception as e:
            logger.error(f"Worker {issue.identifier} failed: {e}")
            reason = "abnormal"
            error_msg = str(e)

        await self._on_worker_exit(issue.id, reason, error_msg)

    async def _on_worker_exit(self, issue_id: str, reason: str, error_msg: Optional[str] = None):
        running_entry = self.state.running.pop(issue_id, None)
        if not running_entry:
            return

        # Add to totals
        now = datetime.now(timezone.utc)
        elapsed = (now - running_entry.started_at).total_seconds()
        self.state.codex_totals.seconds_running += elapsed

        if running_entry.session:
            self.state.codex_totals.input_tokens += running_entry.session.codex_input_tokens
            self.state.codex_totals.output_tokens += running_entry.session.codex_output_tokens
            self.state.codex_totals.total_tokens += running_entry.session.codex_total_tokens

        if reason == "normal":
            self.state.completed.add(issue_id)
            await self._schedule_retry(issue_id, running_entry.issue_identifier, 1, delay_ms=1000, error=None)
        else:
            next_attempt = (running_entry.attempt or 0) + 1
            delay_ms = min(10000 * (2 ** (next_attempt - 1)), self.config.max_retry_backoff_ms)
            await self._schedule_retry(issue_id, running_entry.issue_identifier, next_attempt, delay_ms=delay_ms, error=error_msg)

    async def _schedule_retry(self, issue_id: str, identifier: str, attempt: int, delay_ms: float, error: Optional[str]):
        logger.info(f"Scheduling retry for {identifier} in {delay_ms}ms (attempt {attempt})")
        timer = self._loop.call_later(delay_ms / 1000.0, lambda: asyncio.create_task(self._on_retry_timer(issue_id)))

        self.state.retry_attempts[issue_id] = RetryEntry(
            issue_id=issue_id,
            identifier=identifier,
            attempt=attempt,
            due_at_ms=datetime.now(timezone.utc).timestamp() * 1000 + delay_ms,
            timer_handle=timer,
            error=error
        )

    async def _on_retry_timer(self, issue_id: str):
        retry_entry = self.state.retry_attempts.pop(issue_id, None)
        if not retry_entry:
            return

        try:
            candidates = await asyncio.to_thread(self.tracker.fetch_candidate_issues)
        except Exception as e:
            logger.error(f"Retry fetch failed: {e}")
            await self._schedule_retry(issue_id, retry_entry.identifier, retry_entry.attempt + 1, 10000, "retry poll failed")
            return

        issue = next((i for i in candidates if i.id == issue_id), None)
        if not issue:
            self.state.claimed.discard(issue_id)
            return

        # Hack for available slots check without duplicate logic
        if not self._has_available_slots(issue):
            await self._schedule_retry(issue_id, issue.identifier, retry_entry.attempt + 1, 10000, "no available orchestrator slots")
            return

        self.state.claimed.discard(issue_id) # remove before dispatch so it passes should_dispatch
        await self._dispatch_issue(issue, attempt=retry_entry.attempt)

    async def _reconcile_running_issues(self):
        # Part A: Stall detection
        now = datetime.now(timezone.utc)
        stall_ms = self.config.codex_stall_timeout_ms
        stalled_ids = []

        if stall_ms > 0:
            for issue_id, run_attempt in list(self.state.running.items()):
                last_ts = run_attempt.session.last_codex_timestamp if run_attempt.session and run_attempt.session.last_codex_timestamp else run_attempt.started_at
                elapsed_ms = (now - last_ts).total_seconds() * 1000
                if elapsed_ms > stall_ms:
                    logger.warning(f"Run {run_attempt.issue_identifier} stalled")
                    stalled_ids.append(issue_id)

        for issue_id in stalled_ids:
            run = self.state.running[issue_id]
            if run.worker_handle:
                run.worker_handle.cancel()
            await self._on_worker_exit(issue_id, "abnormal", "stalled")

        # Part B: Tracker state refresh
        running_ids = list(self.state.running.keys())
        if not running_ids:
            return

        try:
            refreshed = await asyncio.to_thread(self.tracker.fetch_issue_states_by_ids, running_ids)
        except Exception as e:
            logger.warning(f"State refresh failed, keeping workers: {e}")
            return

        terminal_states = [s.lower().strip() for s in self.config.tracker_terminal_states]
        active_states = [s.lower().strip() for s in self.config.tracker_active_states]

        for issue in refreshed:
            state_key = issue.state.lower().strip()
            run = self.state.running.get(issue.id)
            if not run:
                continue

            if state_key in terminal_states:
                logger.info(f"Run {issue.identifier} is terminal, stopping")
                if run.worker_handle:
                    run.worker_handle.cancel()
                self.workspace_manager.cleanup_workspace(issue.identifier)
                await self._on_worker_exit(issue.id, "normal")
            elif state_key in active_states:
                run.issue = issue
            else:
                logger.info(f"Run {issue.identifier} is no longer active, stopping without cleanup")
                if run.worker_handle:
                    run.worker_handle.cancel()
                await self._on_worker_exit(issue.id, "normal")
