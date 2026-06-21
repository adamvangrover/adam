from src.governance.gatekeeper import GovernanceGatekeeper, GovernanceError
import asyncio
import json
import logging
import time
import uuid
import traceback
import aiosqlite
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
logger = logging.getLogger('AdamOrchestrator')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(ch)


class TokenOverflowException(Exception):
    pass


class CircularDependencyError(Exception):
    pass


class TaskState:
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    SUSPENDED_AWAITING_INPUT = 'SUSPENDED_AWAITING_INPUT'
    SKIPPED = 'SKIPPED'


@dataclass
class TaskNode:
    task_id: str
    coroutine_func: Callable[..., Coroutine[Any, Any, Any]]
    dependencies: List[str] = field(default_factory=list)
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 0
    retry_backoff: float = 2.0
    conditional_router: Optional[Callable[[Dict[str, Any]], bool]] = None
    is_human_gate: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncSQLiteManager:
    """Non-blocking background worker for persisting orchestrator state."""

    def __init__(self, db_path: str='orchestrator_state.db'):
        self.db_path = db_path
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def initialize_schema(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    trace_id TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    global_context TEXT
                )
            """
                )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT,
                    workflow_id TEXT,
                    span_id TEXT,
                    parent_id TEXT,
                    status TEXT,
                    duration_sec REAL,
                    updated_at TEXT,
                    PRIMARY KEY (task_id, workflow_id)
                )
            """
                )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    task_id TEXT,
                    span_id TEXT,
                    event_type TEXT,
                    timestamp TEXT,
                    metrics TEXT
                )
            """
                )
            await db.commit()

    async def start(self):
        await self.initialize_schema()
        self.worker_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        async with aiosqlite.connect(self.db_path) as db:
            while not self._stop_event.is_set() or not self.queue.empty():
                try:
                    action, data = await asyncio.wait_for(self.queue.get(),
                        timeout=1.0)
                    if action == 'upsert_workflow':
                        await db.execute(
                            """
                            INSERT INTO workflows (workflow_id, trace_id, status, start_time, end_time, global_context)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(workflow_id) DO UPDATE SET
                            status=excluded.status, end_time=excluded.end_time, global_context=excluded.global_context
                        """
                            , (data['workflow_id'], data['trace_id'], data[
                            'status'], data['start_time'], data['end_time'],
                            json.dumps(data['global_context'])))
                    elif action == 'upsert_task':
                        await db.execute(
                            """
                            INSERT INTO tasks (task_id, workflow_id, span_id, parent_id, status, duration_sec, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(task_id, workflow_id) DO UPDATE SET
                            status=excluded.status, span_id=excluded.span_id, duration_sec=excluded.duration_sec, updated_at=excluded.updated_at
                        """
                            , (data['task_id'], data['workflow_id'], data[
                            'span_id'], data['parent_id'], data['status'],
                            data.get('duration_sec'), data['updated_at']))
                    elif action == 'insert_log':
                        await db.execute(
                            """
                            INSERT INTO execution_logs (trace_id, task_id, span_id, event_type, timestamp, metrics)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """
                            , (data['trace_id'], data['task_id'], data[
                            'span_id'], data['event'], data['timestamp'],
                            json.dumps(data.get('metrics', {}))))
                    await db.commit()
                    self.queue.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f'Database worker error: {str(e)}')

    async def stop(self):
        self._stop_event.set()
        if self.worker_task:
            await self.worker_task


class StateLedger:

    def __init__(self, workflow_id: str, db_manager: AsyncSQLiteManager,
        trace_id: Optional[str]=None):
        self.workflow_id = workflow_id
        self.trace_id = trace_id or f'tr-{uuid.uuid4().hex[:12]}'
        self.db = db_manager
        self.status = TaskState.PENDING
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.global_context: Dict[str, Any] = {}
        self.task_registry: Dict[str, Dict[str, Any]] = {}

    def _get_timestamp(self) ->str:
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    def _sync_workflow_state(self):
        """Pushes the global workflow state to the async queue."""
        self.db.queue.put_nowait(('upsert_workflow', {'workflow_id': self.
            workflow_id, 'trace_id': self.trace_id, 'status': self.status,
            'start_time': self.start_time, 'end_time': self.end_time,
            'global_context': self.global_context}))

    def log_event(self, event: str, task_id: str, span_id: str, metrics:
        Optional[Dict[str, Any]]=None):
        entry = {'timestamp': self._get_timestamp(), 'event': event,
            'task_id': task_id, 'span_id': span_id, 'trace_id': self.
            trace_id, 'metrics': metrics or {}}
        self.db.queue.put_nowait(('insert_log', entry))
        log_msg = f'{event} | Task: {task_id} | Span: {span_id}'
        if event == 'TASK_FAILED':
            logger.error(
                f"{log_msg} | Reason: {entry['metrics'].get('error', 'Unknown')}"
                )
        else:
            logger.debug(log_msg)

    def update_task_status(self, task_id: str, status: str, span_id: str,
        parent_id: Optional[str]=None, duration: Optional[float]=None):
        if task_id not in self.task_registry:
            self.task_registry[task_id] = {}
        update_data = {'status': status, 'span_id': span_id, 'parent_id':
            parent_id, 'updated_at': self._get_timestamp()}
        if duration is not None:
            update_data['duration_sec'] = duration
        self.task_registry[task_id].update(update_data)
        db_payload = {'task_id': task_id, 'workflow_id': self.workflow_id,
            **update_data}
        self.db.queue.put_nowait(('upsert_task', db_payload))
        self._sync_workflow_state()


class OrchestratorEngine:

    def __init__(self, workflow_id: str):
        self.db_manager = AsyncSQLiteManager()
        self.ledger = StateLedger(workflow_id, self.db_manager)
        self.tasks: Dict[str, TaskNode] = {}
        self._pending_queue: Set[str] = set()
        self.gatekeeper = GovernanceGatekeeper(schema={'type': 'object'})
        self.gatekeeper = GovernanceGatekeeper(schema={'type': 'object'})
        self.gatekeeper = GovernanceGatekeeper(schema={'type': 'object'})
        self.gatekeeper = GovernanceGatekeeper(schema={'type': 'object'})
        self.gatekeeper = GovernanceGatekeeper(schema={'type': 'object'})
        self.gatekeeper = GovernanceGatekeeper(schema={'type': 'object'})

    def add_task(self, task: TaskNode):
        self.tasks[task.task_id] = task
        self._pending_queue.add(task.task_id)
        if task.task_id not in self.ledger.task_registry:
            self.ledger.update_task_status(task.task_id, TaskState.PENDING,
                f'sp-{uuid.uuid4().hex[:8]}', None)

    def validate_dag(self):
        visited = set()
        path = set()

        def dfs(node_id: str):
            if node_id in path:
                raise CircularDependencyError(
                    f'Circular dependency detected involving task: {node_id}')
            if node_id in visited:
                return
            path.add(node_id)
            for dep in self.tasks.get(node_id, TaskNode(node_id, lambda : None)
                ).dependencies:
                if dep in self.tasks:
                    dfs(dep)
            path.remove(node_id)
            visited.add(node_id)
        for task_id in self.tasks:
            dfs(task_id)
        logger.info('DAG validation passed. No circular dependencies found.')

    def _get_parent_span_id(self, task: TaskNode) ->Optional[str]:
        if not task.dependencies:
            return None
        return self.ledger.task_registry.get(task.dependencies[0], {}).get(
            'span_id')

    async def _execute_with_retry(self, task: TaskNode, span_id: str,
        parent_id: Optional[str]) ->str:
        attempt = 0
        start_time = time.perf_counter()
        while attempt <= task.max_retries:
            try:
                result = await asyncio.wait_for(task.coroutine_func(self.
                    ledger.global_context), timeout=task.timeout)
                if isinstance(result, dict) and 'provenance_trace' in result:
                    try:
                        result = self.gatekeeper.exit_gate(result)
                    except GovernanceError as e:
                        raise Exception(
                            f'Governance validation failed: {str(e)}')
                if task.is_human_gate and result == 'SUSPENDED_AWAITING_INPUT':
                    self.ledger.update_task_status(task.task_id, TaskState.
                        SUSPENDED_AWAITING_INPUT, span_id, parent_id)
                    self.ledger.log_event('TASK_SUSPENDED', task.task_id,
                        span_id, metrics={'reason': 'Awaiting human input'})
                    return TaskState.SUSPENDED_AWAITING_INPUT
                tokens_used = result.get('tokens_used', 0) if isinstance(result
                    , dict) else 0
                if tokens_used > task.max_tokens:
                    raise TokenOverflowException(
                        f'Token overflow: {tokens_used} > {task.max_tokens}')
                duration = time.perf_counter() - start_time
                if isinstance(result, dict) and 'updates' in result:
                    self.ledger.global_context.update(result['updates'])
                self.ledger.update_task_status(task.task_id, TaskState.
                    COMPLETED, span_id, parent_id, duration=duration)
                self.ledger.log_event('TASK_SUCCESS', task.task_id, span_id,
                    metrics={'tokens_used': tokens_used, 'duration_sec':
                    round(duration, 3)})
                return TaskState.COMPLETED
            except asyncio.TimeoutError:
                err_msg = 'Timeout'
            except TokenOverflowException as e:
                err_msg = str(e)
            except Exception as e:
                err_msg = traceback.format_exc()
            attempt += 1
            if attempt <= task.max_retries:
                delay = task.retry_backoff ** attempt
                self.ledger.log_event('TASK_RETRY', task.task_id, span_id,
                    metrics={'attempt': attempt, 'delay': delay, 'error':
                    err_msg})
                await asyncio.sleep(delay)
            else:
                duration = time.perf_counter() - start_time
                self.ledger.update_task_status(task.task_id, TaskState.
                    FAILED, span_id, parent_id, duration=duration)
                self.ledger.log_event('TASK_FAILED', task.task_id, span_id,
                    metrics={'error': err_msg})
                return TaskState.FAILED

    async def _run_task_wrapper(self, task: TaskNode, semaphore: asyncio.
        Semaphore):
        async with semaphore:
            registry_entry = self.ledger.task_registry.get(task.task_id, {})
            span_id = registry_entry.get('span_id'
                ) or f'sp-{uuid.uuid4().hex[:8]}'
            parent_id = self._get_parent_span_id(task)
            if task.conditional_router and not task.conditional_router(self
                .ledger.global_context):
                self.ledger.update_task_status(task.task_id, TaskState.
                    SKIPPED, span_id, parent_id)
                self.ledger.log_event('TASK_SKIPPED', task.task_id, span_id,
                    metrics={'reason': 'Conditional Router evaluated to False'}
                    )
                return task.task_id, TaskState.SKIPPED
            self.ledger.update_task_status(task.task_id, TaskState.RUNNING,
                span_id, parent_id)
            self.ledger.log_event('TASK_START', task.task_id, span_id)
            final_status = await self._execute_with_retry(task, span_id,
                parent_id)
            return task.task_id, final_status

    async def run(self, max_concurrency: int=10):
        await self.db_manager.start()
        self.validate_dag()
        if not self.ledger.start_time:
            self.ledger.start_time = self.ledger._get_timestamp()
        if self.ledger.status not in [TaskState.SUSPENDED_AWAITING_INPUT,
            TaskState.FAILED]:
            self.ledger.status = TaskState.RUNNING
            self.ledger._sync_workflow_state()
        semaphore = asyncio.Semaphore(max_concurrency)
        running_coros = set()
        pending_tasks = {t_id for t_id, state in self.ledger.task_registry.
            items() if state.get('status') == TaskState.PENDING}
        while self.ledger.status == TaskState.RUNNING:
            ready_to_launch = []
            for task_id in list(pending_tasks):
                task = self.tasks[task_id]
                deps_statuses = [self.ledger.task_registry.get(dep, {}).get
                    ('status') for dep in task.dependencies]
                if any(s in (TaskState.FAILED, TaskState.SKIPPED) for s in
                    deps_statuses):
                    span_id = self.ledger.task_registry.get(task_id, {}).get(
                        'span_id') or f'sp-{uuid.uuid4().hex[:8]}'
                    self.ledger.update_task_status(task_id, TaskState.
                        SKIPPED, span_id, self._get_parent_span_id(task))
                    self.ledger.log_event('TASK_SKIPPED', task_id, span_id,
                        metrics={'reason':
                        'Upstream dependency failed or skipped'})
                    pending_tasks.remove(task_id)
                elif all(s == TaskState.COMPLETED for s in deps_statuses):
                    ready_to_launch.append(task)
                    pending_tasks.remove(task_id)
            for task in ready_to_launch:
                task_future = asyncio.create_task(self._run_task_wrapper(
                    task, semaphore))
                running_coros.add(task_future)
            if not running_coros:
                break
            done, running_coros = await asyncio.wait(running_coros,
                return_when=asyncio.FIRST_COMPLETED)
            for finished_coro in done:
                task_id, final_status = finished_coro.result()
                if final_status == TaskState.SUSPENDED_AWAITING_INPUT:
                    self.ledger.status = TaskState.SUSPENDED_AWAITING_INPUT
        all_completed = all(self.ledger.task_registry.get(t, {}).get(
            'status') in (TaskState.COMPLETED, TaskState.SKIPPED) for t in
            self.tasks)
        if self.ledger.status != TaskState.SUSPENDED_AWAITING_INPUT:
            if all_completed:
                self.ledger.status = TaskState.COMPLETED
            elif any(self.ledger.task_registry.get(t, {}).get('status') ==
                TaskState.FAILED for t in self.tasks):
                self.ledger.status = TaskState.FAILED
            self.ledger.end_time = self.ledger._get_timestamp()
            self.ledger._sync_workflow_state()
        logger.info(
            f'Orchestration run ended with status: {self.ledger.status}')
        await self.db_manager.stop()

    def resolve_human_gate(self, task_id: str, callback_payload: Dict[str, Any]
        ):
        if self.ledger.task_registry.get(task_id, {}).get('status'
            ) == TaskState.SUSPENDED_AWAITING_INPUT:
            span_id = self.ledger.task_registry.get(task_id, {}).get('span_id')
            parent_id = self.ledger.task_registry.get(task_id, {}).get(
                'parent_id')
            self.ledger.global_context.update(callback_payload.get(
                'updates', {}))
            tokens_used = callback_payload.get('tokens_used', 0)
            self.ledger.update_task_status(task_id, TaskState.COMPLETED,
                span_id, parent_id)
            self.ledger.log_event('TASK_SUCCESS', task_id, span_id, metrics
                ={'tokens_used': tokens_used, 'event': 'HUMAN_GATE_RESOLVED'})
            self.ledger.status = TaskState.PENDING
            self.ledger._sync_workflow_state()
