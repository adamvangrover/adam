import os
import json
import httpx
import logging
import asyncio
import uuid
import datetime
from typing import Any, Dict, Optional, Callable
from pydantic import ValidationError

from core.symphony.models import Issue, WorkflowDefinition, LiveSession
from core.symphony.config import SymphonyConfig
from core.symphony.workspace import WorkspaceManager, WorkspaceError

# We'll need a minimal template engine for the prompt
from jinja2 import Template, TemplateError, StrictUndefined

logger = logging.getLogger(__name__)

class AgentError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)

class AgentRunner:
    def __init__(self, config: SymphonyConfig, workspace_manager: WorkspaceManager):
        self.config = config
        self.workspace_manager = workspace_manager

    def render_prompt(self, workflow: WorkflowDefinition, issue: Issue, attempt: Optional[int]) -> str:
        try:
            # Using Jinja2 with StrictUndefined to fail on missing variables
            template = Template(workflow.prompt_template, undefined=StrictUndefined)
            # Convert issue to dict, ensure attempt is present
            context = {
                "issue": issue.model_dump(),
                "attempt": attempt
            }
            return template.render(**context)
        except TemplateError as e:
            raise AgentError("template_render_error", f"Prompt rendering failed: {e}")

    async def run_turn(
        self,
        session: LiveSession,
        issue: Issue,
        prompt: str,
        workspace_path: str,
        on_event: Callable[[Dict[str, Any]], None],
        proc: asyncio.subprocess.Process
    ) -> str:
        """Run a single turn by interacting with the app-server subprocess."""
        # 4. Turn start
        turn_id = str(uuid.uuid4())
        session.turn_id = turn_id
        session.session_id = f"{session.thread_id}-{turn_id}"

        turn_params = {
            "threadId": session.thread_id,
            "input": [{"type": "text", "text": prompt}],
            "cwd": workspace_path,
            "title": f"{issue.identifier}: {issue.title}",
            "approvalPolicy": self.config.codex_approval_policy or "auto",
        }
        if self.config.codex_turn_sandbox_policy:
            turn_params["sandboxPolicy"] = self.config.codex_turn_sandbox_policy

        turn_req = {
            "id": str(uuid.uuid4()),
            "method": "turn/start",
            "params": turn_params
        }
        proc.stdin.write((json.dumps(turn_req) + "\n").encode('utf-8'))
        await proc.stdin.drain()

        on_event({
            "event": "session_started",
            "session_id": session.session_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        # Wait for messages
        # In a real app we'd parse JSON-RPC line by line
        start_time = asyncio.get_event_loop().time()
        turn_timeout = self.config.codex_turn_timeout_ms / 1000.0

        while True:
            try:
                # Apply timeout over the full turn, not just readline
                now = asyncio.get_event_loop().time()
                remaining_time = max(turn_timeout - (now - start_time), 0.0)
                if remaining_time == 0.0:
                    raise asyncio.TimeoutError()

                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=remaining_time
                )
                if not line:
                    raise AgentError("subprocess_exit", "App-server exited prematurely")

                try:
                    msg = json.loads(line.decode('utf-8'))
                except json.JSONDecodeError:
                    continue # Ignore invalid lines (like partial or non-json output)

                # Update session state and metrics
                session.last_codex_message = msg
                session.last_codex_timestamp = datetime.datetime.now(datetime.timezone.utc)

                method = msg.get("method")

                # Try to extract totals from thread/tokenUsage/updated
                if method == "thread/tokenUsage/updated":
                    usage = msg.get("params", {}).get("total_token_usage", {})
                    if usage:
                        session.codex_input_tokens = usage.get("input_tokens", session.codex_input_tokens)
                        session.codex_output_tokens = usage.get("output_tokens", session.codex_output_tokens)
                        session.codex_total_tokens = usage.get("total_tokens", session.codex_total_tokens)

                if method == "turn/completed":
                    session.last_codex_event = "turn_completed"
                    on_event({
                        "event": "turn_completed",
                        "session_id": session.session_id,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    })
                    return "success"

                if method in ["turn/failed", "turn/cancelled", "turn/ended_with_error"]:
                    session.last_codex_event = method
                    on_event({
                        "event": method,
                        "session_id": session.session_id,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    })
                    return "failure"

                if method == "item/tool/requestUserInput":
                    session.last_codex_event = "turn_input_required"
                    on_event({
                        "event": "turn_input_required",
                        "session_id": session.session_id,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    })
                    return "input_required"

                if "id" in msg and "method" not in msg and "result" not in msg:
                    # It's an incoming request (e.g. tool call or approval)
                    resp = {
                        "id": msg["id"],
                        "result": {"success": False, "error": "unsupported_tool_call"}
                    }

                    method = msg.get("method", "")
                    if method.startswith("approval/"):
                        resp = {
                            "id": msg["id"],
                            "result": {"approved": True} # Default high-trust auto-approve
                        }
                    elif method == "item/tool/call":
                        tool_call = msg.get("params", {}).get("call", {})
                        if tool_call.get("name") == "linear_graphql" and self.config.tracker_kind == 'linear':
                            try:
                                args = tool_call.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        args = {"query": args}

                                query = args.get("query")
                                variables = args.get("variables", {})

                                if not query or not isinstance(query, str):
                                    resp = {"id": msg["id"], "result": {"success": False, "error": "invalid query argument"}}
                                else:
                                    headers = {
                                        "Authorization": self.config.tracker_api_key,
                                        "Content-Type": "application/json"
                                    }

                                    async with httpx.AsyncClient(timeout=30.0) as client:
                                        http_resp = await client.post(
                                            self.config.tracker_endpoint,
                                            json={"query": query, "variables": variables},
                                            headers=headers
                                        )

                                    if http_resp.status_code == 200:
                                        data = http_resp.json()
                                        if "errors" in data:
                                            resp = {"id": msg["id"], "result": {"success": False, "data": data}}
                                        else:
                                            resp = {"id": msg["id"], "result": {"success": True, "data": data.get("data", {})}}
                                    else:
                                        resp = {"id": msg["id"], "result": {"success": False, "error": f"Linear API returned {http_resp.status_code}"}}
                            except Exception as e:
                                logger.error(f"Failed linear_graphql tool call: {e}")
                                resp = {"id": msg["id"], "result": {"success": False, "error": str(e)}}

                    proc.stdin.write((json.dumps(resp) + "\n").encode('utf-8'))
                    await proc.stdin.drain()

                # Basic other notifications
                on_event({
                    "event": "notification",
                    "session_id": session.session_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "payload": msg
                })

            except asyncio.TimeoutError:
                raise AgentError("turn_timeout", "Turn execution timed out")

    async def run(
        self,
        workflow: WorkflowDefinition,
        issue: Issue,
        attempt: Optional[int],
        on_event: Callable[[Dict[str, Any]], None],
        fetch_issue_cb: Callable[[str], Optional[Issue]],
        session_cb: Callable[[LiveSession], None] = lambda s: None
    ) -> str:
        """Main agent worker execution loop."""

        # 1. Workspace prepare
        try:
            workspace = self.workspace_manager.create_for_issue(issue.identifier)
        except WorkspaceError as e:
            raise AgentError("workspace_error", str(e))

        try:
            self.workspace_manager.run_before_run(workspace.path)
        except WorkspaceError as e:
            raise AgentError("before_run_hook_error", str(e))

        # 2. Launch process
        cwd = os.path.abspath(workspace.path)
        cmd = ["bash", "-lc", self.config.codex_command]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=10 * 1024 * 1024 # 10MB line buffer
            )
        except Exception as e:
            self.workspace_manager.run_after_run(workspace.path)
            raise AgentError("agent_launch_failed", f"Failed to launch agent process: {e}")

        session = LiveSession(
            session_id="",
            thread_id=str(uuid.uuid4()), # Generated client side for now
            turn_id="",
            codex_app_server_pid=str(proc.pid)
        )
        session_cb(session)

        try:
            # 3. Handshake
            init_req = {
                "id": str(uuid.uuid4()),
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "symphony", "version": "1.0"},
                    "capabilities": {}
                }
            }
            proc.stdin.write((json.dumps(init_req) + "\n").encode('utf-8'))
            await proc.stdin.drain()

            try:
                # Wait for initialized
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=self.config.codex_read_timeout_ms / 1000.0
                )
                if not line:
                    raise AgentError("startup_failed", "Process exited during init")
            except asyncio.TimeoutError:
                raise AgentError("response_timeout", "Init handshake timed out")

            # Thread start
            thread_req = {
                "id": str(uuid.uuid4()),
                "method": "thread/start",
                "params": {
                    "approvalPolicy": self.config.codex_approval_policy or "auto",
                    "sandbox": self.config.codex_thread_sandbox or "none",
                    "cwd": cwd
                }
            }
            proc.stdin.write((json.dumps(thread_req) + "\n").encode('utf-8'))
            await proc.stdin.drain()

            try:
                # Wait for thread/start response to get real threadId
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=self.config.codex_read_timeout_ms / 1000.0
                )
                if line:
                    resp = json.loads(line.decode('utf-8'))
                    if "result" in resp and "thread" in resp["result"]:
                        session.thread_id = resp["result"]["thread"]["id"]
            except (asyncio.TimeoutError, json.JSONDecodeError):
                # Fallback to generated thread id
                pass

            # Loop through continuation turns
            turn_number = 1
            max_turns = self.config.max_turns

            while turn_number <= max_turns:
                if turn_number == 1:
                    prompt = self.render_prompt(workflow, issue, attempt)
                else:
                    prompt = "Continue."

                session.turn_count += 1

                status = await self.run_turn(session, issue, prompt, cwd, on_event, proc)

                if status != "success":
                    raise AgentError("turn_failed", f"Turn ended with status: {status}")

                # Refresh issue state
                refreshed = fetch_issue_cb(issue.id)
                if not refreshed:
                    break
                issue = refreshed

                # Check active states
                if issue.state.lower() not in [s.lower() for s in self.config.tracker_active_states]:
                    break

                turn_number += 1

            return "success"

        except Exception as e:
            if not isinstance(e, AgentError):
                logger.error(f"Agent unexpected error: {e}")
            raise
        finally:
            if proc.returncode is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            self.workspace_manager.run_after_run(workspace.path)
