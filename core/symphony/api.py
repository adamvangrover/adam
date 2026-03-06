from flask import Flask, jsonify, request
import logging
from datetime import datetime, timezone
import asyncio

from core.symphony.orchestrator import SymphonyOrchestrator

logger = logging.getLogger(__name__)

def create_app(orchestrator: SymphonyOrchestrator) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def dashboard():
        return "Symphony Orchestrator is running. View API at /api/v1/state", 200

    @app.route("/api/v1/state")
    def get_state():
        state = orchestrator.state

        running_data = []
        for issue_id, run in state.running.items():
            running_data.append({
                "issue_id": run.issue_id,
                "issue_identifier": run.issue_identifier,
                "state": run.issue.state if run.issue else "Unknown",
                "session_id": run.session.session_id if run.session else None,
                "turn_count": run.session.turn_count if run.session else 0,
                "last_event": run.session.last_codex_event if run.session else None,
                "last_message": run.session.last_codex_message if run.session else None,
                "started_at": run.started_at.isoformat(),
                "last_event_at": run.session.last_codex_timestamp.isoformat() if run.session and run.session.last_codex_timestamp else None,
                "tokens": {
                    "input_tokens": run.session.codex_input_tokens if run.session else 0,
                    "output_tokens": run.session.codex_output_tokens if run.session else 0,
                    "total_tokens": run.session.codex_total_tokens if run.session else 0
                }
            })

        retrying_data = []
        for issue_id, retry in state.retry_attempts.items():
            retrying_data.append({
                "issue_id": retry.issue_id,
                "issue_identifier": retry.identifier,
                "attempt": retry.attempt,
                "due_at": datetime.fromtimestamp(retry.due_at_ms / 1000.0, tz=timezone.utc).isoformat(),
                "error": retry.error
            })

        return jsonify({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "running": len(state.running),
                "retrying": len(state.retry_attempts)
            },
            "running": running_data,
            "retrying": retrying_data,
            "codex_totals": state.codex_totals.model_dump(),
            "rate_limits": state.codex_rate_limits
        })

    @app.route("/api/v1/<issue_identifier>")
    def get_issue(issue_identifier):
        state = orchestrator.state

        # Search running
        for run in state.running.values():
            if run.issue_identifier == issue_identifier:
                return jsonify({
                    "issue_identifier": issue_identifier,
                    "issue_id": run.issue_id,
                    "status": "running",
                    "workspace": {"path": run.workspace_path},
                    "attempts": {"current_retry_attempt": run.attempt},
                    "running": {
                        "session_id": run.session.session_id if run.session else None,
                        "turn_count": run.session.turn_count if run.session else 0,
                        "state": run.issue.state if run.issue else "Unknown",
                        "started_at": run.started_at.isoformat(),
                        "last_event": run.session.last_codex_event if run.session else None,
                        "last_message": run.session.last_codex_message if run.session else None,
                        "last_event_at": run.session.last_codex_timestamp.isoformat() if run.session and run.session.last_codex_timestamp else None,
                        "tokens": {
                            "input_tokens": run.session.codex_input_tokens if run.session else 0,
                            "output_tokens": run.session.codex_output_tokens if run.session else 0,
                            "total_tokens": run.session.codex_total_tokens if run.session else 0
                        }
                    },
                    "retry": None,
                    "logs": {},
                    "recent_events": [],
                    "last_error": None,
                    "tracked": {}
                })

        # Search retrying
        for retry in state.retry_attempts.values():
            if retry.identifier == issue_identifier:
                return jsonify({
                    "issue_identifier": issue_identifier,
                    "issue_id": retry.issue_id,
                    "status": "retrying",
                    "attempts": {"current_retry_attempt": retry.attempt},
                    "retry": {
                        "due_at": datetime.fromtimestamp(retry.due_at_ms / 1000.0, tz=timezone.utc).isoformat(),
                        "error": retry.error
                    }
                })

        return jsonify({"error": {"code": "issue_not_found", "message": f"Issue {issue_identifier} not found in active state."}}), 404

    @app.route("/api/v1/refresh", methods=["POST"])
    def refresh():
        # Trigger an immediate tick if possible
        loop = orchestrator._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(orchestrator._on_tick(), loop)

        return jsonify({
            "queued": True,
            "coalesced": False,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "operations": ["poll", "reconcile"]
        }), 202

    return app
