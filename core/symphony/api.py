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
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Symphony Orchestrator</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        h1, h2 { color: #00ffcc; }
        .container { max-width: 1200px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background-color: #1e1e1e; padding: 20px; border-radius: 8px; border: 1px solid #333; }
        .stat { font-size: 2em; font-weight: bold; margin: 10px 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #333; }
        th { color: #888; text-transform: uppercase; font-size: 0.85em; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
        .badge-running { background-color: #1e3a8a; color: #93c5fd; }
        .badge-retrying { background-color: #78350f; color: #fcd34d; }
        pre { background-color: #000; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; border: 1px solid #333; }
        .refresh-btn { background-color: #00ffcc; color: #000; border: none; padding: 10px 20px; border-radius: 4px; font-weight: bold; cursor: pointer; float: right; }
        .refresh-btn:hover { background-color: #00cca3; }
    </style>
</head>
<body>
    <div class="container">
        <button class="refresh-btn" onclick="triggerRefresh()">Trigger Poll</button>
        <h1>Symphony Orchestrator</h1>
        <p>Last updated: <span id="last-updated">...</span></p>

        <div class="grid">
            <div class="card">
                <h2>Active Sessions</h2>
                <div class="stat" id="count-running">0</div>
            </div>
            <div class="card">
                <h2>Queued Retries</h2>
                <div class="stat" id="count-retrying">0</div>
            </div>
            <div class="card">
                <h2>Total Tokens (All Time)</h2>
                <div class="stat" id="count-tokens">0</div>
            </div>
            <div class="card">
                <h2>Total Runtime (Seconds)</h2>
                <div class="stat" id="count-runtime">0.0</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <h2>Running Issues</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Identifier</th>
                            <th>State</th>
                            <th>Session ID</th>
                            <th>Turn</th>
                            <th>Last Event</th>
                            <th>Tokens (In/Out/Total)</th>
                            <th>Running Since</th>
                        </tr>
                    </thead>
                    <tbody id="table-running">
                        <tr><td colspan="7" style="text-align:center">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <h2>Retry Queue</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Identifier</th>
                            <th>Attempt</th>
                            <th>Due At</th>
                            <th>Error</th>
                        </tr>
                    </thead>
                    <tbody id="table-retrying">
                        <tr><td colspan="4" style="text-align:center">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        function formatTime(isoString) {
            if (!isoString) return 'N/A';
            return new Date(isoString).toLocaleTimeString();
        }

        async function triggerRefresh() {
            try {
                await fetch('/api/v1/refresh', { method: 'POST' });
                fetchState();
            } catch (e) {
                console.error('Refresh failed', e);
            }
        }

        async function fetchState() {
            try {
                const res = await fetch('/api/v1/state');
                const data = await res.json();

                document.getElementById('last-updated').textContent = new Date(data.generated_at).toLocaleString();
                document.getElementById('count-running').textContent = data.counts.running;
                document.getElementById('count-retrying').textContent = data.counts.retrying;
                document.getElementById('count-tokens').textContent = data.codex_totals.total_tokens.toLocaleString();
                document.getElementById('count-runtime').textContent = data.codex_totals.seconds_running.toFixed(1);

                const runningHtml = data.running.length === 0 ?
                    '<tr><td colspan="7" style="text-align:center;color:#888;">No active sessions</td></tr>' :
                    data.running.map(r => `
                        <tr>
                            <td><strong>${r.issue_identifier}</strong></td>
                            <td><span class="badge badge-running">${r.state}</span></td>
                            <td style="font-family:monospace;font-size:0.9em">${r.session_id || 'Starting...'}</td>
                            <td>${r.turn_count}</td>
                            <td>${r.last_event || '-'}</td>
                            <td>${r.tokens.input_tokens} / ${r.tokens.output_tokens} / ${r.tokens.total_tokens}</td>
                            <td>${formatTime(r.started_at)}</td>
                        </tr>
                    `).join('');
                document.getElementById('table-running').innerHTML = runningHtml;

                const retryingHtml = data.retrying.length === 0 ?
                    '<tr><td colspan="4" style="text-align:center;color:#888;">No queued retries</td></tr>' :
                    data.retrying.map(r => `
                        <tr>
                            <td><strong>${r.issue_identifier}</strong></td>
                            <td>${r.attempt}</td>
                            <td>${formatTime(r.due_at)}</td>
                            <td style="color:#ef4444">${r.error || 'N/A'}</td>
                        </tr>
                    `).join('');
                document.getElementById('table-retrying').innerHTML = retryingHtml;
            } catch (e) {
                console.error('Failed to fetch state', e);
            }
        }

        // Fetch immediately and then every 5 seconds
        fetchState();
        setInterval(fetchState, 5000);
    </script>
</body>
</html>
""", 200

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
