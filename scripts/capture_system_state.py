import json
import os
import sys
import datetime
import subprocess
import re

def capture_system_state():
    telemetry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "system_id": "ADAM-v23.5-Apex", # Mocking for consistency with the log
    }

    # OS Info - resilient to non-unix
    try:
        telemetry["os_info"] = subprocess.check_output(["uname", "-a"]).decode().strip()
    except Exception:
         telemetry["os_info"] = f"{os.name} generic"

    # Fetch logs/adam.log
    try:
        with open("logs/adam.log", "r") as f:
            log_content = f.readlines()
            telemetry["recent_logs"] = "".join(log_content[-20:])
    except FileNotFoundError:
        telemetry["recent_logs"] = "WARNING: logs/adam.log not found."

    # Fetch v23_integration_log.json
    try:
        with open("data/v23_integration_log.json", "r") as f:
            data = json.load(f)
            meta = data.get("v23_integration_log", {}).get("meta", {})
            delta = data.get("v23_integration_log", {}).get("delta_analysis", {})
            telemetry["model_divergence"] = {
                "source_model": meta.get("source_model", "Unknown"),
                "valuation_divergence": delta.get("valuation_divergence", {})
            }
    except (FileNotFoundError, json.JSONDecodeError):
         telemetry["model_divergence"] = "WARNING: Integration log data unavailable."

    return telemetry

def update_html_with_telemetry(telemetry):
    filepath = "showcase/adam_convergence_live_neural_link.html"

    with open(filepath, "r") as f:
        html_content = f.read()

    # Prepare JSON snippets for embedding
    # We want valid JSON output in the HTML display

    val_div = telemetry.get('model_divergence', {}).get('valuation_divergence', {})
    val_div_str = json.dumps(val_div, indent=4)
    # indentation adjustment for the pre block (add 4 spaces to each line except the first?)
    # simplified: just dump it.

    recent_logs = telemetry['recent_logs'].strip().splitlines()[-5:]
    recent_logs_str = json.dumps(recent_logs, indent=4)

    # Generate the HTML block for telemetry
    # We use explicit start/end comments for easier replacement
    telemetry_html = f"""<!-- LIVE SYSTEM TELEMETRY -->
        <div class="space-y-8">
            <div class="flex items-center gap-4 opacity-50">
                <div class="h-px flex-1 bg-gradient-to-r from-transparent to-accent-purple"></div>
                <span class="font-mono text-sm font-bold text-accent-purple">RUNTIME // TELEMETRY</span>
                <div class="h-px flex-1 bg-gradient-to-l from-transparent to-accent-purple"></div>
            </div>

            <section class="code-card glass-panel rounded-2xl overflow-hidden group">
                <div class="px-6 py-4 bg-white/5 border-b border-white/5 flex items-center justify-between">
                    <h2 class="font-mono text-sm text-accent-purple/90 flex items-center gap-2">
                        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
                        system_state_capture.json
                    </h2>
                    <span class="text-[10px] font-mono text-white/30 uppercase tracking-tighter">live_metrics</span>
                </div>
                <div class="relative">
                    <pre class="p-6 overflow-x-auto bg-code-bg font-mono text-sm leading-relaxed"><code class="block text-white/90">
{{
  <span class="syntax-string">"timestamp"</span>: <span class="syntax-string">"{telemetry['timestamp']}"</span>,
  <span class="syntax-string">"environment"</span>: {{
    <span class="syntax-string">"python_version"</span>: <span class="syntax-string">"{telemetry['python_version']}"</span>,
    <span class="syntax-string">"system_id"</span>: <span class="syntax-string">"{telemetry['system_id']}"</span>,
    <span class="syntax-string">"os_kernel"</span>: <span class="syntax-string">"{telemetry['os_info'][:30]}..."</span>
  }},
  <span class="syntax-string">"model_limitations"</span>: {{
    <span class="syntax-string">"divergence_source"</span>: <span class="syntax-string">"{telemetry.get('model_divergence', {}).get('source_model', 'N/A')}"</span>,
    <span class="syntax-string">"valuation_delta"</span>: {val_div_str}
  }},
  <span class="syntax-string">"recent_logs"</span>: {recent_logs_str}
}}</code></pre>
                </div>
            </section>
        </div>
        <!-- END LIVE SYSTEM TELEMETRY -->"""

    # Replacement Logic
    # 1. Try to find existing block with start/end markers (if I had added them before)
    #    (Not applicable for first run of *this* version, but good for future)
    # 2. Try to find existing block with just start marker + heuristic end (current state)
    # 3. Else append

    start_marker = "<!-- LIVE SYSTEM TELEMETRY -->"
    end_marker = "<!-- END LIVE SYSTEM TELEMETRY -->"

    # Check if we have the fully wrapped block from a previous run of *this* script
    if start_marker in html_content and end_marker in html_content:
        print("Updating existing full telemetry block...")
        pattern = re.compile(f"{re.escape(start_marker)}.*?{re.escape(end_marker)}", re.DOTALL)
        updated_html = pattern.sub(telemetry_html, html_content)

    elif start_marker in html_content:
        # We have the partial block from the previous imperfect script
        print("Upgrading partial telemetry block...")
        # Match from start marker until just before </main>
        # We assume the block is at the end of main.
        pattern = re.compile(f"{re.escape(start_marker)}.*?(?=</main>)", re.DOTALL)
        # We need to make sure we replace everything up to </main> but keep </main>
        # The replacement `telemetry_html` does NOT contain </main>.
        # So we replace the match with `telemetry_html + \n    `
        updated_html = pattern.sub(telemetry_html + "\n    ", html_content)

    else:
        # First injection
        print("Injecting new telemetry block...")
        updated_html = html_content.replace("</main>", f"{telemetry_html}\n    </main>")

    with open(filepath, "w") as f:
        f.write(updated_html)
    print(f"Successfully updated {filepath} with live telemetry.")

if __name__ == "__main__":
    data = capture_system_state()
    update_html_with_telemetry(data)
