import json
import os

RUNTIME_FILE = "data/ARCHITECT_INFINITE/execution_runtime.json"
OUTPUT_FILE = "public/architect-infinite/index.html"

def generate_html():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(RUNTIME_FILE, "r") as f:
        runtime_data = json.load(f)

    nodes_html = ""
    for node in runtime_data.get("execution_nodes", []):
        validation_flag = node.get('primary_node_audit', {}).get('validation_flag', '')
        color = '#ec4899' if 'WARN' in validation_flag or 'WATCH' in validation_flag or 'TIGHT' in validation_flag else '#00ff41'

        nodes_html += f"""
        <div style="border: 1px solid #3b82f6; margin: 10px; padding: 15px; border-radius: 5px; background: #0a0a0c;">
            <h3 style="color: #00ff41; margin-top: 0;">Sector: {node.get('industry')}</h3>
            <p><strong>Entities:</strong> {', '.join(node.get('entities', []))}</p>
            <div style="background: #111827; padding: 10px; border-radius: 3px;">
                <p><strong>Primary Node Audit:</strong> {node.get('primary_node_audit', {}).get('entity')}</p>
                <p><strong>PD Alpha:</strong> {node['primary_node_audit'].get('pd_alpha')} | <strong>PD Beta:</strong> {node['primary_node_audit'].get('pd_beta')}</p>
                <p><strong>Validation Flag:</strong> <span style="color: {color};">{validation_flag}</span></p>
                <p style="font-family: monospace; font-size: 0.8em; color: #64748b; margin-bottom: 0;">Hash: {node['primary_node_audit'].get('prov_hash')}</p>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en" data-theme="adam-os">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARCHITECT_INFINITE - Risk Governance Dashboard</title>
    <style>
        body {{ background-color: #0a0a0c; color: #00ff41; font-family: monospace; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #3b82f6; border-bottom: 1px solid #3b82f6; padding-bottom: 10px; }}
        .meta-panel {{ background-color: #111827; padding: 15px; border-radius: 5px; margin-bottom: 20px; border: 1px solid #64748b; }}
        .nodes-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ARCHITECT_INFINITE: UNIFIED RISK ENGINEERING & INTELLIGENCE DOSSIER</h1>

        <div class="meta-panel">
            <h2 style="margin-top: 0;">Execution Metadata (W3C PROV-O)</h2>
            <p><strong>Run ID:</strong> {runtime_data['_meta']['run_id']}</p>
            <p><strong>Timestamp (UTC):</strong> {runtime_data['_meta']['timestamp_utc']}</p>
            <p><strong>Status:</strong> {runtime_data['_meta']['status']}</p>
            <p style="margin-bottom: 0;"><strong>Operator Review:</strong> {runtime_data['operator_review']['status']} (Token: {runtime_data['operator_review']['supervisor_token']})</p>
        </div>

        <h2>Execution Nodes</h2>
        <div class="nodes-grid">
            {nodes_html}
        </div>
    </div>
</body>
</html>
"""

    with open(OUTPUT_FILE, "w") as f:
        f.write(html_content)
    print(f"Generated dashboard at {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_html()
