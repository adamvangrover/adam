import json
import os
import glob
import uuid
import re

# Paths
DATA_DIR = 'showcase/data'
PROVENANCE_DIR = 'showcase/data/provenance'
OUTPUT_FILE = os.path.join(PROVENANCE_DIR, 'lineage.json')

def generate_uuid():
    return str(uuid.uuid4())

def parse_credit_memo(filepath):
    filename = os.path.basename(filepath)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    artifact_id = filename

    if isinstance(data, list):
        # If list, assume first item is the main one or skip
        if len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        else:
            print(f"Skipping {filename}: Top level is list but empty or not dicts.")
            return None
    elif not isinstance(data, dict):
        print(f"Skipping {filename}: Top level is not dict or list.")
        return None

    artifact_name = data.get('borrower_name', 'Unknown Borrower') + " Credit Memo"

    nodes = []
    edges = []

    # Artifact Node
    nodes.append({
        "id": artifact_id,
        "label": artifact_name,
        "type": "Artifact",
        "icon": "memo"
    })

    # Agent Log
    agent_log = data.get('agent_log', [])

    # Track unique agents to avoid duplicate nodes
    agents_seen = set()
    sources_seen = set()

    for log in agent_log:
        agent_name = log.get('user_id', 'Unknown Agent')
        action = log.get('action', 'UNKNOWN_ACTION')

        # Agent Node
        if agent_name not in agents_seen:
            nodes.append({
                "id": agent_name,
                "label": agent_name,
                "type": "Agent",
                "icon": "agent"
            })
            agents_seen.add(agent_name)

        # Edges based on action
        if action == 'SEARCH_FILINGS':
            # Source Node (implied from inputs)
            ticker = log.get('inputs', {}).get('ticker', 'Unknown')
            doc_type = log.get('inputs', {}).get('doc_type', '10-K')
            source_id = f"{ticker}_{doc_type}"
            source_label = f"{ticker} {doc_type}"

            if source_id not in sources_seen:
                nodes.append({
                    "id": source_id,
                    "label": source_label,
                    "type": "Source",
                    "icon": "file"
                })
                sources_seen.add(source_id)

            # Agent searched Source
            edges.append({
                "source": agent_name,
                "target": source_id,
                "label": "searched"
            })
            # Artifact derived from Source
            edges.append({
                "source": artifact_id,
                "target": source_id,
                "label": "derived_from"
            })

        elif action == 'SPREAD_FINANCIALS':
            # Agent processed Source (previous source or generic)
            edges.append({
                "source": agent_name,
                "target": artifact_id,
                "label": "contributed_financials"
            })

        elif action == 'IDENTIFY_RISKS':
             edges.append({
                "source": agent_name,
                "target": artifact_id,
                "label": "identified_risks"
            })

        elif action == 'CRITIQUE_MEMO':
             edges.append({
                "source": agent_name,
                "target": artifact_id,
                "label": "critiqued"
            })

    # If no log, add generic creator
    if not agent_log:
        nodes.append({"id": "CreditAnalyst", "label": "Credit Analyst", "type": "Agent", "icon": "agent"})
        edges.append({"source": "CreditAnalyst", "target": artifact_id, "label": "generated"})

    return {
        "artifactId": artifact_id,
        "nodes": nodes,
        "edges": edges
    }

def parse_equity_report(filepath):
    filename = os.path.basename(filepath)
    # Mock lineage for HTML reports
    artifact_id = filename
    artifact_name = filename.replace('_', ' ').replace('.html', '')

    nodes = [
        {"id": artifact_id, "label": artifact_name, "type": "Artifact", "icon": "report"},
        {"id": "EquityAnalyst", "label": "Equity Analyst", "type": "Agent", "icon": "agent"},
        {"id": "MarketData", "label": "Market Data Feed", "type": "Source", "icon": "database"}
    ]
    edges = [
        {"source": "EquityAnalyst", "target": artifact_id, "label": "authored"},
        {"source": artifact_id, "target": "MarketData", "label": "derived_from"}
    ]

    return {
        "artifactId": artifact_id,
        "nodes": nodes,
        "edges": edges
    }

def main():
    lineage_data = []

    # Process JSON Credit Memos
    json_files = glob.glob(os.path.join(DATA_DIR, 'credit_memo_*.json'))
    for f in json_files:
        lineage = parse_credit_memo(f)
        if lineage:
            lineage_data.append(lineage)

    # Process HTML Equity Reports (Mock)
    equity_files = glob.glob(os.path.join(DATA_DIR, 'equity_reports', '*.html'))
    for f in equity_files:
        lineage = parse_equity_report(f)
        if lineage:
            lineage_data.append(lineage)

    # Also add sample HTML credit reports if any
    credit_html_files = glob.glob(os.path.join(DATA_DIR, 'credit_reports', '*.html'))
    for f in credit_html_files:
        lineage = parse_equity_report(f) # Reuse mock logic
        if lineage:
             lineage['nodes'][1]['label'] = "Credit Analyst" # Fix agent name
             lineage_data.append(lineage)

    # Process Financial Models
    model_files = glob.glob(os.path.join(DATA_DIR, 'models', '*.json'))
    for f in model_files:
        # Check if v2 model
        if 'Financial_Model_v2' in f:
            try:
                with open(f, 'r') as jf:
                    data = json.load(jf)
                audit = data.get('audit_trail', {})
                lineage = {
                    "artifactId": os.path.basename(f),
                    "nodes": [
                        {"id": os.path.basename(f), "label": f"{data.get('ticker')} FinModel v2", "type": "Artifact", "icon": "graph"},
                        {"id": audit.get('auditor_id', 'Unknown Auditor'), "label": "Auditor Bot", "type": "Agent", "icon": "shield"},
                        {"id": "FinGPT-v4", "label": "FinGPT-v4", "type": "Model", "icon": "cpu"}
                    ],
                    "edges": [
                        {"source": "FinGPT-v4", "target": os.path.basename(f), "label": "generated"},
                        {"source": audit.get('auditor_id', 'Unknown Auditor'), "target": os.path.basename(f), "label": "audited"}
                    ]
                }
                lineage_data.append(lineage)
            except Exception as e:
                print(f"Error parsing v2 model {f}: {e}")
        else:
            lineage = parse_equity_report(f) # Reuse mock logic
            if lineage:
                lineage['nodes'][0]['icon'] = "graph"
                lineage['nodes'][1]['label'] = "Quant Analyst"
                lineage_data.append(lineage)

    # Process History
    history_files = glob.glob(os.path.join(DATA_DIR, 'history', '*.html'))
    for f in history_files:
        lineage = parse_equity_report(f)
        if lineage:
             lineage['nodes'][1]['label'] = "Archivist"
             lineage_data.append(lineage)

    # Process Portfolios
    portfolio_files = glob.glob(os.path.join(DATA_DIR, 'portfolios', '*.json'))
    for f in portfolio_files:
        try:
            with open(f, 'r') as jf:
                data = json.load(jf)
            audit = data.get('audit_trail', {})
            lineage = {
                "artifactId": os.path.basename(f),
                "nodes": [
                    {"id": os.path.basename(f), "label": data.get('fund_name'), "type": "Portfolio", "icon": "portfolio"},
                    {"id": audit.get('fund_manager', 'PM'), "label": "Portfolio Manager", "type": "Agent", "icon": "agent"},
                    {"id": audit.get('compliance_officer', 'COMP'), "label": "Compliance", "type": "Agent", "icon": "shield"}
                ],
                "edges": [
                    {"source": audit.get('fund_manager', 'PM'), "target": os.path.basename(f), "label": "managed"},
                    {"source": audit.get('compliance_officer', 'COMP'), "target": os.path.basename(f), "label": "approved"}
                ]
            }
            lineage_data.append(lineage)
        except Exception as e:
            print(f"Error parsing portfolio {f}: {e}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(lineage_data, f, indent=2)

    print(f"Generated lineage for {len(lineage_data)} artifacts in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
