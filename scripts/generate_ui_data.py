import os
import json
import time
import re
import random
import datetime
import sys

# Ensure core modules are accessible
sys.path.append(os.getcwd())

from core.data_processing.universal_ingestor import UniversalIngestor, ArtifactType

# --- 1. The Generator (The "Gold Standard") ---
class DataFactory:
    @staticmethod
    def generate_deep_dive(ticker="AAPL", scenario="neutral"):
        """
        Generates a synthetic v23_knowledge_graph report.
        Allows for dynamic modification based on scenario.
        """
        base_valuation = 245.50 if ticker == "AAPL" else 150.00
        
        # Adjust numbers based on scenario
        if scenario == "bear":
            conviction = 3
            recommendation = "Sell"
            modifier = 0.8
        elif scenario == "bull":
            conviction = 9
            recommendation = "Buy"
            modifier = 1.2
        else:
            conviction = 5
            recommendation = "Hold"
            modifier = 1.0

        return {
            "title": f"{ticker} Deep Dive Analysis ({scenario.title()} Case)",
            "file_path": f"synthetic/{ticker}_{scenario}.json",
            "type": "synthetic_report",
            "v23_knowledge_graph": {
                "meta": {
                    "target": ticker,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "model_version": "Adam-v23.5-Synthetic"
                },
                "nodes": {
                    "entity_ecosystem": {
                        "legal_entity": {
                            "name": f"{ticker} Inc.",
                            "jurisdiction": "California, USA"
                        },
                        "management_assessment": {
                            "capital_allocation_score": 9.2 if scenario == "bull" else 6.5,
                            "alignment_analysis": "High ownership structure. Buybacks active.",
                            "key_person_risk": "Low"
                        },
                        "competitive_positioning": {
                            "moat_status": "Wide" if scenario != "bear" else "Narrowing",
                            "technology_risk_vector": "Generative AI integration neutralizes disruption risk."
                        }
                    },
                    "equity_analysis": {
                        "fundamentals": {
                            "revenue_cagr_3yr": "8.5%" if scenario == "bull" else "4.2%",
                            "ebitda_margin_trend": "Expanding" if scenario == "bull" else "Contracting"
                        },
                        "valuation_engine": {
                            "dcf_model": {
                                "wacc": 0.085,
                                "terminal_growth": 0.03,
                                "intrinsic_value": round(3200000.0 * modifier, 2),
                                "intrinsic_share_price": round(base_valuation * modifier, 2)
                            },
                            "multiples_analysis": {
                                "current_ev_ebitda": 22.5,
                                "peer_median_ev_ebitda": 25.0
                            },
                            "price_targets": {
                                "bear_case": round(base_valuation * 0.7, 2),
                                "base_case": round(base_valuation, 2),
                                "bull_case": round(base_valuation * 1.3, 2)
                            }
                        }
                    },
                    "credit_analysis": {
                        "snc_rating_model": {
                            "overall_borrower_rating": "Pass",
                            "facilities": [
                                {"id": "Revolver", "amount": "$5B", "regulatory_rating": "Pass", "collateral_coverage": "Unsecured", "covenant_headroom": ">50%"}
                            ]
                        },
                        "cds_market_implied_rating": "AA",
                        "covenant_risk_analysis": {
                            "primary_constraint": "None",
                            "current_level": 0.0,
                            "breach_threshold": 0.0,
                            "risk_assessment": "Minimal"
                        }
                    },
                    "simulation_engine": {
                        "monte_carlo_default_prob": 0.0001,
                        "quantum_scenarios": [
                            {"name": "Supply Chain Decoupling", "probability": 0.15, "estimated_impact_ev": "-12%"},
                            {"name": "Antitrust Breakup", "probability": 0.05, "estimated_impact_ev": "-25%"}
                        ],
                        "trading_dynamics": {
                            "short_interest": "0.8%",
                            "liquidity_risk": "None"
                        }
                    },
                    "strategic_synthesis": {
                        "m_and_a_posture": "Buyer",
                        "final_verdict": {
                            "recommendation": recommendation,
                            "conviction_level": conviction,
                            "time_horizon": "Long Term",
                            "rationale_summary": f"Automated scenario generation for {scenario} market conditions.",
                            "justification_trace": ["Strong Cash Flow", "Wide Moat", "Valuation Analysis"]
                        }
                    }
                }
            }
        }

# --- 2. Helper Functions ---

def clean_json_text(text):
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text

def get_file_tree(root_dir):
    file_tree = []
    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.github', 'venv', 'env', 'site-packages', 'dist', 'build'}

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        path = root.split(os.sep)
        if any(p.startswith('.') and p != '.' for p in path):
            continue

        rel_path = os.path.relpath(root, root_dir)
        if rel_path == '.':
            rel_path = ''

        for file in files:
            if file.startswith('.'):
                continue
            file_tree.append({
                'path': os.path.join(rel_path, file),
                'type': 'blob',
                'size': 0,
                'last_modified': 0
            })
    return file_tree

def parse_agents_md(root_dir):
    agents_md_path = os.path.join(root_dir, 'AGENTS.md')
    agents = []
    if os.path.exists(agents_md_path):
        with open(agents_md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('### Sub-Agents'): current_section = 'Sub-Agent'
            elif line.startswith('### Meta-Agents'): current_section = 'Meta-Agent'
            elif line.startswith('### Orchestrator Agents'): current_section = 'Orchestrator'

            if current_section and line.startswith('* **') and '**:' in line:
                parts = line.replace('* **', '').split('**:', 1)
                if len(parts) == 2:
                    agents.append({
                        'name': parts[0].strip(),
                        'type': current_section,
                        'description': parts[1].strip(),
                        'status': 'Active'
                    })
    return agents

def get_company_data(root_dir):
    path = os.path.join(root_dir, 'data/company_data.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.loads(clean_json_text(f.read()))
        except: pass
    return {}

def get_market_baseline(root_dir):
    path = os.path.join(root_dir, 'data/adam_market_baseline.json')
    if os.path.exists(path):
        try:
            content = clean_json_text(open(path).read())
            # Simple heuristic for concatenated JSON
            if '}{' in content:
                return json.loads(content.split('}{')[1].replace('}', '', 1)) # Very rough
            return json.loads(content)
        except: pass
    return {}

# --- 3. Integration with Universal Ingestor ---

def get_ingested_data(ingestor: UniversalIngestor, root_dir):
    # 1. Reports
    reports_artifacts = ingestor.get_artifacts_by_type(ArtifactType.REPORT)
    reports = []
    for art in reports_artifacts:
        # Convert artifact format to UI format
        report_data = art['content']
        if isinstance(report_data, dict):
            # Ensure essential keys
            report_data['title'] = art['title']
            report_data['file_path'] = os.path.relpath(art['source_path'], root_dir)
            reports.append(report_data)
        else:
             # Fallback for text reports
             reports.append({
                 "title": art['title'],
                 "file_path": os.path.relpath(art['source_path'], root_dir),
                 "content": str(report_data)[:500] + "..."
             })
    
    # Synthetic Backfill
    if len(reports) < 5:
        synthetic_tickers = ["MSFT", "GOOGL", "NVDA", "TSLA", "AAPL"]
        scenarios = ["bull", "bear", "neutral"]
        for ticker in synthetic_tickers:
            reports.append(DataFactory.generate_deep_dive(ticker, random.choice(scenarios)))

    # 2. Newsletters
    news_artifacts = ingestor.get_artifacts_by_type(ArtifactType.NEWSLETTER)
    newsletters = []
    for art in news_artifacts:
        newsletters.append({
            "title": art['title'],
            "file_path": os.path.relpath(art['source_path'], root_dir),
            "content": str(art['content'])[:200]
        })

    # 3. Prompts
    prompt_artifacts = ingestor.get_artifacts_by_type(ArtifactType.PROMPT)
    prompts = []
    for art in prompt_artifacts:
        prompts.append({
            "name": art['title'],
            "path": os.path.relpath(art['source_path'], root_dir),
            "category": "General" # Could extract from path
        })

    return reports, newsletters, prompts

def main():
    root_dir = os.path.abspath('.')
    print(f"Scanning {root_dir}...")

    # Run Gold Standard Ingestor
    ingestor = UniversalIngestor()
    ingestor.scan_directory(os.path.join(root_dir, "core/libraries_and_archives"))
    ingestor.scan_directory(os.path.join(root_dir, "prompt_library"))
    ingestor.scan_directory(os.path.join(root_dir, "data")) # For specific data

    # Save the Gold Standard JSONL
    os.makedirs(os.path.join(root_dir, "data/gold_standard"), exist_ok=True)
    ingestor.save_to_jsonl(os.path.join(root_dir, "data/gold_standard/knowledge_artifacts.jsonl"))

    # Map to UI Data
    reports, newsletters, prompts = get_ingested_data(ingestor, root_dir)

    data = {
        'generated_at': time.time(),
        'files': get_file_tree(root_dir),
        'agents': parse_agents_md(root_dir),
        'reports': reports,
        'newsletters': newsletters,
        'company_data': get_company_data(root_dir),
        'market_data': get_market_baseline(root_dir),
        'prompts': prompts,
        'system_stats': {
            'cpu_usage': 12.5,
            'memory_usage': 45.2,
            'active_tasks': 3,
            'queued_tasks': 12,
            'version': "23.5 Partner"
        }
    }

    output_path = os.path.join(root_dir, 'showcase/data/ui_data.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Data written to {output_path}")

    js_output_path = os.path.join(root_dir, 'showcase/js/mock_data.js')
    with open(js_output_path, 'w', encoding='utf-8') as f:
        f.write(f"window.MOCK_DATA = {json.dumps(data, indent=2)};")
    print(f"JS Data written to {js_output_path}")

if __name__ == "__main__":
    main()
