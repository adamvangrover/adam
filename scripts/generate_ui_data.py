import os
import json
import time
import re
import random
import datetime

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

# --- 2. Helper Functions & Scanners ---

def clean_json_text(text):
    # Remove single line comments
    text = re.sub(r'//.*', '', text)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Fix trailing commas (simple case)
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

            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                modified = os.path.getmtime(file_path)
            except OSError:
                size = 0
                modified = 0

            file_tree.append({
                'path': os.path.join(rel_path, file),
                'type': 'blob',
                'size': size,
                'last_modified': modified
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
            if line.startswith('### Sub-Agents'):
                current_section = 'Sub-Agent'
            elif line.startswith('### Meta-Agents'):
                current_section = 'Meta-Agent'
            elif line.startswith('### Orchestrator Agents'):
                current_section = 'Orchestrator'

            if current_section and line.startswith('* **') and '**:' in line:
                parts = line.replace('* **', '').split('**:', 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    desc = parts[1].strip()
                    agents.append({
                        'name': name,
                        'type': current_section,
                        'description': desc,
                        'status': 'Active'
                    })
    return agents

def get_reports(root_dir):
    reports = []
    reports_dir = os.path.join(root_dir, 'core/libraries_and_archives/reports')
    if os.path.exists(reports_dir):
        for root, _, files in os.walk(reports_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)

                if file.endswith('.json'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            clean_content = clean_json_text(content)
                            data = json.loads(clean_content)
                            data['file_path'] = rel_path
                            # Ensure we have a title or company name
                            if 'title' not in data and 'company' in data:
                                data['title'] = f"{data['company']} Report"
                            reports.append(data)
                    except Exception as e:
                        # Fallback for really broken JSON: just verify existence
                        reports.append({
                            "title": file.replace('.json', '').replace('_', ' ').title(),
                            "file_path": rel_path,
                            "error": str(e),
                            "raw_content_preview": content[:200] if 'content' in locals() else ""
                        })
                elif file.endswith('.md'):
                     reports.append({
                         "title": file.replace('.md', '').replace('_', ' ').title(),
                         "type": "markdown",
                         "file_path": rel_path,
                         "date": "2025-01-01"
                     })
    return reports

def get_newsletters(root_dir):
    newsletters = []
    news_dir = os.path.join(root_dir, 'core/libraries_and_archives/newsletters')
    if os.path.exists(news_dir):
        for file in os.listdir(news_dir):
            file_path = os.path.join(news_dir, file)
            rel_path = os.path.join('core/libraries_and_archives/newsletters', file)

            if file.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        clean_content = clean_json_text(content)
                        data = json.loads(clean_content)
                        data['file_path'] = rel_path
                        newsletters.append(data)
                except Exception as e:
                      print(f"Error reading newsletter {file}: {e}")
            elif file.endswith('.md'):
                 newsletters.append({
                     "title": file.replace('.md', '').replace('_', ' ').title(),
                     "type": "markdown",
                     "file_path": rel_path,
                     "date": "2025-01-01"
                 })
    return newsletters

def get_company_data(root_dir):
    path = os.path.join(root_dir, 'data/company_data.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                clean_content = clean_json_text(content)
                return json.loads(clean_content)
        except Exception as e:
            print(f"Error reading company data: {e}")
    return {}

def get_market_baseline(root_dir):
    path = os.path.join(root_dir, 'data/adam_market_baseline.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Handle concatenated JSON objects if present
                clean_content = clean_json_text(content)
                try:
                    return json.loads(clean_content)
                except:
                     # Attempt to split if multiple objects
                     parts = clean_content.split('\n{')
                     if len(parts) > 1:
                         return json.loads('{' + parts[-1])
        except Exception as e:
            print(f"Error reading market baseline: {e}")
    return {}

def get_prompts(root_dir):
    prompts = []
    prompt_dir = os.path.join(root_dir, 'prompt_library')
    if os.path.exists(prompt_dir):
        for root, _, files in os.walk(prompt_dir):
            for file in files:
                if file.endswith('.md') or file.endswith('.json') or file.endswith('.yaml'):
                    prompts.append({
                        'name': file,
                        'path': os.path.relpath(os.path.join(root, file), root_dir),
                        'category': os.path.basename(root) if root != prompt_dir else 'General'
                    })
    return prompts

# --- 3. The Hybrid Orchestrator ---
def get_combined_reports(root_dir):
    """
    Combines real reports scanned from disk with synthetic data if needed.
    """
    # 1. Try to get real data
    real_reports = get_reports(root_dir)
    
    # 2. Check if we need synthetic backfill (e.g. if we have fewer than 5 reports)
    if len(real_reports) < 5:
        print("⚠️  Insufficient real data found. Injecting synthetic Deep Dives...")
        synthetic_tickers = ["MSFT", "GOOGL", "NVDA", "TSLA", "AAPL"]
        scenarios = ["bull", "bear", "neutral"]
        
        for ticker in synthetic_tickers:
            # Randomize scenario for variety
            scenario = random.choice(scenarios)
            # Generate the mock report
            synthetic_report = DataFactory.generate_deep_dive(ticker, scenario)
            # Add to the list
            real_reports.append(synthetic_report)
            
    return real_reports

# --- Main Execution ---
def main():
    root_dir = os.path.abspath('.')
    print(f"Scanning {root_dir}...")

    # Note: We use get_combined_reports here instead of just get_reports
    data = {
        'generated_at': time.time(),
        'files': get_file_tree(root_dir),
        'agents': parse_agents_md(root_dir),
        'reports': get_combined_reports(root_dir), 
        'newsletters': get_newsletters(root_dir),
        'company_data': get_company_data(root_dir),
        'market_data': get_market_baseline(root_dir),
        'prompts': get_prompts(root_dir),
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
