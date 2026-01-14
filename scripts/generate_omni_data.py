import os
import json
import time
import random
import sys
import ast
import glob
import re

# Ensure core modules are accessible
sys.path.append(os.getcwd())

# Attempt imports, handle missing dependencies gracefully
try:
    from core.data_processing.universal_ingestor import UniversalIngestor, ArtifactType
except ImportError:
    UniversalIngestor = None

# --- Helper Functions ---

def clean_json_text(text):
    if not text: return "{}"
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove trailing commas in JSON objects/arrays
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text

def get_file_tree(root_dir):
    file_tree = []
    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.github', 'venv', 'env', 'site-packages', 'dist', 'build', '.idea', '.vscode'}

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        rel_path = os.path.relpath(root, root_dir)
        if rel_path == '.': rel_path = ''
        
        path_parts = rel_path.split(os.sep)
        if any(p.startswith('.') and p != '.' for p in path_parts):
            continue

        for file in files:
            if file.startswith('.'): continue
            full_path = os.path.join(root, file)
            size = os.path.getsize(full_path)
            file_tree.append({
                'path': os.path.join(rel_path, file),
                'type': 'file',
                'size': size,
                'last_modified': os.path.getmtime(full_path)
            })
    return file_tree

def parse_agent_file(filepath, root_dir):
    """Extracts agent metadata from python source file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        docstring = ast.get_docstring(tree)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        
        agent_info = []
        for cls in classes:
            if 'Agent' in cls.name:
                agent_info.append({
                    'name': cls.name,
                    'path': os.path.relpath(filepath, root_dir),
                    'docstring': ast.get_docstring(cls) or docstring or "No description available.",
                    'methods': [m.name for m in cls.body if isinstance(m, ast.FunctionDef) and not m.name.startswith('_')],
                    'bases': [b.id for b in cls.bases if isinstance(b, ast.Name)]
                })
        return agent_info
    except Exception as e:
        return []

def scan_agents(root_dir):
    agents = []
    agent_dir = os.path.join(root_dir, 'core/agents')
    if os.path.exists(agent_dir):
        for root, _, files in os.walk(agent_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    agents.extend(parse_agent_file(os.path.join(root, file), root_dir))
    return agents

def get_knowledge_graph_data(root_dir):
    nodes = []
    edges = []
    
    # scan for .ttl or .json graph files
    data_dir = os.path.join(root_dir, 'data')
    gold_dir = os.path.join(data_dir, 'gold_standard')
    
    # Priority: Gold Standard v23.5 -> Root Knowledge Graph -> Others
    candidates = [
        os.path.join(gold_dir, 'v23_5_knowledge_graph.json'),
        os.path.join(data_dir, 'knowledge_graph_v2.json'),
        os.path.join(data_dir, 'knowledge_graph.json')
    ]
    # Add globs
    candidates.extend(glob.glob(os.path.join(data_dir, '*knowledge_graph*.json')))

    selected_file = None
    for f in candidates:
        if os.path.exists(f):
            selected_file = f
            break

    if selected_file:
        try:
            with open(selected_file, 'r', encoding='utf-8') as f:
                content = json.load(f)

                # Handle List (V23.5 Batch Dump) vs Dict (Single Graph)
                if isinstance(content, list):
                    # It's a list of knowledge graphs. We'll merge them or take the first one.
                    # For simplicity, let's flatten all nodes from all items.
                    for item in content:
                        if isinstance(item, dict):
                            # V23.5 Deep Dive Structure
                            if 'nodes' in item:
                                # nodes might be nested deeper in V23.5 structure: nodes -> entity_ecosystem ...
                                # Or it might be a direct dictionary of nodes.
                                raw_nodes = item['nodes']
                                if isinstance(raw_nodes, dict):
                                    # It's hierarchical. We need to flatten it or extract meaningful entities.
                                    # Heuristic: Extract the main entity
                                    meta = item.get('meta', {})
                                    target = meta.get('target', 'Unknown')
                                    nodes.append({'id': target, 'label': target, 'group': 'Target', 'meta': meta})

                                    # Attempt to find other entities in specific sub-keys
                                    for section, data in raw_nodes.items():
                                        if isinstance(data, dict):
                                            # e.g. "competitors" inside "market_radar_node" or similar
                                            pass
                                elif isinstance(raw_nodes, list):
                                    nodes.extend(raw_nodes)

                            # relationships
                            if 'relationships' in item:
                                edges.extend(item['relationships'])

                elif isinstance(content, dict):
                    # Check for "nodes"
                    node_list = content.get('nodes', [])
                    if isinstance(node_list, dict): # Sometimes nodes are a dict by ID
                        for nid, nops in node_list.items():
                            if isinstance(nops, dict):
                                nops['id'] = nid
                                nodes.append(nops)
                    elif isinstance(node_list, list):
                        nodes = node_list

                    # Check for "edges" or "relationships"
                    edge_list = content.get('edges', content.get('relationships', []))
                    edges = edge_list

        except Exception as e:
            print(f"Error reading KG {selected_file}: {e}")

    # Fallback/Augment with dummy data if empty
    if not nodes:
        nodes = [
            {'id': 'Adam_System', 'label': 'ADAM System', 'group': 'Core'},
            {'id': 'Market_Data', 'label': 'Market Data', 'group': 'Input'},
            {'id': 'Risk_Model', 'label': 'Risk Model', 'group': 'Logic'}
        ]
        edges = [
            {'from': 'Market_Data', 'to': 'Adam_System'},
            {'from': 'Adam_System', 'to': 'Risk_Model'}
        ]

    # Normalize for VisJS (id, label, group)
    normalized_nodes = []
    for n in nodes:
        if isinstance(n, str): continue
        nid = n.get('id') or n.get('name')
        if not nid: continue
        label = n.get('label') or n.get('name') or nid
        group = n.get('group') or n.get('type') or 'Entity'
        normalized_nodes.append({'id': nid, 'label': label, 'group': group})

    return {'nodes': normalized_nodes, 'edges': edges}

def get_financial_data(root_dir):
    # Scan downloads for CSVs
    market_data = {}
    downloads_dir = os.path.join(root_dir, 'downloads')

    # Also check data/market_data (though it's parquet mostly)

    if os.path.exists(downloads_dir):
        csv_files = glob.glob(os.path.join(downloads_dir, '*.csv'))
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            try:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()[-50:]
                    parsed = []
                    # Simple parsing
                    for line in lines[1:]:
                        cols = line.strip().split(',')
                        if len(cols) >= 2:
                            # Heuristic: Find first float col
                            val = 0.0
                            for c in cols[1:]:
                                try:
                                    val = float(c)
                                    break
                                except: continue
                            parsed.append({'time': cols[0], 'value': val})
                    market_data[filename] = parsed
            except: pass
    
    # Fallback mock data
    if not market_data:
        # Generate synthetic SPY data
        base = 400
        data = []
        for i in range(100):
            import math
            val = base + 10 * math.sin(i/10) + random.random() * 5
            data.append({'time': f"2023-01-{i%30+1:02d}", 'value': round(val, 2)})
        market_data['SPY_synthetic.csv'] = data

    return market_data

def get_reports(root_dir):
    reports = []
    report_dirs = [
        os.path.join(root_dir, 'core/libraries_and_archives/reports'),
        os.path.join(root_dir, 'data/gold_standard') # check here too
    ]

    files = []
    for d in report_dirs:
        if os.path.exists(d):
            files.extend(glob.glob(os.path.join(d, '*.json')))
            files.extend(glob.glob(os.path.join(d, '*.jsonl')))

    for f in files:
        try:
            with open(f, 'r') as file:
                content_str = file.read().strip()
                if not content_str: continue

                if f.endswith('.jsonl'):
                    # Try first line
                    data = json.loads(content_str.split('\n')[0])
                else:
                    data = json.loads(content_str)

                reports.append({
                    'title': data.get('title', data.get('company_name', os.path.basename(f))),
                    'path': os.path.relpath(f, root_dir),
                    'date': data.get('date', data.get('assessment_date', 'Unknown')),
                    'executive_summary': data.get('executive_summary', data.get('summary', '')),
                    'content': data
                })
        except Exception as e:
            # print(f"Error parsing {f}: {e}")
            pass
    return reports

def get_prompts(root_dir):
    prompts = []
    prompt_dir = os.path.join(root_dir, 'prompt_library')
    if os.path.exists(prompt_dir):
        for root, _, files in os.walk(prompt_dir):
            for f in files:
                if f.endswith('.md') or f.endswith('.json'):
                    path = os.path.join(root, f)
                    try:
                        with open(path, 'r', encoding='utf-8') as file:
                            prompts.append({
                                'name': f,
                                'path': os.path.relpath(path, root_dir),
                                'content': file.read()
                            })
                    except: pass
    return prompts

def get_training_sets(root_dir):
    sets = []
    ts_dir = os.path.join(root_dir, 'data/artisanal_training_sets')
    if os.path.exists(ts_dir):
        for f in glob.glob(os.path.join(ts_dir, '*.jsonl')):
            filename = os.path.basename(f)
            size = os.path.getsize(f)
            # Count lines for samples
            line_count = 0
            try:
                with open(f, 'r') as file:
                    for _ in file: line_count += 1
            except: pass

            # Determine type
            t_type = "Generic"
            if "snc" in filename: t_type = "Instruction (SNC)"
            elif "esg" in filename: t_type = "Classification (ESG)"
            elif "risk" in filename: t_type = "Reasoning (Risk)"
            elif "compliance" in filename: t_type = "Audit (Compliance)"

            sets.append({
                'filename': filename,
                'path': os.path.relpath(f, root_dir),
                'size': size,
                'samples': line_count,
                'type': t_type,
                'status': 'Ready'
            })
    return sets

def get_strategies(root_dir):
    strats = []
    s_dir = os.path.join(root_dir, 'data/strategies')
    if os.path.exists(s_dir):
        for f in glob.glob(os.path.join(s_dir, '*.json')):
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    strats.append({
                        'name': data.get('name', data.get('strategy_name', os.path.basename(f).replace('.json','').replace('_',' ').title())),
                        'filename': os.path.basename(f),
                        'description': data.get('description', ''),
                        'risk_profile': data.get('risk_profile', 'Unknown'),
                        'content': data
                    })
            except: pass
    return strats

def main():
    root_dir = os.path.abspath('.')
    print(f"Scanning {root_dir} for OMNI data...")

    data = {
        'stats': {
            'version': '23.5',
            'status': 'HYBRID_ONLINE',
            'cpu_load': random.randint(10, 40),
            'memory_usage': random.randint(30, 60),
            'active_tasks': random.randint(2, 8)
        },
        'files': get_file_tree(root_dir),
        'agents': scan_agents(root_dir),
        'reports': get_reports(root_dir),
        'prompts': get_prompts(root_dir),
        'strategies': get_strategies(root_dir),
        'training_sets': get_training_sets(root_dir),
        'knowledge_graph': get_knowledge_graph_data(root_dir),
        'financial_data': get_financial_data(root_dir)
    }

    output_path = os.path.join(root_dir, 'showcase/js/mock_data.js')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"window.MOCK_DATA = {json.dumps(data, indent=2)};")
    
    print(f"OMNI State written to {output_path}")

if __name__ == "__main__":
    main()
