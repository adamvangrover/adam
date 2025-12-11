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
    
    # Also parse AGENTS.md for high-level status if not found in code
    # (Simplified logic: just return what we found in code, maybe augment later)
    return agents

def get_knowledge_graph_data(root_dir):
    nodes = []
    edges = []
    
    # scan for .ttl or .json graph files
    data_dir = os.path.join(root_dir, 'data')
    
    # 1. Try to load v23_knowledge_graph.json or similar
    kg_files = glob.glob(os.path.join(data_dir, '*knowledge_graph*.json'))
    for kg_file in kg_files:
        try:
            with open(kg_file, 'r') as f:
                content = json.load(f)
                # Heuristic to find nodes/edges
                if 'nodes' in content:
                    # Normalize nodes
                    for n in content['nodes']:
                        # Handle different formats
                        nid = n.get('id') or n.get('name')
                        if nid:
                            nodes.append({'id': nid, 'label': n.get('label', 'Entity'), 'group': n.get('type', 'Unknown')})
                if 'edges' in content:
                    edges.extend(content['edges'])
                if 'relationships' in content: # Alternate key
                    edges.extend(content['relationships'])
        except: pass

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

    return {'nodes': nodes, 'edges': edges}

def get_financial_data(root_dir):
    # Scan downloads for CSVs
    market_data = {}
    downloads_dir = os.path.join(root_dir, 'downloads')
    if os.path.exists(downloads_dir):
        csv_files = glob.glob(os.path.join(downloads_dir, '*.csv'))
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            # Read last 50 lines to keep it light
            try:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()[-50:]
                    # Simple parsing assuming Date,Close or similar
                    parsed = []
                    header = lines[0].strip().split(',')
                    for line in lines[1:]:
                        cols = line.strip().split(',')
                        if len(cols) >= 2:
                            parsed.append({'time': cols[0], 'value': float(cols[4]) if len(cols) > 4 else float(cols[1])}) # Guessing Close col
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

def get_vault_content(root_dir):
    vault = {
        'reports': [],
        'prompts': [],
        'code_docs': []
    }
    
    # Reports
    report_dir = os.path.join(root_dir, 'core/libraries_and_archives/reports')
    if os.path.exists(report_dir):
        for f in glob.glob(os.path.join(report_dir, '*.json')):
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    vault['reports'].append({
                        'title': data.get('title', os.path.basename(f)),
                        'path': os.path.relpath(f, root_dir),
                        'content': data
                    })
            except: pass

    # Prompts
    prompt_dir = os.path.join(root_dir, 'prompt_library')
    if os.path.exists(prompt_dir):
        for root, _, files in os.walk(prompt_dir):
            for f in files:
                if f.endswith('.md') or f.endswith('.json'):
                    path = os.path.join(root, f)
                    try:
                        with open(path, 'r', encoding='utf-8') as file:
                            vault['prompts'].append({
                                'name': f,
                                'path': os.path.relpath(path, root_dir),
                                'content': file.read()
                            })
                    except: pass
    
    return vault

def main():
    root_dir = os.path.abspath('.')
    print(f"Scanning {root_dir} for OMNI data...")

    data = {
        'generated_at': time.time(),
        'system_info': {
            'version': '23.5',
            'status': 'HYBRID_ONLINE',
            'cpu_load': 12,
            'memory_usage': 34
        },
        'files': get_file_tree(root_dir),
        'agents': scan_agents(root_dir),
        'knowledge_graph': get_knowledge_graph_data(root_dir),
        'financial_data': get_financial_data(root_dir),
        'vault': get_vault_content(root_dir)
    }

    # Inject specific source code if requested (e.g. robo_advisor)
    robo_path = os.path.join(root_dir, 'core/advisory/robo_advisor.py')
    if os.path.exists(robo_path):
        with open(robo_path, 'r') as f:
            data['vault']['code_docs'].append({
                'name': 'robo_advisor.py',
                'content': f.read(),
                'language': 'python'
            })

    output_path = os.path.join(root_dir, 'showcase/js/mock_data.js')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"window.ADAM_STATE = {json.dumps(data, indent=2)};")
    
    print(f"OMNI State written to {output_path}")

if __name__ == "__main__":
    main()
