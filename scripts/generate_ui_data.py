import os
import json
import time
import random
import sys
import ast
import glob

# Ensure core modules are accessible
sys.path.append(os.getcwd())

from core.data_processing.universal_ingestor import UniversalIngestor, ArtifactType
from core.data_processing.synthetic_data_factory import DataFactory

# --- Helper Functions ---

def clean_json_text(text):
    if not text: return "{}"
    import re
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text

def get_file_content(filepath):
    """Reads file content with strict size limit and text decoding."""
    MAX_SIZE = 1024 * 50 # 50 KB limit

    try:
        size = os.path.getsize(filepath)
        if size > MAX_SIZE:
            return f"File content too large to display inline ({size / 1024:.1f} KB). Please download to view."

        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return "Binary file (not displayed)."

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# --- Data Indexing Functions ---

def scan_strategies(root_dir):
    """Scans data/strategies for investment strategies."""
    strategies = []
    strategy_dir = os.path.join(root_dir, 'data', 'strategies')

    if os.path.exists(strategy_dir):
        for f in glob.glob(os.path.join(strategy_dir, '*.json')):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.loads(clean_json_text(file.read()))
                    strategies.append({
                        'name': data.get('name', os.path.basename(f)),
                        'description': data.get('description', 'No description available.'),
                        'risk_level': data.get('risk_level', 'Medium'),
                        'assets': data.get('assets', []),
                        'path': os.path.relpath(f, root_dir)
                    })
            except Exception as e:
                print(f"Error parsing strategy {f}: {e}")
    return strategies

def scan_training_sets(root_dir):
    """Scans data/artisanal_training_sets for high-value datasets."""
    sets = []
    train_dir = os.path.join(root_dir, 'data', 'artisanal_training_sets')
    
    if os.path.exists(train_dir):
        for f in glob.glob(os.path.join(train_dir, '*.jsonl')):
            try:
                count = 0
                preview = []
                with open(f, 'r', encoding='utf-8') as file:
                    for line in file:
                        count += 1
                        if count <= 3:
                            preview.append(json.loads(line))

                sets.append({
                    'name': os.path.basename(f),
                    'path': os.path.relpath(f, root_dir),
                    'count': count,
                    'preview': preview
                })
            except Exception as e:
                print(f"Error parsing training set {f}: {e}")
    return sets

def scan_omni_graph(root_dir):
    """Indexes the Omni-Graph structure."""
    omni_data = {
        'constellations': [],
        'dossiers': [],
        'relationships': []
    }
    base_dir = os.path.join(root_dir, 'data', 'omni_graph')

    if os.path.exists(base_dir):
        # Constellations (Breadth)
        for f in glob.glob(os.path.join(base_dir, 'constellations', '*.json')):
             omni_data['constellations'].append(os.path.basename(f))

        # Dossiers (Depth)
        for f in glob.glob(os.path.join(base_dir, 'dossiers', '*.json')):
             omni_data['dossiers'].append(os.path.basename(f))

        # Relationships (Edges)
        for f in glob.glob(os.path.join(base_dir, 'relationships', '*.json')):
             omni_data['relationships'].append(os.path.basename(f))

    return omni_data

def scan_agents_metadata(root_dir):
    """Parses Python agent files to extract docstrings and metadata."""
    agents = []
    agents_dir = os.path.join(root_dir, 'core', 'agents')

    # Walk through core/agents and subdirectories
    for root, dirs, files in os.walk(agents_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())

                    docstring = ast.get_docstring(tree)
                    class_name = None
                    # Find first class
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            class_name = node.name
                            if not docstring: # Prefer class docstring
                                docstring = ast.get_docstring(node)
                            break

                    if class_name:
                        agents.append({
                            'name': class_name,
                            'file': file,
                            'path': os.path.relpath(filepath, root_dir),
                            'docstring': docstring if docstring else "No documentation found.",
                            'type': 'Specialized' if 'specialized' in root else 'Core'
                        })
                except Exception as e:
                    print(f"Error parsing agent {file}: {e}")
    return agents

def main():
    root_dir = os.path.abspath('.')
    print(f"Scanning {root_dir}...")

    # 1. Run Universal Ingestor for standard artifacts
    ingestor = UniversalIngestor()
    try:
        ingestor.scan_directory(os.path.join(root_dir, "core/libraries_and_archives"))
        ingestor.scan_directory(os.path.join(root_dir, "prompt_library"))
        # We handle data manually below for specific structures
    except Exception as e:
        print(f"Ingestor warning: {e}")

    # Map Ingested Data
    reports_artifacts = ingestor.get_artifacts_by_type(ArtifactType.REPORT)
    reports = []
    for art in reports_artifacts:
        content = art['content'] if isinstance(art['content'], dict) else {'text': str(art['content'])[:500]}
        reports.append({
            'title': art['title'],
            'path': os.path.relpath(art['source_path'], root_dir),
            'content': content
        })

    prompts_artifacts = ingestor.get_artifacts_by_type(ArtifactType.PROMPT)
    prompts = [{
        'name': p['title'],
        'path': os.path.relpath(p['source_path'], root_dir),
        'content': str(p['content'])
    } for p in prompts_artifacts]

    # 2. Deep Scan for New Intelligence
    strategies = scan_strategies(root_dir)
    training_sets = scan_training_sets(root_dir)
    omni_graph = scan_omni_graph(root_dir)
    agent_metadata = scan_agents_metadata(root_dir)

    # 3. Assemble Data Object
    data = {
        'generated_at': time.time(),
        'stats': {
            'agent_count': len(agent_metadata),
            'report_count': len(reports),
            'prompt_count': len(prompts),
            'strategy_count': len(strategies),
            'training_set_count': len(training_sets)
        },
        'agents': agent_metadata,
        'reports': reports,
        'prompts': prompts,
        'strategies': strategies,
        'training_sets': training_sets,
        'omni_graph': omni_graph,
        'files': [] # simplified for this view, or we can do a partial tree if needed
    }

    # 4. Output
    js_output_path = os.path.join(root_dir, 'showcase/js/mock_data.js')
    os.makedirs(os.path.dirname(js_output_path), exist_ok=True)
    with open(js_output_path, 'w', encoding='utf-8') as f:
        f.write(f"window.MOCK_DATA = {json.dumps(data, indent=2)};")

    # Output to webapp/public/data/manifest.json
    manifest_path = os.path.join(root_dir, 'webapp/public/data/manifest.json')
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Successfully generated UI data at {js_output_path} and {manifest_path}")
    print(f"Indexed: {len(strategies)} Strategies, {len(training_sets)} Training Sets, {len(agent_metadata)} Agents.")

if __name__ == "__main__":
    main()
