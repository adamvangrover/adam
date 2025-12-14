import os
import json
import time
import sys
import ast
import glob

# Ensure core modules are accessible
sys.path.append(os.getcwd())

# Mock Ingestor if not available
try:
    from core.data_processing.universal_ingestor import UniversalIngestor, ArtifactType
except ImportError:
    UniversalIngestor = None

def get_agent_metadata(root_dir):
    agents = []
    agents_dir = os.path.join(root_dir, 'core', 'agents')

    for root, dirs, files in os.walk(agents_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ast.parse(content)

                    class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
                    if class_node:
                        doc = ast.get_docstring(class_node) or "No description."
                        agents.append({
                            'id': class_node.name.lower(),
                            'name': class_node.name,
                            'status': 'Active', # Mock status
                            'specialization': doc.split('\n')[0][:50] if doc else 'General',
                            'path': os.path.relpath(filepath, root_dir)
                        })
                except Exception as e:
                    print(f"Error parsing {file}: {e}")
    return agents

def get_reports(root_dir):
    reports = []
    report_dirs = [
        os.path.join(root_dir, 'core', 'libraries_and_archives', 'reports'),
        os.path.join(root_dir, 'core', 'libraries_and_archives', 'newsletters'),
        os.path.join(root_dir, 'data')
    ]

    for d in report_dirs:
        if not os.path.exists(d): continue
        for f in glob.glob(os.path.join(d, '*.*')):
            if f.endswith('.json') or f.endswith('.md'):
                name = os.path.basename(f)
                reports.append({
                    'id': name.replace('.', '_'),
                    'title': name,
                    'date': time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(f))),
                    'type': f.split('.')[-1].upper(),
                    'path': os.path.relpath(f, root_dir)
                })
    return reports

def main():
    root_dir = os.path.abspath('.')
    print(f"Scanning {root_dir}...")

    agents = get_agent_metadata(root_dir)
    reports = get_reports(root_dir)

    data = {
        'generated_at': time.time(),
        'agents': agents,
        'reports': reports
    }

    # Output path for React App
    # Using ui_manifest.json to avoid conflict with PWA manifest.json
    output_path = os.path.join(root_dir, 'services/webapp/client/public/ui_manifest.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Manifest generated at {output_path}")
    print(f"Found {len(agents)} agents and {len(reports)} reports.")

if __name__ == "__main__":
    main()
