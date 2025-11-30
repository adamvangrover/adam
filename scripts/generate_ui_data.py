import os
import json
import time

def get_file_tree(root_dir):
    file_tree = []
    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.github', 'venv', 'env'}

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        path = root.split(os.sep)
        # Skip hidden directories
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
        with open(agents_md_path, 'r') as f:
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
                        'status': 'Active' # Simulated status
                    })
    return agents

def main():
    root_dir = os.path.abspath('.')
    print(f"Scanning {root_dir}...")

    data = {
        'generated_at': time.time(),
        'files': get_file_tree(root_dir),
        'agents': parse_agents_md(root_dir),
        'system_stats': {
            'cpu_usage': 12.5, # Mock data
            'memory_usage': 45.2, # Mock data
            'active_tasks': 3,
            'queued_tasks': 12
        }
    }

    output_path = os.path.join(root_dir, 'showcase/data/ui_data.json')
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Data written to {output_path}")

if __name__ == "__main__":
    main()
