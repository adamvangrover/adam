import os
import json
import time
import random
import sys

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
    # REDUCED LIMIT to avoid blowing up the JS file size
    MAX_SIZE = 1024 * 30 # 30 KB limit for inline viewing

    try:
        size = os.path.getsize(filepath)
        if size > MAX_SIZE:
            return f"File content too large to display inline ({size / 1024:.1f} KB). Please download to view."

        # Binary check
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return "Binary file (not displayed)."

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def get_file_tree(root_dir):
    file_tree = []
    # Exclude directories
    exclude_dirs = {
        '.git', '__pycache__', 'node_modules', '.github', 'venv', 'env',
        'site-packages', 'dist', 'build', '.idea', '.vscode', 'pytest_cache',
        'coverage', 'htmlcov', '.mypy_cache', 'tmp', 'temp'
    }
    # Include extensions
    include_exts = {
        '.md', '.json', '.jsonl', '.html', '.py', '.ttl',
        '.yaml', '.yml', '.sh', '.css', '.js', '.txt', '.csv'
    }

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

            ext = os.path.splitext(file)[1].lower()
            if ext not in include_exts and file not in ['AGENTS.md', 'Dockerfile', 'Makefile']:
                continue

            filepath = os.path.join(root, file)

            # Decide whether to read content
            content = None
            # Only read content if it's a target extension
            if ext in include_exts or file == 'AGENTS.md':
                content = get_file_content(filepath)

            file_tree.append({
                'path': os.path.join(rel_path, file).replace('\\', '/'),
                'type': 'file',
                'size': os.path.getsize(filepath),
                'last_modified': os.path.getmtime(filepath),
                'content': content
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
            with open(path, 'r', encoding='utf-8') as f:
                content = clean_json_text(f.read())
            # Simple heuristic for concatenated JSON
            if '}{' in content:
                return json.loads(content.split('}{')[1].replace('}', '', 1)) # Very rough
            return json.loads(content)
        except: pass
    return {}

# --- Integration with Universal Ingestor ---

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
            "category": "General", # Could extract from path
            "content": str(art['content'])
        })

    return reports, newsletters, prompts

def main():
    root_dir = os.path.abspath('.')
    print(f"Scanning {root_dir}...")

    # Run Gold Standard Ingestor
    ingestor = UniversalIngestor()
    ingestor.scan_directory(os.path.join(root_dir, "core/libraries_and_archives"))
    ingestor.scan_directory(os.path.join(root_dir, "prompt_library"))
    ingestor.scan_directory(os.path.join(root_dir, "data"))
    ingestor.scan_directory(os.path.join(root_dir, "docs"))

    # Save the Gold Standard JSONL - SKIP if too large to avoid sandbox limits
    # os.makedirs(os.path.join(root_dir, "data/gold_standard"), exist_ok=True)
    # ingestor.save_to_jsonl(os.path.join(root_dir, "data/gold_standard/knowledge_artifacts.jsonl"))

    # Map to UI Data
    reports, newsletters, prompts = get_ingested_data(ingestor, root_dir)

    # Get Full File Tree with Content
    files = get_file_tree(root_dir)

    data = {
        'generated_at': time.time(),
        'files': files,
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

    # Don't write the huge raw JSON for now, just the JS file
    # output_path = os.path.join(root_dir, 'showcase/data/ui_data.json')
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=2)

    js_output_path = os.path.join(root_dir, 'showcase/js/mock_data.js')
    os.makedirs(os.path.dirname(js_output_path), exist_ok=True)
    with open(js_output_path, 'w', encoding='utf-8') as f:
        f.write(f"window.MOCK_DATA = {json.dumps(data, indent=2)};")
    print(f"JS Data written to {js_output_path}")

if __name__ == "__main__":
    main()
