import os

import markdown

# --- Configuration ---
ROOT_DIR = os.path.abspath(".")
SHOWCASE_DIR = os.path.join(ROOT_DIR, "showcase")
CSS_REL_PATH = "showcase/css/style.css"  # From root

# Limit generation to key directories to avoid overwhelming the file system/tool limits
TARGET_DIRS = [
    ".",
    "core",
    "core/agents",
    "core/engine",
    "core/system",
    "core/vertical_risk_agent",
    "docs",
    "scripts",
    "showcase",
    "services",
    "services/webapp",
    "prompt_library",
    "config"
]

EXCLUDED_DIRS = {
    ".git", ".github", "__pycache__", "node_modules", "venv", ".idea", ".vscode", "build", "dist", ".pytest_cache"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adam System :: {current_path}</title>
    <link rel="stylesheet" href="{css_path}">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        .readme-content {{ padding: 20px; }}
        .readme-content h1 {{ font-size: 1.5rem; margin-bottom: 1rem; color: var(--primary-color); }}
        .readme-content h2 {{ font-size: 1.25rem; margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 1px solid var(--panel-border); padding-bottom: 0.25rem; }}
        .readme-content p {{ margin-bottom: 1rem; line-height: 1.6; color: var(--text-secondary); }}
        .readme-content pre {{ background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 4px; overflow-x: auto; border: 1px solid var(--panel-border); }}
        .readme-content code {{ font-family: var(--font-mono); color: var(--accent-color); }}

        .file-icon {{ display: inline-block; width: 20px; text-align: center; margin-right: 8px; color: var(--text-muted); }}
        .dir-icon {{ color: var(--primary-color); }}
    </style>
</head>
<body>
    <div class="scan-line"></div>
    <div class="scan-glow"></div>

    <div class="container" style="max-width: 1200px; margin: 0 auto; padding: 20px;">

        <!-- Header -->
        <header style="margin-bottom: 40px; border-bottom: 1px solid var(--panel-border); padding-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="glitch-text" data-text="ADAM v23.5" style="font-size: 2rem; margin: 0; color: var(--primary-color);">ADAM v23.5</h1>
                <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted); margin-top: 5px;">
                    PATH: <span style="color: var(--text-primary);">/{current_path}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div class="cyber-badge badge-cyan">SYSTEM ONLINE</div>
                <div style="margin-top: 5px; font-size: 0.7rem; color: var(--text-muted);">Mode: SHOWCASE</div>
            </div>
        </header>

        <!-- Navigation -->
        <nav style="margin-bottom: 30px; display: flex; gap: 10px;">
            <a href="{parent_link}" class="cyber-btn">&uarr; UP LEVEL</a>
            <a href="{root_link}/index.html" class="cyber-btn">ROOT</a>
            <button onclick="toggleRuntime()" class="cyber-btn">LIVE RUNTIME</button>
            <a href="https://github.com/adamvangrover/adam" class="cyber-btn" target="_blank">GITHUB</a>
        </nav>

        <div style="display: grid; grid-template-columns: 250px 1fr; gap: 20px;">

            <!-- Sidebar / File Explorer -->
            <aside class="glass-panel" style="padding: 20px; border-radius: 4px; height: fit-content;">
                <h3 style="margin-top: 0; font-size: 0.9rem; text-transform: uppercase; color: var(--text-secondary); border-bottom: 1px solid var(--panel-border); padding-bottom: 10px; margin-bottom: 15px;">
                    Directory Contents
                </h3>
                <ul style="list-style: none; padding: 0; margin: 0; font-family: var(--font-mono); font-size: 0.85rem;">
                    {file_list_items}
                </ul>
            </aside>

            <!-- Main Content -->
            <main>
                <!-- Runtime Overlay (Hidden by default) -->
                <div id="runtime-panel" class="glass-panel fade-in" style="display: none; padding: 20px; margin-bottom: 20px; border-color: var(--success-color);">
                    <h3 style="color: var(--success-color); margin-top: 0;">Live System Interface</h3>
                    <p style="font-size: 0.9rem;">Connecting to Orchestrator at localhost:5001...</p>
                    <div style="background: black; padding: 10px; border-radius: 4px; font-family: var(--font-mono); height: 100px; overflow-y: auto; font-size: 0.8rem; color: #0f0;">
                        > Connecting...<br>
                        > Error: API Unreachable (Mock Mode Active)<br>
                        > System Status: STANDBY
                    </div>
                </div>

                <!-- Readme Content -->
                <div class="glass-panel" style="border-radius: 4px;">
                    {readme_html}
                </div>
            </main>
        </div>

        <!-- Footer -->
        <footer style="margin-top: 50px; border-top: 1px solid var(--panel-border); padding-top: 20px; text-align: center; color: var(--text-muted); font-size: 0.8rem;">
            ADAM AUTO-GENERATED SHOWCASE | <span class="mono">REF: {current_path}</span>
        </footer>
    </div>

    <script>
        function toggleRuntime() {{
            const panel = document.getElementById('runtime-panel');
            if (panel.style.display === 'none') {{
                panel.style.display = 'block';
            }} else {{
                panel.style.display = 'none';
            }}
        }}
    </script>
</body>
</html>
"""

def get_css_path(current_dir):
    rel_path = os.path.relpath(ROOT_DIR, current_dir)
    return os.path.join(rel_path, CSS_REL_PATH).replace("\\", "/")

def get_parent_link(current_dir):
    rel_path = os.path.relpath(os.path.dirname(current_dir), current_dir)
    return os.path.join(rel_path, "index.html").replace("\\", "/")

def get_root_link(current_dir):
    rel_path = os.path.relpath(ROOT_DIR, current_dir)
    return rel_path.replace("\\", "/")

def generate_file_list(current_dir):
    items = sorted(os.listdir(current_dir))
    directories = []
    files = []

    for item in items:
        if item in EXCLUDED_DIRS or item.startswith('.'):
            continue

        full_path = os.path.join(current_dir, item)
        if os.path.isdir(full_path):
            directories.append(item)
        else:
            files.append(item)

    html = ""

    # Directories first
    for d in directories:
        link = f"{d}/index.html"
        html += f'<li style="margin-bottom: 6px;"><span class="dir-icon">📂</span> <a href="{link}" style="color: var(--primary-color);">{d}/</a></li>'

    # Files
    for f in files:
        if f == "index.html": continue
        html += f'<li style="margin-bottom: 6px;"><span class="file-icon">📄</span> <a href="{f}" style="color: var(--text-secondary);">{f}</a></li>'

    return html

def render_readme(current_dir):
    readme_files = [f for f in os.listdir(current_dir) if f.lower().startswith('readme')]
    if not readme_files:
        return '<div class="readme-content" style="text-align: center; color: var(--text-muted); padding: 40px;">No README found in this directory.</div>'

    readme_path = os.path.join(current_dir, readme_files[0])
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            text = f.read()
            html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
            return f'<div class="readme-content">{html}</div>'
    except Exception as e:
        return f'<div class="readme-content">Error rendering README: {e}</div>'

def process_directory(current_dir):
    print(f"Processing: {current_dir}")

    # Calculate paths
    css_path = get_css_path(current_dir)
    current_path_display = os.path.relpath(current_dir, ROOT_DIR)
    if current_path_display == ".": current_path_display = "ROOT"

    parent_link = get_parent_link(current_dir)
    root_link = get_root_link(current_dir)

    file_list_items = generate_file_list(current_dir)
    readme_html = render_readme(current_dir)

    html_content = HTML_TEMPLATE.format(
        current_path=current_path_display,
        css_path=css_path,
        parent_link=parent_link,
        root_link=root_link,
        file_list_items=file_list_items,
        readme_html=readme_html
    )

    with open(os.path.join(current_dir, "index.html"), "w", encoding='utf-8') as f:
        f.write(html_content)

def main():
    # Only process target directories
    for target in TARGET_DIRS:
        abs_path = os.path.join(ROOT_DIR, target)
        if os.path.isdir(abs_path):
            process_directory(abs_path)
        else:
            print(f"Skipping missing target: {target}")

if __name__ == "__main__":
    main()
