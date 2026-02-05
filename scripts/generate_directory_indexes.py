import os
import html

# Configuration
SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'webapp', 'services/webapp/client', '.idea', '.vscode', 'venv', 'env', '.env'}
CUSTOM_INDEX_MARKER = "ADAM AUTO-GENERATED SHOWCASE"

# Template
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Index of {current_path}</title>
    <link rel="stylesheet" href="{css_path}">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        .file-list {{ list-style: none; padding: 0; }}
        .file-item {{ padding: 8px 12px; border-bottom: 1px solid var(--panel-border); display: flex; align-items: center; justify-content: space-between; transition: background 0.2s; }}
        .file-item:hover {{ background: rgba(255, 255, 255, 0.05); }}
        .file-icon {{ margin-right: 10px; width: 20px; text-align: center; display: inline-block; }}
        .file-name {{ flex-grow: 1; font-family: var(--font-mono); font-size: 0.9rem; color: var(--text-primary); }}
        .file-tag {{ font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; margin-left: 10px; }}
        .tag-dir {{ background: rgba(6, 182, 212, 0.1); color: var(--primary-color); border: 1px solid var(--primary-color); }}
        .tag-html {{ background: rgba(245, 158, 11, 0.1); color: var(--warning-color); border: 1px solid var(--warning-color); }}
        .tag-py {{ background: rgba(16, 185, 129, 0.1); color: var(--success-color); border: 1px solid var(--success-color); }}
        .readme-container {{ margin-top: 30px; padding: 20px; background: rgba(0, 0, 0, 0.2); border: 1px solid var(--panel-border); border-radius: 4px; }}
        .readme-title {{ color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 1px; border-bottom: 1px solid var(--panel-border); padding-bottom: 5px; }}
        .readme-body {{ font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-muted); white-space: pre-wrap; overflow-x: auto; line-height: 1.5; }}
    </style>
</head>
<body>
    <div class="scan-line"></div>

    <header class="cyber-header" style="padding: 15px 20px; border-bottom: 1px solid var(--primary-color); background: rgba(5, 11, 20, 0.95); display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="mono" style="margin: 0; font-size: 1.2rem; color: var(--primary-color);">ADAM REPO BROWSER</h1>
            <div class="mono" style="font-size: 0.8rem; color: var(--text-muted);">PATH: <span style="color: var(--text-primary);">{display_path}</span></div>
        </div>
        <div style="text-align: right;">
            <div class="cyber-badge badge-cyan">SYSTEM ONLINE</div>
        </div>
    </header>

    <div style="padding: 20px; max-width: 1200px; margin: 0 auto;">
        <nav style="margin-bottom: 20px; display: flex; gap: 10px;">
            <a href="{parent_link}" class="cyber-btn">&uarr; UP LEVEL</a>
            <a href="{root_link}index.html" class="cyber-btn">ROOT</a>
            <a href="{root_link}showcase/index.html" class="cyber-btn" style="border-color: var(--accent-color); color: var(--accent-color);">MISSION CONTROL</a>
        </nav>

        <main class="glass-panel" style="padding: 0; overflow: hidden; border-radius: 4px;">
            <div style="padding: 10px 15px; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--panel-border); color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase;">
                Directory Contents
            </div>
            <ul class="file-list">
                {file_list_html}
            </ul>
        </main>

        {readme_html}
    </div>

    <footer style="margin-top: 50px; text-align: center; color: var(--text-muted); font-size: 0.8rem; padding-bottom: 20px;">
        ADAM AUTO-GENERATED SHOWCASE | REF: {current_dir_name}
    </footer>
</body>
</html>
"""

def generate_indexes(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter directories in place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.')]

        # Determine paths
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == '.':
            depth = 0
            display_path = "/ROOT"
            parent_link = "."
            root_link = "./"
        else:
            depth = len(rel_path.split(os.sep))
            display_path = "/" + rel_path
            parent_link = "../"
            root_link = "../" * depth

        # CSS Path
        css_path = root_link + "showcase/css/style.css"

        # Check existing index.html
        index_path = os.path.join(dirpath, "index.html")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Force update if it matches the marker (overwrite previous generated ones)
                    # Skip if it DOES NOT match marker and seems complex
                    if CUSTOM_INDEX_MARKER not in content and "Directory Listing" not in content and len(content) > 500:

                        # Root and Showcase Root are special
                        if rel_path == '.':
                            print(f"Skipping root: {dirpath}")
                            continue
                        if rel_path == "showcase" or dirpath.endswith("/showcase"):
                             print(f"Skipping showcase root: {dirpath}")
                             continue

                        print(f"Skipping custom index: {index_path}")
                        continue
            except:
                pass

        # Generate File List
        items = []

        # Directories first
        for d in sorted(dirnames):
            items.append({
                'name': d + "/",
                'href': d + "/index.html",
                'icon': "📂",
                'tag': "DIR",
                'tag_class': "tag-dir"
            })

        # Files
        for f in sorted(filenames):
            if f == "index.html": continue
            if f.startswith('.'): continue

            icon = "📄"
            tag = ""
            tag_class = ""

            if f.endswith(".html"):
                icon = "🖥️"
                tag = "DASHBOARD"
                tag_class = "tag-html"
            elif f.endswith(".py"):
                icon = "🐍"
                tag = "CODE"
                tag_class = "tag-py"
            elif f.endswith(".md"):
                icon = "📝"
            elif f.endswith(".json"):
                icon = "{}"

            items.append({
                'name': f,
                'href': f,
                'icon': icon,
                'tag': tag,
                'tag_class': tag_class
            })

        file_list_html = ""
        for item in items:
            tag_html = f'<span class="file-tag {item["tag_class"]}">{item["tag"]}</span>' if item["tag"] else ""
            file_list_html += f"""
                <li class="file-item">
                    <a href="{item['href']}" style="display: flex; align-items: center; width: 100%; color: inherit; text-decoration: none;">
                        <span class="file-icon">{item['icon']}</span>
                        <span class="file-name">{item['name']}</span>
                        {tag_html}
                    </a>
                </li>
            """

        # Readme
        readme_html = ""
        readme_file = next((f for f in filenames if f.lower() == "readme.md"), None)
        if readme_file:
            try:
                with open(os.path.join(dirpath, readme_file), 'r', encoding='utf-8') as f:
                    readme_content = html.escape(f.read())
                    readme_html = f"""
                    <div class="readme-container">
                        <div class="readme-title">📄 {readme_file} Preview</div>
                        <div class="readme-body"><pre>{readme_content}</pre></div>
                    </div>
                    """
            except:
                pass

        # Final HTML
        html_content = TEMPLATE.format(
            current_path=display_path,
            css_path=css_path,
            display_path=display_path,
            parent_link=parent_link,
            root_link=root_link,
            file_list_html=file_list_html,
            readme_html=readme_html,
            current_dir_name=os.path.basename(dirpath) or "ROOT"
        )

        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Generated: {index_path}")

if __name__ == "__main__":
    generate_indexes(".")
