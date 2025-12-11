import os
import sys
import json
import argparse

class ShowcaseGenerator:
    def __init__(self):
        pass

    def generate(self, directory, depth, files):
        relative_root = "../" * depth if depth > 0 else "./"
        current_dir_name = os.path.basename(directory) if directory != "." else "ROOT"

        # 1. Generate Sidebar File List
        try:
            all_items = sorted(os.listdir(directory))
        except PermissionError:
            return

        file_list_html = ""

        # Directories first
        for item in all_items:
            if item.startswith('.') or item == 'node_modules': continue
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                file_list_html += f'<li style="margin-bottom: 0.5rem;">📂 <a href="{item}/index.html" style="color: var(--text-primary);">{item}/</a></li>\n'

        # Then files
        for item in all_items:
            if item.startswith('.') or item == 'node_modules' or item == 'index.html': continue
            full_path = os.path.join(directory, item)
            if os.path.isfile(full_path):
                file_list_html += f'<li style="margin-bottom: 0.5rem;">📄 <span style="color: var(--text-secondary);">{item}</span></li>\n'

        # 2. Context-Aware Content
        main_content = ""
        readme_path = os.path.join(directory, "README.md")
        if os.path.exists(readme_path):
            main_content += f'<div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">'
            main_content += f'<h3 style="margin-top:0;">📜 README.md Detected</h3>'
            main_content += f'<p>Documentation available. <button class="cyber-btn" onclick="alert(\'Markdown rendering simulated\')">VIEW DOCS</button></p>'
            main_content += f'</div>'

        # Generic File Table
        main_content += '<table class="cyber-table"><thead><tr><th>Type</th><th>Name</th><th>Size</th></tr></thead><tbody>'
        for item in all_items:
            if item.startswith('.') or item == 'node_modules' or item == 'index.html': continue
            full_path = os.path.join(directory, item)
            is_dir = os.path.isdir(full_path)
            icon = "📂" if is_dir else "📄"
            size = "-" if is_dir else f"{os.path.getsize(full_path)} B"
            link = f'<a href="{item}/index.html">{item}</a>' if is_dir else item

            main_content += f'<tr><td>{icon}</td><td>{link}</td><td class="mono">{size}</td></tr>'
        main_content += '</tbody></table>'

        # 3. HTML Template
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adam v23.5 - {current_dir_name}</title>
    <link rel="stylesheet" href="{relative_root}showcase/css/style.css">
</head>
<body>
    <div class="scan-line"></div>
    <div class="scan-glow"></div>
    <div class="grid-bg" style="min-height: 100vh; padding: 2rem;">
        <header style="margin-bottom: 2rem; border-bottom: 1px solid var(--panel-border); padding-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="glitch-text" data-text="ADAM v23.5" style="margin: 0; font-size: 1.5rem; color: var(--primary-color);">ADAM v23.5</h1>
                    <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                        <span style="color: var(--text-muted);">{directory}</span>
                    </div>
                </div>
                <div class="cyber-badge badge-green">SYSTEM ONLINE</div>
            </div>
            <nav style="margin-top: 1rem; display: flex; gap: 1rem; font-family: var(--font-mono); font-size: 0.8rem;">
                <a href="{relative_root}index.html" class="cyber-btn">[UP LEVEL]</a>
                <a href="{relative_root}index.html" class="cyber-btn">[ROOT]</a>
                <a href="#" onclick="window.toggleRuntime()" class="cyber-btn">[LIVE RUNTIME]</a>
                <a href="https://github.com/your-repo" class="cyber-btn">[GITHUB]</a>
            </nav>
        </header>

        <div style="display: grid; grid-template-columns: 250px 1fr; gap: 2rem;">
            <aside class="glass-panel" style="padding: 1rem; height: fit-content;">
                <h3 style="font-size: 0.9rem; color: var(--text-muted); margin-top: 0;">EXPLORER</h3>
                <ul style="list-style: none; padding: 0; margin: 0; font-family: var(--font-mono); font-size: 0.85rem;">
                    {file_list_html}
                </ul>
            </aside>

            <main class="glass-panel" style="padding: 2rem; min-height: 500px;">
                <h2 style="margin-top: 0; color: var(--primary-color);">Directory Contents</h2>
                <div class="fade-in">
                    {main_content}
                </div>
            </main>
        </div>

        <footer style="margin-top: 3rem; text-align: center; font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted);">
            ADAM AUTO-GENERATED SHOWCASE | REF: {current_dir_name}
        </footer>
    </div>

    <script>
    (function() {{
        const CONFIG = {{
            mockMode: true,
            rootPath: "{relative_root}",
            currentDir: "{current_dir_name}"
        }};

        function initNav() {{
            console.log("Nav initialized for " + CONFIG.currentDir);
        }}

        async function loadData() {{
            try {{
                const response = await fetch('./AGENTS.md');
                if(response.ok) {{
                    console.log("AGENTS.md found");
                    // Implement markdown rendering logic here
                }}
            }} catch (e) {{
                console.warn("No local documentation found.");
            }}
        }}

        window.toggleRuntime = function() {{
            alert('Runtime Toggle: Active');
        }};

        document.addEventListener('DOMContentLoaded', () => {{
            initNav();
            loadData();
        }});
    }})();
    </script>
</body>
</html>"""

        # Write index.html
        index_path = os.path.join(directory, "index.html")
        with open(index_path, "w") as f:
            f.write(html)

        # Write manifest
        manifest_path = os.path.join(directory, "directory_manifest.jsonld")
        manifest = {
            "@context": "https://schema.org",
            "@type": "Dataset",
            "name": f"Adam v23.5 - {current_dir_name}",
            "description": f"Auto-generated showcase manifest for {current_dir_name}",
            "url": "./index.html",
            "hasPart": [{"@type": "Thing", "name": f} for f in files if f != "index.html"],
            "variableMeasured": "Interactivity Level: Static"
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Generated showcase for {directory}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Max directories to process")
    parser.add_argument("--target", default=".", help="Root directory")
    args = parser.parse_args()

    generator = ShowcaseGenerator()

    count = 0
    for root, dirs, files in os.walk(args.target):
        # Exclude hidden dirs and node_modules
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules']

        if ".git" in root or "node_modules" in root:
            continue

        # Skip if index.html already exists to preserve manual work
        # Also skip the root directory to avoid overwriting the main entry point if it exists
        if "index.html" in files:
            print(f"Skipping {root} (index.html exists)")
            continue

        depth = root.count(os.sep)
        if root == ".": depth = 0

        generator.generate(root, depth, files)
        count += 1
        if args.limit > 0 and count >= args.limit:
            print(f"Limit of {args.limit} reached. Stopping.")
            break

if __name__ == "__main__":
    main()
