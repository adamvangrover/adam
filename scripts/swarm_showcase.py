import os
import json
import urllib.parse
import ast

ROOT_DIR = "."
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".ipynb_checkpoints", "venv", "env", ".vscode", ".idea",
    "archive", "services", "src", "webapp", "downloads", "tests", "adam_v21_upgrade", "verification",
    "target", "build", "dist", "site-packages", "evaluation", "evals",
    "skills", "templates", "assets", "mock_data", "examples", "legacy", "migrations",
    "libraries_and_archives", "financial_suite"  # Skip deep data archives to focus on logic
}
STYLE_PATH = "showcase/css/style.css"
MAX_DIRS = 35  # Tighter limit to avoid sandbox file limits

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 | {dir_name}</title>
    <link rel="stylesheet" href="{style_path}">
    <style>
        /* Fallback styles if stylesheet is missing */
        body {{ background-color: #050b14; color: #06b6d4; font-family: monospace; margin: 0; padding: 0; }}
        .header {{ border-bottom: 1px solid #06b6d4; padding: 20px; display: flex; justify-content: space-between; align-items: center; }}
        .breadcrumb {{ font-size: 1.2em; color: #e2e8f0; }}
        .badge {{ border: 1px solid #06b6d4; padding: 5px 10px; border-radius: 4px; font-size: 0.8em; text-transform: uppercase; }}
        .container {{ display: flex; height: calc(100vh - 80px); }}
        .sidebar {{ width: 300px; border-right: 1px solid #334155; padding: 20px; overflow-y: auto; background: rgba(5, 11, 20, 0.95); }}
        .main-stage {{ flex: 1; padding: 30px; overflow-y: auto; }}
        .file-item {{ padding: 8px; cursor: pointer; display: block; color: #94a3b8; text-decoration: none; }}
        .file-item:hover {{ color: #06b6d4; background: rgba(6, 182, 212, 0.1); }}
        .folder {{ color: #fbbf24; font-weight: bold; }}
        .footer {{ border-top: 1px solid #334155; padding: 10px; text-align: center; font-size: 0.8em; color: #64748b; background: #020617; }}
        .nav-links a {{ margin-left: 15px; color: #06b6d4; text-decoration: none; border: 1px solid #06b6d4; padding: 5px 10px; transition: all 0.2s; }}
        .nav-links a:hover {{ background: #06b6d4; color: #000; }}
        pre {{ background: #0f172a; padding: 15px; border-radius: 6px; overflow-x: auto; border: 1px solid #1e293b; }}
        h1, h2, h3 {{ color: #f8fafc; text-shadow: 0 0 10px rgba(6, 182, 212, 0.5); }}
        .card {{ background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; padding: 20px; margin-bottom: 20px; border-radius: 8px; }}
        .glitch {{ position: relative; }}
        .module-doc {{ margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px dashed #334155; }}
        .doc-text {{ color: #94a3b8; font-style: italic; }}
    </style>
    <!-- Mermaid JS -->
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
    </script>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; display:inline-block;" class="glitch">ADAM v23.5</h1>
            <span class="breadcrumb" style="margin-left: 20px;">{current_path}</span>
        </div>
        <div class="nav-links">
            <a href="{up_level}">[UP LEVEL]</a>
            <a href="{root_link}">[ROOT]</a>
            <a href="#" onclick="window.toggleRuntime()">[LIVE RUNTIME]</a>
            <a href="https://github.com/your-repo/adam" target="_blank">[GITHUB]</a>
        </div>
        <span class="badge">SYSTEM ONLINE</span>
    </div>

    <div class="container">
        <div class="sidebar">
            <h3>Directory Structure</h3>
            {file_list}
        </div>
        <div class="main-stage" id="main-content">
            <div class="card">
                <h2>Overview</h2>
                <div id="readme-content">
                    {content_preview}
                </div>
            </div>

            {mermaid_section}

            {modules_section}

            <div id="runtime-panel" style="display:none; border-top: 1px solid #06b6d4; margin-top: 20px; padding-top: 20px;">
                <h3>Runtime Diagnostics</h3>
                <pre>Mock Mode: ACTIVE\nAgent Swarm: STANDBY</pre>
            </div>
        </div>
    </div>

    <div class="footer">
        ADAM AUTO-GENERATED SHOWCASE | REF: {dir_name}
    </div>

    <script>
    // AUTO-GENERATED: Showcase Runtime
    (function() {{
        const CONFIG = {{
            mockMode: true,
            rootPath: "{root_link}",
            currentDir: "{dir_name}"
        }};

        // 1. Navigation Handler
        function initNav() {{
            console.log("Nav initialized for " + CONFIG.currentDir);
        }}

        // 2. Data Loader (Graceful Fallback)
        async function loadData() {{
            try {{
                // Attempt to load local descriptor
                const response = await fetch('./AGENTS.md');
                if(response.ok) {{
                    const text = await response.text();
                    // Simple text render for now, simulated markdown
                    document.getElementById('readme-content').innerHTML = '<pre>' + text + '</pre>';
                }}
            }} catch (e) {{
                console.warn("No local documentation found.");
            }}
        }}

        // 3. Runtime Toggle
        window.toggleRuntime = function() {{
            const panel = document.getElementById('runtime-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        }};

        document.addEventListener('DOMContentLoaded', () => {{
            initNav();
            // loadData(); // Disabled to prefer generated content
        }});
    }})();
    </script>
</body>
</html>
"""

def analyze_directory(directory):
    modules = []
    classes = []

    for f in sorted(os.listdir(directory)):
        path = os.path.join(directory, f)
        if f.endswith('.py') and f != "__init__.py":
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    tree = ast.parse(content)
                    docstring = ast.get_docstring(tree)
                    modules.append({'name': f, 'doc': docstring or "No description available."})

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            classes.append({'module': f, 'name': node.name})
            except Exception as e:
                pass

    return modules, classes

def generate_mermaid_graph(classes):
    if not classes: return ""
    graph = '<div class="card"><h2>Class Structure</h2><pre class="mermaid">\nclassDiagram\n'
    for c in classes:
        graph += f"    class {c['name']} {{\n        {c['module']}\n    }}\n"
    graph += '</pre></div>'
    return graph

def generate_modules_section(modules):
    if not modules: return ""
    html = '<div class="card"><h2>Module Documentation</h2>'
    for m in modules:
        html += f'<div class="module-doc"><h3>{m["name"]}</h3><p class="doc-text">{m["doc"]}</p></div>'
    html += '</div>'
    return html

def get_relative_path(from_dir, to_file):
    return os.path.relpath(to_file, from_dir)

def generate_file_list_html(directory):
    try:
        items = sorted(os.listdir(directory))
    except OSError:
        return ""

    html = []
    if directory != ".":
         html.append(f'<a href="../index.html" class="file-item folder">📂 .. (Parent)</a>')

    for item in items:
        if item.startswith('.') or item == "index.html" or item == "directory_manifest.jsonld":
            continue
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            if item in SKIP_DIRS: continue
            html.append(f'<a href="{item}/index.html" class="file-item folder">📂 {item}</a>')
        else:
            icon = "📄"
            if item.endswith(".json"): icon = "📊"
            if item.endswith(".md"): icon = "📝"
            html.append(f'<a href="{item}" class="file-item">{icon} {item}</a>')
    return "\n".join(html)

def generate_content_preview(directory):
    readme_path = os.path.join(directory, "README.md")
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple line-break to BR
                html = content.replace('\n', '<br>')
                return f"<h3>README.md</h3><div style='line-height:1.6;'>{html[:3000]}...</div>"
        except:
            pass
    return "<p>No README.md found. See Module Documentation below.</p>"

def generate_manifest(directory, files):
    manifest = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": f"Adam v23.5 - {os.path.basename(os.path.abspath(directory))}",
        "url": "./index.html",
        "hasPart": [],
        "variableMeasured": "Interactivity Level: Static"
    }
    for f in files:
        if f.endswith('.py'):
            manifest["hasPart"].append({"@type": "SoftwareSourceCode", "name": f, "programmingLanguage": "Python"})
        elif f.endswith('.md'):
            manifest["hasPart"].append({"@type": "TechArticle", "name": f})
    return json.dumps(manifest, indent=2)

def main():
    print("Starting Showcase Swarm Generator v2 (Enhanced)...")

    count = 0
    # Walk the directory tree
    for root, dirs, files in os.walk(ROOT_DIR):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

        # Prioritize prompts first (to ensure they are covered)
        dirs.sort(key=lambda x: 0 if x == "prompts" else (1 if x == "core" else (2 if x == "docs" else 3)))

        if count >= MAX_DIRS:
            print(f"Reached MAX_DIRS limit ({MAX_DIRS}). Stopping.")
            break

        # Calculate paths
        rel_path = os.path.relpath(root, ROOT_DIR)
        if rel_path == ".": rel_path = ""
        depth = 0 if rel_path == "" else rel_path.count(os.sep) + 1

        if rel_path == "":
            style_rel = STYLE_PATH
            root_link = "./index.html"
            up_level = "#"
        else:
            style_rel = "../" * depth + STYLE_PATH
            root_link = "../" * depth + "index.html"
            up_level = "../index.html"

        current_dir_name = os.path.basename(os.path.abspath(root))
        if not current_dir_name: current_dir_name = "ROOT"

        # Analyze content
        modules, classes = analyze_directory(root)
        mermaid_section = generate_mermaid_graph(classes)
        modules_section = generate_modules_section(modules)
        file_list_html = generate_file_list_html(root)
        content_preview = generate_content_preview(root)

        html_content = HTML_TEMPLATE.format(
            dir_name=current_dir_name,
            style_path=style_rel,
            current_path="/" + rel_path.replace(os.sep, "/"),
            up_level=up_level,
            root_link=root_link,
            file_list=file_list_html,
            content_preview=content_preview,
            mermaid_section=mermaid_section,
            modules_section=modules_section
        )

        # Write files
        index_path = os.path.join(root, "index.html")
        if root in ["./showcase", "showcase", "."]:
             if os.path.exists(index_path):
                 try:
                     with open(index_path, 'r', encoding='utf-8') as f:
                         if "Mission Control" in f.read(): continue
                 except: pass

        try:
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            with open(os.path.join(root, "directory_manifest.jsonld"), "w", encoding="utf-8") as f:
                f.write(generate_manifest(root, files))

            print(f"Generated enhanced showcase for: {root}")
            count += 1
        except Exception as e:
            print(f"Failed {root}: {e}")

if __name__ == "__main__":
    main()
