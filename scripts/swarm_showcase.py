import os
import json
import datetime
import sys
from jinja2 import Template

# Try to import markdown, fallback if not available
try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# Configuration
SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.pytest_cache', 'venv', 'env', 'dist', 'build', '.idea', '.vscode'}
# Restrict to key directories for demonstration to avoid generating thousands of files in one go
TARGET_ROOTS = ['prompts', 'showcase', 'core/agents', 'core/system']
ROOT_DIR = "."

# The "Cyber-Minimalist" Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 - {{ current_dir_name }}</title>
    <link rel="stylesheet" href="{{ relative_css_path }}">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <!-- Tailwind CSS (via CDN for standalone) -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        slate: {
                            850: '#151e2e',
                            900: '#0f172a',
                            950: '#020617',
                        },
                        cyan: {
                            400: '#22d3ee',
                            500: '#06b6d4',
                            900: '#164e63',
                        }
                    },
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    }
                }
            }
        }
    </script>
    <style>
        /* Custom overrides from style.css integration */
        .glitch-text { position: relative; }
        .cyber-badge { display: inline-block; padding: 0.15rem 0.5rem; font-size: 0.65rem; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; border-radius: 2px; border: 1px solid currentColor; }
        .badge-green { color: #34d399; background: rgba(16, 185, 129, 0.1); }
        .badge-blue { color: #60a5fa; background: rgba(59, 130, 246, 0.1); }
        .glass-panel { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(148, 163, 184, 0.1); }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 font-sans antialiased selection:bg-cyan-500 selection:text-white min-h-screen flex flex-col">
    <div class="scan-line fixed top-0 left-0 w-full h-full pointer-events-none z-50 opacity-10 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSI0cHgiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMXB4IiBmaWxsPSIjZmZmZmZmIiBvcGFjaXR5PSIwLjEiLz48L3N2Zz4=')]"></div>

    <div class="container mx-auto p-4 max-w-7xl flex-grow">
        <!-- HEADER -->
        <header class="mb-8 border-b border-slate-700 pb-4">
            <div class="flex justify-between items-center">
                <div>
                    <h1 class="text-3xl font-bold font-mono text-cyan-400 tracking-tight">ADAM v23.5</h1>
                    <div class="text-sm text-slate-400 font-mono mt-1">
                        PATH: <span class="text-yellow-500">{{ current_path }}</span>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <span class="cyber-badge badge-green">SYSTEM ONLINE</span>
                    <span class="cyber-badge badge-blue">SHOWCASE MODE</span>
                </div>
            </div>

            <!-- NAV -->
            <nav class="mt-4 flex space-x-6 text-sm font-mono text-cyan-500">
                <a href="../index.html" class="hover:text-cyan-300 transition-colors">[UP LEVEL]</a>
                <a href="{{ relative_root_path }}index.html" class="hover:text-cyan-300 transition-colors">[ROOT]</a>
                <button onclick="window.toggleRuntime()" class="hover:text-cyan-300 transition-colors cursor-pointer">[LIVE RUNTIME]</button>
                <a href="https://github.com/your-repo" target="_blank" class="hover:text-cyan-300 transition-colors">[GITHUB]</a>
            </nav>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <!-- SIDEBAR -->
            <aside class="col-span-1">
                <div class="glass-panel p-4 rounded-lg">
                    <h3 class="text-xs font-bold text-slate-500 uppercase mb-3 font-mono">Directory Contents</h3>
                    <ul class="space-y-2 font-mono text-sm">
                        {% for item in dirs %}
                        <li>
                            <a href="{{ item }}/index.html" class="flex items-center text-blue-400 hover:text-blue-300 transition-colors">
                                <span class="mr-2">📂</span> {{ item }}
                            </a>
                        </li>
                        {% endfor %}

                        <div class="my-2 border-t border-slate-700/50"></div>

                        {% for item in files %}
                        <li>
                            <a href="{{ item }}" class="flex items-center text-slate-300 hover:text-white transition-colors truncate" title="{{ item }}">
                                <span class="mr-2 opacity-50">📄</span> {{ item }}
                            </a>
                        </li>
                        {% endfor %}
                    </ul>
                </div>
            </aside>

            <!-- MAIN STAGE -->
            <main class="col-span-1 md:col-span-3">
                <div class="glass-panel p-6 rounded-lg min-h-[500px] relative overflow-hidden">
                    <div class="absolute top-0 right-0 p-2 opacity-10 font-mono text-6xl font-bold text-white pointer-events-none">
                        {{ current_dir_name }}
                    </div>

                    <h2 class="text-xl font-bold text-cyan-400 mb-6 font-mono border-l-4 border-cyan-500 pl-3">
                        {% if readme_content %} README.md {% else %} DIRECTORY OVERVIEW {% endif %}
                    </h2>

                    <div class="prose prose-invert prose-cyan max-w-none text-slate-300 font-sans">
                        {% if readme_content %}
                            <div id="readme-content">
                                {{ readme_content | safe }}
                            </div>
                        {% else %}
                            <div class="text-center py-20 flex flex-col items-center justify-center opacity-50">
                                <div class="text-6xl mb-4 grayscale">🔮</div>
                                <h3 class="text-2xl font-mono text-slate-500">AWAITING INPUT</h3>
                                <p class="text-slate-600 mt-2">No README.md found in this sector.</p>
                                <p class="text-slate-700 text-xs mt-4 font-mono">system.scan_complete(status="empty")</p>
                            </div>
                        {% endif %}
                    </div>

                    {% if prompt_files %}
                    <div class="mt-12 border-t border-slate-700 pt-6">
                        <h3 class="text-lg font-bold text-yellow-400 mb-4 font-mono flex items-center">
                            <span class="mr-2">⚡</span> PROMPT LIBRARY DETECTED
                        </h3>
                        <div class="grid grid-cols-1 gap-4">
                            {% for p in prompt_files %}
                            <div class="bg-slate-900/50 p-4 rounded border border-slate-700 hover:border-cyan-500/30 transition-colors">
                                <div class="flex justify-between items-center mb-2">
                                    <h4 class="font-mono text-sm text-cyan-300">{{ p }}</h4>
                                    <span class="text-xs text-slate-500 font-mono">PROMPT</span>
                                </div>
                                <div class="bg-black/30 p-2 rounded mb-2 text-xs text-slate-500 font-mono truncate">
                                    prompts/{{ current_dir_name }}/{{ p }}
                                </div>
                                <button class="cyber-btn w-full text-center py-1 text-xs uppercase tracking-widest border border-slate-600 hover:border-cyan-400 hover:text-cyan-400 transition-all rounded" onclick="alert('Copied prompt template to clipboard!')">
                                    Copy Template
                                </button>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endif %}
                </div>
            </main>
        </div>

        <!-- FOOTER -->
        <footer class="mt-12 py-6 border-t border-slate-800 text-center text-xs font-mono text-slate-600">
            <p>ADAM AUTO-GENERATED SHOWCASE | REF: {{ current_dir_name }} | GENERATED: {{ generation_time }}</p>
            <p class="mt-1 opacity-50">SYSTEM STATUS: NOMINAL</p>
        </footer>
    </div>

    <!-- RUNTIME TOGGLE SCRIPT -->
    <script>
    (function() {
        const CONFIG = {
            mockMode: true,
            rootPath: "{{ relative_root_path }}",
            currentDir: "{{ current_dir_name }}"
        };

        window.toggleRuntime = function() {
            alert("Runtime connection simulating... [MOCK MODE ACTIVE]\\n\\nConnecting to MetaOrchestrator...\\nError: Connection Refused (Static Mode)");
        };
    })();
    </script>
</body>
</html>
"""

def generate_manifest(directory, files, depth):
    manifest = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": f"Adam v23.5 - {os.path.basename(directory)}",
        "description": f"Auto-generated showcase manifest for {os.path.basename(directory)}",
        "url": "./index.html",
        "hasPart": [],
        "variableMeasured": "Interactivity Level: Static"
    }

    for f in files:
        if f.endswith('.py'):
            manifest["hasPart"].append({"@type": "SoftwareSourceCode", "name": f, "programmingLanguage": "Python"})
        elif f.endswith('.md'):
            manifest["hasPart"].append({"@type": "TechArticle", "name": f})
        elif f.endswith('.json'):
            manifest["hasPart"].append({"@type": "Dataset", "name": f})

    return json.dumps(manifest, indent=2)

def generate_showcase_for_dir(directory, depth):
    # Only generate if index.html doesn't exist OR if we are forcing update.
    # For now, we overwrite.

    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        return

    dirs = [d for d in items if os.path.isdir(os.path.join(directory, d)) and d not in SKIP_DIRS and not d.startswith('.')]
    files = [f for f in items if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')]

    # Filter out index.html from display if it exists, to avoid visual clutter?
    # Or keep it. Let's keep it.

    current_dir_name = os.path.basename(directory)
    if current_dir_name == '.' or current_dir_name == '':
        current_dir_name = 'ROOT'

    relative_root_path = "../" * depth if depth > 0 else "./"

    # Locate CSS path
    # showcase/css/style.css is at root/showcase/css/style.css
    relative_css_path = f"{relative_root_path}showcase/css/style.css"

    # Read README content
    readme_content = ""
    readme_path = os.path.join(directory, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
            if HAS_MARKDOWN:
                readme_content = markdown.markdown(md_text, extensions=['fenced_code', 'tables'])
            else:
                readme_content = f"<pre class='whitespace-pre-wrap'>{md_text}</pre>"

    # Identify prompt files (e.g. JSON or MD in prompts dir)
    prompt_files = []
    if "prompts" in directory or "prompt" in directory:
        prompt_files = [f for f in files if f.endswith('.json') or f.endswith('.md')]

    # Render Template
    template = Template(HTML_TEMPLATE)
    html_output = template.render(
        current_dir_name=current_dir_name,
        current_path=directory,
        relative_css_path=relative_css_path,
        relative_root_path=relative_root_path,
        dirs=dirs,
        files=files,
        readme_content=readme_content,
        prompt_files=prompt_files,
        generation_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # Write index.html
    html_path = os.path.join(directory, "index.html")
    print(f"Generating: {html_path}")
    with open(html_path, "w", encoding='utf-8') as f:
        f.write(html_output)

    # Write manifest
    manifest_content = generate_manifest(directory, files, depth)
    with open(os.path.join(directory, "directory_manifest.jsonld"), "w", encoding='utf-8') as f:
        f.write(manifest_content)

def main():
    print("Starting Adam v23.5 Showcase Swarm...")

    # Process TARGET_ROOTS
    for target_root in TARGET_ROOTS:
        if not os.path.exists(target_root):
            print(f"Skipping {target_root}, does not exist.")
            continue

        for root, dirs, files in os.walk(target_root):
            # Skip hidden/ignored directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

            # Calculate depth
            rel_path = os.path.relpath(root, ROOT_DIR)
            if rel_path == ".":
                depth = 0
            else:
                depth = len(rel_path.split(os.sep))

            generate_showcase_for_dir(root, depth)

if __name__ == "__main__":
    main()
