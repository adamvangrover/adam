import os
import html

# --- Configuration ---
ROOT_DIR = "."
OUTPUT_FILE = "showcase/comprehensive_index.html"
SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', 'venv', 'env', '.env',
    '.idea', '.vscode', 'dist', 'build', 'coverage', '.pytest_cache',
    'site-packages', 'webapp', 'services/webapp/client',
    'verification_artifacts', 'verification_images', 'verification_screenshots'
}
SKIP_FILES = {
    'index.html',  # We'll handle index.html specially or skip duplicates
    '404.html',
    'comprehensive_index.html' # Don't list itself
}

TEMPLATE_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v26.0 :: SYSTEM MAP</title>
    <link rel="stylesheet" href="css/style.css">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #00f3ff;
            --accent-color: #0aff60;
            --bg-color: #050b14;
            --panel-bg: rgba(5, 11, 20, 0.8);
            --text-primary: #e0e0e0;
            --text-muted: #888;
            --panel-border: #333;
        }
        body { margin: 0; background: var(--bg-color); color: var(--text-primary); font-family: 'Inter', sans-serif; overflow-x: hidden; }
        .mono { font-family: 'JetBrains Mono', monospace; }

        /* Header */
        .cyber-header {
            height: 60px; display: flex; align-items: center; justify-content: space-between;
            padding: 0 20px; border-bottom: 1px solid var(--primary-color);
            background: rgba(5, 11, 20, 0.95); position: sticky; top: 0; z-index: 100;
        }

        .scan-line { position: fixed; top: 0; left: 0; width: 100%; height: 2px; background: rgba(0, 243, 255, 0.1); animation: scan 3s linear infinite; pointer-events: none; z-index: 999; }
        @keyframes scan { 0% { top: 0; } 100% { top: 100%; } }

        .cyber-btn {
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 6px 12px;
            border: 1px solid #444; color: var(--text-primary); background: rgba(0,0,0,0.3);
            border-radius: 2px; text-transform: uppercase; text-decoration: none; display: inline-block;
        }
        .cyber-btn:hover { border-color: var(--primary-color); color: var(--primary-color); }

        /* Grid */
        .grid-container {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; padding: 40px;
        }

        .dir-card {
            background: rgba(255, 255, 255, 0.03); border: 1px solid var(--panel-border);
            border-radius: 4px; padding: 20px; transition: all 0.2s;
        }
        .dir-card:hover { border-color: var(--primary-color); background: rgba(255, 255, 255, 0.05); }

        .dir-title {
            font-family: 'JetBrains Mono', monospace; color: var(--accent-color);
            border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px; font-size: 0.9rem;
            display: flex; justify-content: space-between; align-items: center;
        }

        .file-list { list-style: none; padding: 0; margin: 0; }
        .file-item { margin-bottom: 8px; font-size: 0.85rem; display: flex; align-items: center; }
        .file-item a { color: var(--text-muted); text-decoration: none; transition: color 0.2s; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .file-item a:hover { color: var(--primary-color); }
        .file-icon { margin-right: 8px; opacity: 0.7; font-size: 0.8rem; }

        /* Search */
        .search-bar {
            padding: 20px 40px; background: rgba(0,0,0,0.3); border-bottom: 1px solid #333;
            display: flex; justify-content: center;
        }
        #search-input {
            width: 100%; max-width: 600px; padding: 12px 20px;
            background: rgba(0,0,0,0.5); border: 1px solid #444; color: white;
            font-family: 'JetBrains Mono', monospace; font-size: 1rem; border-radius: 4px;
        }
        #search-input:focus { outline: none; border-color: var(--primary-color); }
    </style>
</head>
<body>
    <div class="scan-line"></div>

    <header class="cyber-header">
        <div style="display: flex; align-items: center; gap: 20px;">
            <h1 class="mono" style="margin: 0; font-size: 1.5rem; color: var(--primary-color); letter-spacing: 2px;">ADAM v26.0</h1>
            <div class="mono" style="font-size: 0.8rem; color: #666; border-left: 1px solid #333; padding-left: 10px;">SYSTEM MAP</div>
        </div>
        <nav style="display: flex; gap: 10px;">
            <a href="../index.html" class="cyber-btn">&uarr; ROOT</a>
            <a href="index.html" class="cyber-btn">LEGACY SHOWCASE</a>
            <a href="neural_dashboard.html" class="cyber-btn" style="border-color: var(--accent-color); color: var(--accent-color);">DASHBOARD</a>
        </nav>
    </header>

    <div class="search-bar">
        <input type="text" id="search-input" placeholder="Search system artifacts (e.g., 'report', 'dashboard', 'agent')..." onkeyup="filterFiles()">
    </div>

    <div class="grid-container" id="grid">
"""

TEMPLATE_FOOTER = """
    </div>

    <footer style="border-top: 1px solid var(--panel-border); padding: 40px; text-align: center; color: var(--text-muted); font-size: 0.8rem;">
        ADAM v26.0 | SYSTEM INDEX GENERATED AUTOMATICALLY
    </footer>

    <script>
        function filterFiles() {
            const input = document.getElementById('search-input');
            const filter = input.value.toUpperCase();
            const grid = document.getElementById('grid');
            const cards = grid.getElementsByClassName('dir-card');

            for (let i = 0; i < cards.length; i++) {
                let card = cards[i];
                let items = card.getElementsByClassName('file-item');
                let cardVisible = false;

                for (let j = 0; j < items.length; j++) {
                    let item = items[j];
                    let text = item.textContent || item.innerText;
                    if (text.toUpperCase().indexOf(filter) > -1) {
                        item.style.display = "";
                        cardVisible = true;
                    } else {
                        item.style.display = "none";
                    }
                }

                if (cardVisible) {
                    card.style.display = "";
                } else {
                    card.style.display = "none";
                }
            }
        }
    </script>
</body>
</html>
"""

def generate_index():
    file_map = {} # Dir -> List of files

    print("Scanning for HTML files...")
    for root, dirs, files in os.walk(ROOT_DIR):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

        for file in files:
            if file.endswith(".html") and file not in SKIP_FILES:
                rel_path = os.path.relpath(os.path.join(root, file), ROOT_DIR)

                # Check exclusion patterns in path
                if any(skip in rel_path.split(os.sep) for skip in SKIP_DIRS):
                    continue

                directory = os.path.dirname(rel_path)
                if directory == "": directory = "ROOT"

                if directory not in file_map:
                    file_map[directory] = []

                # Adjust link relative to showcase/
                # The output file is in showcase/, so we need to go up one level for files outside showcase/
                # except for files IN showcase/

                if rel_path.startswith("showcase/"):
                    link_href = rel_path.replace("showcase/", "")
                else:
                    link_href = "../" + rel_path

                file_map[directory].append({
                    'name': file,
                    'path': rel_path,
                    'href': link_href
                })

    # Sort directories
    sorted_dirs = sorted(file_map.keys())

    # Generate HTML content
    html_content = TEMPLATE_HEADER

    for directory in sorted_dirs:
        files = sorted(file_map[directory], key=lambda x: x['name'])

        # Icon logic
        dir_icon = "📂"
        if "showcase" in directory: dir_icon = "🚀"
        if "core" in directory: dir_icon = "🧠"
        if "services" in directory: dir_icon = "⚙️"
        if "test" in directory: dir_icon = "🧪"

        html_content += f"""
        <div class="dir-card">
            <div class="dir-title">
                <span>{dir_icon} {directory}</span>
                <span style="font-size:0.7rem; opacity:0.5;">{len(files)} files</span>
            </div>
            <ul class="file-list">
        """

        for file in files:
            icon = "📄"
            if "dashboard" in file['name'].lower(): icon = "🖥️"
            if "report" in file['name'].lower(): icon = "📊"
            if "agent" in file['name'].lower(): icon = "🤖"

            html_content += f"""
                <li class="file-item">
                    <span class="file-icon">{icon}</span>
                    <a href="{file['href']}">{file['name']}</a>
                </li>
            """

        html_content += """
            </ul>
        </div>
        """

    html_content += TEMPLATE_FOOTER

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Index generated at: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_index()
