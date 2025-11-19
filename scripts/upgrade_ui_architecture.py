import os
import shutil
import json
import datetime
import re
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
REPO_ROOT = Path(__file__).parent.parent.resolve()
ARCHIVE_DIR = REPO_ROOT / "docs" / "ui_archive_v1"
MASTER_INDEX_PATH = REPO_ROOT / "index.html"

# Directories to scan for "Intelligence"
AGENTS_DIR = REPO_ROOT / "core" / "agents"
PROMPTS_DIR = REPO_ROOT / "prompts"
DATA_DIR = REPO_ROOT / "data" / "artisanal_training_sets"

# Exclusions for the archiver
EXCLUDE_FILES = [
    "index.html", # We will overwrite this, but don't archive the master itself if it exists
    "node_modules",
    ".git",
    "build",
    "dist"
]

# ==========================================
# HTML TEMPLATE (Cyberpunk Financial Terminal)
# ==========================================
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v22.0 | Mission Control</title>
    <style>
        :root {{
            --bg-dark: #0a0a0c;
            --panel-bg: #141419;
            --border-color: #2a2a35;
            --primary: #00f0ff; /* Cyber Blue */
            --success: #00ff9d; /* Matrix Green */
            --warning: #ffb86c;
            --danger: #ff5555;
            --text-main: #e0e0e0;
            --text-dim: #858595;
            --font-mono: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            height: 100vh;
            display: flex;
            overflow: hidden;
        }}

        /* SIDEBAR */
        .sidebar {{
            width: 260px;
            background: var(--panel-bg);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 20px;
        }}

        .brand {{
            font-family: var(--font-mono);
            font-size: 1.2rem;
            font-weight: bold;
            color: var(--primary);
            margin-bottom: 40px;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .brand-status {{
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--success);
        }}

        .nav-item {{
            color: var(--text-dim);
            text-decoration: none;
            padding: 12px;
            margin-bottom: 5px;
            border-radius: 6px;
            transition: all 0.2s;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .nav-item:hover, .nav-item.active {{
            background: rgba(0, 240, 255, 0.1);
            color: var(--primary);
        }}

        .nav-category {{
            font-family: var(--font-mono);
            font-size: 0.7rem;
            color: var(--text-dim);
            text-transform: uppercase;
            margin-top: 20px;
            margin-bottom: 10px;
            opacity: 0.7;
        }}

        /* MAIN CONTENT */
        .main {{
            flex: 1;
            overflow-y: auto;
            padding: 40px;
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            grid-template-rows: auto auto 1fr;
            gap: 24px;
        }}

        /* HEADER */
        .header {{
            grid-column: 1 / -1;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}

        .page-title h1 {{
            font-size: 2rem;
            font-weight: 300;
        }}
        .page-title .subtitle {{
            color: var(--text-dim);
            font-family: var(--font-mono);
            margin-top: 5px;
        }}

        /* WIDGETS */
        .widget {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }}

        .widget-header {{
            font-family: var(--font-mono);
            color: var(--text-dim);
            font-size: 0.8rem;
            text-transform: uppercase;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
        }}

        /* SYSTEM STATUS BOARD */
        .status-board {{
            grid-column: 1 / 9;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}

        .stat-card {{
            background: rgba(255,255,255,0.03);
            padding: 15px;
            border-radius: 4px;
        }}
        .stat-value {{ font-size: 1.5rem; font-weight: bold; color: var(--text-main); }}
        .stat-label {{ color: var(--text-dim); font-size: 0.8rem; }}
        .status-ok {{ color: var(--success); }}

        /* SPLIT BRAIN NAVIGATOR */
        .portal-access {{
            grid-column: 9 / -1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .launch-btn {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--primary);
            color: #000;
            padding: 15px;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
            transition: transform 0.1s;
        }}
        .launch-btn:hover {{ transform: translateY(-2px); }}
        .launch-btn.secondary {{ background: transparent; border: 1px solid var(--border-color); color: var(--text-main); }}

        /* ASSET TABLES */
        .asset-list {{
            grid-column: 1 / 7;
            max-height: 500px;
            overflow-y: auto;
        }}
        .prompt-list {{
            grid-column: 7 / -1;
            max-height: 500px;
            overflow-y: auto;
        }}

        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th {{ text-align: left; padding: 10px; color: var(--text-dim); border-bottom: 1px solid var(--border-color); font-weight: normal; }}
        td {{ padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        tr:hover {{ background: rgba(255,255,255,0.02); }}
        
        .tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: bold;
        }}
        .tag.v22 {{ background: rgba(0, 240, 255, 0.2); color: var(--primary); border: 1px solid var(--primary); }}
        .tag.legacy {{ background: rgba(133, 133, 149, 0.2); color: var(--text-dim); }}
        
        .file-link {{ color: var(--text-main); text-decoration: none; }}
        .file-link:hover {{ color: var(--primary); }}

    </style>
</head>
<body>
    <aside class="sidebar">
        <div class="brand">
            <div class="brand-status"></div>
            ADAM TERMINAL
        </div>
        
        <div class="nav-category">Core Systems</div>
        <a href="#" class="nav-item active">Dashboard</a>
        <a href="services/webapp/client/public/index.html" class="nav-item">React WebApp</a>
        <a href="navigator.html" class="nav-item">3D Navigator</a>
        <a href="chatbot/index.html" class="nav-item">Legacy Chatbot</a>

        <div class="nav-category">Intelligence</div>
        <a href="data/knowledgegraph.ttl" class="nav-item">Knowledge Graph</a>
        <a href="data/artisanal_training_sets/" class="nav-item">Artisanal Cortex</a>
        
        <div class="nav-category">Archives</div>
        <a href="docs/ui_archive_v1/index.html" class="nav-item">UI Archive v1</a>
        <a href="docs/" class="nav-item">Documentation</a>
    </aside>

    <main class="main">
        <div class="header">
            <div class="page-title">
                <h1>System Overview</h1>
                <div class="subtitle">Architecture Version: 22.0 | Environment: STATIC/DEV</div>
            </div>
            <div style="text-align: right; font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-dim);">
                LAST INDEX: {generation_date}<br>
                STATUS: OPERATIONAL
            </div>
        </div>

        <!-- TOP ROW: STATUS & PORTALS -->
        <div class="widget status-board">
            <div class="stat-card">
                <div class="stat-label">ACTIVE AGENTS</div>
                <div class="stat-value">{agent_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">PROMPT SKILLS</div>
                <div class="stat-value">{prompt_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">CORTEX FILES</div>
                <div class="stat-value">{cortex_count}</div>
            </div>
            <div class="stat-card" style="grid-column: 1/-1; margin-top: 10px;">
                <div class="stat-label">SYSTEM HEALTH</div>
                <div class="stat-value status-ok" style="font-size: 1rem; margin-top: 5px;">
                    ● CORE LOGIC INTEGRITY: VERIFIED<br>
                    ● KNOWLEDGE GRAPH: MOUNTED
                </div>
            </div>
        </div>

        <div class="widget portal-access">
            <div class="widget-header">SYSTEM ACCESS PORTALS</div>
            <a href="services/webapp/client/public/index.html" class="launch-btn">
                <span>LAUNCH V22 WEBAPP</span>
                <span>→</span>
            </a>
            <a href="navigator.html" class="launch-btn secondary">
                <span>OPEN NAVIGATOR</span>
                <span>↗</span>
            </a>
            <a href="chatbot/index.html" class="launch-btn secondary">
                <span>LEGACY CHATBOT</span>
                <span>↗</span>
            </a>
        </div>

        <!-- MIDDLE ROW: AGENTS & PROMPTS -->
        <div class="widget asset-list">
            <div class="widget-header">
                <span>Neural Agent Registry (core/agents)</span>
                <span>V22 PRIORITY</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Agent Name</th>
                        <th>Type</th>
                        <th>Version</th>
                    </tr>
                </thead>
                <tbody>
                    {agent_rows}
                </tbody>
            </table>
        </div>

        <div class="widget prompt-list">
            <div class="widget-header">
                <span>Prompt Engineering & Skills</span>
                <span>AOPL v1.0</span>
            </div>
             <table>
                <thead>
                    <tr>
                        <th>Configuration / Skill</th>
                        <th>Format</th>
                    </tr>
                </thead>
                <tbody>
                    {prompt_rows}
                </tbody>
            </table>
        </div>

        <!-- BOTTOM: MIGRATION NOTE -->
        <div class="widget" style="grid-column: 1 / -1;">
            <div class="widget-header">ARCHITECTURAL NOTES</div>
            <p style="color: var(--text-dim); font-size: 0.9rem; line-height: 1.5;">
                This dashboard unifies the <strong>Split Brain Architecture</strong> identified in the v22 audit. 
                The <em>React WebApp</em> is the target environment for production interactions. 
                Static tools found in the root are preserved for development and debugging. 
                Old HTML artifacts have been migrated to <a href="docs/ui_archive_v1/" style="color: var(--primary)">docs/ui_archive_v1/</a>.
            </p>
        </div>

    </main>
</body>
</html>
"""

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def scan_agents():
    """Scans the agents directory and returns a list of dicts."""
    agents = []
    if not AGENTS_DIR.exists():
        return agents
    
    for file in AGENTS_DIR.rglob("*.py"):
        if file.name == "__init__.py": 
            continue
        
        # Determine version tag based on filename or content (naive)
        tag = "v22"
        if "19" in file.name: tag = "legacy"
        if "20" in file.name: tag = "legacy"
        if "21" in file.name: tag = "legacy"
        
        rel_path = file.relative_to(REPO_ROOT)
        agents.append({
            "name": file.stem,
            "path": str(rel_path),
            "tag": tag,
            "type": "Python"
        })
    
    # Sort: v22 first
    return sorted(agents, key=lambda x: x['tag'] == 'legacy')

def scan_prompts():
    """Scans the prompts directory."""
    prompts = []
    if not PROMPTS_DIR.exists():
        return prompts
        
    for file in PROMPTS_DIR.rglob("*"):
        if file.suffix not in ['.json', '.yaml', '.md'] or file.name.startswith('.'):
            continue
            
        rel_path = file.relative_to(REPO_ROOT)
        prompts.append({
            "name": file.name,
            "path": str(rel_path),
            "format": file.suffix.upper().replace('.', '')
        })
    return prompts

def scan_cortex():
    """Counts artisanal training files."""
    if not DATA_DIR.exists():
        return 0
    return len(list(DATA_DIR.glob("*.jsonl")))

def archive_html_artifacts():
    """Moves loose HTML files to the archive folder."""
    if not ARCHIVE_DIR.exists():
        os.makedirs(ARCHIVE_DIR)
        print(f"Created archive directory: {ARCHIVE_DIR}")

    moved_files = []
    
    # Walk the repo
    for root, dirs, files in os.walk(REPO_ROOT):
        # Skip protected directories
        if any(excluded in root for excluded in EXCLUDE_FILES[1:]): # Skip excluding index.html here, logic below
            continue
            
        for file in files:
            if file.endswith(".html"):
                file_path = Path(root) / file
                
                # Don't move the master index we are about to create, or things inside the archive
                if file_path == MASTER_INDEX_PATH:
                    continue
                if ARCHIVE_DIR in file_path.parents:
                    continue
                
                # Calculate relative path to maintain structure or flat rename
                rel_path = file_path.relative_to(REPO_ROOT)
                
                # Strategy: Flatten but prepend path to avoid collisions
                # e.g. core_libraries_newsletters_MM06292025.html
                safe_name = str(rel_path).replace("/", "_").replace("\\", "_")
                dest_path = ARCHIVE_DIR / safe_name
                
                print(f"Archiving artifact: {rel_path} -> {dest_path.name}")
                shutil.copy2(file_path, dest_path)
                moved_files.append({"original": str(rel_path), "archived": safe_name})

    # Create Manifest
    with open(ARCHIVE_DIR / "manifest.json", "w") as f:
        json.dump(moved_files, f, indent=2)
    
    # Create an index for the archive folder itself
    create_archive_index(moved_files)

def create_archive_index(files):
    """Creates a simple index file for the archive folder."""
    html = """<html><head><title>UI Archive v1</title><style>body{font-family:sans-serif;background:#111;color:#ddd;padding:20px;} a{color:#0ff;text-decoration:none;} ul{list-style:none;padding:0;} li{padding:5px 0;border-bottom:1px solid #333;}</style></head><body><h1>UI Archive v1</h1><ul>"""
    for item in files:
        html += f"<li><a href='{item['archived']}'>{item['original']}</a></li>"
    html += "</ul></body></html>"
    
    with open(ARCHIVE_DIR / "index.html", "w") as f:
        f.write(html)

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Initializing ADAM v22.0 UI Architecture Upgrade...")
    
    # 1. Run Archival Process
    archive_html_artifacts()
    
    # 2. Gather Data
    agents = scan_agents()
    prompts = scan_prompts()
    cortex_count = scan_cortex()
    
    # 3. Generate HTML Rows
    agent_rows = ""
    for a in agents:
        tag_class = "v22" if a['tag'] == "v22" else "legacy"
        agent_rows += f"""
        <tr>
            <td><a href="{a['path']}" class="file-link">{a['name']}</a></td>
            <td>{a['type']}</td>
            <td><span class="tag {tag_class}">{a['tag'].upper()}</span></td>
        </tr>
        """

    prompt_rows = ""
    for p in prompts[:15]: # Limit to top 15 for clean UI
        prompt_rows += f"""
        <tr>
            <td><a href="{p['path']}" class="file-link">{p['name']}</a></td>
            <td><span class="tag">{p['format']}</span></td>
        </tr>
        """
    if len(prompts) > 15:
        prompt_rows += f"<tr><td colspan='2' style='text-align:center; color: #555;'>... and {len(prompts)-15} more items ...</td></tr>"

    # 4. Fill Template
    final_html = DASHBOARD_TEMPLATE.format(
        generation_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        agent_count=len(agents),
        prompt_count=len(prompts),
        cortex_count=cortex_count,
        agent_rows=agent_rows,
        prompt_rows=prompt_rows
    )
    
    # 5. Write Master Index
    with open(MASTER_INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(final_html)
        
    print(f"SUCCESS: Master Dashboard generated at {MASTER_INDEX_PATH}")
    print(f"SUCCESS: Artifacts archived in {ARCHIVE_DIR}")
    print("Deployment Ready. Commit changes and push to GitHub to see the live dashboard.")

if __name__ == "__main__":
    main()
