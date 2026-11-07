#!/usr/bin/env python3
"""
generate_neural_browser_index.py

Generates a standalone, self-contained HTML page (`adam_neural_browser_index.html`)
presenting the adamvangrover/adam repository as an interactive neural network visualizer
framed within an authentic 1990s Windows web browser interface (Netscape / Mosaic / IE style).

Features:
- Windows 95 / 98 chrome: beveled title bar, menus, retro 3D buttons, address bar, animated throbber, status bar
- Interactive neural network graph (Canvas API): animated action potentials, force-directed layout, synapse pulses, node inspector
- Repository perusal: complete file catalog of 7,200+ repo files, categorization, live filter/search, file tree viewer
- Built-in retro document viewer with embedded institutional files (README.md, AGENTS.md, CONSTITUTION.md, etc.)
- Web Audio API retro 56k dial-up synthesizer and classic UI sound effects
- CRT scanline effect toggle
"""

import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Neural network layers and modules categorization
CLUSTER_MAP = {
    "system_1_swarm": {
        "title": "System 1 Swarm (High-Frequency Execution)",
        "color": "#00f3ff",
        "desc": "Reactive order flow, sentiment reflex, algorithmic execution and fast-path heuristics.",
        "dirs": ["adam_swarm", "skills", "mcp", "terminals", "market-mayhem"],
    },
    "system_2_dag": {
        "title": "System 2 DAG (Deliberative Symbolic Reasoning)",
        "color": "#9d4edd",
        "desc": "Deep credit underwriting, counterparty risk trees, scenario simulations, and balance-sheet stress.",
        "dirs": ["core", "core_engine", "core_kernel", "adam_finance", "financial_digital_twin"],
    },
    "pdil_governance": {
        "title": "PDIL Governance & Fiduciary Arbiters",
        "color": "#06d6a0",
        "desc": "Probabilistic-to-Deterministic Integration Layer, confidence gates, W3C PROV-O audit trails.",
        "dirs": ["adam_governance", "policies", "rules", "human_gates", "reasoning_frameworks"],
    },
    "zero_trust_security": {
        "title": "Zero Trust Cryptographic Core",
        "color": "#ef476f",
        "desc": "ECDSA mTLS, 15-minute STS credentials, air-gapped quarantine logging, SHA-256 verification.",
        "dirs": ["src", "kubernetes", "backend", "server", "servers", "services"],
    },
    "showcase_intelligence": {
        "title": "Institutional Showcase & Daily Rituals",
        "color": "#ffd166",
        "desc": "Daily market mayhem briefs, hedge fund & private equity models, war rooms, HTML dashboards.",
        "dirs": ["showcase", "newsletters", "briefings", "daily_ritual", "exports", "reports"],
    },
    "knowledge_docs": {
        "title": "Knowledge Graph & Documentation Matrix",
        "color": "#118ab2",
        "desc": "Diataxis framework, agent catalogs, mathematical specifications, research monographs.",
        "dirs": ["docs", "knowledge_base", "research", "technical_specification", "specs", "prompt_library", "prompts"],
    },
    "evals_testing": {
        "title": "Adversarial Evals & Verification Harness",
        "color": "#f77f00",
        "desc": "Deterministic property testing, clean room verification, red team zombie stress test suites.",
        "dirs": ["evals", "tests", "verification", "verification_images", "verification_screenshots", "benchmarks"],
    },
    "runtime_infra": {
        "title": "Runtime Infrastructure & Orchestration",
        "color": "#8338ec",
        "desc": "Docker orchestration, Python packaging, scripts, configuration matrices, and build pipelines.",
        "dirs": ["docker", "config", "ops", "scripts", "rust_ext", "packages", "applications", "apps", "webapp", "public"],
    },
    "experimental_lab": {
        "title": "Tinker Lab & Experimental Prototypes",
        "color": "#05ffa1",
        "desc": "Forward-looking R&D, liquid neural networks, memory bus prototypes, UFOS sandbox.",
        "dirs": ["experimental", "tinker_lab", "prototype", "ufos", "ufos_showcase", "data", "runs", "logs", "archive", "artifacts"],
    },
    "root_manifest": {
        "title": "Repository Root & Meta Directives",
        "color": "#ffffff",
        "desc": "Primary repo entrypoints, legal constitutions, package manifests, and institutional indexes.",
        "dirs": ["."],
    }
}


def build_repo_catalog(repo_root):
    catalog = []
    category_counts = {k: 0 for k in CLUSTER_MAP}

    # Priority embedded files
    embedded_paths = [
        "README.md",
        "AGENTS.md",
        "CONSTITUTION.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "LLM_README.md",
        "pyproject.toml",
        "api_schema.yaml",
        "ADAM_NEUTRALITY_V2.4.MD",
        "Architectural_Review_Refined.md",
        "docs/setup_guide.md",
        "docs/architecture/PDIL_MAPPING.md",
        "core/engine/unified_knowledge_graph.py",
        "src/zero_trust/sts.py",
        "skills/hunt_and_kill.py"
    ]
    embedded_data = {}

    for path in embedded_paths:
        full = os.path.join(repo_root, path)
        if os.path.exists(full) and os.path.isfile(full):
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    # Truncate if massive to keep HTML manageable
                    if len(content) > 30000:
                        content = content[:30000] + "\n\n... [TRUNCATED FOR STANDALONE VIEWER DISPLAY] ..."
                    embedded_data[path] = content
            except Exception:
                pass

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "dist", "build", "__pycache__", ".venv", "adam_project.egg-info")]
        for f in files:
            if f.startswith("."):
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, repo_root)
            top_dir = rel_path.split(os.sep)[0] if os.sep in rel_path else "."

            # Determine cluster
            assigned_cluster = "root_manifest"
            for cluster_id, meta in CLUSTER_MAP.items():
                if top_dir in meta["dirs"]:
                    assigned_cluster = cluster_id
                    break

            category_counts[assigned_cluster] = category_counts.get(assigned_cluster, 0) + 1

            size = os.path.getsize(full_path)
            ext = os.path.splitext(f)[1].lower()

            catalog.append({
                "p": rel_path,
                "n": f,
                "c": assigned_cluster,
                "s": size,
                "e": ext
            })

    # Sort catalog by path
    catalog.sort(key=lambda x: x["p"])
    return catalog, category_counts, embedded_data


def generate_html_content(catalog, category_counts, embedded_data):
    clusters_json = json.dumps(CLUSTER_MAP)
    catalog_json = json.dumps(catalog)
    embedded_json = json.dumps(embedded_data)
    total_files = len(catalog)

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Netscape Navigator - [http://adamvangrover.org/neural/net.htm]</title>
  <style>
    /* ═══════════════════════════════════════════════════════════════
       AUTHENTIC 1990s WINDOWS / NETSCAPE NAVIGATOR DESIGN SYSTEM
       ═══════════════════════════════════════════════════════════════ */
    :root {{
      --win-gray: #c0c0c0;
      --win-dark-gray: #808080;
      --win-light: #ffffff;
      --win-shadow: #000000;
      --win-title-blue: #000080;
      --win-title-blue-2: #1084d0;
      --win-title-text: #ffffff;
      --retro-canvas-bg: #030813;
      --terminal-green: #00ff66;
      --cyan-bright: #00f3ff;
      --retro-font: 'MS Sans Serif', Geneva, Tahoma, sans-serif;
      --retro-mono: 'Fixedsys', 'Courier New', Courier, monospace;
    }}

    * {{
      box-sizing: border-box;
      user-select: none;
    }}

    body, html {{
      margin: 0;
      padding: 0;
      height: 100%;
      width: 100%;
      background-color: #008080; /* Windows 95 classic teal desktop */
      font-family: var(--retro-font);
      font-size: 11px;
      overflow: hidden;
      color: #000000;
    }}

    /* Bevel Utility Classes */
    .win-outset {{
      border-top: 2px solid var(--win-light);
      border-left: 2px solid var(--win-light);
      border-right: 2px solid var(--win-dark-gray);
      border-bottom: 2px solid var(--win-dark-gray);
      box-shadow: inset 1px 1px 0px #dfdfdf, 1px 1px 0px #000000;
    }}

    .win-inset {{
      border-top: 2px solid var(--win-dark-gray);
      border-left: 2px solid var(--win-dark-gray);
      border-right: 2px solid var(--win-light);
      border-bottom: 2px solid var(--win-light);
      box-shadow: inset 1px 1px 0px #000000, 1px 1px 0px #dfdfdf;
    }}

    .win-button {{
      background: var(--win-gray);
      border-top: 2px solid var(--win-light);
      border-left: 2px solid var(--win-light);
      border-right: 2px solid var(--win-dark-gray);
      border-bottom: 2px solid var(--win-dark-gray);
      padding: 2px 8px;
      font-size: 11px;
      font-family: var(--retro-font);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      outline: none;
      box-shadow: 1px 1px 0px #000000;
    }}

    .win-button:active, .win-button.active {{
      border-top: 2px solid var(--win-dark-gray);
      border-left: 2px solid var(--win-dark-gray);
      border-right: 2px solid var(--win-light);
      border-bottom: 2px solid var(--win-light);
      box-shadow: inset 1px 1px 0px #000000;
      transform: translate(1px, 1px);
    }}

    /* Main Window Frame */
    #desktop {{
      width: 100vw;
      height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 4px;
      background: #008080;
    }}

    #browser-window {{
      flex: 1;
      display: flex;
      flex-direction: column;
      background-color: var(--win-gray);
      overflow: hidden;
    }}

    /* Windows 95 Window Titlebar */
    .titlebar {{
      height: 20px;
      background: linear-gradient(90deg, var(--win-title-blue), var(--win-title-blue-2));
      color: var(--win-title-text);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 3px;
      font-weight: bold;
      font-size: 11px;
      letter-spacing: 0.5px;
    }}

    .titlebar-title {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .titlebar-icon {{
      width: 14px;
      height: 14px;
      border: 1px solid #fff;
      background: radial-gradient(circle at 30% 30%, #ff0, #f00);
      border-radius: 2px;
    }}

    .titlebar-controls {{
      display: flex;
      gap: 2px;
    }}

    .titlebar-btn {{
      width: 16px;
      height: 14px;
      background: var(--win-gray);
      border-top: 1px solid var(--win-light);
      border-left: 1px solid var(--win-light);
      border-right: 1px solid var(--win-dark-gray);
      border-bottom: 1px solid var(--win-dark-gray);
      box-shadow: 1px 1px 0px #000;
      font-size: 9px;
      font-weight: bold;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      color: #000;
      line-height: 1;
    }}

    .titlebar-btn:active {{
      border-top: 1px solid var(--win-dark-gray);
      border-left: 1px solid var(--win-dark-gray);
      border-right: 1px solid var(--win-light);
      border-bottom: 1px solid var(--win-light);
    }}

    /* Menu Bar */
    .menubar {{
      height: 22px;
      background: var(--win-gray);
      display: flex;
      align-items: center;
      padding: 0 4px;
      border-bottom: 1px solid var(--win-dark-gray);
      gap: 12px;
      font-size: 11px;
    }}

    .menu-item {{
      padding: 2px 6px;
      cursor: pointer;
    }}

    .menu-item:hover {{
      background: var(--win-title-blue);
      color: #fff;
    }}

    .menu-item u {{
      text-decoration: underline;
    }}

    /* Retro Toolbar */
    .toolbar {{
      background: var(--win-gray);
      padding: 3px 6px;
      display: flex;
      align-items: center;
      gap: 6px;
      border-bottom: 1px solid var(--win-dark-gray);
      box-shadow: inset 0 1px 0 var(--win-light);
    }}

    .toolbar-group {{
      display: flex;
      align-items: center;
      gap: 3px;
    }}

    .tool-btn {{
      height: 38px;
      min-width: 44px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2px 4px;
      font-size: 10px;
      gap: 2px;
    }}

    .tool-icon {{
      font-size: 15px;
      line-height: 1;
    }}

    .divider-v {{
      width: 2px;
      height: 34px;
      border-left: 1px solid var(--win-dark-gray);
      border-right: 1px solid var(--win-light);
      margin: 0 3px;
    }}

    /* Animated 1990s Netscape Throbber */
    .throbber-box {{
      margin-left: auto;
      width: 40px;
      height: 36px;
      background: #000000;
      position: relative;
      overflow: hidden;
      cursor: pointer;
    }}

    .throbber-n {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-family: var(--retro-mono);
      font-size: 20px;
      font-weight: bold;
      color: #00f3ff;
      text-shadow: 0 0 6px #00f3ff;
      animation: throbber-pulse 2s infinite ease-in-out;
    }}

    .throbber-stars {{
      position: absolute;
      width: 100%;
      height: 100%;
      background: radial-gradient(circle, #fff 1px, transparent 1px);
      background-size: 8px 8px;
      opacity: 0.3;
      animation: throbber-rotate 8s linear infinite;
    }}

    @keyframes throbber-pulse {{
      0%, 100% {{ color: #00f3ff; text-shadow: 0 0 8px #00f3ff; }}
      50% {{ color: #ff0077; text-shadow: 0 0 12px #ff0077; }}
    }}

    @keyframes throbber-rotate {{
      0% {{ transform: rotate(0deg); }}
      100% {{ transform: rotate(360deg); }}
    }}

    /* Location Bar / URL */
    .location-bar {{
      padding: 3px 6px;
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--win-gray);
      border-bottom: 2px solid var(--win-dark-gray);
    }}

    .location-label {{
      font-weight: bold;
      font-size: 11px;
    }}

    .location-input-wrap {{
      flex: 1;
      background: #fff;
      display: flex;
      align-items: center;
      padding: 1px 4px;
    }}

    .location-input {{
      width: 100%;
      border: none;
      outline: none;
      font-family: var(--retro-mono);
      font-size: 12px;
      background: transparent;
      user-select: text;
    }}

    /* Main Viewport Splitter */
    #viewport {{
      flex: 1;
      display: flex;
      overflow: hidden;
      background: #000;
      position: relative;
    }}

    /* Left Pane: Neural Network Visualizer */
    #neural-pane {{
      flex: 1.35;
      display: flex;
      flex-direction: column;
      position: relative;
      background: var(--retro-canvas-bg);
      border-right: 2px solid var(--win-dark-gray);
      overflow: hidden;
    }}

    #neural-canvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }}

    #neural-canvas:active {{
      cursor: grabbing;
    }}

    /* Overlay HUD in Neural Pane */
    .hud-overlay {{
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(10, 20, 35, 0.88);
      border: 1px solid #00f3ff;
      color: #00f3ff;
      padding: 8px 12px;
      font-family: var(--retro-mono);
      font-size: 10px;
      line-height: 1.4;
      pointer-events: none;
      box-shadow: 0 0 12px rgba(0, 243, 255, 0.25);
      z-index: 10;
    }}

    .hud-title {{
      font-size: 12px;
      font-weight: bold;
      color: #fff;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }}

    .hud-controls {{
      position: absolute;
      bottom: 10px;
      left: 8px;
      display: flex;
      gap: 6px;
      z-index: 10;
    }}

    /* Right Pane: Repository & File Perusal */
    #repo-pane {{
      flex: 1;
      display: flex;
      flex-direction: column;
      background: var(--win-gray);
      overflow: hidden;
      min-width: 380px;
    }}

    .pane-header {{
      background: linear-gradient(90deg, #404040, #606060);
      color: #fff;
      padding: 3px 8px;
      font-weight: bold;
      font-size: 11px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    /* Filter & Search Bar */
    .filter-bar {{
      padding: 5px;
      background: var(--win-gray);
      display: flex;
      gap: 6px;
      border-bottom: 1px solid var(--win-dark-gray);
    }}

    .filter-input {{
      flex: 1;
      font-family: var(--retro-font);
      font-size: 11px;
      padding: 2px 4px;
      background: #fff;
      border: 2px solid var(--win-dark-gray);
      user-select: text;
    }}

    .cluster-filter {{
      font-family: var(--retro-font);
      font-size: 11px;
      background: #fff;
      border: 2px solid var(--win-dark-gray);
    }}

    /* Split: File List & Document Inspector */
    #explorer-split {{
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    #file-table-container {{
      flex: 1;
      overflow-y: auto;
      background: #fff;
      margin: 4px;
    }}

    table.retro-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      font-family: var(--retro-font);
    }}

    table.retro-table th {{
      position: sticky;
      top: 0;
      background: var(--win-gray);
      border-top: 1px solid var(--win-light);
      border-left: 1px solid var(--win-light);
      border-right: 1px solid var(--win-dark-gray);
      border-bottom: 1px solid var(--win-dark-gray);
      padding: 2px 6px;
      text-align: left;
      font-weight: normal;
      cursor: pointer;
    }}

    table.retro-table td {{
      padding: 2px 6px;
      border-bottom: 1px solid #e0e0e0;
      white-space: nowrap;
      user-select: text;
    }}

    table.retro-table tr:hover {{
      background: #eef4ff;
    }}

    table.retro-table tr.selected {{
      background: var(--win-title-blue);
      color: #fff;
    }}

    /* Document Inspector Pane */
    #doc-viewer {{
      height: 240px;
      margin: 0 4px 4px 4px;
      display: flex;
      flex-direction: column;
      background: #ffffff;
      border-top: 2px solid var(--win-dark-gray);
    }}

    .doc-viewer-header {{
      background: var(--win-gray);
      padding: 3px 6px;
      font-weight: bold;
      font-size: 11px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--win-dark-gray);
    }}

    .doc-content {{
      flex: 1;
      overflow: auto;
      padding: 6px;
      font-family: var(--retro-mono);
      font-size: 11px;
      line-height: 1.45;
      color: #000000;
      background: #fdfdfd;
      white-space: pre-wrap;
      word-break: break-all;
      user-select: text;
    }}

    /* Bottom Status Bar */
    .statusbar {{
      height: 20px;
      background: var(--win-gray);
      border-top: 1px solid var(--win-dark-gray);
      display: flex;
      align-items: center;
      padding: 0 4px;
      gap: 4px;
      font-size: 11px;
    }}

    .status-panel {{
      padding: 1px 6px;
      height: 16px;
      display: flex;
      align-items: center;
      background: var(--win-gray);
    }}

    .status-msg {{
      flex: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .status-progress-box {{
      width: 100px;
      height: 14px;
      background: #ffffff;
      position: relative;
    }}

    .status-progress-fill {{
      height: 100%;
      width: 100%;
      background: repeating-linear-gradient(90deg, #000080, #000080 8px, #ffffff 8px, #ffffff 10px);
    }}

    .status-security {{
      width: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: bold;
      color: #008000;
    }}

    /* CRT Scanline Filter Overlay (Optional) */
    .crt-overlay {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%);
      background-size: 100% 4px;
      z-index: 999;
      opacity: 0.6;
      display: none;
    }}

    .crt-overlay.active {{
      display: block;
    }}

    /* Modal / Popups */
    .retro-modal {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 420px;
      background: var(--win-gray);
      z-index: 1000;
      display: none;
    }}

    .modal-body {{
      padding: 12px;
      font-size: 12px;
      line-height: 1.5;
    }}

    .modal-actions {{
      display: flex;
      justify-content: flex-end;
      padding: 8px 12px;
      gap: 6px;
    }}
  </style>
</head>
<body>

  <div id="desktop">
    <div id="browser-window" class="win-outset">

      <!-- 1. Window Titlebar -->
      <div class="titlebar">
        <div class="titlebar-title">
          <div class="titlebar-icon"></div>
          <span>Netscape Navigator - [adamvangrover/adam :: Neural Architecture Index 1996]</span>
        </div>
        <div class="titlebar-controls">
          <div class="titlebar-btn" title="Minimize">_</div>
          <div class="titlebar-btn" title="Maximize">□</div>
          <div class="titlebar-btn" title="Close" onclick="showModal('System Notice', 'ADAM Neuro-Symbolic Sovereign cannot be closed in simulation mode.')">✕</div>
        </div>
      </div>

      <!-- 2. Classic Menu Bar -->
      <div class="menubar">
        <div class="menu-item" onclick="playSound('click'); showModal('File Menu', 'File operations: Save Page As... Export Neural Graph... Exit')"><u>F</u>ile</div>
        <div class="menu-item" onclick="playSound('click'); triggerPulse();"><u>E</u>dit</div>
        <div class="menu-item" onclick="playSound('click'); toggleCRT();"><u>V</u>iew</div>
        <div class="menu-item" onclick="playSound('click'); resetGraph();"><u>G</u>o</div>
        <div class="menu-item" onclick="playSound('click'); showModal('Bookmarks', 'ADAM Financial OS :: Quick Hubs<br>• hub.html<br>• index.html<br>• showcase/neural_dashboard.html<br>• adam_convergence_live_neural_link.html')"><u>B</u>ookmarks</div>
        <div class="menu-item" onclick="playSound('click'); playModemSound();"><u>O</u>ptions</div>
        <div class="menu-item" onclick="playSound('click'); document.getElementById('search-filter').focus();"><u>D</u>irectory</div>
        <div class="menu-item" onclick="playSound('click'); showAboutDialog();"><u>H</u>elp</div>
      </div>

      <!-- 3. Navigation Toolbar -->
      <div class="toolbar">
        <div class="toolbar-group">
          <button class="win-button tool-btn" onclick="historyBack();" title="Go Back">
            <span class="tool-icon">⬅</span>
            <span>Back</span>
          </button>
          <button class="win-button tool-btn" onclick="historyForward();" title="Go Forward">
            <span class="tool-icon">➡</span>
            <span>Forward</span>
          </button>
          <button class="win-button tool-btn" onclick="resetGraph();" title="Home Node">
            <span class="tool-icon">🏠</span>
            <span>Home</span>
          </button>
          <button class="win-button tool-btn" onclick="stimulateNetwork();" title="Reload / Re-stimulate Synapses">
            <span class="tool-icon">🔄</span>
            <span>Reload</span>
          </button>
        </div>

        <div class="divider-v"></div>

        <div class="toolbar-group">
          <button class="win-button tool-btn" onclick="toggleCRT();" title="Toggle 1990s CRT Scanlines">
            <span class="tool-icon">📺</span>
            <span>CRT</span>
          </button>
          <button class="win-button tool-btn" onclick="playModemSound();" title="Dial 56k Modem Sound">
            <span class="tool-icon">☎</span>
            <span>Dial-up</span>
          </button>
          <button class="win-button tool-btn" onclick="expandDocs();" title="View Documentation">
            <span class="tool-icon">📖</span>
            <span>ReadMe</span>
          </button>
          <button class="win-button tool-btn" onclick="triggerPulse();" title="Fire Neural Action Potential">
            <span class="tool-icon">⚡</span>
            <span>Pulse</span>
          </button>
        </div>

        <!-- Netscape Spinning Throbber -->
        <div class="throbber-box win-inset" onclick="playModemSound(); stimulateNetwork();" title="Connected to ADAM Swarm">
          <div class="throbber-stars"></div>
          <div class="throbber-n">N</div>
        </div>
      </div>

      <!-- 4. Location Address Bar -->
      <div class="location-bar">
        <span class="location-label">Location:</span>
        <div class="location-input-wrap win-inset">
          <input type="text" id="url-input" class="location-input" value="http://www.adamvangrover.org/neural/net.htm" readonly>
        </div>
        <button class="win-button" style="padding: 2px 10px;" onclick="window.open('index.html', '_self')">Index</button>
        <button class="win-button" style="padding: 2px 10px;" onclick="window.open('hub.html', '_self')">Hub</button>
      </div>

      <!-- 5. Main Split Viewport -->
      <div id="viewport">
        <div class="crt-overlay" id="crt-screen"></div>

        <!-- Left Pane: Interactive Neural Canvas -->
        <div id="neural-pane">
          <div class="hud-overlay">
            <div class="hud-title">🧠 NEURAL TOPOLOGY MATRIX</div>
            <div>STATUS: <span style="color:#00ff66;">CONVERGED // ZERO DRIFT</span></div>
            <div>SYNAPSE ACTIVITY: <span id="synapse-counter">1,428 FIRINGS/SEC</span></div>
            <div>ACTIVE CLUSTER: <span id="active-cluster-name" style="color:#ffcc00;">ALL NODES</span></div>
            <div>ACTION POTENTIAL: <span id="pot-level" style="color:#00f3ff;">98.4 mV</span></div>
          </div>

          <canvas id="neural-canvas"></canvas>

          <div class="hud-controls">
            <button class="win-button" onclick="zoomIn();">Zoom +</button>
            <button class="win-button" onclick="zoomOut();">Zoom -</button>
            <button class="win-button" onclick="resetZoom();">Fit Graph</button>
            <button class="win-button" onclick="togglePhysics();" id="btn-physics">Freeze Synapses</button>
          </div>
        </div>

        <!-- Right Pane: Windows 95 Explorer & Document Reader -->
        <div id="repo-pane">
          <div class="pane-header">
            <span>REPOSITORY EXPLORER :: {total_files} FILES</span>
            <span id="active-category-pill" style="color: #00ff66; font-size: 10px;">SYSTEM ARCHITECTURE</span>
          </div>

          <!-- Live Search & Filtering -->
          <div class="filter-bar">
            <input type="text" id="search-filter" class="filter-input" placeholder="Peruse files (e.g. core, risk, schema, test)..." oninput="onFilterChange()">
            <select id="cluster-select" class="cluster-filter" onchange="onClusterSelect(this.value)">
              <option value="ALL">All Architecture Clusters ({total_files})</option>
              <option value="system_1_swarm">System 1 Swarm ({category_counts.get("system_1_swarm", 0)})</option>
              <option value="system_2_dag">System 2 Deliberative DAG ({category_counts.get("system_2_dag", 0)})</option>
              <option value="pdil_governance">PDIL Governance ({category_counts.get("pdil_governance", 0)})</option>
              <option value="zero_trust_security">Zero Trust Security ({category_counts.get("zero_trust_security", 0)})</option>
              <option value="showcase_intelligence">Showcase & Daily Rituals ({category_counts.get("showcase_intelligence", 0)})</option>
              <option value="knowledge_docs">Docs & Knowledge Base ({category_counts.get("knowledge_docs", 0)})</option>
              <option value="evals_testing">Evals & Testing ({category_counts.get("evals_testing", 0)})</option>
              <option value="runtime_infra">Runtime & Infra ({category_counts.get("runtime_infra", 0)})</option>
              <option value="experimental_lab">Tinker Lab ({category_counts.get("experimental_lab", 0)})</option>
              <option value="root_manifest">Root Manifest ({category_counts.get("root_manifest", 0)})</option>
            </select>
          </div>

          <!-- Split Explorer: File List Table and Document Viewer -->
          <div id="explorer-split">
            <div id="file-table-container" class="win-inset">
              <table class="retro-table" id="file-table">
                <thead>
                  <tr>
                    <th style="width: 20px;"></th>
                    <th onclick="sortFiles('n')">File Name</th>
                    <th onclick="sortFiles('p')">Full Path</th>
                    <th onclick="sortFiles('s')" style="width: 60px;">Size</th>
                  </tr>
                </thead>
                <tbody id="file-tbody">
                  <!-- Injected via JavaScript -->
                </tbody>
              </table>
            </div>

            <!-- Embedded Document Viewer -->
            <div id="doc-viewer" class="win-outset">
              <div class="doc-viewer-header">
                <span id="doc-title">📄 README.md (Preview)</span>
                <div>
                  <button class="win-button" style="padding:1px 6px;" onclick="openSelectedFileInNewTab()">Launch File ↗</button>
                  <button class="win-button" style="padding:1px 6px;" onclick="copyDocContent()">Copy</button>
                </div>
              </div>
              <div class="doc-content win-inset" id="doc-content">
Select any file above to peruse its institutional contents or double-click to launch interactive showcases.
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 6. Status Bar -->
      <div class="statusbar">
        <div class="status-panel win-inset status-msg" id="status-text">
          Document: Done. Neural topology converged with 7,200+ nodes.
        </div>
        <div class="status-panel win-inset">
          <span id="node-stat">Nodes: 10 Clusters</span>
        </div>
        <div class="status-panel win-inset status-progress-box">
          <div class="status-progress-fill"></div>
        </div>
        <div class="status-panel win-inset status-security" title="Zero Trust Verified">
          🔐 SSL
        </div>
      </div>

    </div>
  </div>

  <!-- Retro Dialog Modal -->
  <div class="retro-modal win-outset" id="retro-modal">
    <div class="titlebar">
      <div class="titlebar-title" id="modal-title">System Message</div>
      <div class="titlebar-btn" onclick="closeModal()">✕</div>
    </div>
    <div class="modal-body win-inset" id="modal-content" style="background: #fff; margin: 4px;"></div>
    <div class="modal-actions">
      <button class="win-button" style="min-width: 60px;" onclick="closeModal()">OK</button>
    </div>
  </div>

  <!-- Data Payloads -->
  <script>
    const CLUSTERS = {clusters_json};
    const CATALOG = {catalog_json};
    const EMBEDDED = {embedded_json};
    let currentSelectedPath = "README.md";
    let filteredCatalog = CATALOG;
  </script>

  <!-- Interactive Logic & Neural Canvas Simulation -->
  <script>
    // ═════════════════════════════════════════════════════════════════
    // 1. WEB AUDIO API - RETRO 90S SOUND SYNTHESIS
    // ═════════════════════════════════════════════════════════════════
    let audioCtx = null;

    function getAudioCtx() {{
      if (!audioCtx) {{
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }}
      if (audioCtx.state === 'suspended') {{
        audioCtx.resume();
      }}
      return audioCtx;
    }}

    function playSound(type) {{
      try {{
        const ctx = getAudioCtx();
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        if (type === 'click') {{
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(800, now);
          osc.frequency.exponentialRampToValueAtTime(300, now + 0.04);
          gain.gain.setValueAtTime(0.12, now);
          gain.gain.linearRampToValueAtTime(0.01, now + 0.04);
          osc.start(now);
          osc.stop(now + 0.04);
        }} else if (type === 'synapse') {{
          osc.type = 'sine';
          const f = 400 + Math.random() * 800;
          osc.frequency.setValueAtTime(f, now);
          osc.frequency.exponentialRampToValueAtTime(f * 1.5, now + 0.08);
          gain.gain.setValueAtTime(0.08, now);
          gain.gain.linearRampToValueAtTime(0.001, now + 0.08);
          osc.start(now);
          osc.stop(now + 0.08);
        }}
      }} catch (e) {{}}
    }}

    function playModemSound() {{
      try {{
        const ctx = getAudioCtx();
        const now = ctx.currentTime;

        // Simulate dial tone + baud rate handshake chirp
        const osc1 = ctx.createOscillator();
        const osc2 = ctx.createOscillator();
        const gain = ctx.createGain();

        osc1.type = 'square';
        osc2.type = 'sawtooth';
        osc1.frequency.setValueAtTime(350, now);
        osc2.frequency.setValueAtTime(440, now);

        // Handshake frequency squeals
        osc1.frequency.setValueAtTime(1200, now + 0.3);
        osc1.frequency.linearRampToValueAtTime(2400, now + 0.6);
        osc1.frequency.setValueAtTime(1800, now + 0.7);

        gain.gain.setValueAtTime(0.1, now);
        gain.gain.linearRampToValueAtTime(0.08, now + 0.8);
        gain.gain.linearRampToValueAtTime(0.001, now + 1.1);

        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(ctx.destination);

        osc1.start(now);
        osc2.start(now);
        osc1.stop(now + 1.1);
        osc2.stop(now + 0.3);

        updateStatus("Dialing 56k V.90 gateway to ADAM swarm... CONNECT 57600/ARQ");
      }} catch (e) {{}}
    }}

    // ═════════════════════════════════════════════════════════════════
    // 2. NEURAL NETWORK VISUALIZER (CANVAS)
    // ═════════════════════════════════════════════════════════════════
    const canvas = document.getElementById('neural-canvas');
    const ctx = canvas.getContext('2d');

    let width = 0;
    let height = 0;
    let scale = 1.0;
    let offsetX = 0;
    let offsetY = 0;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let physicsActive = true;

    // Define Network Graph Nodes based on clusters
    const nodes = [];
    const links = [];
    const actionPotentials = []; // Signals traveling through axons

    function initNetworkGraph() {{
      nodes.length = 0;
      links.length = 0;
      actionPotentials.length = 0;

      // Center Sovereign Node
      const centerNode = {{
        id: "adam_sovereign",
        label: "ADAM SOVEREIGN",
        cluster: "core",
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        radius: 26,
        color: "#ffffff",
        pulse: 0,
        isCenter: true,
        desc: "Autonomous Deterministic Alpha Matrix - Neuro-Symbolic Core"
      }};
      nodes.push(centerNode);

      // Create Cluster Hub Nodes
      const clusterKeys = Object.keys(CLUSTERS);
      const angleStep = (Math.PI * 2) / clusterKeys.length;

      clusterKeys.forEach((key, idx) => {{
        const c = CLUSTERS[key];
        const angle = idx * angleStep;
        const dist = 160 + (idx % 2) * 50;
        const x = Math.cos(angle) * dist;
        const y = Math.sin(angle) * dist;

        const node = {{
          id: key,
          label: c.title.split(' (')[0],
          cluster: key,
          x: x,
          y: y,
          vx: (Math.random() - 0.5) * 0.2,
          vy: (Math.random() - 0.5) * 0.2,
          targetDist: dist,
          angle: angle,
          radius: 18,
          color: c.color,
          pulse: Math.random() * Math.PI,
          desc: c.desc
        }};
        nodes.push(node);

        // Connect to center sovereign
        links.push({{
          source: centerNode,
          target: node,
          weight: 2.5,
          color: c.color
        }});

        // Create satellite sub-nodes for each cluster
        const subCount = 3 + (idx % 3);
        for (let s = 0; s < subCount; s++) {{
          const subAngle = angle + (s - (subCount-1)/2) * 0.35;
          const subDist = dist + 70 + (s % 2) * 30;
          const subX = Math.cos(subAngle) * subDist;
          const subY = Math.sin(subAngle) * subDist;

          const dirSample = c.dirs[s % c.dirs.length];
          const subNode = {{
            id: `${{key}}_${{s}}`,
            label: dirSample,
            cluster: key,
            x: subX,
            y: subY,
            vx: 0,
            vy: 0,
            radius: 8 + (s % 3) * 2,
            color: c.color,
            pulse: Math.random() * Math.PI,
            desc: `Subsystem module: ${{dirSample}}`
          }};
          nodes.push(subNode);

          links.push({{
            source: node,
            target: subNode,
            weight: 1.2,
            color: c.color
          }});
        }}
      }});

      // Inter-cluster synapses (Lateral inhibitory & associative connections)
      for (let i = 1; i < clusterKeys.length; i++) {{
        links.push({{
          source: nodes[i],
          target: nodes[((i + 2) % (clusterKeys.length)) + 1],
          weight: 0.8,
          color: "rgba(0, 243, 255, 0.4)",
          lateral: true
        }});
      }}
    }}

    function resizeCanvas() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = width * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }}

    // Spawn an action potential pulse along a link
    function fireActionPotential(link) {{
      actionPotentials.push({{
        link: link,
        progress: 0,
        speed: 0.015 + Math.random() * 0.02,
        color: link.color || "#00f3ff",
        size: 3 + Math.random() * 2
      }});
    }}

    function updateSimulation() {{
      if (physicsActive) {{
        // Soft spring physics & radial stabilization
        for (let i = 1; i < nodes.length; i++) {{
          const n = nodes[i];
          n.pulse += 0.04;
          n.x += n.vx;
          n.y += n.vy;

          // Drag damping
          n.vx *= 0.96;
          n.vy *= 0.96;

          // Return to orbit
          if (n.targetDist) {{
            const curDist = Math.sqrt(n.x * n.x + n.y * n.y) || 1;
            const diff = n.targetDist - curDist;
            n.vx += (n.x / curDist) * diff * 0.002;
            n.vy += (n.y / curDist) * diff * 0.002;
          }}
        }}

        // Spontaneous action potential firing
        if (Math.random() < 0.25 && links.length > 0) {{
          const randomLink = links[Math.floor(Math.random() * links.length)];
          fireActionPotential(randomLink);
        }}
      }}

      // Advance action potentials
      for (let i = actionPotentials.length - 1; i >= 0; i--) {{
        const ap = actionPotentials[i];
        ap.progress += ap.speed;
        if (ap.progress >= 1.0) {{
          ap.link.target.pulse = Math.PI * 1.5;
          actionPotentials.splice(i, 1);
        }}
      }}
    }}

    function renderCanvas() {{
      ctx.save();
      ctx.clearRect(0, 0, width, height);

      // Translate to center + pan/zoom
      ctx.translate(width / 2 + offsetX, height / 2 + offsetY);
      ctx.scale(scale, scale);

      // 1. Draw Synapses (Axons/Dendrites)
      for (const link of links) {{
        ctx.beginPath();
        ctx.moveTo(link.source.x, link.source.y);
        ctx.lineTo(link.target.x, link.target.y);

        ctx.strokeStyle = link.lateral ? "rgba(0, 243, 255, 0.15)" : "rgba(255, 255, 255, 0.25)";
        ctx.lineWidth = link.weight || 1;
        ctx.stroke();
      }}

      // 2. Draw Traveling Action Potentials
      for (const ap of actionPotentials) {{
        const x = ap.link.source.x + (ap.link.target.x - ap.link.source.x) * ap.progress;
        const y = ap.link.source.y + (ap.link.target.y - ap.link.source.y) * ap.progress;

        ctx.beginPath();
        ctx.arc(x, y, ap.size, 0, Math.PI * 2);
        ctx.fillStyle = ap.color;
        ctx.shadowColor = ap.color;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      }}

      // 3. Draw Neurons (Nodes)
      for (const node of nodes) {{
        const glow = Math.sin(node.pulse) * 4 + 4;

        // Outer aura
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius + glow, 0, Math.PI * 2);
        ctx.fillStyle = node.color + "22";
        ctx.fill();

        // Main neuron body
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.shadowColor = node.color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label
        ctx.fillStyle = "#ffffff";
        ctx.font = node.isCenter ? "bold 11px Courier New" : "9px Courier New";
        ctx.textAlign = "center";
        ctx.fillText(node.label, node.x, node.y + node.radius + 12);
      }}

      ctx.restore();
    }}

    function animLoop() {{
      updateSimulation();
      renderCanvas();
      requestAnimationFrame(animLoop);
    }}

    // Canvas Mouse Interaction (Pan & Click)
    canvas.addEventListener('mousedown', (e) => {{
      isDragging = true;
      dragStartX = e.clientX - offsetX;
      dragStartY = e.clientY - offsetY;

      // Hit test node
      const rect = canvas.getBoundingClientRect();
      const mouseX = (e.clientX - rect.left - width / 2 - offsetX) / scale;
      const mouseY = (e.clientY - rect.top - height / 2 - offsetY) / scale;

      for (const node of nodes) {{
        const dx = node.x - mouseX;
        const dy = node.y - mouseY;
        if (Math.sqrt(dx * dx + dy * dy) <= node.radius + 5) {{
          onNodeClick(node);
          break;
        }}
      }}
    }});

    window.addEventListener('mousemove', (e) => {{
      if (isDragging) {{
        offsetX = e.clientX - dragStartX;
        offsetY = e.clientY - dragStartY;
      }}
    }});

    window.addEventListener('mouseup', () => {{
      isDragging = false;
    }});

    canvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      scale = Math.min(Math.max(scale * zoomFactor, 0.4), 3.0);
    }});

    function onNodeClick(node) {{
      playSound('synapse');
      document.getElementById('active-cluster-name').innerText = node.label.toUpperCase();
      document.getElementById('status-text').innerText = `Neuron selected: [${{node.label}}] - ${{node.desc}}`;

      if (node.cluster && node.cluster !== "core") {{
        document.getElementById('cluster-select').value = node.cluster;
        onClusterSelect(node.cluster);
      }} else {{
        document.getElementById('cluster-select').value = "ALL";
        onClusterSelect("ALL");
      }}
    }}

    function zoomIn() {{ scale = Math.min(scale * 1.25, 3.0); }}
    function zoomOut() {{ scale = Math.max(scale * 0.8, 0.4); }}
    function resetZoom() {{ scale = 1.0; offsetX = 0; offsetY = 0; }}
    function resetGraph() {{ resetZoom(); document.getElementById('cluster-select').value = "ALL"; onClusterSelect("ALL"); }}

    function togglePhysics() {{
      physicsActive = !physicsActive;
      document.getElementById('btn-physics').innerText = physicsActive ? "Freeze Synapses" : "Unfreeze";
      document.getElementById('btn-physics').classList.toggle('active', !physicsActive);
    }}

    function stimulateNetwork() {{
      playSound('synapse');
      for (const link of links) {{
        if (Math.random() < 0.6) fireActionPotential(link);
      }}
      updateStatus("Network stimulated. Multi-agent action potentials propagating across DAG.");
    }}

    function triggerPulse() {{
      stimulateNetwork();
      document.getElementById('pot-level').innerText = (90 + Math.random() * 15).toFixed(1) + " mV";
    }}

    // ═════════════════════════════════════════════════════════════════
    // 3. REPOSITORY PERUSAL & FILE TABLE ENGINE
    // ═════════════════════════════════════════════════════════════════
    let currentSortCol = 'p';
    let sortAsc = true;

    function renderFileTable(files) {{
      const tbody = document.getElementById('file-tbody');
      tbody.innerHTML = '';

      // Limit initial rendering to first 300 to maintain silky 60fps retro responsiveness
      const displayFiles = files.slice(0, 300);

      displayFiles.forEach((file) => {{
        const tr = document.createElement('tr');
        if (file.p === currentSelectedPath) tr.classList.add('selected');

        let icon = '📄';
        if (file.e === '.html') icon = '🌐';
        else if (file.e === '.py') icon = '🐍';
        else if (file.e === '.md') icon = '📝';
        else if (file.e === '.json') icon = '📊';

        tr.innerHTML = `
          <td>${{icon}}</td>
          <td><b>${{escapeHtml(file.n)}}</b></td>
          <td style="color:#555;">${{escapeHtml(file.p)}}</td>
          <td style="text-align:right;">${{formatSize(file.s)}}</td>
        `;

        tr.onclick = () => selectFile(file);
        tr.ondblclick = () => openFileDirect(file.p);
        tbody.appendChild(tr);
      }});

      document.getElementById('node-stat').innerText = `Showing ${{displayFiles.length}} of ${{files.length}} files`;
    }}

    function selectFile(file, trElement) {{
      playSound('click');
      currentSelectedPath = file.p;
      document.getElementById('url-input').value = `http://www.adamvangrover.org/${{file.p}}`;

      // Update table selection highlight
      document.querySelectorAll('#file-tbody tr').forEach(r => r.classList.remove('selected'));
      if (trElement) {{
        trElement.classList.add('selected');
      }}

      document.getElementById('doc-title').innerText = `📄 ${{file.n}} (${{formatSize(file.s)}})`;
      document.getElementById('status-text').innerText = `Inspecting file: ${{file.p}}`;

      const viewer = document.getElementById('doc-content');
      if (EMBEDDED[file.p]) {{
        viewer.textContent = EMBEDDED[file.p];
      }} else {{
        viewer.textContent = `[REPOSITORY FILE RECORD]\n\nPath: ${{file.p}}\nCluster: ${{file.c}}\nSize: ${{file.s}} bytes\nExtension: ${{file.e || 'none'}}\n\nThis file is cataloged in the ADAM v30.1 Autonomous Sovereign repository index.\nDouble-click the file entry or click "Launch File ↗" above to inspect the live file.`;
      }}
    }}

    function onFilterChange() {{
      const query = document.getElementById('search-filter').value.toLowerCase().trim();
      const cluster = document.getElementById('cluster-select').value;

      filteredCatalog = CATALOG.filter(f => {{
        const matchesQuery = !query || f.p.toLowerCase().includes(query) || f.n.toLowerCase().includes(query);
        const matchesCluster = cluster === 'ALL' || f.c === cluster;
        return matchesQuery && matchesCluster;
      }});

      renderFileTable(filteredCatalog);
    }}

    function onClusterSelect(cluster) {{
      playSound('click');
      const clusterObj = CLUSTERS[cluster];
      if (clusterObj) {{
        document.getElementById('active-category-pill').innerText = clusterObj.title.toUpperCase();
      }} else {{
        document.getElementById('active-category-pill').innerText = "ALL ARCHITECTURE CLUSTERS";
      }}
      onFilterChange();
    }}

    function sortFiles(col) {{
      playSound('click');
      if (currentSortCol === col) {{
        sortAsc = !sortAsc;
      }} else {{
        currentSortCol = col;
        sortAsc = true;
      }}

      filteredCatalog.sort((a, b) => {{
        let vA = a[col];
        let vB = b[col];
        if (typeof vA === 'string') vA = vA.toLowerCase();
        if (typeof vB === 'string') vB = vB.toLowerCase();
        if (vA < vB) return sortAsc ? -1 : 1;
        if (vA > vB) return sortAsc ? 1 : -1;
        return 0;
      }});

      renderFileTable(filteredCatalog);
    }}

    function openSelectedFileInNewTab() {{
      if (currentSelectedPath) {{
        window.open(currentSelectedPath, '_blank');
      }}
    }}

    function openFileDirect(path) {{
      window.open(path, '_blank');
    }}

    function copyDocContent() {{
      const text = document.getElementById('doc-content').textContent;
      navigator.clipboard.writeText(text);
      updateStatus("Copied document content to clipboard.");
    }}

    function expandDocs() {{
      const readme = CATALOG.find(f => f.p === "README.md");
      if (readme) selectFile(readme);
    }}

    // ═════════════════════════════════════════════════════════════════
    // 4. RETRO 1990S BROWSER CHROMES & DIALOGS
    // ═════════════════════════════════════════════════════════════════
    function toggleCRT() {{
      playSound('click');
      const crt = document.getElementById('crt-screen');
      crt.classList.toggle('active');
      updateStatus(crt.classList.contains('active') ? "CRT 1995 phosphorescent scanlines ENABLED" : "CRT scanlines DISABLED");
    }}

    function showModal(title, text) {{
      playSound('click');
      document.getElementById('modal-title').innerText = title;
      document.getElementById('modal-content').innerHTML = text;
      document.getElementById('retro-modal').style.display = 'block';
    }}

    function closeModal() {{
      playSound('click');
      document.getElementById('retro-modal').style.display = 'none';
    }}

    function showAboutDialog() {{
      showModal("About ADAM Neural Navigator", `
        <b>Autonomous Deterministic Alpha Matrix (ADAM v30.1)</b><br>
        <i>Institutional-Grade Neuro-Symbolic Sovereign System</i><br><br>
        • Repository: <b>adamvangrover/adam</b><br>
        • Cataloged Files: <b>${{CATALOG.length}} files</b><br>
        • Architecture: Dual-system DAG / Swarm / PDIL / Zero-Trust<br>
        • Interface: Windows 95 / Netscape Navigator 3.0 Aesthetic<br><br>
        Engineered for deep credit underwriting, market mayhem scenarios, and autonomous alpha exploration.
      `);
    }}

    function historyBack() {{
      playSound('click');
      updateStatus("Navigated back in session stack.");
    }}

    function historyForward() {{
      playSound('click');
      updateStatus("Navigated forward in session stack.");
    }}

    function updateStatus(msg) {{
      document.getElementById('status-text').innerText = msg;
    }}

    function escapeHtml(str) {{
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }}

    function formatSize(bytes) {{
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / 1048576).toFixed(1) + ' MB';
    }}

    // Initialization
    window.addEventListener('DOMContentLoaded', () => {{
      initNetworkGraph();
      resizeCanvas();
      window.addEventListener('resize', () => {{
        resizeCanvas();
      }});
      animLoop();
      renderFileTable(CATALOG);

      // Default select README.md
      const readme = CATALOG.find(f => f.p === "README.md");
      if (readme) selectFile(readme);
    }});
  </script>
</body>
</html>
"""
    return html_code


def main():
    print(f"Scanning repository at: {REPO_ROOT}...")
    catalog, category_counts, embedded_data = build_repo_catalog(REPO_ROOT)
    print(f"Collected {len(catalog)} active files across {len(category_counts)} architectural clusters.")
    print(f"Embedded full text content for {len(embedded_data)} key institutional files.")

    html_content = generate_html_content(catalog, category_counts, embedded_data)
    out_file = os.path.join(REPO_ROOT, "adam_neural_browser_index.html")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Successfully generated standalone HTML: {out_file} ({len(html_content)} bytes)")


if __name__ == "__main__":
    main()
