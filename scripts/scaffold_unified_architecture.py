import os

FILES_TO_UPGRADE = ["index.html", "hub.html", "index_all.html"]
SCAFFOLD_HTML = """
<!-- UNIFIED NAVIGATOR SCAFFOLD -->
<style>
    #unified-nav-scaffold {
        position: fixed; top: 80px; right: 20px;
        background: rgba(10, 15, 25, 0.9); border: 1px solid #ff00ff;
        border-radius: 8px; padding: 15px; z-index: 10000;
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.2);
        display: flex; flex-direction: column; gap: 10px;
        font-family: 'JetBrains Mono', monospace;
    }
    .unified-btn {
        color: #e0e0e0; text-decoration: none; padding: 8px 12px;
        border: 1px solid #444; border-radius: 4px; font-size: 0.8rem;
        text-align: center; transition: all 0.2s;
    }
    .unified-btn:hover { background: rgba(255, 0, 255, 0.1); border-color: #ff00ff; color: #ff00ff; }
    .scaffold-title { color: #ff00ff; font-size: 0.9rem; margin-bottom: 5px; text-align: center; border-bottom: 1px solid #333; padding-bottom: 5px; }
</style>
<div id="unified-nav-scaffold">
    <div class="scaffold-title">[ CAPABILITIES ]</div>
    <a href="system_graph_viewer.html" class="unified-btn">SYSTEM GRAPH</a>
    <a href="json_data_viewer.html" class="unified-btn">DATA / JSON</a>
    <a href="prompt_library_viewer.html" class="unified-btn">PROMPT LIB</a>
    <a href="agent_viewer.html" class="unified-btn">AGENT VIEWER</a>
    <a href="data_topography_viewer.html" class="unified-btn">DATA TOPO</a>
</div>
<!-- END UNIFIED NAVIGATOR SCAFFOLD -->
"""

for filename in FILES_TO_UPGRADE:
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        if "<!-- UNIFIED NAVIGATOR SCAFFOLD -->" not in content:
            new_content = content.replace("</body>", f"{SCAFFOLD_HTML}\n</body>")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Injected scaffold into {filename}")
        else:
            print(f"Scaffold already present in {filename}")
    else:
        print(f"File {filename} not found.")