#!/usr/bin/env python3
"""
ADAM v24.0 :: MODULE EXPORTER
-----------------------------------------------------------------------------
Captures repo subsets into modular, self-contained, portable units.
Each module runs standalone with its own index.html and dependencies.
-----------------------------------------------------------------------------
"""

import os
import shutil
import glob
import argparse
from pathlib import Path

# Base Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_DIR = REPO_ROOT / "showcase"
EXPORT_BASE = REPO_ROOT / "exports"

# Common Assets required by all modules
COMMON_ASSETS = [
    "js/nav.js",
    "js/app.js",
    "js/mock_data.js",
    "css/style.css",
    "site_map.json",
    "js/command_palette.js" # Often used
]

# Module Definitions
# Keys: Module Name
# Values:
#   - entry: The main HTML file (will be renamed to index.html)
#   - includes: List of files or glob patterns relative to showcase/
MODULES = {
    "market_mayhem": {
        "entry": "market_mayhem_archive.html",
        "includes": [
            "js/market_mayhem_viewer.js",
            "data/market_mayhem_index.json",
            "data/newsletter_data.json",
            "newsletter_*.html",
            "Daily_Briefing_*.html",
            "Market_Pulse_*.html",
            "House_View_*.html",
            "*_Market_Mayhem.html",
            "market_mayhem_archive_v*.html",
            "market_mayhem_rebuild.html",
            "market_mayhem_conviction.html"
        ]
    },
    "system_brain": {
        "entry": "system_brain.html",
        "includes": [
            "js/system_brain_data.js",
            "data/system_knowledge_graph.json",
            "system_knowledge_graph.html",
            "js/hud.js"
        ]
    },
    "neural_nexus": {
        "entry": "neural_nexus.html",
        "includes": [
            "js/neural_nexus.js",
            "data/repo_metadata.json",
            "js/hud.js"
        ]
    },
    "repository": {
        "entry": "repository_v2.html",
        "includes": [
            "js/repository_v2.js",
            "js/editor_logic.js",
            "editor_studio.html",
            "data/market_mayhem_index.json"
        ]
    }
}

def setup_export_dir(module_name, output_base):
    """Creates the export directory."""
    target_dir = output_base / module_name
    if target_dir.exists():
        print(f"[*] Cleaning existing export: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

def copy_asset(asset_rel_path, target_dir):
    """Copies a file from showcase/ to target_dir, preserving structure."""
    src = SHOWCASE_DIR / asset_rel_path
    dst = target_dir / asset_rel_path

    if not src.exists():
        # Try globbing
        if "*" in asset_rel_path:
            for file in SHOWCASE_DIR.glob(asset_rel_path):
                rel_path = file.relative_to(SHOWCASE_DIR)
                dst_path = target_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dst_path)
            return
        else:
            print(f"[!] Warning: Asset not found: {src}")
            return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

def patch_nav_js(target_dir):
    """Patches nav.js to work in the root of the exported module."""
    nav_js_path = target_dir / "js/nav.js"
    if nav_js_path.exists():
        print("[*] Patching js/nav.js for standalone mode...")
        content = nav_js_path.read_text()

        # Replace the showcase path logic
        # Original: this.showcasePath = `${cleanRoot}/showcase`;
        # New:      this.showcasePath = `${cleanRoot}`;
        new_content = content.replace(
            "this.showcasePath = `${cleanRoot}/showcase`;",
            "this.showcasePath = `${cleanRoot}`;"
        )

        # Also handle the specific check for /showcase/ in path
        # Original: if (window.location.pathname.includes('/showcase/') && !dataRoot)
        # We can just make it always use cleanRoot (current dir)

        nav_js_path.write_text(new_content)

def create_launcher(target_dir, module_name):
    """Creates a simple python server launcher."""
    launcher_path = target_dir / "run_module.py"
    content = f"""
import http.server
import socketserver
import webbrowser
import os

PORT = 8000
DIRECTORY = "."

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

print(f"[*] Starting {module_name} module on port {{PORT}}...")
print(f"[*] Opening browser to http://localhost:{{PORT}}")

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        webbrowser.open(f"http://localhost:{{PORT}}")
        httpd.serve_forever()
except OSError:
    print(f"[!] Port {{PORT}} in use. Try running: python3 -m http.server")
"""
    launcher_path.write_text(content.strip())
    os.chmod(launcher_path, 0o755)

def export_module(module_name, output_dir):
    """Main export logic."""
    if module_name not in MODULES:
        print(f"[!] Error: Unknown module '{module_name}'")
        print(f"Available modules: {', '.join(MODULES.keys())}")
        return

    print(f"\n>>> EXPORTING MODULE: {module_name}")
    config = MODULES[module_name]
    target_dir = setup_export_dir(module_name, Path(output_dir))

    # 1. Copy Common Assets
    print("[*] Copying common assets...")
    for asset in COMMON_ASSETS:
        copy_asset(asset, target_dir)

    # 2. Copy Module Specifics
    print("[*] Copying module assets...")
    for item in config["includes"]:
        copy_asset(item, target_dir)

    # 3. Handle Entry Point
    entry_src = SHOWCASE_DIR / config["entry"]
    if entry_src.exists():
        entry_dst = target_dir / "index.html"
        shutil.copy2(entry_src, entry_dst)
        print(f"[*] Entry point set: {config['entry']} -> index.html")
    else:
        print(f"[!] Error: Entry point {entry_src} not found!")

    # 4. Patch Navigation
    patch_nav_js(target_dir)

    # 5. Create Launcher
    create_launcher(target_dir, module_name)

    print(f"[+] Export Complete: {target_dir}")
    print(f"    Run: python3 {target_dir}/run_module.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ADAM repo modules.")
    parser.add_argument("module", help="Name of the module to export (or 'all')")
    parser.add_argument("--output", default="exports", help="Output directory")

    args = parser.parse_args()

    if args.module == "all":
        for mod in MODULES:
            export_module(mod, args.output)
    else:
        export_module(args.module, args.output)
