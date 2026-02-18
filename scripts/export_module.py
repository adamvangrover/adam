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
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

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
    "js/command_palette.js"
]

# Module Definitions
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
    },
    "strategic_intelligence": {
        "entry": "modular_dashboard.html",
        "includes": [
            "js/modular_loader.js",
            "data/modular/manifest.json",
            "data/modular/reports.json",
            "data/modular/market_data.json",
            "data/modular/system_status.json",
            "daily_briefings_library.html",
            "market_pulse_library.html",
            "house_view_library.html",
            "js/library_logic.js"
        ]
    },
    "credit_command": {
        "entry": "credit_analyst_workstation.html",
        "includes": [
            "js/modular_loader.js",
            "data/modular/manifest.json",
            "data/modular/credit_memos.json",
            "data/modular/market_data.json",
            "data/modular/system_status.json",
            "credit_memo_automation.html",
            "risk_topography.html",
            "unified_credit_console.html",
            "js/credit_memo.js"
        ]
    },
    "simulation_deck": {
        "entry": "crisis_simulator.html",
        "includes": [
            "js/modular_loader.js",
            "data/modular/manifest.json",
            "data/modular/market_data.json",
            "data/modular/system_status.json",
            "war_room.html",
            "wargame_dashboard.html",
            "scenario_lab.html",
            "js/simulation_viewer.js"
        ]
    },
    "modular_dashboard": {
        "entry": "modular_dashboard.html",
        "includes": [
            "js/data_manager_modular.js",
            "data/seed_reports.json",
            "data/seed_credit_memos.json",
            "data/seed_file_index.json",
            "data/seed_agents.json",
            "data/seed_prompts.json",
            "data/seed_training_data.json"
        ]
    },
    "13f_tracker": {
        "entry": "13f_tracker.html",
        "includes": [
            "data/13f_data.json"
        ]
    },
    "evolution_v2": {
        "entry": "evolution_v2.html",
        "includes": [
            "data/evolution_data.json",
            "js/command_palette.js"
        ]
    },
    "intelligence_library": {
        "entry": "intelligence_library.html",
        "includes": [
            "css/library.css",
            "js/library_logic.js",
            "data/market_mayhem_index.json"
        ]
    },
    "deep_dive_viewer": {
        "entry": "deep_dive_viewer.html",
        "includes": [
            "data/deep_dive_sample.json",
            "data/deep_dive_*.json",
            "css/cyberpunk-core.css"
        ]
    },
    "simulation_dashboard": {
        "entry": "simulation_dashboard.html",
        "includes": [
            "js/simulation_viewer.js",
            "css/market_mayhem_tiers.css"
        ]
    },
    "research_lab": {
        "entry": "scenario_lab.html",
        "includes": [
            "data/mock_scenario_lab_data.json"
        ]
    },
    "financial_twin": {
        "entry": "financial_twin.html",
        "includes": [
            "data/unified_banking_scenarios.json"
        ]
    },
    "contagion_analysis": {
        "entry": "risk_topography.html",
        "includes": [
            "data/phase3_portfolio_demo.json"
        ]
    },
    "slm_adam": {
        "entry": "chat.html",
        "includes": [
            "js/mock_data.js"
        ]
    },
    "optimization": {
        "entry": "portfolio_dashboard.html",
        "includes": [
            "data/portfolio_history.json"
        ]
    },
    "quantum_toolkit": {
        "entry": "quantum_search.html",
        "includes": [
            "data/quantum_search_data.json"
        ]
    }
}

# -----------------------------------------------------------------------------
# Export Logic
# -----------------------------------------------------------------------------

def setup_export_dir(module_name, output_base):
    """Creates the export directory, cleaning it if it exists."""
    target_dir = output_base / module_name
    if target_dir.exists():
        print(f"[*] Cleaning existing export: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

def copy_asset(asset_rel_path, target_dir):
    """
    Copies a file or directory from showcase/ to target_dir.
    Handles standard paths and glob patterns (e.g., data/file_*.json).
    """
    src_path = SHOWCASE_DIR / asset_rel_path
    
    # 1. Direct File/Directory Match
    if src_path.exists():
        dst_path = target_dir / asset_rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
        return

    # 2. Glob Match (Pattern Matching)
    # glob() matches against the actual filesystem
    found_files = list(SHOWCASE_DIR.glob(asset_rel_path))
    
    if found_files:
        for file in found_files:
            rel_path = file.relative_to(SHOWCASE_DIR)
            dst_path = target_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if file.is_file():
                shutil.copy2(file, dst_path)
        return

    # 3. Not Found
    print(f"[!] Warning: Asset not found or pattern matched nothing: {asset_rel_path}")

def patch_nav_js(target_dir):
    """
    Patches nav.js to work in the root of the exported module.
    It changes the showcasePath reference to the current root.
    """
    nav_js_path = target_dir / "js/nav.js"
    if nav_js_path.exists():
        print("[*] Patching js/nav.js for standalone mode...")
        try:
            content = nav_js_path.read_text(encoding='utf-8')
            
            # Replace the showcase path logic
            target_str = "this.showcasePath = `${cleanRoot}/showcase`;"
            replacement_str = "this.showcasePath = `${cleanRoot}`;"
            
            new_content = content.replace(target_str, replacement_str)
            
            if new_content == content:
                print(f"[!] Warning: Target string for patching not found in nav.js.")
            else:
                nav_js_path.write_text(new_content, encoding='utf-8')
                print("[+] nav.js patched successfully.")
        except Exception as e:
            print(f"[!] Error patching nav.js: {e}")

def create_launcher(target_dir, module_name):
    """Creates a standalone Python server launcher script."""
    launcher_path = target_dir / "run_module.py"
    
    # Using raw string for the script content to avoid f-string complexity with inner curlies
    content = f"""
import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIRECTORY = "."

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    print(f"[*] Starting {module_name} module...")
    print(f"[*] Serving at http://localhost:{{PORT}}")
    
    # Change to script directory to ensure relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            webbrowser.open(f"http://localhost:{{PORT}}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\\n[*] Stopping server...")
                sys.exit(0)
    except OSError as e:
        print(f"[!] Error: Port {{PORT}} might be in use. Details: {{e}}")
        print("    Try running manually: python3 -m http.server")

if __name__ == "__main__":
    main()
"""
    launcher_path.write_text(content.strip(), encoding='utf-8')
    # Make executable on Unix-like systems
    try:
        current_perms = os.stat(launcher_path).st_mode
        os.chmod(launcher_path, current_perms | 0o111)
    except Exception:
        pass

def export_module(module_name, output_dir_str):
    """Main export execution logic."""
    if module_name not in MODULES:
        print(f"[!] Error: Unknown module '{module_name}'")
        print(f"    Available: {', '.join(sorted(MODULES.keys()))}")
        return

    print(f"\n>>> EXPORTING MODULE: {module_name}")
    config = MODULES[module_name]
    output_base = Path(output_dir_str).resolve()
    target_dir = setup_export_dir(module_name, output_base)

    # 1. Copy Common Assets
    print("[*] Copying common assets...")
    for asset in COMMON_ASSETS:
        copy_asset(asset, target_dir)

    # 2. Copy Module Specifics
    print("[*] Copying module includes...")
    for item in config.get("includes", []):
        copy_asset(item, target_dir)

    # 3. Handle Entry Point
    entry_file = config.get("entry")
    if entry_file:
        entry_src = SHOWCASE_DIR / entry_file
        
        if entry_src.exists():
            # Copy to index.html (Main Entry)
            entry_dst = target_dir / "index.html"
            shutil.copy2(entry_src, entry_dst)
            print(f"[*] Entry point set: {entry_file} -> index.html")
            
            # Also copy to original filename (Preserve Links)
            if entry_file != "index.html":
                orig_dst = target_dir / entry_file
                shutil.copy2(entry_src, orig_dst)
        else:
            print(f"[!] Error: Entry point {entry_src} not found!")

    # 4. Patch Navigation
    patch_nav_js(target_dir)

    # 5. Create Launcher
    create_launcher(target_dir, module_name)

    print(f"[+] Export Complete: {target_dir}")
    print(f"    Run: python3 {target_dir / 'run_module.py'}")

# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ADAM repo modules.")
    parser.add_argument("module", nargs='?', help="Name of the module to export (or 'all').")
    parser.add_argument("--output", default="exports", help="Output directory (default: exports)")
    parser.add_argument("--list", action="store_true", help="List available modules")

    args = parser.parse_args()

    if args.list:
        print("Available Modules:")
        for mod in sorted(MODULES.keys()):
            print(f" - {mod}")
    elif args.module == "all":
        for mod in MODULES:
            export_module(mod, args.output)
    elif args.module:
        export_module(args.module, args.output)
    else:
        parser.print_help()