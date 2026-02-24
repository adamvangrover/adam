#!/usr/bin/env python3
"""
ADAM v26.0 :: INTERACTIVE BUILDER
-----------------------------------------------------------------------------
Custom Build & Deployment Wizard.
Generates portable environments, Dockerfiles, and dependency sets based on user selection.
-----------------------------------------------------------------------------
"""

import os
import sys
import shutil
import datetime
from pathlib import Path

# Import export logic
try:
    import export_module
except ImportError:
    # Add local dir to path if running from scripts/
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import export_module

# Constants
BUILD_DIR = Path("builds")
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Import System Logger
try:
    from core.utils.system_logger import SystemLogger
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.utils.system_logger import SystemLogger

def print_header():
    print("\033[1;36m" + "=" * 60)
    print("   ADAM v26.0 - Custom Environment Builder")
    print("   Modular | Portable | Adaptive")
    print("=" * 60 + "\033[0m")
    print()

def get_user_selection(options, prompt_text="Select options (comma separated, e.g. 1,3):"):
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options):
        print(f"  [{i+1}] {opt}")

    choice = input("\n> ").strip()
    if not choice:
        return []

    selected_indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
    selected = [options[i] for i in selected_indices if 0 <= i < len(options)]
    return selected

def select_profile():
    print("\n[?] Select Runtime Profile:")
    profiles = [
        "Lite (HTML/JS Only - Static Hosting)",
        "Standard (Python + Flask - Local Dev)",
        "Full (Docker + ML Stack - Production)"
    ]
    for i, p in enumerate(profiles):
        print(f"  [{i+1}] {p}")

    choice = input("\n> ").strip()
    if choice == "1": return "lite"
    if choice == "2": return "standard"
    if choice == "3": return "full"
    return "standard" # Default

def generate_landing_page(target_dir, modules):
    """Creates a root index.html linking to all exported modules."""
    links_html = ""
    for mod in modules:
        # Assuming export_module creates a folder named 'mod'
        links_html += f"""
        <a href="{mod}/index.html" class="module-card">
            <div class="module-icon">📦</div>
            <div class="module-info">
                <h3>{mod.replace('_', ' ').upper()}</h3>
                <p>Launch Module</p>
            </div>
            <div class="arrow">→</div>
        </a>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM Custom Build</title>
    <style>
        body {{ background: #050b14; color: #e0e0e0; font-family: 'Inter', sans-serif; display: flex; flex-direction: column; align-items: center; min-height: 100vh; margin: 0; }}
        h1 {{ margin-top: 50px; font-family: 'JetBrains Mono', monospace; color: #00f3ff; letter-spacing: 2px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; width: 100%; max-width: 800px; padding: 20px; }}
        .module-card {{ background: rgba(255,255,255,0.05); border: 1px solid #333; padding: 20px; border-radius: 8px; text-decoration: none; color: #fff; transition: all 0.2s; display: flex; align-items: center; gap: 15px; }}
        .module-card:hover {{ border-color: #00f3ff; background: rgba(0, 243, 255, 0.05); transform: translateY(-2px); }}
        .module-icon {{ font-size: 2rem; }}
        .module-info h3 {{ margin: 0 0 5px 0; font-size: 1rem; color: #00f3ff; font-family: 'JetBrains Mono'; }}
        .module-info p {{ margin: 0; font-size: 0.8rem; color: #888; }}
        .arrow {{ margin-left: auto; color: #555; }}
        .footer {{ margin-top: auto; padding: 20px; font-size: 0.8rem; color: #555; font-family: 'JetBrains Mono'; }}
    </style>
</head>
<body>
    <h1>ADAM // SYSTEM HUB</h1>
    <div class="grid">
        {links_html}
    </div>
    <div class="footer">GENERATED BUILD: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body>
</html>
"""
    (target_dir / "index.html").write_text(html)

def generate_requirements(target_dir, profile):
    """Copies the appropriate requirements file."""
    if profile == "standard":
        src = TEMPLATES_DIR / "requirements_standard.txt"
    elif profile == "full":
        src = TEMPLATES_DIR / "requirements_full.txt"
    else:
        # Lite profile
        return

    if src.exists():
        shutil.copy(src, target_dir / "requirements.txt")
        print(f"[*] Generated requirements.txt ({profile} profile)")
    else:
        print(f"[!] Warning: Template {src} not found.")

def generate_dockerfile(target_dir, profile, modules):
    """Generates a Dockerfile based on the template."""
    template_path = TEMPLATES_DIR / "Dockerfile.template"
    if not template_path.exists():
        print("[!] Dockerfile template not found.")
        return

    content = template_path.read_text()

    # Customization logic
    base_image = "python:3.10-slim" if profile == "standard" else "python:3.10"

    copy_cmds = "COPY . /app"
    env_type = "production" if profile == "full" else "development"

    content = content.replace("{{ BASE_IMAGE }}", base_image)
    content = content.replace("{{ COPY_COMMANDS }}", copy_cmds)
    content = content.replace("{{ PORT }}", "8000")
    content = content.replace("{{ ENV_TYPE }}", env_type)

    (target_dir / "Dockerfile").write_text(content)
    print(f"[*] Generated Dockerfile ({base_image})")

def generate_launcher(target_dir):
    """Generates a root-level launcher script."""
    launcher_path = target_dir / "run_module.py"
    content = """
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

print(f"[*] Starting ADAM Custom Build on port {PORT}...")
print(f"[*] Opening browser to http://localhost:{PORT}")

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        webbrowser.open(f"http://localhost:{PORT}")
        httpd.serve_forever()
except OSError:
    print(f"[!] Port {PORT} in use. Try running: python3 -m http.server")
except KeyboardInterrupt:
    print("\\n[*] Server stopped.")
    sys.exit(0)
"""
    launcher_path.write_text(content.strip())
    os.chmod(launcher_path, 0o755)
    print("[*] Generated Universal Launcher (run_module.py)")

def main():
    # Log Build Start
    SystemLogger().log_event("SERVER_BUILD", {"status": "START"})

    print_header()

    # 1. Select Modules
    available_modules = list(export_module.MODULES.keys())
    selected_modules = get_user_selection(available_modules, "Select Modules to Include:")

    if not selected_modules:
        print("[!] No modules selected. Exiting.")
        return

    # 2. Select Profile
    profile = select_profile()

    # 3. Setup Build Dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    build_name = f"adam_build_{timestamp}"
    target_dir = BUILD_DIR / build_name

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    print(f"\n[*] Initializing Build in: {target_dir}")

    # 4. Export Modules
    print("[*] Exporting Modules...")
    for mod in selected_modules:
        export_module.export_module(mod, str(target_dir))

    # 5. Generate Landing Page
    generate_landing_page(target_dir, selected_modules)

    # 6. Generate Configs
    if profile != "lite":
        generate_requirements(target_dir, profile)
        generate_dockerfile(target_dir, profile, selected_modules)

    # 7. Generate Launcher (for all profiles as convenience)
    generate_launcher(target_dir)

    # 8. Final Instructions
    print("\n" + "=" * 60)
    print("   BUILD COMPLETE")
    print("=" * 60)
    print(f"   Location: {target_dir}")
    print("\n   [Run Locally]")
    if profile == "lite":
        print(f"   cd {target_dir}")
        print("   python3 run_module.py  (or any static server)")
    else:
        print(f"   cd {target_dir}")
        print("   pip install -r requirements.txt")
        print("   python3 run_module.py")

    if profile != "lite":
        print("\n   [Run Docker]")
        print(f"   cd {target_dir}")
        print(f"   docker build -t {build_name} .")
        print(f"   docker run -p 8000:8000 {build_name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
