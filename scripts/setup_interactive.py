#!/usr/bin/env python3
import os
import sys
import subprocess

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

def print_banner():
    print(f"{CYAN}")
    print("=" * 60)
    print("      ADAM SYSTEM - INTERACTIVE SETUP WIZARD")
    print("=" * 60)
    print(f"{RESET}")

def check_python():
    print(f"{CYAN}[*] Checking Python environment...{RESET}")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"{RED}[!] Error: Python 3.9+ is required. You are running {version.major}.{version.minor}{RESET}")
        sys.exit(1)
    print(f"{GREEN}[✓] Python {version.major}.{version.minor} detected.{RESET}")

def check_docker():
    print(f"{CYAN}[*] Checking Docker...{RESET}")
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"{GREEN}[✓] Docker detected.{RESET}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{YELLOW}[!] Docker not found. You will be limited to local execution.{RESET}")
        return False

def check_node():
    print(f"{CYAN}[*] Checking Node.js (for UI)...{RESET}")
    try:
        subprocess.run(["npm", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"{GREEN}[✓] Node/npm detected.{RESET}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{YELLOW}[!] npm not found. The React UI cannot be built locally.{RESET}")
        return False

def setup_env():
    print(f"\n{CYAN}[*] Configuring Environment (.env)...{RESET}")

    if os.path.exists(".env"):
        overwrite = input(f"{YELLOW}[?] .env already exists. Overwrite? (y/N): {RESET}").lower()
        if overwrite != 'y':
            print(f"{GREEN}[*] Keeping existing .env.{RESET}")
            return

    env_content = []

    print("\n--- Configuration Mode ---")
    print("1. Mock / Demo Mode (No API keys required)")
    print("2. Production Mode (Requires OpenAI/Neo4j keys)")
    choice = input("Select mode [1]: ").strip() or "1"

    if choice == "1":
        print(f"{GREEN}[*] Configuring for Mock Mode.{RESET}")
        env_content = [
            "# Adam System - Mock Mode Configuration",
            "FLASK_CONFIG=development",
            "CORE_INTEGRATION=False",  # Disable heavy core integration for pure UI demo
            "SECRET_KEY=dev-secret-key-123",
            "CORS_ALLOWED_ORIGINS=*",
            "NEO4J_URI=bolt://localhost:7687", # Still useful if local neo4j exists
            "NEO4J_USER=neo4j",
            "NEO4J_PASSWORD=password"
        ]
    else:
        print(f"{GREEN}[*] Configuring for Production.{RESET}")
        openai_key = input("Enter OpenAI API Key: ").strip()
        neo4j_uri = input("Enter Neo4j URI [bolt://localhost:7687]: ").strip() or "bolt://localhost:7687"
        neo4j_user = input("Enter Neo4j User [neo4j]: ").strip() or "neo4j"
        neo4j_pass = input("Enter Neo4j Password: ").strip()

        env_content = [
            "# Adam System - Production Configuration",
            "FLASK_CONFIG=production",
            "CORE_INTEGRATION=True",
            f"OPENAI_API_KEY={openai_key}",
            f"NEO4J_URI={neo4j_uri}",
            f"NEO4J_USER={neo4j_user}",
            f"NEO4J_PASSWORD={neo4j_pass}",
            "SECRET_KEY=prod-secret-key-change-me",
            "CORS_ALLOWED_ORIGINS=*"
        ]

    with open(".env", "w") as f:
        f.write("\n".join(env_content))
    print(f"{GREEN}[✓] .env file created.{RESET}")

def install_dependencies():
    print(f"\n{CYAN}[*] Installing Python Dependencies...{RESET}")
    confirm = input("Run pip install? (Y/n): ").lower() or 'y'
    if confirm == 'y':
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print(f"{GREEN}[✓] Dependencies installed.{RESET}")
        except subprocess.CalledProcessError:
            print(f"{RED}[!] Failed to install dependencies.{RESET}")

def main():
    print_banner()
    check_python()
    has_docker = check_docker()
    has_node = check_node()

    setup_env()
    install_dependencies()

    print(f"\n{CYAN}" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60 + f"{RESET}")
    print("\nTo start the system:")

    if has_docker:
        print(f"  {GREEN}docker-compose up -d{RESET}  (Recommended)")
    else:
        print(f"  {GREEN}./run_adam.sh{RESET}       (Local fallback)")

    print("\nAccess the UI at: http://localhost:80 (Docker) or http://localhost:3000 (Local)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{RED}[!] Setup cancelled.{RESET}")
        sys.exit(0)
