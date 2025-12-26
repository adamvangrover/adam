#!/usr/bin/env python3
"""
Interactive Setup Script for Adam v2.0
Helps users configure the environment, API keys, and Docker settings.
"""

import os
import sys
import getpass
import shutil

def print_header():
    print("=" * 60)
    print("   ADAM v2.0 - Interactive Setup")
    print("   Transforming Static Data into Agentic Intelligence")
    print("=" * 60)
    print()

def check_file_exists(filepath):
    return os.path.exists(filepath)

def setup_env_file():
    print("[*] Configuring Environment Variables (.env)...")

    if check_file_exists(".env"):
        overwrite = input("    .env file already exists. Overwrite? (y/N): ").lower()
        if overwrite != 'y':
            print("    Skipping .env creation.")
            return

    openai_key = getpass.getpass("    Enter OpenAI API Key (leave blank for Mock Mode): ")

    env_content = "# Adam v2.0 Configuration\n"
    if openai_key:
        env_content += f"OPENAI_API_KEY={openai_key}\n"
        print("    > OpenAI Key set.")
    else:
        env_content += "OPENAI_API_KEY=mock\n"
        env_content += "ADAM_MOCK_MODE=true\n"
        print("    > Mock Mode enabled.")

    # Default settings
    env_content += "LOG_LEVEL=INFO\n"
    env_content += "DATABASE_URL=sqlite:///finance_data.db\n"

    with open(".env", "w") as f:
        f.write(env_content)

    print("    .env file created successfully.")

def setup_docker():
    print("\n[*] Checking Docker Configuration...")
    if not check_file_exists("Dockerfile"):
        print("    Error: Dockerfile not found in root.")
        return

    print("    Docker configuration appears valid.")
    # In a real scenario, we might check if docker daemon is running
    # but strictly python logic here is safer.

def create_startup_script():
    print("\n[*] Creating 'adam' CLI alias...")
    # This is symbolic, creates a helper script
    with open("adam", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("python3 scripts/run_adam.py \"$@\"\n")

    os.chmod("adam", 0o755)
    print("    Created './adam' script. You can run './adam --help'")

def main():
    print_header()

    try:
        setup_env_file()
        setup_docker()
        create_startup_script()

        print("\n" + "=" * 60)
        print("   Setup Complete!")
        print("   Run './adam' to start the system.")
        print("   Run './scripts/run_ui.sh' to start the Web App.")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nSetup aborted.")
        sys.exit(1)

if __name__ == "__main__":
    main()
