#!/usr/bin/env python3
"""
Repo Chat Tool - v1.0
---------------------
A portable, modular runtime setup to:
1. Assess the environment (rate limits, token usage, available LLM endpoints/models).
2. Pack the entire repository into a prompt context.
3. Run an interactive chat loop with the codebase context.

Usage:
    python scripts/repo_chat.py [--assess] [--dry-run] [--model MODEL] [--max-tokens LIMIT]
"""

import os
import sys
import json
import time
import argparse
import subprocess
import platform
import socket
from typing import List, Dict, Optional, Tuple, Any

# Conditional Imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AssessEnvironment:
    """Assess the runtime environment for LLM capabilities."""

    def __init__(self):
        self.api_keys = {}
        self.local_models = []
        self.rate_limits = {}

    def check_api_keys(self):
        """Check for known API keys in environment variables."""
        keys_to_check = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "MISTRAL_API_KEY"
        ]
        print(f"[*] Checking for API Keys...")
        found_any = False
        for key in keys_to_check:
            val = os.environ.get(key)
            if val:
                masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "****"
                print(f"    [+] Found {key}: {masked}")
                self.api_keys[key] = True
                found_any = True
            else:
                self.api_keys[key] = False

        if not found_any:
            print("    [!] No API keys found. Functionality may be limited to local models.")
        print("")

    def check_local_endpoints(self):
        """Check for local LLM endpoints (e.g., Ollama, LM Studio)."""
        print(f"[*] Checking Local LLM Endpoints...")

        # Ollama Default Port
        ollama_url = "http://localhost:11434/api/tags"
        if REQUESTS_AVAILABLE:
            try:
                response = requests.get(ollama_url, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    print(f"    [+] Ollama running on port 11434. Models: {', '.join(models)}")
                    self.local_models.extend(models)
                else:
                    print(f"    [-] Ollama returned status {response.status_code}")
            except requests.exceptions.RequestException:
                print(f"    [-] Ollama not detected on port 11434")
        else:
            print(f"    [!] 'requests' library not installed. Skipping local endpoint check.")
        print("")

    def estimate_rate_limits(self):
        """Estimate rate limits based on detected keys."""
        print(f"[*] Estimating Rate Limits (Heuristic)...")
        if self.api_keys.get("OPENAI_API_KEY"):
            print("    [OpenAI] Tier 1: ~10k TPM / 500 RPM (Default)")
            print("    [OpenAI] Tier 2+: ~30k+ TPM")

        if self.api_keys.get("ANTHROPIC_API_KEY"):
            print("    [Anthropic] Tier 1: ~20k TPM / 5 RPM")
            print("    [Anthropic] Tier 2: ~40k TPM / 1000 RPM")

        if self.local_models:
            print("    [Local] Rate Limit: Infinite (Hardware Dependent)")
        print("")

    def run(self):
        print("=== Environment Assessment ===")
        print(f"Python Version: {sys.version.split()[0]}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Libraries: requests={REQUESTS_AVAILABLE}, tiktoken={TIKTOKEN_AVAILABLE}, openai={OPENAI_AVAILABLE}, anthropic={ANTHROPIC_AVAILABLE}")
        print("-" * 30)
        self.check_api_keys()
        self.check_local_endpoints()
        self.estimate_rate_limits()
        print("==============================")


class RepoIngestor:
    """Walks the repository and consolidates files into a context string."""

    IGNORE_DIRS = {
        ".git", ".svn", ".hg", "node_modules", "venv", ".venv", "env", ".env",
        "__pycache__", ".pytest_cache", ".mypy_cache", "dist", "build", "target",
        "htmlcov", "site-packages", "gems", "vendor", ".vite", "exports", "downloads",
        "verification_screenshots", "verification_images", "showcase/data" # Explicitly ignore large data dirs
    }

    IGNORE_EXTS = {
        # Images
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".bmp", ".tiff",
        # Archives
        ".zip", ".tar", ".gz", ".7z", ".rar", ".jar",
        # Binaries / Compiled
        ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin", ".class", ".o", ".obj",
        # Database / Big Data
        ".db", ".sqlite", ".sqlite3", ".parquet", ".h5", ".pkl", ".pt", ".pth", ".onnx",
        # Fonts
        ".eot", ".ttf", ".woff", ".woff2",
        # Misc
        ".map", ".lock", ".log", ".jsonl" # skip jsonl generally as they are often data
    }

    IGNORE_FILES = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock",
        ".DS_Store"
    }

    def __init__(self, root_dir: str = ".", max_tokens: int = 128000):
        self.root_dir = root_dir
        self.max_tokens = max_tokens
        self.files_processed = 0
        self.files_skipped = 0
        self.total_tokens = 0
        self.context_string = ""
        self.file_map = [] # List of tuples (path, tokens)

    def is_ignored(self, filepath: str) -> bool:
        """Check if a file should be ignored."""
        parts = filepath.split(os.sep)
        filename = os.path.basename(filepath)

        # Check directories
        for part in parts:
            if part in self.IGNORE_DIRS:
                return True

        # Check filename
        if filename in self.IGNORE_FILES:
            return True

        # Check extensions
        _, ext = os.path.splitext(filename)
        if ext.lower() in self.IGNORE_EXTS:
            return True

        return False

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken or heuristic."""
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                pass
        # Fallback: 1 token ~= 4 chars
        return len(text) // 4

    def ingest(self):
        """Walk directory and build context."""
        print(f"[*] Ingesting Repository from {self.root_dir}...")
        if self.max_tokens > 0:
            print(f"    [!] Max Token Limit: {self.max_tokens:,}")

        buffer = []

        # Prioritize key directories for smaller context?
        # For now, just walk top-down.

        for root, dirs, files in os.walk(self.root_dir):
            # Modify dirs in-place to skip ignored directories during walk
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]

            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, self.root_dir)

                if self.is_ignored(rel_path):
                    self.files_skipped += 1
                    continue

                try:
                    # Check size (skip > 500KB to be safer)
                    if os.path.getsize(filepath) > 500 * 1024:
                        self.files_skipped += 1
                        continue

                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Basic XML-style wrapping for clear context separation
                    file_block = f"<file path=\"{rel_path}\">\n{content}\n</file>\n\n"
                    tokens = self.count_tokens(file_block)

                    if self.max_tokens > 0 and (self.total_tokens + tokens) > self.max_tokens:
                        print(f"    [!] Token limit reached at {rel_path}. Stopping ingestion.")
                        self.context_string = "".join(buffer)
                        return

                    self.total_tokens += tokens
                    self.files_processed += 1
                    self.file_map.append((rel_path, tokens))
                    buffer.append(file_block)

                except Exception as e:
                    self.files_skipped += 1

        self.context_string = "".join(buffer)
        print(f"    [+] Processed {self.files_processed} files.")
        print(f"    [+] Skipped {self.files_skipped} files/directories.")
        print(f"    [+] Total estimated tokens: {self.total_tokens:,}")

    def get_stats(self):
        return {
            "files": self.files_processed,
            "tokens": self.total_tokens,
            "skipped": self.files_skipped
        }


class LLMClient:
    """Unified client for interacting with LLMs."""

    def __init__(self, model: str):
        self.model = model
        self.client_type = self._determine_client_type(model)
        self.client = None
        self._init_client()

    def _determine_client_type(self, model: str) -> str:
        if model.startswith("gpt"):
            return "openai"
        elif model.startswith("claude"):
            return "anthropic"
        else:
            return "ollama" # Assume local if not standard cloud

    def _init_client(self):
        if self.client_type == "openai" and OPENAI_AVAILABLE:
            self.client = openai.OpenAI()
        elif self.client_type == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic()
        elif self.client_type == "ollama" and not REQUESTS_AVAILABLE:
            print("[!] Error: 'requests' library required for Ollama.")

    def chat(self, system_prompt: str, user_prompt: str):
        """Send chat request and stream response."""

        print(f"\n[AI - {self.model}]: ", end="", flush=True)

        try:
            if self.client_type == "openai" and self.client:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                print("") # Newline

            elif self.client_type == "anthropic" and self.client:
                # Anthropic doesn't support 'system' role in messages list the same way, usually it's a separate param
                stream = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True
                )
                for chunk in stream:
                    if chunk.type == "content_block_delta":
                        print(chunk.delta.text, end="", flush=True)
                print("")

            elif self.client_type == "ollama" and REQUESTS_AVAILABLE:
                # Local Ollama via requests
                url = "http://localhost:11434/api/chat"
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": True
                }

                with requests.post(url, json=payload, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            try:
                                json_line = json.loads(line.decode('utf-8'))
                                content = json_line.get("message", {}).get("content", "")
                                print(content, end="", flush=True)
                                if json_line.get("done"):
                                    break
                            except:
                                pass
                print("")
            else:
                print(f"\n[!] Error: Client for {self.client_type} not initialized properly (missing keys or libraries).")

        except Exception as e:
            print(f"\n[!] Error during chat: {e}")


def main():
    parser = argparse.ArgumentParser(description="Repo Chat & Environment Assessment Tool")
    parser.add_argument("--assess", action="store_true", help="Run environment assessment only.")
    parser.add_argument("--dry-run", action="store_true", help="Ingest repo and count tokens without calling LLM.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use (default: gpt-4o).")
    parser.add_argument("--max-tokens", type=int, default=128000, help="Max tokens to ingest (default: 128,000). Set to 0 for unlimited.")

    args = parser.parse_args()

    assessor = AssessEnvironment()

    if args.assess:
        assessor.run()
        return

    # Ingest Repo
    print(f"[*] Initializing Repo Chat with model: {args.model}")
    ingestor = RepoIngestor(max_tokens=args.max_tokens)
    ingestor.ingest()

    if args.dry_run:
        print("\n=== Dry Run Stats ===")
        print(f"Files: {ingestor.files_processed}")
        print(f"Tokens: {ingestor.total_tokens:,}")
        print("Top 10 Largest Files (by token count):")
        sorted_files = sorted(ingestor.file_map, key=lambda x: x[1], reverse=True)[:10]
        for f, t in sorted_files:
            print(f"  - {f}: {t:,} tokens")
        return

    # Chat Loop
    client = LLMClient(args.model)

    system_prompt = f"""You are an expert software engineer and architect.
You have been provided with the context of the current repository below.
Answer the user's questions based on this codebase.

=== REPOSITORY CONTEXT START ===
{ingestor.context_string}
=== REPOSITORY CONTEXT END ===
"""

    print("\n" + "="*40)
    print("Repo Chat Ready! Type 'exit' to quit.")
    print("="*40)

    while True:
        try:
            user_input = input("\n[User]: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            client.chat(system_prompt, user_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
