import os
import sys
import shutil
import argparse
import subprocess
import json

# Configuration
MAX_FILE_SIZE_MB = 1.0
BINARY_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.webp',
    '.pdf', '.zip', '.tar', '.gz', '.7z', '.rar',
    '.parquet', '.h5', '.pkl', '.pt', '.pth', '.onnx', '.bin',
    '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe',
    '.db', '.sqlite', '.sqlite3',
    '.eot', '.ttf', '.woff', '.woff2'
}

def clean_repo():
    """Cleans temporary files and build artifacts."""
    print("Cleaning repository...")
    patterns_to_remove = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "htmlcov",
        "target", # Rust
        "node_modules", # Node
        "dist",
        "build",
        ".coverage",
        ".DS_Store"
    ]

    # Remove directories
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d in patterns_to_remove or d.endswith(".egg-info"):
                path = os.path.join(root, d)
                print(f"Removing directory: {path}")
                shutil.rmtree(path, ignore_errors=True)

        # Remove files
        for f in files:
            if f.endswith(".pyc") or f.endswith(".pyo") or f.endswith(".pyd") or f == ".DS_Store":
                path = os.path.join(root, f)
                # print(f"Removing file: {path}") # Too verbose
                os.remove(path)

def is_binary(filepath):
    """Checks if a file is binary based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in BINARY_EXTENSIONS

def get_tracked_files():
    """Returns a list of files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            check=True
        )
        files = result.stdout.splitlines()
        return files
    except subprocess.CalledProcessError:
        print("Warning: git ls-files failed. Falling back to os.walk (might include ignored files).")
        files = []
        for root, _, filenames in os.walk("."):
            if ".git" in root: continue
            for f in filenames:
                files.append(os.path.join(root, f))
        return files

def ingest_repo(output_file):
    """Ingests the repository into a JSONL file."""
    print(f"Ingesting repository to {output_file}...")

    files = get_tracked_files()
    count = 0
    skipped = 0

    with open(output_file, 'w', encoding='utf-8') as out:
        for filepath in files:
            if not os.path.exists(filepath):
                continue

            if os.path.isdir(filepath):
                continue

            if is_binary(filepath):
                # print(f"Skipping binary file: {filepath}")
                skipped += 1
                continue

            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                if size_mb > MAX_FILE_SIZE_MB:
                    print(f"Skipping large file ({size_mb:.2f} MB): {filepath}")
                    skipped += 1
                    continue

                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                record = {
                    "path": filepath,
                    "content": content
                }
                out.write(json.dumps(record) + "\n")
                count += 1

            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                skipped += 1

    print(f"Ingestion complete. {count} files ingested, {skipped} skipped.")

def main():
    parser = argparse.ArgumentParser(description="Clean and ingest repository.")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts before ingestion.")
    parser.add_argument("--output", default="adam_seed_data.jsonl", help="Output JSONL file.")

    args = parser.parse_args()

    if args.clean:
        clean_repo()

    ingest_repo(args.output)

if __name__ == "__main__":
    main()
