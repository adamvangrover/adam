import os
import json
import time
from datetime import datetime

# Configuration
ROOT_DIR = "."
OUTPUT_FILE = "showcase/data/filesystem_manifest.json"
OUTPUT_JS_FILE = "showcase/data/filesystem_manifest.js"
EXCLUDED_DIRS = {'.git', '.idea', '.vscode', '__pycache__', 'node_modules', 'venv', '.jules', '.devcontainer', '.github'}
EXCLUDED_FILES = {'.DS_Store'}

def get_file_info(path):
    """
    Get file metadata.
    """
    try:
        stats = os.stat(path)
        return {
            "name": os.path.basename(path),
            "type": "file",
            "path": path,
            "size": stats.st_size,
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
        }
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return None

def build_tree(directory):
    """
    Recursively build the file system tree.
    """
    tree = []
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return []

    for entry in entries:
        full_path = os.path.join(directory, entry)

        # Skip excluded directories and files
        if entry in EXCLUDED_DIRS or entry in EXCLUDED_FILES:
            continue
        if entry.startswith('.') and entry != '.gitignore': # Skip hidden files generally
            continue

        if os.path.isdir(full_path):
            node = {
                "name": entry,
                "type": "directory",
                "path": full_path,
                "children": build_tree(full_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
            }
            tree.append(node)
        else:
            info = get_file_info(full_path)
            if info:
                tree.append(info)

    return tree

def main():
    print(f"Generating filesystem manifest from {os.path.abspath(ROOT_DIR)}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Build the tree
    manifest = build_tree(ROOT_DIR)

    # Write to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest generated at {OUTPUT_FILE}")

    # Write to JS (Global Variable) for offline/local support
    with open(OUTPUT_JS_FILE, 'w') as f:
        json_str = json.dumps(manifest, indent=2)
        f.write(f"window.FILESYSTEM_MANIFEST = {json_str};")
    print(f"JS Manifest generated at {OUTPUT_JS_FILE}")

if __name__ == "__main__":
    main()
