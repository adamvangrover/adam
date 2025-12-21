import json
import os
import shutil
from datetime import datetime

# Configuration
SOURCE_ROOT = "."  # Root of the repo
ARCHIVE_DIR = "docs/ui_archive_v1"
MANIFEST_FILE = "manifest.json"

# Files to exclude to prevent recursion or archiving the archive itself
EXCLUDE_DIRS = {".git", ".github", ".vite", "node_modules", "__pycache__", "docs/ui_archive_v1"}

def setup_archive_dir():
    """Creates the archive directory if it doesn't exist."""
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        print(f"Created archive directory: {ARCHIVE_DIR}")
    else:
        print(f"Archive directory exists: {ARCHIVE_DIR}")

def scan_and_copy_html_files():
    """Scans for .html files and copies them to the archive with conflict resolution."""
    html_files = []
    manifest = {
        "archive_created_at": datetime.now().isoformat(),
        "files": []
    }

    print("Scanning for HTML files...")

    for root, dirs, files in os.walk(SOURCE_ROOT):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and os.path.join(root, d) != ARCHIVE_DIR]

        for file in files:
            if file.endswith(".html"):
                original_path = os.path.join(root, file)
                
                # Determine a unique name for the archive to avoid collisions
                # Strategy: replace path separators with underscores
                # e.g., services/webapp/client/public/index.html -> services_webapp_client_public_index.html
                relative_path = os.path.relpath(original_path, SOURCE_ROOT)
                clean_name = relative_path.replace(os.sep, "_").replace(" ", "_")
                
                target_path = os.path.join(ARCHIVE_DIR, clean_name)
                
                # Perform the copy
                try:
                    shutil.copy2(original_path, target_path)
                    
                    entry = {
                        "original_path": relative_path,
                        "archive_name": clean_name,
                        "archived_path": os.path.join(ARCHIVE_DIR, clean_name)
                    }
                    manifest["files"].append(entry)
                    print(f"Archived: {relative_path} -> {clean_name}")
                except Exception as e:
                    print(f"Error copying {original_path}: {e}")

    # Write Manifest
    manifest_path = os.path.join(ARCHIVE_DIR, MANIFEST_FILE)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    print("--- Archive Complete ---")
    print(f"Total files archived: {len(manifest['files'])}")
    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    setup_archive_dir()
    scan_and_copy_html_files()
