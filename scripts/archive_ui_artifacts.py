import os
import shutil
import json
from pathlib import Path

def archive_ui_artifacts():
    repo_root = Path(".")
    archive_dir = repo_root / "docs" / "ui_archive_v1"

    print(f"Cleaning archive directory: {archive_dir}")
    # Clean existing archive
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    files_to_archive = []

    print("Scanning for HTML files...")
    # Walk the repo
    for root, dirs, files in os.walk(repo_root):
        # Modifying dirs in-place to prune traversal
        if ".git" in dirs:
            dirs.remove(".git")
        if "node_modules" in dirs:
            dirs.remove("node_modules")
        if ".venv" in dirs:
            dirs.remove(".venv")
        if "venv" in dirs:
            dirs.remove("venv")
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")

        root_path = Path(root)

        # Skip the archive directory itself
        if root_path.resolve() == archive_dir.resolve():
            continue

        for file in files:
            if file.endswith(".html"):
                src_path = root_path / file
                # Double check we aren't in the archive dir
                if archive_dir.resolve() in src_path.resolve().parents:
                    continue

                files_to_archive.append(src_path)

    print(f"Found {len(files_to_archive)} HTML files.")

    for src_path in files_to_archive:
        rel_path = src_path.relative_to(repo_root)

        # Flatten name: replace path separators with underscores
        flat_name = str(rel_path).replace(os.sep, "_")

        # Rename root index.html to avoid overwriting the gallery index
        if flat_name == "index.html":
            flat_name = "root_index.html"

        dest_path = archive_dir / flat_name

        try:
            shutil.copy2(src_path, dest_path)
            manifest.append({
                "original_path": str(rel_path),
                "archived_path": flat_name # relative to archive dir
            })
            print(f"Archived: {rel_path} -> {flat_name}")
        except Exception as e:
            print(f"Error archiving {src_path}: {e}")

    # Write manifest
    manifest_path = archive_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"files": manifest}, f, indent=2)

    # Generate Gallery Index
    gallery_path = archive_dir / "index.html"
    with open(gallery_path, "w") as f:
        f.write("<!DOCTYPE html><html><head><title>UI Archive v1</title>")
        f.write("<style>body{font-family:sans-serif;background:#0f172a;color:#f8fafc;padding:20px;} a{color:#3b82f6;text-decoration:none;} a:hover{text-decoration:underline;} ul{list-style:none;padding:0;} li{margin:5px 0;padding:10px;background:#1e293b;border-radius:5px;}</style>")
        f.write("</head><body>")
        f.write("<h1>UI Archive v1</h1>")
        f.write(f"<p>Preserved snapshot of {len(manifest)} static HTML files.</p>")
        f.write("<ul>")
        # Sort manifest by original path
        manifest.sort(key=lambda x: x["original_path"])
        for item in manifest:
            f.write(f'<li><a href="{item["archived_path"]}">{item["original_path"]}</a></li>')
        f.write("</ul></body></html>")

    print(f"Archive complete. Manifest written to {manifest_path}")
    print(f"Gallery index written to {gallery_path}")

if __name__ == "__main__":
    archive_ui_artifacts()
