import os
import json
import time

def generate_repo_index(root_dir="."):
    index = {
        "generated_at": time.time(),
        "files": [],
        "directories": []
    }

    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.pytest_cache', 'venv', 'env', 'dist', 'build'}
    exclude_files = {'.DS_Store'}

    for root, dirs, files in os.walk(root_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        rel_root = os.path.relpath(root, root_dir)
        if rel_root == ".":
            rel_root = ""

        if rel_root:
            index["directories"].append(rel_root)

        for f in files:
            if f in exclude_files:
                continue

            path = os.path.join(rel_root, f)
            full_path = os.path.join(root, f)

            try:
                size = os.path.getsize(full_path)
                ext = os.path.splitext(f)[1].lower()

                # Simple type inference
                ftype = "text"
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                    ftype = "image"
                elif ext in ['.json', '.js', '.py', '.html', '.css', '.md', '.txt']:
                    ftype = "code"

                index["files"].append({
                    "path": path,
                    "name": f,
                    "size": size,
                    "type": ftype,
                    "extension": ext
                })
            except OSError:
                pass # Skip files we can't read

    return index

if __name__ == "__main__":
    data = generate_repo_index()

    # Ensure directory exists
    os.makedirs("showcase/data", exist_ok=True)

    output_path = "showcase/data/repo_full_index.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Repository index generated at {output_path} with {len(data['files'])} files.")
