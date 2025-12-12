import os

ROOT_DIR = "."
NAV_JS_PATH = "showcase/js/nav.js"

# Whitelist approach is safer given the limit
TARGET_DIRS = [
    ".",
    "showcase",
    "prompts",
    "core/agents",
    "core/engine",
    "financial_digital_twin"
]

EXCLUDES = [
    "docs/ui_archive_v1",
    "core/newsletter_layout/templates",
    ".git",
    "node_modules",
    "build",
    "dist",
    "site-packages",
    "tinker_lab", # Skip training docs for now
    "artifacts",  # Skip DB seeds
    "downloads",
    "services",   # Skip React app and backend templates
    "webapp",     # Skip mockups for now
    "logs",
    "tests"
]

def get_relative_root(file_path):
    rel_path = os.path.relpath(file_path, ROOT_DIR)
    parts = rel_path.split(os.sep)
    depth = len(parts) - 1
    if depth == 0:
        return "."
    return "/".join([".."] * depth)

def process_file(file_path):
    # Double check excludes
    for ex in EXCLUDES:
        if ex in file_path:
            return

    rel_root = get_relative_root(file_path)

    if rel_root == ".":
        nav_src = NAV_JS_PATH
    else:
        nav_src = f"{rel_root}/{NAV_JS_PATH}"

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if "showcase/js/nav.js" in content and "data-root" in content:
        print(f"Skipping {file_path} (already has nav)")
        return

    script_tag = f'\n    <!-- Adam Global Nav -->\n    <script src="{nav_src}" data-root="{rel_root}"></script>\n'

    if "</body>" in content:
        new_content = content.replace("</body>", script_tag + "</body>")
    else:
        new_content = content + script_tag

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Injected nav into {file_path}")
    except Exception as e:
        print(f"Error writing {file_path}: {e}")

# Walk
for root, dirs, files in os.walk(ROOT_DIR):
    dirs[:] = [d for d in dirs if not d.startswith('.')]

    # Check if root is interesting
    is_target = False
    for target in TARGET_DIRS:
        if root == "." or root.startswith(target) or root.startswith("./"+target):
            is_target = True
            break

    if not is_target:
        continue

    for file in files:
        if file.endswith(".html"):
            process_file(os.path.join(root, file))
