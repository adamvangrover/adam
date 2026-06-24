import os
import ast

def map_dependencies(root_dir="."):
    target_dirs = ["agents", "core", "docs", "human_gates", "logs", "servers", "src/pdil", "src/schemas", "tests"]
    print("# Architectural Dependency Map\n")
    for t_dir in target_dirs:
        if not os.path.exists(t_dir):
            continue
        print(f"## Module: {t_dir}")
        for root, _, files in os.walk(t_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        try:
                            tree = ast.parse(f.read(), filename=filepath)
                            imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
                            imports_from = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
                            all_imports = imports + imports_from
                            if all_imports:
                                print(f"- `{filepath}` depends on: {', '.join(all_imports)}")
                        except Exception:
                            pass
    print("\nMapping complete.")

if __name__ == "__main__":
    map_dependencies()
