import os
import ast
import json

def generate_showcase(root_dir):
    """
    Recursively traverses the repo, parses code ASTs, and generates
    Cyber-Minimalist HTML dashboards.
    """
    report = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories and known artifacts
        # Fix: Allow '.' as current directory component
        if any(d.startswith('.') and d != '.' and d != './' for d in dirpath.split(os.sep)):
            continue

        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r') as f:
                        tree = ast.parse(f.read())
                        docstring = ast.get_docstring(tree)
                        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                        report[filepath] = {
                            "docstring": docstring,
                            "classes": classes
                        }
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")

    # Generate HTML
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Adam v23 Mission Control</title>
        <style>
            body { background-color: #050b14; color: #06b6d4; font-family: monospace; }
            .module { border: 1px solid #06b6d4; margin: 10px; padding: 10px; }
            h1 { text-align: center; }
        </style>
    </head>
    <body>
        <h1>Adam v23 Mission Control</h1>
    """

    for path, data in report.items():
        html_content += f"""
        <div class="module">
            <h3>{path}</h3>
            <p>{data['docstring'] or 'No docstring'}</p>
            <ul>
                {''.join(f'<li>{c}</li>' for c in data['classes'])}
            </ul>
        </div>
        """

    html_content += "</body></html>"

    with open("showcase/mission_control.html", "w") as f:
        f.write(html_content)

    print("Mission Control generated at showcase/mission_control.html")

if __name__ == "__main__":
    if not os.path.exists("showcase"):
        os.makedirs("showcase")
    generate_showcase("./core")
