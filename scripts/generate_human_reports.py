import os
import glob
import ast

def generate_reports():
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Architecture Overview Report
    arch_html = """<!DOCTYPE html>
<html>
<head>
    <title>Adam OS - Agentic Architecture Overview</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #333; }
        .component { border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        .layer { background-color: #f9f9f9; padding: 10px; margin-top: 10px; border-left: 4px solid #0066cc; }
    </style>
</head>
<body>
    <h1>Adam OS: Agentic AI System Overview</h1>
    <p>This report explains the core components of the Adam OS platform in simple terms.</p>

    <div class="component">
        <h2>1. The Multi-Agent System (LangGraph)</h2>
        <p>Instead of relying on a single AI to do everything, Adam uses a "team" of specialized AI agents working together.</p>
        <div class="layer">
            <strong>The Supervisor:</strong> The manager agent that assigns tasks to other agents.
        </div>
        <div class="layer">
            <strong>Quant Agent:</strong> Analyzes numerical data, financial statements, and calculations.
        </div>
        <div class="layer">
            <strong>Legal Agent:</strong> Reads text documents, contracts, and checks for rules (covenants).
        </div>
        <div class="layer">
            <strong>Market Agent:</strong> Searches for real-time news and broader economic context.
        </div>
    </div>

    <div class="component">
        <h2>2. Glass Box Architecture</h2>
        <p>Unlike "Black Box" AI where decisions are hidden, Adam is a "Glass Box". Every step, calculation, and decision is recorded and can be explained.</p>
        <p>We use strict data formats (Pydantic schemas) to ensure the AI always outputs numbers when it should, instead of vague text.</p>
    </div>

    <div class="component">
        <h2>3. Human-in-the-Loop (HITL)</h2>
        <p>The AI acts as an assistant (co-pilot), not a replacement. For important decisions, a human must review and approve the AI's work before anything is finalized.</p>
    </div>
</body>
</html>
"""
    with open(os.path.join(reports_dir, "architecture_overview.html"), "w") as f:
        f.write(arch_html)

    print("Generated reports/architecture_overview.html")

    # 2. Dynamic Codebase Report
    modules_to_scan = ["core/vertical_risk_agent/agents", "src/pdil"]

    codebase_html = """<!DOCTYPE html>
<html>
<head>
    <title>Adam OS - Codebase Capabilities Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #333; }
        .file-section { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        .class-section { background-color: #f0f8ff; padding: 10px; margin-top: 10px; border-left: 4px solid #4682b4; }
        .func-section { margin-left: 20px; color: #555; }
    </style>
</head>
<body>
    <h1>Codebase Capabilities Report</h1>
    <p>A non-technical breakdown of the functions and capabilities of key files in the repository.</p>
"""
    for folder in modules_to_scan:
        codebase_html += f"<h2>Module: {folder}</h2>\n"
        py_files = glob.glob(f"{folder}/**/*.py", recursive=True)
        for filepath in py_files:
            if "__init__" in filepath:
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                tree = ast.parse(content)
            except Exception:
                continue

            file_doc = ast.get_docstring(tree)
            file_doc_str = f"<p><em>{file_doc}</em></p>" if file_doc else "<p><em>No general description available.</em></p>"

            codebase_html += f"<div class='file-section'><h3>File: {filepath}</h3>{file_doc_str}"

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node)
                    class_doc_str = class_doc if class_doc else "A specialized component."
                    codebase_html += f"<div class='class-section'><strong>Component: {node.name}</strong><p>{class_doc_str}</p>"

                    for sub_node in node.body:
                        if isinstance(sub_node, ast.FunctionDef) and not sub_node.name.startswith('_'):
                            func_doc = ast.get_docstring(sub_node)
                            func_doc_str = func_doc if func_doc else "Performs a specific task."
                            codebase_html += f"<div class='func-section'>- <strong>Action: {sub_node.name}</strong>: {func_doc_str}</div>"

                    codebase_html += "</div>"
            codebase_html += "</div>"

    codebase_html += "</body></html>"

    with open(os.path.join(reports_dir, "codebase_capabilities.html"), "w") as f:
        f.write(codebase_html)

    print("Generated reports/codebase_capabilities.html")

if __name__ == "__main__":
    generate_reports()
