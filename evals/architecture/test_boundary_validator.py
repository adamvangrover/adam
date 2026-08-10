import os

def test_no_financial_math_in_agents():
    # Simple heuristic check: prevent raw math libraries in agent dir
    forbidden_imports = ["import numpy", "import scipy"]

    agent_dir = "agents"
    if os.path.exists(agent_dir):
        for root, dirs, files in os.walk(agent_dir):
            for file in files:
                if file.endswith(".py"):
                    with open(os.path.join(root, file)) as f:
                        content = f.read()
                        for imp in forbidden_imports:
                            # Not a perfect AST check, but enforces the pattern
                            assert imp not in content, f"Found '{imp}' in {file}. Financial math belongs in rust_ext or core."
