import sys
import ast
import astor

def update_file(filepath):
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    has_import = any(
        isinstance(node, ast.ImportFrom) and node.module == 'src.governance.gatekeeper' and any(n.name == 'GovernanceGatekeeper' for n in node.names)
        for node in tree.body
    )
    if not has_import:
        import_node = ast.ImportFrom(
            module='src.governance.gatekeeper',
            names=[
                ast.alias(name='GovernanceGatekeeper', asname=None),
                ast.alias(name='GovernanceError', asname=None)
            ],
            level=0
        )
        tree.body.insert(0, import_node)

    class Transformer(ast.NodeTransformer):
        def visit_ClassDef(self, node):
            if node.name == "OrchestratorEngine":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                        init_stmt = ast.parse('self.gatekeeper = GovernanceGatekeeper(schema={"type": "object"})').body[0]
                        child.body.append(init_stmt)

                    if isinstance(child, ast.AsyncFunctionDef) and child.name == "_execute_with_retry":
                        for b_node in child.body:
                            if isinstance(b_node, ast.While):
                                for try_node in b_node.body:
                                    if isinstance(try_node, ast.Try):
                                        new_try_body = []
                                        for stmt in try_node.body:
                                            new_try_body.append(stmt)
                                            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "result":
                                                gate_check_str = """
if isinstance(result, dict) and 'provenance_trace' in result:
    try:
        result = self.gatekeeper.exit_gate(result)
    except GovernanceError as e:
        raise Exception(f'Governance validation failed: {str(e)}')
"""
                                                gate_check = ast.parse(gate_check_str)
                                                new_try_body.extend(gate_check.body)
                                        try_node.body = new_try_body
            self.generic_visit(node)
            return node

    tree = Transformer().visit(tree)

    with open(filepath, "w") as f:
        f.write(astor.to_source(tree))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        update_file(sys.argv[1])
