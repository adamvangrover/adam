import os

import pytest

from core.symphony.config import SymphonyConfig
from core.symphony.workspace import WorkspaceError, WorkspaceManager


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    return SymphonyConfig({
        "workspace": {
            "root": str(root)
        },
        "hooks": {
            "after_create": "echo created > hook.txt",
            "before_run": "echo before > before.txt"
        }
    })

def test_workspace_sanitization(config):
    manager = WorkspaceManager(config)
    workspace = manager.create_for_issue("TICKET-123/bad_chars!")
    assert workspace.workspace_key == "TICKET-123_bad_chars_"
    assert workspace.path.endswith("TICKET-123_bad_chars_")
    assert workspace.created_now is True

def test_workspace_reuse(config):
    manager = WorkspaceManager(config)
    workspace1 = manager.create_for_issue("TICKET-1")
    assert workspace1.created_now is True

    workspace2 = manager.create_for_issue("TICKET-1")
    assert workspace2.created_now is False
    assert workspace1.path == workspace2.path

def test_workspace_after_create_hook(config):
    manager = WorkspaceManager(config)
    workspace = manager.create_for_issue("TICKET-HOOK")
    hook_file = os.path.join(workspace.path, "hook.txt")
    assert os.path.exists(hook_file)

    # Reusing should not run after_create
    os.remove(hook_file)
    workspace2 = manager.create_for_issue("TICKET-HOOK")
    assert workspace2.created_now is False
    assert not os.path.exists(hook_file)

def test_workspace_before_run_hook(config):
    manager = WorkspaceManager(config)
    workspace = manager.create_for_issue("TICKET-BEFORE")
    manager.run_before_run(workspace.path)

    before_file = os.path.join(workspace.path, "before.txt")
    assert os.path.exists(before_file)

def test_workspace_cleanup(config):
    manager = WorkspaceManager(config)
    workspace = manager.create_for_issue("TICKET-DEL")
    assert os.path.exists(workspace.path)

    manager.cleanup_workspace("TICKET-DEL")
    assert not os.path.exists(workspace.path)

def test_workspace_out_of_bounds(config):
    manager = WorkspaceManager(config)

    # Due to sanitization, "../../../etc" becomes "____etc" and is safe under root.
    # We can mock `_sanitize_key` to test the containment invariant.
    original_sanitize = manager._sanitize_key
    manager._sanitize_key = lambda x: x

    with pytest.raises(WorkspaceError) as exc:
        manager.create_for_issue("../../../../etc/passwd")

    assert "Workspace path must be within workspace root" in str(exc.value)

    manager._sanitize_key = original_sanitize
