"""
AFOS v30.1 — Adversarial Security Regression Suite for src/antigravity_ide/server.py
Covers CodeQL py/path-injection vectors, URL double-encoding, sibling prefix traps, and symlinks.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.antigravity_ide.server import (
    JSONLOGIC_VERSION,
    app,
    resolve_confined_workspace_path,
)


@pytest.fixture()
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "notes.md").write_text("AFOS v30.1 Verified Content", encoding="utf-8")

    # Create a sibling directory with matching prefix to test prefix-collision bypasses
    sibling_dir = tmp_path / "workspace_secret"
    sibling_dir.mkdir(parents=True, exist_ok=True)
    (sibling_dir / "credentials.json").write_text('{"secret": "forbidden"}', encoding="utf-8")

    monkeypatch.setenv("ADAM_WORKSPACE_ROOT", str(workspace_dir))
    return workspace_dir


@pytest.fixture()
def client(isolated_workspace: Path) -> TestClient:
    return TestClient(app)


@pytest.mark.security
def test_happy_path_read_and_prov_o_audit(client: TestClient) -> None:
    response = client.post("/api/v1/ide/read", json={"relative_path": "notes.md"})
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "AFOS v30.1 Verified Content"
    assert data["jsonLogic_version"] == JSONLOGIC_VERSION
    assert len(data["prov_o_audit_trail"]) == 1
    audit = data["prov_o_audit_trail"][0]
    assert audit["@type"] == "prov:Activity"
    assert audit["prov:wasAssociatedWith"] == "urn:adam:agent:AFOS-SNT-01"


@pytest.mark.security
def test_happy_path_write_and_list(client: TestClient, isolated_workspace: Path) -> None:
    write_resp = client.post(
        "/api/v1/ide/write",
        json={
            "relative_path": "subdir/model_config.json",
            "content": '{"tier": "PathA"}',
            "create_parents": True,
        },
    )
    assert write_resp.status_code == 200
    assert (isolated_workspace / "subdir" / "model_config.json").read_text(encoding="utf-8") == '{"tier": "PathA"}'

    list_resp = client.post("/api/v1/ide/list", json={"relative_path": "subdir"})
    assert list_resp.status_code == 200
    assert "model_config.json" in list_resp.json()["entries"]


@pytest.mark.security
@pytest.mark.parametrize(
    "malicious_path",
    [
        "../workspace_secret/credentials.json",
        "../../etc/passwd",
        "subdir/../../etc/passwd",
        "%2e%2e/%2e%2e/etc/passwd",
        "%252e%252e%252fworkspace_secret%252fcredentials.json",
        "/etc/passwd",
        "\\Windows\\System32\\drivers\\etc\\hosts",
        "C:\\Windows\\win.ini",
        "notes.md\x00.txt",
        "notes.md%00.txt",
    ],
)
def test_adversarial_traversal_vectors_blocked(
    client: TestClient,
    isolated_workspace: Path,
    malicious_path: str,
) -> None:
    # Direct function guard check
    with pytest.raises((HTTPException, ValueError)):
        resolve_confined_workspace_path(malicious_path, allowed_root=isolated_workspace)

    # HTTP endpoint check (must reject with 403 Forbidden or 422 Unprocessable Entity)
    response = client.post("/api/v1/ide/read", json={"relative_path": malicious_path})
    assert response.status_code in (403, 422)


@pytest.mark.security
def test_symlink_escape_blocked(client: TestClient, isolated_workspace: Path, tmp_path: Path) -> None:
    outside_secret = tmp_path / "workspace_secret" / "credentials.json"
    symlink_file = isolated_workspace / "escaped_link.json"
    symlink_file.symlink_to(outside_secret)

    response = client.post("/api/v1/ide/read", json={"relative_path": "escaped_link.json"})
    assert response.status_code == 403
    assert "AFOS-SNT-01" in response.json()["detail"]
