"""
Integration & Unit Tests for Jules Web IDE Server.
Tests API handlers directly for maximum portability and zero test overhead.
"""

from pathlib import Path
from src.jules.ide.server import (
    ChatRequest,
    FileSaveRequest,
    TaskRequest,
    api_chat,
    api_list_files,
    api_plan,
    api_read_file,
    api_run,
    api_save_file,
    get_ide_html,
    get_status,
)


def test_ide_status_handler() -> None:
    res = get_status()
    assert res["status"] == "online"
    assert "provider" in res
    assert "tools" in res
    assert isinstance(res["tools"], list)


def test_ide_chat_handler() -> None:
    req = ChatRequest(message="hello from test", provider="mock")
    res = api_chat(req)
    assert "reply" in res
    assert "history" in res


def test_ide_plan_handler() -> None:
    req = TaskRequest(prompt="design new service", provider="mock")
    res = api_plan(req)
    assert res["prompt"] == "design new service"
    assert isinstance(res["plan"], list)


def test_ide_run_handler() -> None:
    req = TaskRequest(prompt="execute unit tests", provider="mock")
    res = api_run(req)
    assert res["status"] == "completed"
    assert "audit_log" in res


def test_ide_file_handlers(tmp_path: Path) -> None:
    # Save file
    save_req = FileSaveRequest(filepath=".jules/ide_unit_test.txt", content="unit test content")
    save_res = api_save_file(save_req)
    assert save_res["status"] == "success"

    # Read file
    read_res = api_read_file(".jules/ide_unit_test.txt")
    assert read_res["status"] == "success"
    assert read_res["result"] == "unit test content"

    # List files
    list_res = api_list_files(".")
    assert list_res["status"] == "success"


def test_ide_html_handler() -> None:
    html = get_ide_html()
    assert isinstance(html, str)
    assert "Jules Agent Workbench" in html or "Jules" in html
