"""
Unit & Integration Tests for Jules CLI.
"""

import argparse
from typing import Any
from pytest import CaptureFixture

from src.jules.cli import build_parser, handle_info, handle_plan, handle_run, main


def test_cli_parser_build() -> None:
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    args = parser.parse_args(["info"])
    assert args.command == "info"

    args_run = parser.parse_args(["run", "build new feature", "--provider", "mock"])
    assert args_run.command == "run"
    assert args_run.prompt == "build new feature"
    assert args_run.provider == "mock"


def test_cli_handle_info(capsys: CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["info"])
    code = handle_info(args)
    assert code == 0

    captured = capsys.readouterr().out
    assert "Jules System Status" in captured
    assert "registered_tools_count" in captured


def test_cli_handle_plan(capsys: CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["plan", "test plan generation"])
    code = handle_plan(args)
    assert code == 0

    captured = capsys.readouterr().out
    assert "Execution Plan for" in captured


def test_cli_handle_run(capsys: CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "test task execution"])
    code = handle_run(args)
    assert code == 0

    captured = capsys.readouterr().out
    assert "Executing task" in captured
    assert "Status: completed" in captured


def test_cli_main_entrypoint(capsys: CaptureFixture[str]) -> None:
    ret = main(["info"])
    assert ret == 0
    captured = capsys.readouterr().out
    assert "registered_tools_count" in captured
