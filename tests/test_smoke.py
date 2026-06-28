from __future__ import annotations

import subprocess
import sys

import docsher
from docsher.cli import build_parser, main


def test_package_exposes_version() -> None:
    assert docsher.__version__ == "0.1.0"


def test_cli_parser_has_help_text() -> None:
    help_text = build_parser().format_help()

    assert "usage: docsher" in help_text
    assert "Local Docsher" in help_text


def test_cli_main_returns_success() -> None:
    assert main([]) == 0


def test_module_help_command_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "docsher", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "usage: docsher" in completed.stdout
