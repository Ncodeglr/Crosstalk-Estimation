import os
import click
import pytest
from click.testing import CliRunner

import cvnn.cli as cli_mod
from cvnn.cli import cli_main


def test_help_shows_commands():
    runner = CliRunner()
    result = runner.invoke(cli_main, ["--help"])
    assert result.exit_code == 0
    # Should show usage and options
    assert "Usage:" in result.output
    assert "--mode" in result.output
    assert "--resume-logdir" in result.output


def test_cli_main_full_run(tmp_path, monkeypatch):
    # Create dummy config file
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("dummy: true")
    # Create dummy resume dir
    resume_dir = tmp_path / "old_logs"
    resume_dir.mkdir()

    # Capture run_experiment calls
    called = {}

    def fake_run_experiment(config_path, resume_logdir=None, mode_override=None):
        called["config_path"] = config_path
        called["resume_logdir"] = resume_logdir
        called["mode_override"] = mode_override

    monkeypatch.setattr(cli_mod, "run_experiment", fake_run_experiment)

    runner = CliRunner()
    # Provide three newlines for the input() calls
    result = runner.invoke(
        cli_main,
        [str(cfg_file), "--mode", "full", "--resume-logdir", str(resume_dir)],
        input="\n\n\n",
    )
    assert result.exit_code == 0
    out = result.output
    # Verify printed messages
    assert f"Running experiment with config: {cfg_file}" in out
    assert "Mode override: full" in out
    assert f"Resuming from log directory: {resume_dir}" in out
    # Verify fake_run_experiment was called with correct args
    assert called["config_path"] == str(cfg_file)
    assert called["resume_logdir"] == str(resume_dir)
    assert called["mode_override"] == "full"


def test_cli_main_missing_config_file():
    """Test CLI behavior with non-existent config file"""
    runner = CliRunner()
    result = runner.invoke(cli_main, ["nonexistent.yaml"])
    assert result.exit_code != 0


def test_cli_main_invalid_mode(tmp_path, monkeypatch):
    """Test CLI behavior with invalid mode"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("dummy: true")

    def fake_run_experiment(config_path, resume_logdir=None, mode_override=None):
        pass

    monkeypatch.setattr(cli_mod, "run_experiment", fake_run_experiment)

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(cfg_file), "--mode", "invalid_mode"])
    # Should fail with invalid choice
    assert result.exit_code != 0
    assert (
        "invalid choice" in result.output.lower()
        or "invalid value" in result.output.lower()
    )


def test_cli_main_default_mode(tmp_path, monkeypatch):
    """Test CLI behavior with default mode (no --mode specified)"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("dummy: true")

    called = {}

    def fake_run_experiment(config_path, resume_logdir=None, mode_override=None):
        called["mode_override"] = mode_override

    monkeypatch.setattr(cli_mod, "run_experiment", fake_run_experiment)

    runner = CliRunner()
    result = runner.invoke(cli_main, [str(cfg_file)], input="\n\n\n")
    assert result.exit_code == 0
    # Should have None for mode_override when not specified
    assert called["mode_override"] is None
