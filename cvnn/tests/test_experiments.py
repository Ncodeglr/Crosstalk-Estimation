import os
import sys
import yaml
import pytest
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import cvnn.experiments as exp_mod
from cvnn.experiments import run_experiment


def test_run_experiment_no_task(tmp_path):
    # Config without task should raise ValueError
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump({}))
    with pytest.raises(ValueError):
        run_experiment(str(cfg_path))


def test_run_experiment_invalid_task(tmp_path):
    # Config with unknown task should raise ValueError
    cfg = {"task": "unknown"}
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError) as exc:
        run_experiment(str(cfg_path))
    assert "Config must specify a valid" in str(exc.value)


def test_run_experiment_resume_not_exist(tmp_path):
    # Valid config but non-existent resume directory should error
    cfg = {"task": "reconstruction"}
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(yaml.dump(cfg))
    resume_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        run_experiment(str(cfg_file), resume_logdir=str(resume_dir))


def test_run_experiment_dispatch_fake_task(tmp_path, monkeypatch):
    # Create a fake task experiment class
    class FakeExp:
        def __init__(self, config_path, resume_logdir=None, mode_override=None):
            self.history = {"fake": True}
            self.metrics = [42]
            self.logdir = "fake_logdir"
            self.model = "fake_model"
            # write marker to config file
            with open(config_path, "a") as f:
                f.write("\n# modified")

        def run(self):
            # Return the expected values for the test assertions
            return self.history, self.metrics, self.logdir, self.model

    # Patch the plugin registry (not _TASKS)
    import cvnn.plugins as plugins_mod

    monkeypatch.setitem(plugins_mod._PLUGINS, "fake", FakeExp)

    # Prepare config file
    cfg = {"task": "fake", "logging": {"logdir": str(tmp_path / "logs")}}
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(yaml.dump(cfg))

    # Run the experiment
    history, results, logdir, model = run_experiment(str(cfg_file))
    assert history == {"fake": True}
    assert results == [42]
    assert logdir == "fake_logdir"
    assert model == "fake_model"
    # Also ensure config file was appended (mode override etc.)
    assert "# modified" in cfg_file.read_text()
