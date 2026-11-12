"""Tests for runs.py: finding, listing and deleting runs."""

import json

import pytest

import runs


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """A temporary models directory in place of cartpole/models/."""
    directory = tmp_path / "models"
    directory.mkdir()
    monkeypatch.setattr(runs, "MODELS_DIR", directory)
    return directory


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """A temporary runs directory in place of cartpole/runs/."""
    directory = tmp_path / "runs"
    directory.mkdir()
    monkeypatch.setattr(runs, "RUNS_DIR", directory)
    return directory


def make_run(directory, name, description="", test_mean=None):
    path = directory / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"description": description}))
    if test_mean is not None:
        results = {
            "started": "2026-11-12T10:43:24-08:00",
            "test_lengths": {"mean": test_mean, "limit": 500},
        }
        (path / "results.json").write_text(json.dumps(results))
    return path


def test_find_run_by_name_or_path(runs_dir):
    path = make_run(runs_dir, "r1")
    assert runs.find_run("r1") == path.resolve()
    assert runs.find_run(str(path)) == path.resolve()


@pytest.mark.parametrize("name", ["missing", "not-a-run"])
def test_find_run_rejects_what_isnt_a_run(runs_dir, name):
    (runs_dir / "not-a-run").mkdir()  # no config.json
    with pytest.raises(ValueError, match="not a run"):
        runs.find_run(name)


def test_find_run_rejects_directories_outside_runs(runs_dir, tmp_path):
    model = make_run(tmp_path / "models", "kept")  # looks like a run
    with pytest.raises(ValueError, match="not a run"):
        runs.find_run(str(model))


def test_promote_copies_the_whole_run(runs_dir, models_dir):
    run = make_run(runs_dir, "r1", "kept", test_mean=500)
    (run / "weights.json").write_text("{}")
    model = runs.promote_run(run, "keeper")
    assert model == models_dir / "keeper"
    assert sorted(p.name for p in model.iterdir()) == [
        "config.json",
        "results.json",
        "weights.json",
    ]
    assert run.exists()  # copied, not moved


def test_promote_never_overwrites_a_model(runs_dir, models_dir):
    run = make_run(runs_dir, "r1")
    (models_dir / "r1").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        runs.promote_run(run, "r1")


@pytest.mark.parametrize("name", ["", "a/b", ".hidden"])
def test_promote_rejects_unusable_names(runs_dir, models_dir, name):
    run = make_run(runs_dir, "r1")
    with pytest.raises(ValueError, match="not a usable model name"):
        runs.promote_run(run, name)


def test_delete_when_confirmed(runs_dir, capsys):
    path = make_run(runs_dir, "r1")
    assert runs.delete_runs([path], confirmed=True)
    assert not path.exists()


@pytest.mark.parametrize("answer, deleted", [("y", True), ("n", False), ("", False)])
def test_delete_asks_first(runs_dir, monkeypatch, capsys, answer, deleted):
    path = make_run(runs_dir, "r1")
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    assert runs.delete_runs([path], confirmed=False) == deleted
    assert path.exists() != deleted


def test_list_shows_runs_with_or_without_results(runs_dir, capsys):
    make_run(runs_dir, "finished", "a description", test_mean=487.5)
    make_run(runs_dir, "no-results")
    runs.list_runs()
    output = capsys.readouterr().out
    assert "finished" in output and "488/500" in output
    assert "a description" in output and "2026-11-12 10:43" in output
    assert "no-results" in output


def test_list_with_no_runs(runs_dir, capsys):
    runs.list_runs()
    assert "No runs" in capsys.readouterr().out
