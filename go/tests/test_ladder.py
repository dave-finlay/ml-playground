"""Tests for ladder.py: training against champions, and promotion."""

import csv
import sys

import torch

import ladder
from policy import PolicyNetwork, load_network, save_network, saved_komi


def run(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["ladder.py", *map(str, args)])
    ladder.main()
    return capsys.readouterr().out


def start_network(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "start.pt"
    save_network(path, PolicyNetwork(5, channels=8, layers=1))
    return path


SMALL = ["--games", 4, "--gate-every", 2, "--gate-games", 4, "--seed", 1]


def test_a_learner_that_clears_the_gate_is_promoted(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "ladder"
    out = run(
        monkeypatch,
        capsys,
        "--init",
        start_network(tmp_path),
        "--dir",
        out_dir,
        "--batches",
        4,
        "--promote-at",
        0.0,
        "--komi",
        7.5,
        *SMALL,
    )
    assert "promoted: champion 1" in out and "promoted: champion 2" in out
    for rung in (0, 1, 2):
        champion = out_dir / f"champion-{rung}.pt"
        assert load_network(champion).has_colour  # komi added the plane
        assert saved_komi(champion) == 7.5
    with open(out_dir / "ladder.csv") as f:
        rows = list(csv.DictReader(f))
    assert [row["rung"] for row in rows] == ["1", "2"]
    assert (out_dir / "learner.pt").exists()


def test_without_a_clear_win_the_champion_stays(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "ladder"
    out = run(
        monkeypatch,
        capsys,
        "--init",
        start_network(tmp_path),
        "--dir",
        out_dir,
        "--batches",
        2,
        "--promote-at",
        1.01,
        *SMALL,
    )
    assert "promoted" not in out and "0 promotion(s)" in out
    assert not (out_dir / "champion-1.pt").exists()


def test_head_to_head_of_a_network_with_itself_is_a_rate():
    network = PolicyNetwork(5, channels=8, layers=1)
    black, white, overall = ladder.head_to_head(network, network, 4, 5, 7.5, 0, 2)
    assert 0 <= black <= 1 and 0 <= white <= 1
    assert overall == (black + white) / 2
