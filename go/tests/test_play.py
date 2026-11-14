"""Tests for play.py's options for pitting players against each other."""

import sys

import pytest
import torch

import play
from game import Game
from players import EyeAwareRandomPlayer, RandomOpening
from policy import PolicyNetwork, save_network


def run(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["play.py", *map(str, args)])
    play.main()
    return capsys.readouterr().out


@pytest.fixture
def two_networks(tmp_path):
    paths = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        path = tmp_path / f"net{seed}.pt"
        save_network(path, PolicyNetwork(5, channels=8, layers=1))
        paths.append(path)
    return paths


def test_two_different_networks_play_head_to_head(two_networks, monkeypatch, capsys):
    first, second = two_networks
    out = run(
        monkeypatch,
        capsys,
        "--black",
        "learned",
        "--white",
        "learned",
        "--black-model",
        first,
        "--white-model",
        second,
        "--games",
        6,
        "--opening",
        2,
        "--swap",
        "--seed",
        1,
    )
    assert "Head to head" in out
    assert str(first) in out and str(second) in out
    assert "every game is the same" not in out


def test_without_an_opening_two_networks_repeat_one_game(
    two_networks, monkeypatch, capsys
):
    out = run(
        monkeypatch,
        capsys,
        "--black",
        "learned",
        "--white",
        "learned",
        "--model",
        two_networks[0],
        "--games",
        2,
    )
    assert "every game is the same" in out


@pytest.mark.parametrize(
    "args, message",
    [
        (["--black", "learned"], "needs --model or --black-model"),
        (["--black-model", "x.pt"], "--black-model needs --black learned"),
        (["--window", "--black", "human", "--opening", "2"], "computer players"),
        (["--watch", "--swap"], "--swap is for many games"),
    ],
)
def test_option_errors(args, message, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, capsys, *args)
    assert message in capsys.readouterr().err


def test_random_opening_makes_the_first_moves_random():
    always_pass = RandomOpening(lambda game: game.pass_move, moves=2, seed=3)
    game = Game(5)
    for _ in range(4):  # two moves per side
        game.play(
            always_pass(game)
            if len(game.moves) % 2 == 0
            else EyeAwareRandomPlayer(0)(game)
        )
    assert game.pass_move not in game.moves[::2]
    assert always_pass(game) == game.pass_move  # then it plays as itself
