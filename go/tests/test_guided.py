"""Tests for guided.py: tree search guided by a network."""

import random

import pytest
import torch

from game import BLACK, Game
from guided import GuidedSearchPlayer, evaluate, guided_search
from play import play_game
from players import EyeAwareRandomPlayer
from policy import PolicyNetwork


def uninformed(value=False):
    """A network with all weights zero: equal priors for every legal move,
    and (with a value head) a value of exactly 0."""
    network = PolicyNetwork(5, channels=8, layers=1, value=value)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    return network


def point(game, name):
    return next(p for p in range(game.pass_move) if game.name(p) == name)


def test_evaluate_gives_priors_over_legal_moves_and_a_value():
    game = Game(5)
    game.play(12)
    priors, value = evaluate(uninformed(value=True), game, random.Random(0))
    assert set(priors) == set(game.legal_moves())
    assert sum(priors.values()) == pytest.approx(1.0)
    assert len(set(round(p, 6) for p in priors.values())) == 1  # all equal
    assert value == 0.0


def test_without_a_value_head_a_playout_judges_the_position():
    _, value = evaluate(uninformed(), Game(5), random.Random(0))
    assert value in (-1.0, 0.0, 1.0)


def test_search_visits_add_up():
    root = guided_search(Game(5), uninformed(value=True), 50, rng=random.Random(1))
    assert sum(child.visits for child in root.children) == 50


def test_guided_search_finds_the_only_winning_move():
    """The capture position from test_mcts.py: with no knowledge in the
    network (equal priors, playouts for values) the search still finds
    C2."""
    game = Game.from_rows(
        [
            ". X X . X",
            "X X X O O",
            "X O O O X",
            "O X . X X",
            "O O O X .",
        ],
        to_play=BLACK,
        komi=7.5,
    )
    assert GuidedSearchPlayer(uninformed(), 300, seed=0)(game) == point(game, "C2")


def test_guided_player_plays_whole_games_legally():
    torch.manual_seed(2)
    network = PolicyNetwork(5, channels=8, layers=1, value=True, colour=True)
    game = play_game(
        Game(5, komi=7.5), GuidedSearchPlayer(network, 20), EyeAwareRandomPlayer(2)
    )
    assert game.over
