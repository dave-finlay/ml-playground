"""Tests for mcts.py and Game.copy: looking ahead by tree search."""

import random

from game import BLACK, Game
from mcts import MCTSPlayer, playout, search
from play import play_game
from players import EyeAwareRandomPlayer


def point(game, name):
    return next(p for p in range(game.pass_move) if game.name(p) == name)


def test_copy_is_independent():
    game = Game(5)
    game.play(12)
    copy = game.copy()
    copy.play(0)
    assert len(game.moves) == 1 and game.board[0] == 0
    assert len(copy.moves) == 2 and copy.board[0] != 0
    assert game.to_play != copy.to_play


def test_playout_finishes_the_game_by_passing():
    rng = random.Random(0)
    for _ in range(20):
        game = playout(Game(5), rng)
        assert game.over and game.passes_in_a_row == 2


def test_search_visits_add_up():
    root = search(Game(5), simulations=200, rng=random.Random(1))
    assert sum(child.visits for child in root.children) == 200
    assert root.visits == 200


def test_mcts_finds_the_only_winning_move():
    """C2 captures White's four stones at the bottom left (it's their last
    liberty) and wins; in playouts the next-best move wins only about a
    third of the time. Found by searching random games for such a
    position."""
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
    assert MCTSPlayer(300, seed=2)(game) == point(game, "C2")


def test_mcts_plays_whole_games_legally_and_repeatably():
    first = play_game(Game(5), MCTSPlayer(50, seed=3), EyeAwareRandomPlayer(3))
    second = play_game(Game(5), MCTSPlayer(50, seed=3), EyeAwareRandomPlayer(3))
    assert first.over and first.moves == second.moves
