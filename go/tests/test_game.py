"""Tests for game.py: the rules of Go."""

import pytest

from game import BLACK, EMPTY, WHITE, Game, IllegalMove
from play import play_game
from players import EyeAwareRandomPlayer, RandomPlayer, is_eye


def point(game, name):
    """The point with a name like C3."""
    for p in range(game.size * game.size):
        if game.name(p) == name:
            return p
    raise ValueError(name)


def test_empty_board():
    game = Game(5)
    assert len(game.legal_moves()) == 26  # every point, and pass
    assert game.to_play == BLACK
    assert game.area() == {BLACK: 0, WHITE: 0}  # empty land borders no one


def test_point_names():
    game = Game(5)
    assert game.name(0) == "A5" and game.name(24) == "E1"
    assert game.name(game.pass_move) == "pass"
    nine = Game(9)
    assert nine.name(nine.pass_move - 1) == "J1"  # columns skip I


def test_capture_a_stone_on_the_edge():
    game = Game.from_rows(
        [
            ". X O . .",
            ". . X . .",
            ". . . . .",
            ". . . . .",
            ". . . . .",
        ]
    )
    game.play(point(game, "D5"))
    assert game.board[point(game, "C5")] == EMPTY
    assert game.captures[BLACK] == 1


def test_capture_a_group():
    # White's B5-C5 pair has one liberty left, D5.
    game = Game.from_rows(
        [
            "X O O . .",
            ". X X . .",
            ". . . . .",
            ". . . . .",
            ". . . . .",
        ]
    )
    game.play(point(game, "D5"))
    assert game.board[point(game, "B5")] == EMPTY
    assert game.board[point(game, "C5")] == EMPTY
    assert game.captures[BLACK] == 2


def test_suicide_is_illegal():
    game = Game.from_rows(
        [
            ". O . . .",
            "O . . . .",
            ". . . . .",
            ". . . . .",
            ". . . . .",
        ]
    )
    assert not game.is_legal(point(game, "A5"))
    with pytest.raises(IllegalMove, match="suicide"):
        game.play(point(game, "A5"))


def test_suicide_of_a_group_is_illegal():
    game = Game.from_rows(
        [
            "X . O . .",
            "O O . . .",
            ". . . . .",
            ". . . . .",
            ". . . . .",
        ]
    )
    # B5 would join A5, leaving the pair with no liberties.
    assert not game.is_legal(point(game, "B5"))


def test_a_move_with_no_liberties_is_legal_if_it_captures():
    game = Game.from_rows(
        [
            ". X O . .",
            "X O . . .",
            "O . . . .",
            ". . . . .",
            ". . . . .",
        ],
        to_play=WHITE,
    )
    # A5 is surrounded by Black's B5 and A4, but playing it captures both
    # (B5's last liberty is A5, and so is A4's).
    assert game.is_legal(point(game, "A5"))
    game.play(point(game, "A5"))
    assert game.board[point(game, "B5")] == EMPTY
    assert game.board[point(game, "A4")] == EMPTY
    assert game.captures[WHITE] == 2


KO = [
    ". X O . .",
    "X O . O .",
    ". X O . .",
    ". . . . .",
    ". . . . .",
]


def test_ko_cannot_be_retaken_at_once():
    game = Game.from_rows(KO)
    game.play(point(game, "C4"))  # Black captures White's B4
    assert game.board[point(game, "B4")] == EMPTY
    # White retaking at B4 would recreate the position before: illegal.
    assert not game.is_legal(point(game, "B4"))
    with pytest.raises(IllegalMove, match="superko"):
        game.play(point(game, "B4"))


def test_ko_can_be_retaken_after_moves_elsewhere():
    game = Game.from_rows(KO)
    game.play(point(game, "C4"))
    game.play(point(game, "E1"))  # White plays elsewhere
    game.play(point(game, "E2"))  # so does Black
    assert game.is_legal(point(game, "B4"))  # a new position now


def test_occupied_and_off_board_points_are_illegal():
    game = Game(5)
    game.play(0)
    assert not game.is_legal(0)
    assert not game.is_legal(-1) and not game.is_legal(26)
    with pytest.raises(IllegalMove, match="occupied"):
        game.play(0)


def test_two_passes_end_the_game():
    game = Game(5)
    game.play(game.pass_move)
    assert not game.over
    game.play(0)  # a stone resets the count
    game.play(game.pass_move)
    assert not game.over
    game.play(game.pass_move)
    assert game.over
    assert game.legal_moves() == []
    with pytest.raises(IllegalMove, match="over"):
        game.play(1)


def test_the_move_limit_ends_the_game():
    game = Game(5, max_moves=3)
    for move in (0, 1, 2):
        game.play(move)
    assert game.over


def test_area_scoring_counts_stones_and_territory():
    game = Game.from_rows(
        [
            ". X O . .",
            ". X O . .",
            "X X O . .",
            ". X O O O",
            "X X O . .",
        ]
    )
    area = game.area()
    # Black: 7 stones, plus A5, A4 and A2 (only Black around them).
    assert area[BLACK] == 7 + 3
    # White: 7 stones, plus the 8 empty points on the right.
    assert area[WHITE] == 7 + 8
    assert game.margin() == 10 - 15


def test_neutral_points_score_for_no_one():
    game = Game.from_rows(
        [
            "X . O",
            "X . O",
            "X . O",
        ]
    )
    assert game.area() == {BLACK: 3, WHITE: 3}


def test_komi_and_draws():
    rows = ["X . O", "X . O", "X . O"]
    assert Game.from_rows(rows).winner() == EMPTY
    assert Game.from_rows(rows, komi=0.5).winner() == WHITE
    assert Game.from_rows(rows, komi=0.5).margin() == -0.5


@pytest.mark.parametrize("size", [5, 7])
def test_random_games_finish_legally(size):
    for seed in range(20):
        game = play_game(Game(size), RandomPlayer(seed), RandomPlayer(seed + 100))
        assert game.over
        area = game.area()
        assert area[BLACK] + area[WHITE] <= size * size


def test_random_games_are_repeatable():
    first = play_game(Game(5), RandomPlayer(1), RandomPlayer(2))
    second = play_game(Game(5), RandomPlayer(1), RandomPlayer(2))
    assert first.moves == second.moves


@pytest.mark.parametrize(
    "rows, name, colour, eye",
    [
        # A true eye in the middle: all four neighbours Black.
        (
            [". . . . .", ". . X . .", ". X . X .", ". . X . .", ". . . . ."],
            "C3",
            BLACK,
            True,
        ),
        # One opposing diagonal is allowed away from the edge.
        (
            [". . . . .", ". O X . .", ". X . X .", ". . X . .", ". . . . ."],
            "C3",
            BLACK,
            True,
        ),
        # Two opposing diagonals make it a false eye.
        (
            [". . . . .", ". O X . .", ". X . X .", ". . X O .", ". . . . ."],
            "C3",
            BLACK,
            False,
        ),
        # On the edge, any opposing diagonal makes it false.
        (
            ["X . X . .", "O X . . .", ". . . . .", ". . . . .", ". . . . ."],
            "B5",
            BLACK,
            False,
        ),
        (
            ["X . X . .", ". X . . .", ". . . . .", ". . . . .", ". . . . ."],
            "B5",
            BLACK,
            True,
        ),
        # A corner eye.
        (
            [". X . . .", "X . . . .", ". . . . .", ". . . . .", ". . . . ."],
            "A5",
            BLACK,
            True,
        ),
        # Not an eye for the other colour, or with an empty neighbour.
        (
            [". X . . .", "X . . . .", ". . . . .", ". . . . .", ". . . . ."],
            "A5",
            WHITE,
            False,
        ),
        (
            [". . . . .", ". . X . .", ". X . . .", ". . X . .", ". . . . ."],
            "C3",
            BLACK,
            False,
        ),
    ],
)
def test_is_eye(rows, name, colour, eye):
    game = Game.from_rows(rows)
    assert is_eye(game, point(game, name), colour) == eye


def test_eye_aware_player_never_fills_its_own_eye():
    game = Game.from_rows(
        [
            ". X . . .",
            "X X . . .",
            ". . . . .",
            ". . . . .",
            ". . . . .",
        ]
    )
    player = EyeAwareRandomPlayer(seed=0)
    for _ in range(200):
        assert player(game) != point(game, "A5")


def test_eye_aware_player_passes_when_only_eyes_are_left():
    game = Game.from_rows([". X X", "X X X", "X X ."])
    assert EyeAwareRandomPlayer(seed=0)(game) == game.pass_move


@pytest.mark.parametrize("size", [5, 7])
def test_eye_aware_games_end_by_passing(size):
    for seed in range(20):
        game = play_game(
            Game(size), EyeAwareRandomPlayer(seed), EyeAwareRandomPlayer(seed + 100)
        )
        assert game.passes_in_a_row == 2


def test_territory_marks_the_empty_points_each_side_owns():
    game = Game.from_rows(["X . O", "X . O", "X . O"])
    assert game.territory() == {}  # the middle column touches both
    game = Game.from_rows([". X O", "X X O", ". X O"])
    corners = {point(game, "A3"): BLACK, point(game, "A1"): BLACK}
    assert game.territory() == corners
