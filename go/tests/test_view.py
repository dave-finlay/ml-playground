"""Tests for view.py: drawing the board, without a real window."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # draw without a display

import pygame  # noqa: E402

from game import BLACK, WHITE, Game  # noqa: E402
from play import watch_in_window  # noqa: E402
from players import HumanPlayer  # noqa: E402
from view import (  # noqa: E402
    BACKGROUND,
    BLACK_STONE,
    WHITE_STONE,
    BoardView,
    QuitGame,
    board_geometry,
    draw_board,
)


def test_draw_board_puts_stones_where_they_are():
    pygame.init()
    game = Game.from_rows(
        [
            "X . . . .",
            ". . . . .",
            ". . . . .",
            ". . . . .",
            ". . . . O",
        ]
    )
    cell = 60
    margin, pixels = board_geometry(game.size, cell)
    surface = pygame.Surface((pixels, pixels + 60))
    draw_board(surface, game, cell, caption="test")
    assert surface.get_at((margin, margin))[:3] == BLACK_STONE  # A5
    far = margin + 4 * cell
    assert surface.get_at((far, far))[:3] == WHITE_STONE  # E1
    middle = margin + 2 * cell + cell // 2  # between grid lines, no stone
    assert surface.get_at((middle, middle))[:3] == BACKGROUND
    pygame.quit()


def test_final_board_marks_territory():
    pygame.init()
    game = Game.from_rows([". X O", "X X O", ". X O"])
    cell = 60
    margin, pixels = board_geometry(game.size, cell)
    surface = pygame.Surface((pixels, pixels + 60))
    draw_board(surface, game, cell, final=True)
    assert surface.get_at((margin, margin))[:3] == BLACK_STONE  # A3 territory
    pygame.quit()


def test_board_view_shows_a_game():
    view = BoardView(5)
    game = Game(5)
    game.play(12)
    assert view.show(game, last_move=12, caption="Move 1: Black C3", seconds=0)
    view.close()


def click(view, row, column, offset=(0, 0)):
    x = view.margin + column * view.cell + offset[0]
    y = view.margin + row * view.cell + offset[1]
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x, y)))


def key(name):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=name))


def test_point_at_finds_the_nearest_intersection():
    view = BoardView(5)
    margin, cell = view.margin, view.cell
    assert view.point_at((margin, margin)) == 0  # A5
    assert view.point_at((margin + 2 * cell + 5, margin + 2 * cell - 5)) == 12  # C3
    assert view.point_at((margin + cell // 2, margin)) is None  # between points
    assert view.point_at((2, 2)) is None  # off the grid
    view.close()


def test_a_click_plays_a_point():
    view = BoardView(5)
    pygame.event.clear()
    click(view, 2, 2, offset=(6, -4))
    assert HumanPlayer(view)(Game(5)) == 12
    view.close()


def test_an_illegal_click_waits_for_a_legal_one():
    view = BoardView(5)
    game = Game(5)
    game.play(12)  # Black at C3; now White's turn
    pygame.event.clear()
    click(view, 2, 2)  # occupied
    click(view, 0, 0)  # A5
    assert view.ask_move(game) == 0
    view.close()


def test_p_passes_and_closing_quits():
    view = BoardView(5)
    pygame.event.clear()
    key(pygame.K_p)
    game = Game(5)
    assert view.ask_move(game) == game.pass_move
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    try:
        view.ask_move(game)
        raise AssertionError("expected QuitGame")
    except QuitGame:
        pass
    view.close()


def test_a_human_can_play_a_whole_game(capsys):
    """A human who always passes, against the eye-aware player, in the
    window: the game runs to the end, and a key closes the result."""
    pygame.init()
    pygame.display.set_mode((10, 10))  # so events can be posted before
    pygame.event.clear()
    for _ in range(150):
        key(pygame.K_p)
    game = Game(5)
    watch_in_window(game, "human", "eye-aware", seed=1, delay=0.0)
    assert game.over and game.passes_in_a_row == 2
    assert game.board.count(WHITE) > 0 and game.board.count(BLACK) == 0
    assert "White by" in capsys.readouterr().out
