"""Tests for view.py: drawing the board, without a real window."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # draw without a display

import pygame  # noqa: E402

from game import Game  # noqa: E402
from view import (  # noqa: E402
    BACKGROUND,
    BLACK_STONE,
    WHITE_STONE,
    BoardView,
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
