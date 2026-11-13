"""Players for game.py's Go: each chooses a move for the player to move.

A player is any callable taking a Game and returning a legal move.
"""

import random


def is_eye(game, point, colour):
    """Whether point is an eye of colour's: an empty point whose
    neighbours are all colour's stones, and which can't be broken by the
    opponent holding its diagonals.

    The diagonal test is the usual one in Go programs: away from the edge
    the opponent may hold at most one of the four diagonals, and on the
    edge or in a corner none of them. Otherwise it's a false eye, which
    the opponent can fill after capturing a diagonal.
    """
    board = game.board
    if board[point] != 0 or any(board[n] != colour for n in game.neighbours(point)):
        return False
    size = game.size
    row, column = divmod(point, size)
    diagonals = [
        (row + dr) * size + (column + dc)
        for dr in (-1, 1)
        for dc in (-1, 1)
        if 0 <= row + dr < size and 0 <= column + dc < size
    ]
    opposing = sum(1 for d in diagonals if board[d] not in (0, colour))
    allowed = 1 if len(diagonals) == 4 else 0
    return opposing <= allowed


class RandomPlayer:
    """Chooses uniformly at random among all legal moves, passing
    included. seed makes its choices repeatable."""

    def __init__(self, seed=None):
        self._random = random.Random(seed)

    def __call__(self, game):
        return self._random.choice(game.legal_moves())


class EyeAwareRandomPlayer:
    """Chooses uniformly at random among legal moves that don't fill one of
    its own eyes (see is_eye), and passes only when there are none. This
    is the standard random player of Go programs: real players never fill
    their own eyes, and without that rule random games either stop early
    on chance passes or never end. seed makes its choices repeatable."""

    def __init__(self, seed=None):
        self._random = random.Random(seed)

    def __call__(self, game):
        moves = [
            m
            for m in game.legal_moves()
            if m != game.pass_move and not is_eye(game, m, game.to_play)
        ]
        return self._random.choice(moves) if moves else game.pass_move


class HumanPlayer:
    """Asks a person for the move: view (a BoardView, see view.py) shows
    the board and waits for a click on a point, or P to pass. Raises
    view.QuitGame if the window is closed instead."""

    def __init__(self, view):
        self.view = view

    def __call__(self, game):
        return self.view.ask_move(game)
