"""A Go player that thinks ahead with Monte Carlo tree search (MCTS).

For each move it grows a tree of possible continuations, one simulation
at a time. A simulation:

1. Selects: from the current position, walks down the tree, at each step
   taking the move with the best UCT score: its win rate so far plus a
   bonus for having been tried less often,
       wins / visits + exploration * sqrt(ln(parent's visits) / visits),
   which balances exploiting moves that have done well against exploring
   ones that might.
2. Expands: at the edge of the tree, adds one untried move's position.
3. Evaluates: plays the game out from there with random moves that
   never fill their own eyes (see playout), and scores it by the rules,
   komi included.
4. Backs up: counts the result in every position along the path, as a
   win for the side that made the move into it or a loss.

After all the simulations it plays the move it visited most. No
learning is involved: random playouts, averaged over many simulations,
stand in for judging positions.
"""

import math
import random

from game import EMPTY
from players import is_eye


def playout(game, rng):
    """Play game to the end with random moves that don't fill the mover's
    own eyes, passing when there are none, and return it.

    Faster than EyeAwareRandomPlayer: instead of finding every legal
    move first, it tries empty points in random order and takes the
    first that's legal and not an eye, usually the first or second.
    """
    while not game.over:
        empty = [p for p in range(game.pass_move) if game.board[p] == EMPTY]
        rng.shuffle(empty)
        for point in empty:
            if not is_eye(game, point, game.to_play) and game.is_legal(point):
                game.play(point)
                break
        else:
            game.play(game.pass_move)
    return game


class Node:
    """A position in the search tree, reached by move from its parent.

    wins and visits count the simulations through it, the wins from the
    point of view of the player who made the move into it (draws count
    a half).
    """

    __slots__ = ("move", "parent", "mover", "children", "untried", "wins", "visits")

    def __init__(self, move, parent, mover, legal_moves):
        self.move = move
        self.parent = parent
        self.mover = mover
        self.children = []
        self.untried = legal_moves
        self.wins = 0.0
        self.visits = 0

    def best_child(self, exploration):
        """The child with the highest UCT score."""
        log_visits = math.log(self.visits)
        return max(
            self.children,
            key=lambda c: (
                c.wins / c.visits + exploration * math.sqrt(log_visits / c.visits)
            ),
        )


def search(game, simulations, exploration=1.4, rng=None):
    """Run MCTS from game's position and return the root Node: its
    children hold each move's visits and wins."""
    rng = rng or random.Random()
    root = Node(None, None, None, game.legal_moves())
    rng.shuffle(root.untried)
    for _ in range(simulations):
        node, position = root, game.copy()
        # 1. Select.
        while not node.untried and node.children:
            node = node.best_child(exploration)
            position.play(node.move)
        # 2. Expand.
        if node.untried:
            move = node.untried.pop()
            mover = position.to_play
            position.play(move)
            child = Node(move, node, mover, position.legal_moves())
            rng.shuffle(child.untried)
            node.children.append(child)
            node = child
        # 3. Evaluate.
        winner = playout(position, rng).winner()
        # 4. Back up.
        while node is not None:
            node.visits += 1
            if node.mover is not None:
                node.wins += (
                    1.0 if winner == node.mover else 0.5 if winner == EMPTY else 0.0
                )
            node = node.parent
    return root


class MCTSPlayer:
    """Chooses moves by Monte Carlo tree search with simulations
    simulations per move (more is stronger and slower). seed makes its
    choices repeatable."""

    def __init__(self, simulations=400, exploration=1.4, seed=None):
        self.simulations = simulations
        self.exploration = exploration
        self._random = random.Random(seed)

    def __call__(self, game):
        root = search(game, self.simulations, self.exploration, self._random)
        if not root.children:
            return game.pass_move
        return max(root.children, key=lambda c: c.visits).move
