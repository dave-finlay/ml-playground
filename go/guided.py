"""Tree search guided by a network, as in AlphaZero.

Like mcts.py it grows a tree of continuations one simulation at a time,
but the network does two jobs in it:

- Its policy gives each move a prior P, the probability it would play
  it, and selection prefers moves with high priors (PUCT):
      Q + c_puct * P * sqrt(parent visits) / (1 + visits),
  where Q is the move's average value so far. So the search spends its
  simulations on moves the network thinks are worth a look, instead of
  trying everything.
- Its value head judges each new position directly (V, the expected
  result for the player to move, -1 to +1), instead of a random playout
  to the end. A network without a value head falls back to playouts.

Each simulation walks down the tree by PUCT to a position not yet
evaluated, evaluates it with the network (giving every legal move its
prior, as new children), and backs the value up the path, its sign
flipping at each level since the players alternate. After all the
simulations it plays the move it visited most.
"""

import math
import random

import numpy as np
import torch

from game import EMPTY
from mcts import playout
from policy import encode, legal_mask, masked_logits


class GuidedNode:
    """A position in the tree, reached by move from its parent.

    prior is the network's probability for move; value_sum and visits
    total the values backed up through it, from the point of view of the
    player who made move.
    """

    __slots__ = ("move", "parent", "prior", "children", "visits", "value_sum")

    def __init__(self, move, parent, prior):
        self.move = move
        self.parent = parent
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value_sum = 0.0

    def q(self):
        """The average value of the simulations through this node (0 before
        any)."""
        return self.value_sum / self.visits if self.visits else 0.0


def evaluate(network, game, rng):
    """Return (priors, value) for game's position: the network's
    probability for each legal move, as {move: p}, and the value for the
    player to move (from the value head, or a random playout without one).
    """
    board, mask = encode(game, network.has_colour), legal_mask(game)
    with torch.no_grad():
        scores, values = network.heads(torch.from_numpy(board)[None])
        logits = masked_logits(scores[0], torch.from_numpy(mask))
        probs = torch.softmax(logits, dim=0).numpy()
    priors = {int(move): float(probs[move]) for move in np.flatnonzero(mask)}
    if values is not None:
        value = float(values[0])
    else:
        mover = game.to_play
        winner = playout(game.copy(), rng).winner()
        value = 0.0 if winner == EMPTY else 1.0 if winner == mover else -1.0
    return priors, value


def terminal_value(game):
    """The result of a finished game for the player to move: +1, -1 or 0."""
    winner = game.winner()
    return 0.0 if winner == EMPTY else 1.0 if winner == game.to_play else -1.0


def guided_search(game, network, simulations, c_puct=1.5, rng=None):
    """Run network-guided search from game's position and return the root
    GuidedNode: its children hold each move's visits."""
    rng = rng or random.Random()
    root = GuidedNode(None, None, 1.0)
    priors, _ = evaluate(network, game, rng)
    root.children = [GuidedNode(m, root, p) for m, p in priors.items()]
    root.visits = 1
    for _ in range(simulations):
        node, position = root, game.copy()
        while node.children:
            scale = c_puct * math.sqrt(node.visits)
            node = max(
                node.children,
                key=lambda c: c.q() + scale * c.prior / (1 + c.visits),
            )
            position.play(node.move)
        if position.over:
            value = terminal_value(position)
        else:
            priors, value = evaluate(network, position, rng)
            node.children = [GuidedNode(m, node, p) for m, p in priors.items()]
        # value is for the player to move at the leaf; the leaf's own
        # statistics are for the player who moved into it, the opponent.
        value = -value
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent
    return root


class GuidedSearchPlayer:
    """Chooses moves by search guided by network, with simulations
    simulations per move (more is stronger and slower). seed makes any
    random playouts repeatable."""

    def __init__(self, network, simulations=100, c_puct=1.5, seed=None):
        self.network = network
        self.simulations = simulations
        self.c_puct = c_puct
        self._random = random.Random(seed)

    def __call__(self, game):
        root = guided_search(
            game, self.network, self.simulations, self.c_puct, self._random
        )
        return max(root.children, key=lambda c: c.visits).move
