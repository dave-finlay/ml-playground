"""The board's 8 symmetries, for training on every orientation of a move.

A Go position is worth the same turned a quarter, half or three-quarter
turn, or flipped, and the turned move is as good as the original was.
The network's convolutions don't know that: they treat a pattern alike
wherever it is, but not its mirror image or rotation, so it has to learn
the symmetry, and self-play shows it mostly one orientation of each
position (komi.pt gives B2 an 80% prior after C3 C2, and the identical
D2 under 2%).

augmented() turns recorded moves (board planes, legal mask, move) so
training can use them in all 8 orientations, or one chosen at random,
with the same advantage or result: the turned game would have gone the
same way, move for move, with the same result. symmetry_gap() measures
how lopsided a network is: how differently it treats a position and its
turned versions.

REINFORCE's update is only right on average for moves turning up as
often as the network would play them. A turned move turns up as often as
the network played the original, so unless the network is symmetric it's
off-policy: from komi.pt (policy gap 0.29), training on all 8
orientations collapsed within 150 batches. importance_weights() gives
each turned move the weight rho = pi(turned move | turned board) /
pi(move | board), capped (importance sampling): 1 when the network is
symmetric, smaller for moves it would play less often than the data
shows them.
"""

import numpy as np
import torch

from game import Game
from players import EyeAwareRandomPlayer
from policy import encode, legal_mask, masked_logits

AUGMENT_MODES = ("none", "random", "all")


def symmetries(size):
    """The 8 symmetries of a size x size board, as index arrays: for
    symmetry s, the point that lands on point p is s[p]. The first is the
    identity; then quarter turns, and each of those flipped."""
    grid = np.arange(size * size).reshape(size, size)
    turns = [np.rot90(grid, k) for k in range(4)]
    return [g.ravel().copy() for g in turns + [np.fliplr(g) for g in turns]]


def turn_record(record, source):
    """A recorded move (board planes, legal mask, move) turned by the
    symmetry source (see symmetries): the planes and the mask's points
    move together, the move goes where its point lands, and pass stays
    pass."""
    board, mask, move = record
    channels, size, _ = board.shape
    points = size * size
    turned_board = board.reshape(channels, points)[:, source].reshape(board.shape)
    turned_mask = np.concatenate([mask[:points][source], mask[points:]])
    if move < points:
        move = int(np.flatnonzero(source == move)[0])
    return turned_board, turned_mask, move


def augmented(records, values, mode, rng):
    """records (and each one's advantage or result, values) for training:
    as they are with mode "none", each in one randomly chosen orientation
    with "random", or in all 8 with "all". Returns (records, values,
    origins): values go with their records, unchanged, and origins[i] is
    the index in records of the move the i-th one was turned from. rng is
    a numpy Generator."""
    if mode == "none" or not records:
        return records, values, list(range(len(records)))
    size = records[0][0].shape[1]
    turns = symmetries(size)
    if mode == "random":
        choices = rng.integers(len(turns), size=len(records))
        turned = [
            turn_record(r, turns[c]) for r, c in zip(records, choices, strict=True)
        ]
        return turned, values, list(range(len(records)))
    if mode == "all":
        out_records = [turn_record(r, s) for r in records for s in turns]
        out_values = [v for v in values for _ in turns]
        origins = [i for i in range(len(records)) for _ in turns]
        return out_records, out_values, origins
    raise ValueError(f"unknown augment mode {mode!r}")


def move_probabilities(network, records):
    """network's probability for each record's move on its board."""
    boards = torch.from_numpy(np.stack([b for b, _, _ in records]))
    masks = torch.from_numpy(np.stack([m for _, m, _ in records]))
    moves = torch.tensor([m for _, _, m in records])
    with torch.no_grad():
        scores, _ = network.heads(boards)
        probs = torch.softmax(masked_logits(scores, masks), dim=1)
    return probs[torch.arange(len(records)), moves].numpy()


def importance_weights(network, originals, turned, origins, cap=1.0):
    """The weight for each turned record's policy update: rho = pi(turned
    move | turned board) / pi(original move | original board), capped at
    cap. originals are the records as played, and origins[i] the index
    in originals of the one turned[i] came from."""
    played = move_probabilities(network, originals)[origins]
    rho = move_probabilities(network, turned) / np.maximum(played, 1e-12)
    return np.minimum(rho, cap)


def probe_positions(size, komi, games, seed, colour):
    """Positions to measure lopsidedness on: every position before a move
    in games games between eye-aware random players (seeded with seed),
    recorded as (board planes, legal mask, move) for a network with or
    without a colour plane."""
    records = []
    for i in range(games):
        game = Game(size, komi=komi)
        players = (
            EyeAwareRandomPlayer(seed + 2 * i),
            EyeAwareRandomPlayer(seed + 2 * i + 1),
        )
        while not game.over:
            move = players[len(game.moves) % 2](game)
            records.append((encode(game, colour), legal_mask(game), move))
            game.play(move)
    return records


def symmetry_gap(network, records):
    """How lopsided network is on the recorded positions: (policy gap,
    value gap). The policy gap is the average total variation distance
    between its move probabilities for a position and for each of the 7
    turned versions (turned back): 0 if it treats them alike, 1 if they
    never overlap. The value gap is the average difference between its
    values for them (None without a value head)."""
    size = records[0][0].shape[1]
    turns = symmetries(size)
    points = size * size
    with torch.no_grad():
        policies, values = [], []
        for source in turns:
            turned = [turn_record(r, source) for r in records]
            boards = torch.from_numpy(np.stack([b for b, _, _ in turned]))
            masks = torch.from_numpy(np.stack([m for _, m, _ in turned]))
            scores, value = network.heads(boards)
            probs = torch.softmax(masked_logits(scores, masks), dim=1).numpy()
            # Turn back: the probability at turned point p belongs to the
            # original point source[p].
            back = np.empty_like(probs)
            back[:, source] = probs[:, :points]
            back[:, points:] = probs[:, points:]
            policies.append(back)
            values.append(None if value is None else value.numpy())
    policy_gap = np.mean(
        [0.5 * np.abs(p - policies[0]).sum(axis=1).mean() for p in policies[1:]]
    )
    value_gap = (
        None
        if values[0] is None
        else float(np.mean([np.abs(v - values[0]).mean() for v in values[1:]]))
    )
    return float(policy_gap), value_gap
