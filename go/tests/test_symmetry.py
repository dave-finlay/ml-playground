"""Tests for symmetry.py: turning recorded moves through the board's 8
symmetries, and measuring how lopsided a network is."""

import numpy as np
import torch

from game import BLACK, Game
from players import EyeAwareRandomPlayer
from policy import PolicyNetwork, encode, legal_mask
from symmetry import (
    augmented,
    importance_weights,
    probe_positions,
    symmetries,
    symmetry_gap,
    turn_record,
)


def point(game, name):
    return next(p for p in range(game.pass_move) if game.name(p) == name)


def game_from_planes(board, to_play):
    """The position the planes show ("mine" is the player to move)."""
    mine, theirs = board[0].astype(bool), board[1].astype(bool)
    me, them = ("X", "O") if to_play == BLACK else ("O", "X")
    rows = [
        "".join(me if mine[r, c] else them if theirs[r, c] else "." for c in range(5))
        for r in range(5)
    ]
    return Game.from_rows(rows, to_play=to_play)


def test_there_are_8_distinct_symmetries_starting_with_the_identity():
    turns = symmetries(5)
    assert len({tuple(s) for s in turns}) == 8
    assert list(turns[0]) == list(range(25))
    assert all(sorted(s) == list(range(25)) for s in turns)


def test_the_mirror_of_b2_after_c3_c2_is_d2():
    game = Game(5)
    game.play(point(game, "C3"))
    game.play(point(game, "C2"))
    record = (encode(game), legal_mask(game), point(game, "B2"))
    moves = {game.name(turn_record(record, s)[2]) for s in symmetries(5)}
    assert moves == {"B2", "D2", "B4", "D4"}  # each reached twice
    mirrored = turn_record(record, symmetries(5)[4])
    assert game.name(mirrored[2]) == "D2"


def test_a_turned_mask_is_the_turned_positions_legal_moves():
    """In a mid-game position, with captures and suicide points, each
    turned record's mask is exactly the legal moves of the turned
    position, and its move is legal there."""
    game = Game(5)
    players = EyeAwareRandomPlayer(4), EyeAwareRandomPlayer(5)
    for _ in range(30):
        game.play(players[len(game.moves) % 2](game))
    record = (encode(game), legal_mask(game), players[len(game.moves) % 2](game))
    for source in symmetries(5):
        board, mask, move = turn_record(record, source)
        turned = game_from_planes(board, game.to_play)
        assert (mask == legal_mask(turned)).all()
        assert mask[move]


def test_augmented_keeps_each_value_with_its_records():
    game = Game(5)
    records = [
        (encode(game), legal_mask(game), 12),
        (encode(game), legal_mask(game), 0),
    ]
    rng = np.random.default_rng(0)
    assert augmented(records, [1, -1], "none", rng) == (records, [1, -1], [0, 1])
    all_records, all_values, origins = augmented(records, [1, -1], "all", rng)
    assert len(all_records) == 16 and all_values == [1] * 8 + [-1] * 8
    assert origins == [0] * 8 + [1] * 8
    assert {move for _, _, move in all_records[8:]} == {0, 4, 20, 24}  # corners
    one_each, values, origins = augmented(records, [1, -1], "random", rng)
    assert len(one_each) == 2 and values == [1, -1] and origins == [0, 1]
    assert one_each[0][2] == 12  # the centre stays the centre


def test_symmetry_gap_is_zero_for_a_symmetric_network_and_not_otherwise():
    probes = probe_positions(5, 0.0, 3, 1, colour=False)
    flat = PolicyNetwork(5, channels=8, layers=1, value=True)
    with torch.no_grad():
        for parameter in flat.parameters():
            parameter.zero_()
    policy_gap, value_gap = symmetry_gap(flat, probes)
    assert policy_gap < 1e-6 and value_gap < 1e-6
    torch.manual_seed(0)
    policy_gap, value_gap = symmetry_gap(PolicyNetwork(5, value=True), probes)
    assert policy_gap > 0.001 and value_gap > 0


def test_importance_weights_are_1_for_a_symmetric_network():
    probes = probe_positions(5, 0.0, 2, 1, colour=False)[:20]
    flat = PolicyNetwork(5, channels=8, layers=1)
    with torch.no_grad():
        for parameter in flat.parameters():
            parameter.zero_()
    turned, _, origins = augmented(probes, [0] * 20, "all", np.random.default_rng(0))
    rho = importance_weights(flat, probes, turned, origins, cap=10.0)
    assert np.allclose(rho, 1.0)


class CornerLover:
    """A stand-in network that strongly prefers the top-left corner, A5,
    on any board."""

    def heads(self, boards, detach_value=False):
        scores = torch.zeros(len(boards), 26)
        scores[:, 0] = 8.0
        return scores, None


def test_importance_weights_down_weight_moves_the_network_would_rarely_play():
    """A move at A5, turned to the other corners, gets a tiny weight (the
    network would rarely play there); a move at E1 turned to A5 would be
    weighted up, but the cap holds it at 1."""
    game = Game(5)
    played = [(encode(game), legal_mask(game), 0), (encode(game), legal_mask(game), 24)]
    turned, _, origins = augmented(played, [1, 1], "all", np.random.default_rng(0))
    rho = importance_weights(CornerLover(), played, turned, origins, cap=1.0)
    corners = [game.name(move) for _, _, move in turned]
    for r, corner in zip(rho[:8], corners[:8], strict=True):
        # The original and the diagonal flip keep it at A5.
        assert r == 1.0 if corner == "A5" else r < 0.01
    from_e1_to_a5 = [r for r, c in zip(rho[8:], corners[8:], strict=True) if c == "A5"]
    assert from_e1_to_a5 == [1.0, 1.0] and rho[8:].max() == 1.0
