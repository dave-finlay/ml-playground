"""Tests for selfplay.py: self-play games, per-colour baselines and the
test harness."""

import numpy as np
import pytest
import torch

from game import BLACK, WHITE, Game
from players import EyeAwareRandomPlayer
from policy import PolicyNetwork, encode, legal_mask, policy_gradient_loss
from selfplay import (
    RandomOpening,
    advantages_by_colour,
    play_training_game,
    win_rates,
)


def test_self_play_records_both_sides_with_opposite_results():
    torch.manual_seed(0)
    network = PolicyNetwork(5)
    sides = play_training_game(network, None, BLACK, 5, 0.0, np.random.default_rng(0))
    assert len(sides) == 2
    (black_moves, first, black_result), (white_moves, second, white_result) = sides
    assert (first, second) == (BLACK, WHITE)
    assert black_result == -white_result or black_result == white_result == 0
    assert abs(len(black_moves) - len(white_moves)) <= 1  # they alternate


def test_against_a_past_copy_only_its_own_moves_are_recorded():
    torch.manual_seed(1)
    sides = play_training_game(
        PolicyNetwork(5), PolicyNetwork(5), WHITE, 5, 0.0, np.random.default_rng(1)
    )
    assert len(sides) == 1 and sides[0][1] == WHITE


def test_advantages_use_a_baseline_per_colour():
    record = [(None, None, 0)]
    sides = [
        (record, BLACK, 1.0),
        (record, BLACK, 1.0),
        (record, BLACK, -1.0),  # Black averages 1/3
        (record, WHITE, -1.0),
        (record, WHITE, 1.0),  # White averages 0
    ]
    _, advantages = advantages_by_colour(sides)
    assert advantages == pytest.approx([2 / 3, 2 / 3, -4 / 3, -1.0, 1.0])


def test_entropy_bonus_lowers_the_loss_for_spread_out_policies():
    torch.manual_seed(2)
    network = PolicyNetwork(3, channels=8, layers=1)
    records = [(encode(Game(3)), legal_mask(Game(3)), 4)]
    plain, entropy = policy_gradient_loss(network, records, [0.0])
    with_bonus, _ = policy_gradient_loss(network, records, [0.0], entropy_bonus=0.1)
    assert with_bonus.item() == pytest.approx(plain.item() - 0.1 * entropy)


def test_random_opening_then_the_player():
    played = []

    def player(game):
        played.append(len(game.moves))
        return game.pass_move

    wrapped = RandomOpening(player, moves=2, seed=0)
    game = Game(5)
    for _ in range(6):
        game.play(wrapped(game))
        if game.over:
            break
    assert played and min(played) >= 4  # the first 2 moves per side were random


def test_win_rates_are_between_0_and_1():
    torch.manual_seed(3)
    black, white = win_rates(
        PolicyNetwork(5), EyeAwareRandomPlayer, 4, 5, 0.0, seed=0, opening=1
    )
    assert 0 <= black <= 1 and 0 <= white <= 1
