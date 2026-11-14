"""Tests for policy.py and learn.py: the network, its player, and
REINFORCE training."""

import numpy as np
import pytest
import torch

from game import BLACK, WHITE, Game
from learn import play_batch, reinforce_loss, result_for
from play import play_game
from players import EyeAwareRandomPlayer
from policy import (
    NetworkPlayer,
    PolicyNetwork,
    actor_critic_loss,
    encode,
    legal_mask,
    load_network,
    save_network,
    saved_komi,
    upgraded,
    with_value_head,
)


def test_encode_is_from_the_movers_point_of_view():
    game = Game.from_rows([". X .", ". O .", ". . ."])
    as_black = encode(game)
    assert as_black.shape == (3, 3, 3)
    assert as_black[0, 0, 1] == 1  # Black's stone is the mover's
    assert as_black[1, 1, 1] == 1  # White's is the opponent's
    assert as_black[2].sum() == 7  # empty points
    game.to_play = WHITE
    as_white = encode(game)
    assert as_white[0, 1, 1] == 1 and as_white[1, 0, 1] == 1


def test_legal_mask():
    game = Game.from_rows([". X .", "X . .", ". . ."], to_play=WHITE)
    mask = legal_mask(game)
    assert mask.shape == (10,)
    assert not mask[0]  # suicide for White
    assert not mask[1]  # occupied
    assert mask[9]  # pass


def test_network_scores_every_move():
    network = PolicyNetwork(5)
    scores = network(torch.zeros(4, 3, 5, 5))
    assert scores.shape == (4, 26)


@pytest.mark.parametrize("sample", [False, True])
def test_network_player_only_plays_legal_moves(sample):
    torch.manual_seed(0)
    network = PolicyNetwork(5)
    player = NetworkPlayer(network, sample=sample, rng=np.random.default_rng(0))
    for seed in range(5):
        # game.play raises if a move is illegal
        game = play_game(Game(5), player, EyeAwareRandomPlayer(seed))
        assert game.over


def test_network_player_records_its_moves():
    record = []
    player = NetworkPlayer(PolicyNetwork(5), record=record)
    game = Game(5)
    move = player(game)
    board, mask, recorded = record[0]
    assert recorded == move and mask[move] and board.shape == (3, 5, 5)


def test_result_for():
    game = Game.from_rows(["X X .", "X . .", ". . ."])
    assert result_for(game, BLACK) == 1.0 and result_for(game, WHITE) == -1.0
    assert result_for(Game(3), BLACK) == 0.0  # empty board: a draw


def test_reinforce_makes_moves_in_won_games_more_likely():
    """One update on a won game and a lost game from the same board: the
    move from the won game gains probability, the one from the lost game
    loses it."""
    torch.manual_seed(1)
    network = PolicyNetwork(3, channels=8, layers=1)
    board, mask = encode(Game(3)), legal_mask(Game(3))
    won_move, lost_move = 4, 0
    moves = [[(board, mask, won_move)], [(board, mask, lost_move)]]
    results = [1.0, -1.0]

    def probability(move):
        with torch.no_grad():
            logits = network(torch.from_numpy(board)[None])[0]
        return float(torch.softmax(logits, dim=0)[move])

    before = probability(won_move), probability(lost_move)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.5)
    loss, _ = reinforce_loss(network, moves, results)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert probability(won_move) > before[0]
    assert probability(lost_move) < before[1]


def test_play_batch_alternates_colours():
    torch.manual_seed(2)
    moves, results = play_batch(
        PolicyNetwork(5), EyeAwareRandomPlayer(0), 4, 5, 0.0, np.random.default_rng(0)
    )
    assert len(moves) == len(results) == 4
    assert all(r in (-1.0, 0.0, 1.0) for r in results)
    # As Black the network moves first: its first board is empty.
    assert moves[0][0][0][2].sum() == 25
    assert moves[1][0][0][2].sum() == 24  # as White, after Black's first stone


def test_save_and_load(tmp_path):
    network = PolicyNetwork(5, channels=8, layers=2)
    path = tmp_path / "net.pt"
    save_network(path, network, {"batch": 3})
    loaded = load_network(path)
    boards = torch.rand(2, 3, 5, 5)
    with torch.no_grad():
        assert torch.allclose(network(boards), loaded(boards))
    assert loaded.settings() == {
        "size": 5,
        "channels": 8,
        "layers": 2,
        "value": False,
        "colour": False,
    }


def test_value_head_gives_one_estimate_per_board_between_minus_1_and_1():
    network = PolicyNetwork(5, value=True)
    scores, values = network.heads(torch.rand(4, 3, 5, 5) * 10)
    assert scores.shape == (4, 26) and values.shape == (4,)
    assert (values.abs() <= 1).all()
    assert PolicyNetwork(5).heads(torch.zeros(1, 3, 5, 5))[1] is None


def test_networks_saved_before_value_heads_still_load(tmp_path):
    network = PolicyNetwork(5, channels=8, layers=1)
    path = tmp_path / "old.pt"
    settings = {"size": 5, "channels": 8, "layers": 1}  # no "value" key
    torch.save({"settings": settings, "weights": network.state_dict()}, path)
    assert not load_network(path).has_value


def test_adding_a_value_head_keeps_the_move_choices():
    torch.manual_seed(4)
    network = PolicyNetwork(5, channels=8, layers=2)
    critic = with_value_head(network)
    boards = torch.rand(3, 3, 5, 5)
    with torch.no_grad():
        assert torch.allclose(network(boards), critic(boards))
    assert critic.has_value


def test_actor_critic_update_moves_the_critic_toward_the_result():
    torch.manual_seed(5)
    network = PolicyNetwork(3, channels=8, layers=1, value=True)
    board, mask = encode(Game(3)), legal_mask(Game(3))
    records, results = [(board, mask, 4)], [1.0]

    def value():
        with torch.no_grad():
            return float(network.heads(torch.from_numpy(board)[None])[1][0])

    before = value()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.5)
    for _ in range(5):
        loss, _, value_loss, _ = actor_critic_loss(network, records, results)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert value() > before


def test_actor_critic_advantage_is_result_less_value():
    """With the critic already predicting the result exactly, the move gets
    no push either way: z - V = 0."""
    network = PolicyNetwork(3, channels=8, layers=1, value=True)
    board, mask = encode(Game(3)), legal_mask(Game(3))
    with torch.no_grad():
        network.value_score.weight.zero_()
        network.value_score.bias.fill_(10.0)  # tanh(10) is about 1
    loss, _, value_loss, right = actor_critic_loss(
        network, [(board, mask, 4)], [1.0], value_weight=0.0
    )
    assert abs(loss.item()) < 1e-6 and right == 1.0


def test_a_detached_critic_trains_only_the_value_head():
    torch.manual_seed(6)
    network = PolicyNetwork(5, channels=8, layers=2, value=True)
    _, values = network.heads(torch.rand(2, 3, 5, 5), detach_value=True)
    values.sum().backward()
    assert network.value_score.weight.grad is not None
    assert all(p.grad is None or not p.grad.any() for p in network.body.parameters())


def test_colour_plane_says_who_is_to_move():
    game = Game(3)
    assert encode(game, colour=True).shape == (4, 3, 3)
    assert encode(game, colour=True)[3].min() == 1  # Black to move
    game.play(0)
    assert encode(game, colour=True)[3].max() == 0  # White to move
    assert encode(game).shape == (3, 3, 3)


def test_adding_a_colour_plane_keeps_the_move_choices():
    torch.manual_seed(7)
    network = PolicyNetwork(5, channels=8, layers=2, value=True)
    coloured = upgraded(network, colour=True)
    game = Game(5)
    game.play(12)
    old = torch.from_numpy(encode(game))[None]
    new = torch.from_numpy(encode(game, colour=True))[None]
    with torch.no_grad():
        old_scores, old_value = network.heads(old)
        new_scores, new_value = coloured.heads(new)
    assert torch.allclose(old_scores, new_scores)
    assert torch.allclose(old_value, new_value)
    assert coloured.has_colour and coloured.settings()["colour"]


def test_a_network_with_a_colour_plane_plays(tmp_path):
    network = PolicyNetwork(5, channels=8, layers=1, colour=True)
    game = play_game(Game(5, komi=7.5), NetworkPlayer(network), EyeAwareRandomPlayer(1))
    assert game.over
    path = tmp_path / "coloured.pt"
    save_network(path, network, {"komi": 7.5})
    assert load_network(path).has_colour
    assert saved_komi(path) == 7.5
    old = tmp_path / "old.pt"
    save_network(old, PolicyNetwork(5, channels=8, layers=1))
    assert saved_komi(old) is None
