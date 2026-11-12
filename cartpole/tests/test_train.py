"""Tests for train.py: the REINFORCE maths, settings and training runs."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from cartpole import CartPole
from config import load_config
from train import (
    LinearPolicy,
    TrainConfig,
    discounted_returns,
    policy_gradient,
    scale_to_kp,
    train,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_discounted_returns_by_hand():
    # Backwards: 1; 1 + 0.5 * 1 = 1.5; 1 + 0.5 * 1.5 = 1.75.
    assert discounted_returns([1, 1, 1], 0.5).tolist() == [1.75, 1.5, 1.0]
    assert discounted_returns([1, 1, 1], 1.0).tolist() == [3.0, 2.0, 1.0]
    assert discounted_returns([], 0.99).tolist() == []


def test_returns_cap_at_one_over_one_minus_gamma():
    returns = discounted_returns([1] * 2000, 0.99)
    assert returns[0] == pytest.approx(100, abs=1e-6)
    assert returns[-100] == pytest.approx((1 - 0.99**100) / 0.01)


@pytest.mark.parametrize("action", [0, 1])
def test_gradient_of_log_pi_matches_numerical_derivative(action):
    """(a - p) s, the gradient the update uses, really is the derivative
    of log pi(a | s) with respect to the weights."""
    rng = np.random.default_rng(1)
    weights, inputs = rng.normal(size=4), rng.normal(size=4)

    def log_pi(w):
        p = LinearPolicy(w).prob_right(inputs)
        return math.log(p if action == 1 else 1 - p)

    analytic = (action - LinearPolicy(weights).prob_right(inputs)) * inputs
    h = 1e-6
    numerical = [
        (log_pi(weights + h * e) - log_pi(weights - h * e)) / (2 * h) for e in np.eye(4)
    ]
    assert analytic == pytest.approx(numerical, rel=1e-6)


def episode(inputs, actions, prob_right):
    """An episode for policy_gradient: one step per action, reward 1 each."""
    n = len(actions)
    return (
        np.array([inputs] * n),
        np.array(actions),
        np.full(n, prob_right),
        [1.0] * n,
    )


def test_policy_gradient_by_hand():
    # Pushing right twice lasted 2 steps; pushing left once lasted 1.
    # With gamma 1 the returns are [2, 1] and [1]: mean 4/3, std sqrt(2)/3,
    # so A = [sqrt(2), -1/sqrt(2), -1/sqrt(2)]. With p = 0.5, a - p is
    # [0.5, 0.5, -0.5], so A (a - p) = [1/sqrt(2), -1/(2 sqrt(2)),
    # 1/(2 sqrt(2))], whose mean is sqrt(2)/6. Every step saw the same s.
    s = np.array([0.0, 0.0, 0.05, 0.2])
    episodes = [episode(s, [1, 1], 0.5), episode(s, [0], 0.5)]
    assert policy_gradient(episodes, 1.0) == pytest.approx(math.sqrt(2) / 6 * s)


def test_policy_gradient_favours_actions_followed_by_longer_episodes():
    s = np.array([0.1, -0.3, 0.05, 0.2])
    right_lasted = episode(s, [1] * 20, 0.5)
    left_fell = episode(s, [0] * 3, 0.5)
    gradient = policy_gradient([right_lasted, left_fell], 0.99)
    policy = LinearPolicy(np.zeros(4))
    before = policy.prob_right(s)
    policy.weights += 0.1 * gradient
    assert policy.prob_right(s) > before


def test_scale_to_kp():
    assert scale_to_kp([0.1, 0.2, 2.0, 0.6]).tolist() == pytest.approx(
        [1.0, 2.0, 20.0, 6.0]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"batches": 0},
        {"batch_size": 0},
        {"eval_episodes": 0},
        {"learning_rate": 0.0},
        {"gamma": 0.0},
        {"gamma": 1.5},
    ],
)
def test_train_config_rejects_out_of_range(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(TrainConfig(), **changes)


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.json")), ids=str)
def test_committed_configs_load(path):
    load_config(path, TrainConfig)


def test_standard_config_matches_the_defaults():
    standard = load_config(CONFIGS_DIR / "standard.json", TrainConfig)
    assert dataclasses.replace(standard, description="") == TrainConfig()


def run_training(config, capsys):
    cartpole = CartPole.make(angle_limit=None)
    policy, curve = train(cartpole, config, np.random.default_rng(config.seed))
    cartpole.close()
    capsys.readouterr()  # discard the printed learning curve
    return policy, curve


def test_training_with_a_seed_is_repeatable(capsys):
    config = TrainConfig(seed=5, batches=3, batch_size=2)
    first_policy, first_curve = run_training(config, capsys)
    second_policy, second_curve = run_training(config, capsys)
    assert first_policy.weights.tolist() == second_policy.weights.tolist()
    assert first_curve == second_curve


def test_training_learns(capsys):
    """A loose check that the algorithm works at all: after 50 batches,
    episodes last much longer than the coin-toss start."""
    _, curve = run_training(TrainConfig(seed=3, batches=50), capsys)
    mean_lengths = [row[2] for row in curve]
    assert np.mean(mean_lengths[-10:]) > 3 * np.mean(mean_lengths[:10])
