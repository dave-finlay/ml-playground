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
    ThreeActionPolicy,
    TrainConfig,
    count_reversals,
    dead_band,
    discounted_returns,
    play_episode,
    policy_gradient,
    scale_to_kp,
    train,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "train"
MODELS_DIR = Path(__file__).parent.parent / "models"


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
    gradient = policy_gradient(LinearPolicy(np.zeros(4)), episodes, 1.0)
    assert gradient == pytest.approx(math.sqrt(2) / 6 * s)


def test_policy_gradient_favours_actions_followed_by_longer_episodes():
    s = np.array([0.1, -0.3, 0.05, 0.2])
    right_lasted = episode(s, [1] * 20, 0.5)
    left_fell = episode(s, [0] * 3, 0.5)
    policy = LinearPolicy(np.zeros(4))
    gradient = policy_gradient(policy, [right_lasted, left_fell], 0.99)
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
        {"actions": 3, "push_cost": -0.1},
        {"actions": 3, "push_cost": 1.0},
        {"actions": 2, "push_cost": 0.1},  # every step pushes with two actions
    ],
)
def test_train_config_rejects_out_of_range(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(TrainConfig(), **changes)


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.json")), ids=str)
def test_committed_configs_load(path):
    load_config(path, TrainConfig)


@pytest.mark.parametrize(
    "path",
    sorted(MODELS_DIR.glob("*/config.json")),
    ids=str,
)
def test_model_training_configs_load(path):
    config = load_config(path, TrainConfig)
    assert config.seed is not None  # so the weights can be regenerated


def test_standard_config_matches_the_defaults():
    standard = load_config(CONFIGS_DIR / "standard.json", TrainConfig)
    assert dataclasses.replace(standard, description="") == TrainConfig()


def run_training(config, capsys):
    cartpole = CartPole.make(angle_limit=None)
    weights, curve, scales = train(cartpole, config, np.random.default_rng(config.seed))
    cartpole.close()
    capsys.readouterr()  # discard the printed learning curve
    return weights, curve, scales


def test_training_with_a_seed_is_repeatable(capsys):
    config = TrainConfig(seed=5, batches=3, batch_size=2)
    first_weights, first_curve, _ = run_training(config, capsys)
    second_weights, second_curve, _ = run_training(config, capsys)
    assert first_weights.tolist() == second_weights.tolist()
    assert first_curve == second_curve


def test_training_learns(capsys):
    """A loose check that the algorithm works at all: after 50 batches,
    episodes last much longer than the coin-toss start."""
    _, curve, _ = run_training(TrainConfig(seed=3, batches=50), capsys)
    mean_lengths = [row[2] for row in curve]
    assert np.mean(mean_lengths[-10:]) > 3 * np.mean(mean_lengths[:10])


def test_scales_are_one_without_normalization(capsys):
    _, _, scales = run_training(TrainConfig(seed=1, batches=1), capsys)
    assert scales.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_normalized_inputs_express_the_same_controller():
    """Weights w' on inputs divided by scales act exactly like weights
    w' / scales on the raw inputs, which is how normalized weights are
    converted back for saving."""
    rng = np.random.default_rng(2)
    scales = np.array([0.07, 0.5, 0.09, 0.7])
    normalized_weights = rng.normal(size=4)
    for _ in range(100):
        inputs = rng.normal(size=4) * scales
        assert LinearPolicy(normalized_weights).prob_right(
            inputs / scales
        ) == pytest.approx(LinearPolicy(normalized_weights / scales).prob_right(inputs))


def test_normalized_training_learns(capsys):
    config = TrainConfig(seed=3, batches=50, normalize_inputs=True, learning_rate=3.0)
    _, curve, scales = run_training(config, capsys)
    assert (scales > 0).all() and (scales != 1.0).all()
    mean_lengths = [row[2] for row in curve]
    assert np.mean(mean_lengths[-10:]) > 3 * np.mean(mean_lengths[:10])


@pytest.mark.parametrize("action", [0, 1, 2])
def test_three_action_gradient_matches_numerical_derivative(action):
    """Row k of (onehot(a) - probs) (s, 1), the gradient the update uses,
    really is the derivative of log pi(a | s) with respect to action k's
    weights."""
    rng = np.random.default_rng(4)
    weights, inputs = rng.normal(size=(3, 5)), rng.normal(size=4)
    policy = ThreeActionPolicy(weights)
    analytic = policy.gradient(
        inputs[None, :], np.array([action]), policy.probs(inputs)[None, :], np.ones(1)
    )

    def log_pi(w):
        return math.log(ThreeActionPolicy(w).probs(inputs)[action])

    h = 1e-6
    numerical = np.zeros((3, 5))
    for k in range(3):
        for i in range(5):
            nudge = np.zeros((3, 5))
            nudge[k, i] = h
            numerical[k, i] = (log_pi(weights + nudge) - log_pi(weights - nudge)) / (
                2 * h
            )
    assert analytic == pytest.approx(numerical, rel=1e-5, abs=1e-8)


def test_three_action_probabilities():
    assert ThreeActionPolicy(np.zeros((3, 5))).probs(np.ones(4)) == pytest.approx(
        [1 / 3] * 3
    )
    probs = ThreeActionPolicy(np.random.default_rng(0).normal(size=(3, 5))).probs(
        np.array([0.3, -1.0, 2.0, 0.5])
    )
    assert probs.sum() == pytest.approx(1.0) and (probs > 0).all()


def test_three_action_normalized_weights_convert_back():
    rng = np.random.default_rng(5)
    scales = np.array([0.1, 0.4, 0.08, 0.6])
    policy = ThreeActionPolicy(rng.normal(size=(3, 5)))
    converted = ThreeActionPolicy(policy.in_input_units(scales))
    for _ in range(50):
        inputs = rng.normal(size=4) * scales
        assert policy.probs(inputs / scales) == pytest.approx(converted.probs(inputs))


def test_dead_band_reading_of_mirrored_weights():
    left = [-1.0, -2.0, -20.0, -3.0, 0.0]
    coast = [0.0, 0.0, 0.0, 0.0, 1.5]
    right = [1.0, 2.0, 20.0, 3.0, 0.0]
    gains, threshold = dead_band([left, coast, right])
    assert gains.tolist() == [1.0, 2.0, 20.0, 3.0]
    assert threshold == 1.5


def test_count_reversals_ignores_coasting():
    assert count_reversals([10, 10, 0, -10, 0, -10, 10, 0, 0]) == 2
    assert count_reversals([0, 0, 0]) == 0


class AlwaysDoes:
    """A stand-in policy that always takes one of ThreeActionPolicy's
    actions."""

    def __init__(self, action):
        self.action = action

    def sample(self, inputs, rng):
        return self.action, np.full(3, 1 / 3)

    def force(self, action):
        return ThreeActionPolicy.FORCES[action]


@pytest.mark.parametrize("action, reward", [(1, 1.0), (2, 0.75)])
def test_push_cost_comes_off_pushing_steps_only(action, reward):
    cartpole = CartPole.make(angle_limit=None)
    _, _, _, rewards = play_episode(
        cartpole, AlwaysDoes(action), "velocity", None, seed=1, push_cost=0.25
    )
    cartpole.close()
    assert set(rewards) == {reward}


def test_three_action_training_learns(capsys):
    config = TrainConfig(
        seed=3, batches=40, normalize_inputs=True, learning_rate=3.0, actions=3
    )
    weights, curve, _ = run_training(config, capsys)
    assert weights.shape == (3, 5)
    mean_lengths = [row[2] for row in curve]
    assert np.mean(mean_lengths[-10:]) > 3 * np.mean(mean_lengths[:10])
