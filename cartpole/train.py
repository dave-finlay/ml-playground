"""Train a CartPole controller with REINFORCE, then watch it balance.

The controller is a linear policy. Its inputs are s = (cart position,
its rate of change, pole angle, its rate of change), and it pushes right
with probability sigmoid(w · s), and otherwise left, always at the full
10 N (standard CartPole's "bang-bang" control). By default the rates are
estimated by differencing successive positions and angles, as
cartpole.py's controllers do (see RateEstimator there), so the policy
learns from what angle and position sensors alone would give. Training
adjusts the four weights w by REINFORCE, the simplest policy-gradient
method:

1. Play a batch of episodes with the current weights, sampling each
   action from the policy.
2. Work out each step's discounted return G_t = r_t + gamma * G_{t+1},
   and standardize the returns over the batch to get advantages A_t.
3. Step the weights uphill: w += learning_rate * mean(A_t (a_t - p_t) s_t).

Each weight's step is proportional to its input, and the inputs come in
very different sizes (the angle's rate is typically about 9 times the
angle), so with one learning rate the weights on bigger inputs learn
faster. With normalize_inputs, the policy sees each input divided by its
typical size (its standard deviation in coin-toss episodes, measured
before training), so all four learn at comparable rates. That doesn't
change which controllers it can express: weights w' on s / scales are
weights w' / scales on s, and the weights are converted back like that
for printing and saving.

Training uses standard CartPole: 12 degrees, 2.4 m and 500 steps (10 s).
Afterwards the trained policy plays without sampling, always taking the
likelier action, which makes it a controller like the hand-tuned ones in
cartpole.py: push right when w · s >= 0 (cartpole.py's LinearController).
So the weights play the roles of the PD controller's gains Kx, Kv, Kp and
Kd.

With actions 3, the policy can also not push. Each action k (push left,
don't push, push right) scores z_k = w_k · s + b_k, and the policy picks
it with probability softmax(z)_k. The bias b_k is what lets "don't push"
win near balance: without it, scaling s scales every score alike, so the
choice could depend only on the direction of the state, not on how far
from balance it is. The update has the same form, with
(onehot(a_t) - softmax(z_t)) in place of (a_t - p_t), and (s_t, 1) for
s_t. Once trained it acts as a dead band (cartpole.py's
ThreeActionController): coast while the combined signal is small, push
full force once it's past a threshold. push_cost takes that much off the
reward of every step that pushes, standing in for motor wear; without
it, not pushing is never better than pushing, only sometimes no worse.

Settings come from built-in defaults, overridden by a JSON config file
(--config), overridden by command-line options; see config.py. Each
training run is saved in its own directory under cartpole/runs/:

- config.json: every setting the run used (usable as a config file to
  repeat the run)
- weights.json: the learned weights, for cartpole.py --weights
- curve.csv: the learning curve, one row per batch
- results.json: test results, timing and the git commit of the code

Run from the repo root:

    uv run python cartpole/train.py
    uv run python cartpole/train.py --config cartpole/configs/train/normalized.json
    uv run python cartpole/train.py --learning-rate 1 --run-name slow-learning
    uv run python cartpole/train.py --learning-rate 1 --no-save
    uv run python cartpole/train.py --load cartpole/runs/slow-learning
    uv run python cartpole/cartpole.py --controller learned \\
        --weights cartpole/runs/slow-learning/weights.json
"""

import argparse
import csv
import dataclasses
import datetime
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from cartpole import (
    CONTINUOUS,
    THREE_ACTION,
    TWO_ACTION,
    CartPole,
    ForceRecorder,
    RateEstimator,
    control_quality,
    controller_from_weights,
    count_reversals,
    load_weights,
    print_quality,
    print_summary,
    run_simulation,
    save_weights,
)
from config import (
    add_setting_options,
    given_settings,
    load_config,
    resolve,
    save_config,
    setting,
)

FORCE = 10.0  # standard CartPole's push, in newtons
SCALE_EPISODES = 20  # coin-toss episodes for measuring input scales
# The narrowest a continuous policy's force spread may become, in
# newtons: at too high a learning rate the spread can collapse toward 0,
# after which the gradient blows up.
MIN_FORCE_STD = 0.1
# Added to the seed for checkpoint tests, so they start from different
# positions than the final test: picking the best checkpoint on the same
# episodes it's then tested on would flatter it.
CHECKPOINT_SEED_OFFSET = 1_000_003
RUNS_DIR = Path(__file__).parent / "runs"

# The hand-tuned "gentle" gains from cartpole.py's PD controller, for
# comparison with the learned weights, in the inputs' order: Kx, Kv, Kp, Kd.
PD_GAINS = np.array([0.5, 1.0, 20.0, 3.0])
GAIN_NAMES = [
    "Kx (cart position)",
    "Kv (cart velocity)",
    "Kp (pole angle)",
    "Kd (pole rate)",
]


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Everything that decides what a training run does."""

    description: str = setting("", "what this configuration is for")
    seed: int | None = setting(
        None,
        "random seed for training and for the test episodes; none gives a "
        "different run each time",
    )
    rate_from: str = setting(
        "difference",
        "where the policy's rates of change come from: differences of "
        "successive positions and angles, or the true velocities",
        choices=("difference", "velocity"),
    )
    normalize_inputs: bool = setting(
        False,
        "divide each input by its typical size, measured in coin-toss episodes "
        "before training, so all four weights learn at comparable rates",
    )
    actions: int | str = setting(
        2,
        "2: push left or right every step; 3: push left, don't push or push "
        "right; continuous: push with any force up to the maximum",
        choices=(2, 3, "continuous"),
    )
    force_std: float = setting(
        5.0,
        "continuous actions: the starting standard deviation of the sampled "
        "force, in newtons (it's learned)",
    )
    push_cost: float = setting(
        0.0,
        "taken off the reward (1 per step) for every step that pushes, for "
        "motor wear; needs actions 3",
    )
    force_cost: float = setting(
        0.0,
        "taken off the reward times (force / 10 N) squared, for motor wear; "
        "needs actions continuous",
    )
    angle_cost: float = setting(
        0.0, "taken off the reward times (pole angle / 12 degrees) squared"
    )
    position_cost: float = setting(
        0.0, "taken off the reward times (cart position / 2.4 m) squared"
    )
    batches: int = setting(200, "number of weight updates")
    batch_size: int = setting(10, "episodes played per update")
    learning_rate: float = setting(10.0, "step size for each update")
    gamma: float = setting(0.99, "discount factor for future rewards")
    eval_episodes: int = setting(
        20, "episodes to test the trained policy on, without sampling"
    )
    checkpoint_every: int = setting(
        25,
        "every this many batches (and after the last), test the policy "
        "without sampling, keep the best weights so far and say whether "
        "sampling does better; 0 turns checkpoints off and keeps the final "
        "weights",
    )
    checkpoint_episodes: int = setting(
        30,
        "test episodes at each checkpoint, always from the same starting "
        "positions, which differ from the final test's",
    )

    def __post_init__(self):
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every should be at least 0")
        for name in ("batches", "batch_size", "eval_episodes", "checkpoint_episodes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} should be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate should be more than 0")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma should be more than 0 and at most 1")
        costs = ("push_cost", "force_cost", "angle_cost", "position_cost")
        for name in costs:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} should be at least 0")
        if sum(getattr(self, name) for name in costs) >= 1:
            raise ValueError(
                "the costs should add up to less than 1, so every step is still "
                "worth surviving (otherwise ending the episode early can pay)"
            )
        if self.push_cost and self.actions != 3:
            raise ValueError("push_cost needs actions 3")
        if self.force_cost and self.actions != "continuous":
            raise ValueError("force_cost needs actions continuous")
        if self.force_std <= 0:
            raise ValueError("force_std should be more than 0")
        if self.force_std != 5.0 and self.actions != "continuous":
            raise ValueError("force_std needs actions continuous")


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def discounted_returns(rewards, gamma):
    """Return G_t = r_t + gamma * G_{t+1} for each step, working backwards
    from the end of the episode, where the return is 0."""
    returns = np.zeros(len(rewards))
    later = 0.0
    for t in reversed(range(len(rewards))):
        later = rewards[t] + gamma * later
        returns[t] = later
    return returns


class LinearPolicy:
    """Pushes right with probability sigmoid(weights · inputs), else left.

    For the trained policy without sampling, use cartpole.py's
    LinearController with the same weights.
    """

    def __init__(self, weights):
        self.weights = np.array(weights, dtype=float)

    def prob_right(self, inputs):
        """Probability of pushing right given inputs."""
        return sigmoid(self.weights @ inputs)

    def sample(self, inputs, rng):
        """Choose an action at random from the policy.

        Returns (action, p): action is 1 to push right, 0 to push left,
        and p is the probability of pushing right it was drawn with.
        """
        p = self.prob_right(inputs)
        action = 1 if rng.random() < p else 0
        return action, p

    def force(self, action):
        """The push, in newtons, for an action."""
        return FORCE if action == 1 else -FORCE

    def gradient(self, inputs, actions, probs, advantages):
        """Mean over the steps of A_t (a_t - p_t) s_t: the uphill direction
        for the weights. (a_t - p_t) s_t is the gradient of
        log pi(a_t | s_t)."""
        return ((advantages * (actions - probs))[:, None] * inputs).mean(axis=0)

    def in_input_units(self, scales):
        """The weights for inputs that haven't been divided by scales."""
        return self.weights / scales


class ThreeActionPolicy:
    """Pushes left, doesn't push or pushes right (actions 0, 1, 2), with
    probabilities softmax(z), where z_k = weights[k] · (inputs, 1).

    weights has a row of five per action: one for each input, then a bias.
    For the trained policy without sampling, use cartpole.py's
    ThreeActionController with the same weights.
    """

    FORCES = (-FORCE, 0.0, FORCE)

    def __init__(self, weights):
        self.weights = np.array(weights, dtype=float)

    def probs(self, inputs):
        """Probabilities of the three actions given inputs."""
        scores = self.weights[:, :4] @ inputs + self.weights[:, 4]
        exp = np.exp(scores - scores.max())  # the same softmax, without overflow
        return exp / exp.sum()

    def sample(self, inputs, rng):
        """Choose an action at random from the policy. Returns (action,
        probs): the action, and the probabilities it was drawn with."""
        probs = self.probs(inputs)
        action = int(np.searchsorted(np.cumsum(probs), rng.random(), side="right"))
        return min(action, 2), probs

    def force(self, action):
        """The push, in newtons, for an action."""
        return self.FORCES[action]

    def gradient(self, inputs, actions, probs, advantages):
        """Mean over the steps of A_t (onehot(a_t) - probs_t) (s_t, 1): the
        uphill direction for each action's weights. Row k of
        (onehot(a_t) - probs_t) (s_t, 1) is the gradient of log pi(a_t | s_t)
        with respect to action k's weights."""
        features = np.hstack([inputs, np.ones((len(inputs), 1))])
        taken = np.eye(3)[actions]
        return np.einsum("t,tk,ti->ki", advantages, taken - probs, features) / len(
            inputs
        )

    def in_input_units(self, scales):
        """The weights for inputs that haven't been divided by scales (the
        biases are unchanged)."""
        weights = self.weights.copy()
        weights[:, :4] /= scales
        return weights


class GaussianPolicy:
    """Pushes with a force drawn from a normal distribution whose mean is
    weights[:4] · inputs and whose standard deviation is exp(weights[4])
    newtons, capped at the maximum force when applied.

    The mean is in newtons, so the first four weights are the gains of a
    PD controller (Kx, Kv, Kp, Kd); the last is the log of the spread,
    learned so exploration can narrow as training gains confidence. For
    the trained policy without sampling (the mean force, capped), use
    cartpole.py's PDController with those gains.
    """

    def __init__(self, weights):
        self.weights = np.array(weights, dtype=float)

    @property
    def std(self):
        """The spread of the sampled force, in newtons."""
        return math.exp(self.weights[4])

    def mean(self, inputs):
        """The mean force given inputs, in newtons."""
        return self.weights[:4] @ inputs

    def sample(self, inputs, rng):
        """Draw a force at random from the policy. Returns (force, mean):
        the force before capping, and the mean it was drawn around."""
        mean = self.mean(inputs)
        return mean + self.std * rng.standard_normal(), mean

    def force(self, action):
        """The push, in newtons, for a sampled force: capped at the maximum."""
        return float(np.clip(action, -FORCE, FORCE))

    def gradient(self, inputs, actions, means, advantages):
        """Mean over the steps of A_t times the gradient of log pi(a_t | s_t).
        With z = (a - mean) / std, that's z / std * s for the gains, which
        moves the mean toward forces that did better than average, and
        z^2 - 1 for the log of the spread, which widens it when forces far
        from the mean did better and narrows it when near ones did."""
        z = (actions - means) / self.std
        gains = ((advantages * z / self.std)[:, None] * inputs).mean(axis=0)
        return np.append(gains, (advantages * (z**2 - 1)).mean())

    def in_input_units(self, scales):
        """The weights for inputs that haven't been divided by scales (the
        spread is unchanged)."""
        weights = self.weights.copy()
        weights[:4] /= scales
        return weights


def weights_kind_for(actions):
    """The kind of weights (see cartpole.py) a policy with these actions
    learns."""
    return {2: TWO_ACTION, 3: THREE_ACTION, "continuous": CONTINUOUS}[actions]


def new_policy(config):
    """A policy to start training from: all-zero weights, so every action
    is equally likely, or with continuous actions a mean force of 0 and
    a spread of force_std."""
    if config.actions == 2:
        return LinearPolicy(np.zeros(4))
    if config.actions == 3:
        return ThreeActionPolicy(np.zeros((3, 5)))
    return GaussianPolicy([0, 0, 0, 0, math.log(config.force_std)])


@dataclasses.dataclass(frozen=True)
class Costs:
    """What's taken off each step's reward of 1 for surviving: push for any
    push, force times (force / 10 N) squared, angle times (pole angle /
    angle limit) squared, and position times (cart position / track
    limit) squared. The squared fractions are capped at 1, and the costs
    add up to less than 1 (TrainConfig checks), so every step is still
    worth surviving."""

    push: float = 0.0
    force: float = 0.0
    angle: float = 0.0
    position: float = 0.0

    @staticmethod
    def of(config):
        return Costs(
            config.push_cost, config.force_cost, config.angle_cost, config.position_cost
        )

    def reward(self, survival, state, force, cartpole):
        """The reward for a step that pushed with force and ended in state:
        survival (the environment's reward) less the costs."""
        reward = survival
        if self.push and force != 0:
            reward -= self.push
        if self.force:
            reward -= self.force * min((force / FORCE) ** 2, 1.0)
        if self.angle:
            reward -= self.angle * min((state[2] / cartpole.angle_limit) ** 2, 1.0)
        if self.position:
            reward -= self.position * min((state[0] / cartpole.track_limit) ** 2, 1.0)
        return reward


NO_COSTS = Costs()


def play_episode(
    cartpole, policy, rate_from, rng, seed=None, scales=None, costs=NO_COSTS
):
    """Play one episode, sampling actions from policy.

    The policy's inputs come from the state through a RateEstimator, so
    with rate_from "difference" it sees the rates the way cartpole.py's
    controllers estimate them. If scales is given, each input is divided
    by its scale. Each step's reward is reduced by costs.

    Returns (inputs, actions, probs, rewards), one entry per step: what
    the policy saw, the action, what the policy's sample() returned with
    it (the probability of pushing right, of each action, or the mean
    force), and the reward that followed.
    """
    rates = RateEstimator(cartpole.tau, rate_from)
    inputs, actions, probs, rewards = [], [], [], []
    state = cartpole.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        seen = np.array(rates.update(state))
        if scales is not None:
            seen = seen / scales
        action, p = policy.sample(seen, rng)
        inputs.append(seen)
        actions.append(action)
        probs.append(p)
        force = policy.force(action)
        state, reward, terminated, truncated = cartpole.step(force)
        rewards.append(costs.reward(reward, state, force, cartpole))
    return np.array(inputs), np.array(actions), np.array(probs), rewards


def policy_gradient(policy, episodes, gamma):
    """Estimate the uphill direction for policy's weights from a batch of
    episodes: the mean over every step of A_t times the gradient of
    log pi(a_t | s_t), the direction that makes the action taken more
    likely in that state (see the policy's gradient()).

    A_t is the step's return standardized over the whole batch, so steps
    followed by better-than-average returns count for, and the rest
    against.
    """
    inputs = np.concatenate([s for s, _, _, _ in episodes])
    actions = np.concatenate([a for _, a, _, _ in episodes])
    probs = np.concatenate([p for _, _, p, _ in episodes])
    returns = np.concatenate([discounted_returns(r, gamma) for _, _, _, r in episodes])
    advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
    return policy.gradient(inputs, actions, probs, advantages)


def measure_input_scales(cartpole, config, rng, episodes):
    """Return the standard deviation of each of the policy's inputs over
    episodes played the way training starts (see new_policy)."""
    start = new_policy(config)
    inputs = np.concatenate(
        [
            play_episode(
                cartpole, start, config.rate_from, rng, config.seed if i == 0 else None
            )[0]
            for i in range(episodes)
        ]
    )
    scales = inputs.std(axis=0)
    return np.where(scales > 0, scales, 1.0)


@dataclasses.dataclass
class TestResult:
    """What evaluate() measured: each episode's length in steps, the mean
    reward per episode (survival less costs), and control_quality() over
    all the episodes together."""

    lengths: list
    mean_return: float
    quality: dict


def evaluate(
    cartpole, weights, rate_from, episodes, seed=None, kind=None, costs=NO_COSTS
):
    """Play episodes with the weights (of the given kind) as a controller,
    without sampling, and return a TestResult. seed, if given, sets the
    first starting state, which makes the whole set repeatable."""
    lengths, returns, all_states, all_forces, reversals = [], [], [], [], 0
    for i in range(episodes):
        controller = ForceRecorder(
            controller_from_weights(weights, cartpole.tau, FORCE, rate_from, kind)
        )
        states, _ = run_simulation(
            cartpole,
            controller,
            seed=seed if i == 0 else None,
            start_pause=0,
            end_pause=0,
        )
        lengths.append(len(states) - 1)
        returns.append(
            sum(
                costs.reward(1.0, state, force, cartpole)
                for state, force in zip(states[1:], controller.forces, strict=True)
            )
        )
        all_states += states
        all_forces += controller.forces
        reversals += count_reversals(controller.forces)
    quality = control_quality(all_states, all_forces, cartpole.tau)
    quality["reversals_per_second"] = reversals / (sum(lengths) * cartpole.tau)
    return TestResult(lengths, float(np.mean(returns)), quality)


def scale_to_kp(weights):
    """Scale weights so the pole-angle weight matches the PD gains' Kp.

    As a controller, the policy only uses the sign of weights · inputs, so
    multiplying every weight by the same positive number changes nothing;
    scaling lets the weights be read in the PD gains' units.
    """
    weights = np.asarray(weights)
    return weights * PD_GAINS[2] / weights[2]


def dead_band(weights):
    """Read three-action weights as a dead band, assuming push left and
    push right mirror each other: push right when gains · inputs >
    threshold, left when it's below -threshold, and otherwise coast.
    Returns (gains, threshold), unscaled."""
    left, coast, right = np.asarray(weights)
    gains = (right[:4] - left[:4]) / 2
    threshold = coast[4] - (right[4] + left[4]) / 2
    return gains, threshold


def describe_dead_band(weights):
    """One line: the dead band's gains scaled to Kp = 20, and its
    half-width as a pole angle (how far the pole alone may lean before a
    push)."""
    gains, threshold = dead_band(weights)
    if gains[2] <= 0:
        return "(not yet a dead band: the angle gain isn't positive)"
    scaled = scale_to_kp(gains)
    width = (
        f"{math.degrees(threshold / gains[2]):6.2f} deg"
        if threshold > 0
        else "none (coasting never wins)"
    )
    return f"{scaled[0]:6.2f} {scaled[1]:6.2f} {scaled[3]:6.2f}  {width}"


CHECKPOINT_COLUMNS = [
    "batch",
    "sampled_mean_length",
    "test_mean_length",
    "test_min_length",
    "test_mean_return",
    "test_push_fraction",
    "test_reversals_per_second",
    "test_mean_force",
    "test_rms_angle_degrees",
    "best_so_far",
]


def relies_on_randomness(sampled_mean_length, test_mean_length, max_steps):
    """Whether the policy did clearly better sampling its actions (in the
    training batches) than taking the likeliest one (in a checkpoint test):
    by more than a tenth of the longest episode. Then its randomness is
    doing useful work, which the deterministic controller loses."""
    return sampled_mean_length - test_mean_length > max_steps / 10


@dataclasses.dataclass
class TrainingResult:
    """What train() returns.

    weights are the ones to use: the best checkpoint's, or the final
    weights without checkpoints. Both are in the inputs' own units even
    when training normalized them. scales are what the inputs were divided
    by (all 1 without normalize_inputs). curve has a row per batch and
    checkpoints a row per checkpoint (see CHECKPOINT_COLUMNS); best_batch
    is the batch the kept weights come from. With continuous actions the
    weights are the four gains, and force_std is the spread of the sampled
    force at the end.
    """

    weights: np.ndarray
    final_weights: np.ndarray
    scales: np.ndarray
    curve: list
    checkpoints: list
    best_batch: int
    force_std: float | None = None


def train(cartpole, config, rng, report_every=10):
    """Train a policy (LinearPolicy, or ThreeActionPolicy with actions 3),
    starting from all-zero weights (every action equally likely), and
    return a TrainingResult.

    curve has one row per batch: (batch, episodes so far, mean, shortest
    and longest episode in the batch, then with actions 3 the fraction of
    steps that pushed, then all the weights after the batch's update, in
    the inputs' own units). Every report_every batches, the row is also
    printed.

    With checkpoint_every, every that many batches and after the last, the
    policy is tested without sampling, scored by the reward training uses
    (steps survived, less push_cost on pushing steps), and the best
    weights so far are kept. Ties go to the later checkpoint, which has
    trained longer.
    """
    if config.normalize_inputs:
        scales = measure_input_scales(cartpole, config, rng, SCALE_EPISODES)
        print("Input scales (x, x rate, angle, angle rate):", np.round(scales, 3))
    else:
        scales = np.ones(4)
    policy = new_policy(config)
    kind, costs = weights_kind_for(config.actions), Costs.of(config)
    curve, checkpoints = [], []
    best_weights, best_return, best_batch = None, -math.inf, config.batches
    checkpoint_cartpole = CartPole.make(angle_limit=None)
    checkpoint_seed = (config.seed or 0) + CHECKPOINT_SEED_OFFSET
    if config.actions == 2:
        shown_heading = "weights (x, x rate, angle, angle rate)"
    elif config.actions == 3:
        shown_heading = "pushing  as a dead band: Kx, Kv, Kd at Kp=20; half-width"
    else:
        shown_heading = "gains: Kx      Kv      Kp      Kd  force std (N)"
    print(
        f"{'batch':>5}  {'episodes':>8}  {'mean len':>8}  {'min':>4}  "
        f"{'max':>4}  {shown_heading}"
    )
    for batch in range(1, config.batches + 1):
        episodes = []
        for i in range(config.batch_size):
            episode_seed = config.seed if batch == 1 and i == 0 else None
            episodes.append(
                play_episode(
                    cartpole,
                    policy,
                    config.rate_from,
                    rng,
                    episode_seed,
                    scales,
                    costs,
                )
            )
        policy.weights += config.learning_rate * policy_gradient(
            policy, episodes, config.gamma
        )
        if kind == CONTINUOUS:
            policy.weights[4] = max(policy.weights[4], math.log(MIN_FORCE_STD))
        weights = policy.in_input_units(scales)
        force_std = None
        if kind == CONTINUOUS:  # the gains, and the spread
            weights, force_std = weights[:4], math.exp(weights[4])
        lengths = [len(r) for _, _, _, r in episodes]
        row = (
            batch,
            batch * config.batch_size,
            float(np.mean(lengths)),
            min(lengths),
            max(lengths),
        )
        if config.actions == 3:
            actions = np.concatenate([a for _, a, _, _ in episodes])
            push_fraction = float(np.mean(actions != 1))
            row += (push_fraction,)
        row += tuple(weights.ravel().tolist())
        if force_std is not None:
            row += (force_std,)
        curve.append(row)
        if batch % report_every == 0 or batch == 1 or batch == config.batches:
            if config.actions == 2:
                shown = " ".join(f"{w:7.3f}" for w in weights)
            elif config.actions == 3:
                shown = f"{100 * push_fraction:6.1f}%  {describe_dead_band(weights)}"
            else:
                shown = "       " + " ".join(f"{w:7.2f}" for w in weights)
                shown += f"  {force_std:6.2f}"
            print(
                f"{batch:5}  {row[1]:8}  {row[2]:8.1f}  {row[3]:4}  {row[4]:4}  {shown}"
            )
        if config.checkpoint_every and (
            batch % config.checkpoint_every == 0 or batch == config.batches
        ):
            test = evaluate(
                checkpoint_cartpole,
                weights,
                config.rate_from,
                config.checkpoint_episodes,
                checkpoint_seed,
                kind,
                costs,
            )
            test_mean = float(np.mean(test.lengths))
            test_return = test.mean_return
            since = checkpoints[-1][0] if checkpoints else 0
            sampled_mean = float(np.mean([r[2] for r in curve[since:batch]]))
            is_best = test_return >= best_return
            if is_best:
                best_weights, best_return, best_batch = (
                    weights.copy(),
                    test_return,
                    batch,
                )
            checkpoints.append(
                (
                    batch,
                    sampled_mean,
                    test_mean,
                    min(test.lengths),
                    test_return,
                    test.quality["push_fraction"],
                    test.quality["reversals_per_second"],
                    test.quality["mean_force"],
                    test.quality["rms_angle_degrees"],
                    int(is_best),
                )
            )
            flag = (
                "  sampling does better"
                if relies_on_randomness(
                    sampled_mean, test_mean, checkpoint_cartpole.max_steps
                )
                else ""
            )
            effort = (
                f"mean force {test.quality['mean_force']:.2f} N"
                if kind == CONTINUOUS
                else f"pushing {100 * test.quality['push_fraction']:.0f}%"
            )
            score = f", reward {test_return:.1f}" if costs != NO_COSTS else ""
            print(
                f"      checkpoint: without sampling {test_mean:.1f} steps{score}, "
                f"{effort}; sampled {sampled_mean:.1f}"
                + ("  (best so far)" if is_best else "")
                + flag
            )
    checkpoint_cartpole.close()
    return TrainingResult(
        weights=best_weights if best_weights is not None else weights,
        final_weights=weights,
        scales=scales,
        curve=curve,
        checkpoints=checkpoints,
        best_batch=best_batch,
        force_std=force_std,
    )


def print_comparison(weights, kind):
    """Print the weights (of the given kind) next to the hand-tuned gentle
    PD gains."""
    if kind == THREE_ACTION:
        print_dead_band_comparison(weights)
        return
    if kind == CONTINUOUS:
        print(
            "Learned gains (force = Kx x + Kv x rate + Kp angle + Kd angle rate, "
            "capped\nat 10 N, a PD controller), next to the hand-tuned gentle "
            "gains from\ncartpole.py:"
        )
        print(f"  {'':20}  {'learned':>8}  {'gentle PD':>9}")
        for name, w, g in zip(GAIN_NAMES, weights, PD_GAINS, strict=True):
            print(f"  {name:20}  {w:8.2f}  {g:9.2f}")
        return
    print(
        "Learned weights, scaled so the angle weight matches Kp, next to the\n"
        "hand-tuned gentle gains from cartpole.py (--controller pd --kp 20\n"
        "--kd 3 --kx 0.5 --kv 1):"
    )
    print(f"  {'':20}  {'learned':>8}  {'scaled':>8}  {'gentle PD':>9}")
    rows = zip(GAIN_NAMES, weights, scale_to_kp(weights), PD_GAINS, strict=True)
    for name, w, s, g in rows:
        print(f"  {name:20}  {w:8.3f}  {s:8.2f}  {g:9.2f}")


def print_dead_band_comparison(weights):
    """Print three-action weights read as a dead band, next to the
    hand-tuned gentle PD gains."""
    gains, threshold = dead_band(weights)
    print("Learned weights (rows: push left, don't push, push right; last: bias):")
    for name, row in zip(("left", "coast", "right"), weights, strict=True):
        print(f"  {name:6}" + "".join(f"{w:9.3f}" for w in row))
    if gains[2] <= 0:
        print("No dead band: the angle gain isn't positive.")
        return
    print(
        "Read as a dead band (push when gains · inputs passes ± threshold), "
        "scaled\nso the angle gain matches Kp, next to the hand-tuned gentle "
        "gains:"
    )
    print(f"  {'':20}  {'scaled':>8}  {'gentle PD':>9}")
    rows = zip(GAIN_NAMES, scale_to_kp(gains), PD_GAINS, strict=True)
    for name, s, g in rows:
        print(f"  {name:20}  {s:8.2f}  {g:9.2f}")
    if threshold > 0:
        print(
            f"  {'threshold':20}  {threshold * PD_GAINS[2] / gains[2]:8.3f}  "
            f"(the pole alone may lean {math.degrees(threshold / gains[2]):.2f} "
            "deg before a push)"
        )
    else:
        print("  No dead band: coasting never wins, so it always pushes.")


def print_checkpoint_summary(result, max_steps):
    """Say which checkpoint's weights were kept, and whether the policy
    was relying on its randomness at the end."""
    best = next(c for c in result.checkpoints if c[0] == result.best_batch)
    last = result.checkpoints[-1]
    print()
    effort = f"mean force {best[7]:.2f} N"
    if result.force_std is None:  # discrete actions: how often it pushed
        effort = f"pushing on {100 * best[5]:.0f}% of steps ({effort})"
    print(
        f"Keeping the weights from batch {result.best_batch}, the best "
        f"checkpoint: {best[2]:.1f} steps without sampling, reward {best[4]:.1f}, "
        f"{effort}."
    )
    if result.best_batch != last[0]:
        print(
            f"The final weights (batch {last[0]}: {last[2]:.1f} steps) are saved "
            "as final-weights.json."
        )
    if relies_on_randomness(last[1], last[2], max_steps):
        print(
            f"Note: at the end the policy did better sampling its actions "
            f"({last[1]:.1f} steps) than without sampling ({last[2]:.1f}): it's "
            "relying on its randomness, which the deterministic controller loses."
        )


def git_version():
    """Return (commit, dirty) for the code being run: the git commit hash,
    and whether there are uncommitted changes. (None, None) if git can't
    tell."""
    here = Path(__file__).parent

    def git(*command):
        return subprocess.run(
            ["git", *command], cwd=here, capture_output=True, text=True, check=True
        ).stdout

    try:
        return git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def default_run_name(config_file):
    """Name a run after its config file (or "default") and the time."""
    stem = Path(config_file).stem if config_file else "default"
    return f"{stem}-{datetime.datetime.now():%Y%m%d-%H%M%S}"


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def curve_columns(actions):
    """Column names for train()'s learning curve."""
    columns = ["batch", "episodes", "mean_length", "min_length", "max_length"]
    inputs = ["x", "x_rate", "angle", "angle_rate"]
    if actions == 2:
        return columns + [f"w_{name}" for name in inputs]
    if actions == "continuous":
        return columns + ["kx", "kv", "kp", "kd", "force_std"]
    return (
        columns
        + ["push_fraction"]
        + [
            f"w_{action}_{name}"
            for action in ("left", "coast", "right")
            for name in [*inputs, "bias"]
        ]
    )


def write_curve(path, curve, actions):
    """Write the learning curve from train() as CSV, one row per batch."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(curve_columns(actions))
        writer.writerows(curve)


def write_checkpoints(path, checkpoints):
    """Write train()'s checkpoint tests as CSV, one row per checkpoint."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CHECKPOINT_COLUMNS)
        writer.writerows(checkpoints)


def save_run(run_dir, config, result, test, max_steps, info):
    """Save everything about a training run in a new directory run_dir.

    result is what train() returned and test what evaluate() returned for
    its weights. info holds what else to record in results.json: when the
    run started, how long training took, the git version of the code and
    the input scales.
    """
    weights, curve = result.weights, result.curve
    kind = weights_kind_for(config.actions)
    lengths = test.lengths
    run_dir.mkdir(parents=True)
    save_config(run_dir / "config.json", config)
    save_weights(run_dir / "weights.json", weights, config.rate_from, kind)
    write_curve(run_dir / "curve.csv", curve, config.actions)
    if result.checkpoints:
        save_weights(
            run_dir / "final-weights.json",
            result.final_weights,
            config.rate_from,
            kind,
        )
        write_checkpoints(run_dir / "checkpoints.csv", result.checkpoints)
    _, _, mean, shortest, longest, *_ = curve[-1]
    results = {
        **info,
        "last_batch_lengths": {"mean": mean, "min": shortest, "max": longest},
        "test_lengths": {
            "episodes": len(lengths),
            "mean": float(np.mean(lengths)),
            "min": min(lengths),
            "max": max(lengths),
            "limit": max_steps,
        },
        "test_mean_return": test.mean_return,
        "test_push_fraction": test.quality["push_fraction"],
        "test_reversals_per_second": test.quality["reversals_per_second"],
        "test_quality": test.quality,
    }
    if result.checkpoints:
        best = next(c for c in result.checkpoints if c[0] == result.best_batch)
        last = result.checkpoints[-1]
        results["checkpoints"] = {
            "best_batch": result.best_batch,
            "best": dict(zip(CHECKPOINT_COLUMNS, best[:-1], strict=False)),
            "last": dict(zip(CHECKPOINT_COLUMNS, last[:-1], strict=False)),
            "sampling_does_better_at_last": relies_on_randomness(
                last[1], last[2], max_steps
            ),
        }
    if kind == TWO_ACTION:
        results["weights_scaled_to_kp"] = scale_to_kp(weights).tolist()
    elif kind == CONTINUOUS:
        results["gains"] = dict(
            zip(("kx", "kv", "kp", "kd"), weights.tolist(), strict=True)
        )
        results["final_force_std"] = result.force_std
    else:
        gains, threshold = dead_band(weights)
        results["dead_band"] = {
            "gains_scaled_to_kp": scale_to_kp(gains).tolist(),
            "threshold_scaled_to_kp": float(threshold * PD_GAINS[2] / gains[2]),
            "half_width_degrees": math.degrees(threshold / gains[2]),
        }
    save_json(run_dir / "results.json", results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        dest="config_file",
        default=None,
        help="JSON file of settings, e.g. from cartpole/configs/train/ or a run's "
        "config.json",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="directory name for the run under cartpole/runs/ (default: the "
        "config file's name, or 'default', then the date and time)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="don't save the run: train, test and watch, then keep nothing",
    )
    parser.add_argument(
        "--load",
        metavar="RUN_DIR",
        default=None,
        help="skip training: test and watch a saved run, with its saved settings",
    )
    parser.add_argument(
        "--slow",
        type=float,
        default=1.0,
        help="slow-motion factor for the window (default: 1)",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="don't open a window to watch the trained policy",
    )
    add_setting_options(parser, TrainConfig)
    args = parser.parse_args()
    if args.load:
        extra = given_settings(args, TrainConfig)
        extra += ["--config"] * bool(args.config_file)
        extra += ["--run-name"] * bool(args.run_name)
        extra += ["--no-save"] * args.no_save
        if extra:
            parser.error(
                f"--load uses the run's saved settings: drop {' '.join(extra)}"
            )
    try:
        if args.load:
            args.settings = load_config(Path(args.load) / "config.json", TrainConfig)
        else:
            args.settings = resolve(TrainConfig, args.config_file, args)
    except (OSError, ValueError) as e:
        parser.error(str(e))
    if args.no_save and args.run_name:
        parser.error("--no-save doesn't save a run, so it can't take --run-name")
    args.run_dir = None
    if not (args.load or args.no_save):
        args.run_dir = RUNS_DIR / (args.run_name or default_run_name(args.config_file))
        if args.run_dir.exists():
            parser.error(f"run directory already exists: {args.run_dir}")
    return args


def main():
    args = parse_args()
    config = args.settings
    if config.description:
        print(config.description)

    if args.load:
        weights, rate_from, kind = load_weights(Path(args.load) / "weights.json")
        weights = np.array(weights)
        print(f"Loaded {args.load}")
    else:
        info = {
            "started": datetime.datetime.now().astimezone().isoformat("T", "seconds")
        }
        commit, dirty = git_version()
        cartpole = CartPole.make(angle_limit=None)  # standard CartPole
        start = time.perf_counter()
        result = train(cartpole, config, np.random.default_rng(config.seed))
        info["training_seconds"] = round(time.perf_counter() - start, 2)
        info["git_commit"], info["git_dirty"] = commit, dirty
        info["input_scales"] = (
            result.scales.tolist() if config.normalize_inputs else None
        )
        cartpole.close()
        weights, rate_from = result.weights, config.rate_from
        kind = weights_kind_for(config.actions)
        if result.checkpoints:
            print_checkpoint_summary(result, cartpole.max_steps)

    print()
    print_comparison(weights, kind)
    cartpole = CartPole.make(angle_limit=None)
    test = evaluate(
        cartpole,
        weights,
        rate_from,
        config.eval_episodes,
        config.seed,
        kind,
        Costs.of(config),
    )
    cartpole.close()
    lengths = test.lengths
    print()
    print(
        f"Without sampling, over {config.eval_episodes} episodes: mean "
        f"{np.mean(lengths):.1f} steps, shortest {min(lengths)}, longest "
        f"{max(lengths)} (limit {cartpole.max_steps}); mean reward "
        f"{test.mean_return:.1f}"
    )
    print_quality(test.quality)

    if args.run_dir:
        save_run(args.run_dir, config, result, test, cartpole.max_steps, info)
        print(f"Saved the run to {os.path.relpath(args.run_dir)}")
    elif args.no_save:
        print("Not saved (--no-save).")

    if not args.no_window:
        cartpole = CartPole.make(render_mode="human", slow=args.slow, angle_limit=None)
        controller = controller_from_weights(
            weights, cartpole.tau, FORCE, rate_from, kind
        )
        states, _ = run_simulation(cartpole, controller)
        cartpole.close()
        print_summary(states, cartpole.tau, cartpole.angle_limit, cartpole.track_limit)


if __name__ == "__main__":
    main()
