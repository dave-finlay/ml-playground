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

Training uses standard CartPole: 12 degrees, 2.4 m and 500 steps (10 s).
Afterwards the trained policy plays without sampling, always taking the
likelier action, which makes it a controller like the hand-tuned ones in
cartpole.py: push right when w · s >= 0 (cartpole.py's LinearController).
So the weights play the roles of the PD controller's gains Kx, Kv, Kp and
Kd.

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
    uv run python cartpole/train.py --config cartpole/configs/standard.json --seed 3
    uv run python cartpole/train.py --learning-rate 1 --run-name slow-learning
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
    CartPole,
    LinearController,
    RateEstimator,
    load_weights,
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
    batches: int = setting(200, "number of weight updates")
    batch_size: int = setting(10, "episodes played per update")
    learning_rate: float = setting(10.0, "step size for each update")
    gamma: float = setting(0.99, "discount factor for future rewards")
    eval_episodes: int = setting(
        20, "episodes to test the trained policy on, without sampling"
    )

    def __post_init__(self):
        for name in ("batches", "batch_size", "eval_episodes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} should be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate should be more than 0")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma should be more than 0 and at most 1")


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


def play_episode(cartpole, policy, rate_from, rng, seed=None):
    """Play one episode, sampling actions from policy.

    The policy's inputs come from the state through a RateEstimator, so
    with rate_from "difference" it sees the rates the way cartpole.py's
    controllers estimate them.

    Returns (inputs, actions, probs, rewards), one entry per step: what
    the policy saw, the action (1 right, 0 left), the probability of
    pushing right, and the reward that followed.
    """
    rates = RateEstimator(cartpole.tau, rate_from)
    inputs, actions, probs, rewards = [], [], [], []
    state = cartpole.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        seen = np.array(rates.update(state))
        action, p = policy.sample(seen, rng)
        inputs.append(seen)
        actions.append(action)
        probs.append(p)
        state, reward, terminated, truncated = cartpole.step(
            FORCE if action == 1 else -FORCE
        )
        rewards.append(reward)
    return np.array(inputs), np.array(actions), np.array(probs), rewards


def policy_gradient(episodes, gamma):
    """Estimate the uphill direction for the weights from a batch of
    episodes: the mean over every step of A_t (a_t - p_t) s_t, where s_t
    is what the policy saw.

    A_t is the step's return standardized over the whole batch, so steps
    followed by better-than-average returns count for, and the rest
    against. (a_t - p_t) s_t is the gradient of log pi(a_t | s_t): the
    direction that makes the action taken more likely in that state.
    """
    inputs = np.concatenate([s for s, _, _, _ in episodes])
    actions = np.concatenate([a for _, a, _, _ in episodes])
    probs = np.concatenate([p for _, _, p, _ in episodes])
    returns = np.concatenate([discounted_returns(r, gamma) for _, _, _, r in episodes])
    advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
    return ((advantages * (actions - probs))[:, None] * inputs).mean(axis=0)


def train(cartpole, config, rng, report_every=10):
    """Train a LinearPolicy, starting from all-zero weights (a coin toss
    every step), and return (policy, curve).

    curve has one row per batch: (batch, episodes so far, mean, shortest
    and longest episode in the batch, then the four weights after the
    batch's update). Every report_every batches, the row is also printed.
    """
    policy = LinearPolicy(np.zeros(4))
    curve = []
    print(
        f"{'batch':>5}  {'episodes':>8}  {'mean len':>8}  {'min':>4}  "
        f"{'max':>4}  weights (x, x rate, angle, angle rate)"
    )
    for batch in range(1, config.batches + 1):
        episodes = []
        for i in range(config.batch_size):
            episode_seed = config.seed if batch == 1 and i == 0 else None
            episodes.append(
                play_episode(cartpole, policy, config.rate_from, rng, episode_seed)
            )
        policy.weights += config.learning_rate * policy_gradient(episodes, config.gamma)
        lengths = [len(r) for _, _, _, r in episodes]
        row = (
            batch,
            batch * config.batch_size,
            float(np.mean(lengths)),
            min(lengths),
            max(lengths),
            *policy.weights.tolist(),
        )
        curve.append(row)
        if batch % report_every == 0 or batch == 1 or batch == config.batches:
            weights = " ".join(f"{w:7.3f}" for w in policy.weights)
            print(
                f"{batch:5}  {row[1]:8}  {row[2]:8.1f}  {row[3]:4}  {row[4]:4}  "
                f"{weights}"
            )
    return policy, curve


def evaluate(cartpole, weights, rate_from, episodes, seed=None):
    """Play episodes with the weights as a controller (no sampling) and
    return their lengths in steps. seed, if given, sets the first starting
    state, which makes the whole set repeatable."""
    lengths = []
    for i in range(episodes):
        controller = LinearController(weights, cartpole.tau, FORCE, rate_from)
        states, _ = run_simulation(
            cartpole,
            controller,
            seed=seed if i == 0 else None,
            start_pause=0,
            end_pause=0,
        )
        lengths.append(len(states) - 1)
    return lengths


def scale_to_kp(weights):
    """Scale weights so the pole-angle weight matches the PD gains' Kp.

    As a controller, the policy only uses the sign of weights · inputs, so
    multiplying every weight by the same positive number changes nothing;
    scaling lets the weights be read in the PD gains' units.
    """
    weights = np.asarray(weights)
    return weights * PD_GAINS[2] / weights[2]


def print_comparison(weights):
    """Print the weights next to the hand-tuned gentle PD gains."""
    print(
        "Learned weights, scaled so the angle weight matches Kp, next to the\n"
        "hand-tuned gentle gains from cartpole.py (--controller pd --kp 20\n"
        "--kd 3 --kx 0.5 --kv 1):"
    )
    print(f"  {'':20}  {'learned':>8}  {'scaled':>8}  {'gentle PD':>9}")
    rows = zip(GAIN_NAMES, weights, scale_to_kp(weights), PD_GAINS, strict=True)
    for name, w, s, g in rows:
        print(f"  {name:20}  {w:8.3f}  {s:8.2f}  {g:9.2f}")


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


def write_curve(path, curve):
    """Write the learning curve from train() as CSV, one row per batch."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "batch",
                "episodes",
                "mean_length",
                "min_length",
                "max_length",
                "w_x",
                "w_x_rate",
                "w_angle",
                "w_angle_rate",
            ]
        )
        writer.writerows(curve)


def save_run(run_dir, config, weights, curve, lengths, max_steps, info):
    """Save everything about a training run in a new directory run_dir.

    info holds what else to record in results.json: when the run started,
    how long training took and the git version of the code.
    """
    run_dir.mkdir(parents=True)
    save_config(run_dir / "config.json", config)
    save_weights(run_dir / "weights.json", weights, config.rate_from)
    write_curve(run_dir / "curve.csv", curve)
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
        "weights_scaled_to_kp": scale_to_kp(weights).tolist(),
    }
    save_json(run_dir / "results.json", results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        dest="config_file",
        default=None,
        help="JSON file of settings, e.g. from cartpole/configs/ or a run's "
        "config.json",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="directory name for the run under cartpole/runs/ (default: the "
        "config file's name, or 'default', then the date and time)",
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
    if not args.load:
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
        weights, rate_from = load_weights(Path(args.load) / "weights.json")
        weights = np.array(weights)
        print(f"Loaded {args.load}")
    else:
        info = {
            "started": datetime.datetime.now().astimezone().isoformat("T", "seconds")
        }
        commit, dirty = git_version()
        cartpole = CartPole.make(angle_limit=None)  # standard CartPole
        start = time.perf_counter()
        policy, curve = train(cartpole, config, np.random.default_rng(config.seed))
        info["training_seconds"] = round(time.perf_counter() - start, 2)
        info["git_commit"], info["git_dirty"] = commit, dirty
        cartpole.close()
        weights, rate_from = policy.weights, config.rate_from

    print()
    print_comparison(weights)
    cartpole = CartPole.make(angle_limit=None)
    lengths = evaluate(cartpole, weights, rate_from, config.eval_episodes, config.seed)
    cartpole.close()
    print()
    print(
        f"Without sampling, over {config.eval_episodes} episodes: mean "
        f"{np.mean(lengths):.1f} steps, shortest {min(lengths)}, longest "
        f"{max(lengths)} (limit {cartpole.max_steps})"
    )

    if not args.load:
        save_run(
            args.run_dir, config, weights, curve, lengths, cartpole.max_steps, info
        )
        print(f"Saved the run to {os.path.relpath(args.run_dir)}")

    if not args.no_window:
        cartpole = CartPole.make(render_mode="human", slow=args.slow, angle_limit=None)
        controller = LinearController(weights, cartpole.tau, FORCE, rate_from)
        states, _ = run_simulation(cartpole, controller)
        cartpole.close()
        print_summary(states, cartpole.tau, cartpole.angle_limit, cartpole.track_limit)


if __name__ == "__main__":
    main()
