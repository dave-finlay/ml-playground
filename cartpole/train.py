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
Kd. To run saved weights in cartpole.py, with its longer track and
episodes and its imperfect sensors:

    uv run python cartpole/cartpole.py --controller learned \
        --weights cartpole/linear-policy.json

Run from the repo root:

    uv run python cartpole/train.py
    uv run python cartpole/train.py --seed 3 --batches 100 --no-window
    uv run python cartpole/train.py --learning-rate 1
    uv run python cartpole/train.py --rate-from velocity
    uv run python cartpole/train.py --load cartpole/linear-policy.json
"""

import argparse
import math

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

FORCE = 10.0  # standard CartPole's push, in newtons

# The hand-tuned "gentle" gains from cartpole.py's PD controller, for
# comparison with the learned weights, in the state's order: Kx, Kv, Kp, Kd.
PD_GAINS = np.array([0.5, 1.0, 20.0, 3.0])
GAIN_NAMES = [
    "Kx (cart position)",
    "Kv (cart velocity)",
    "Kp (pole angle)",
    "Kd (pole rate)",
]


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


def train(
    cartpole,
    batches,
    batch_size,
    learning_rate,
    gamma,
    rate_from,
    rng,
    seed=None,
    report_every=10,
):
    """Train a LinearPolicy, starting from all-zero weights (a coin toss
    every step), and return it.

    Prints a learning curve as it goes: every report_every batches, the
    mean, shortest and longest episode in the latest batch, and the
    weights.
    """
    policy = LinearPolicy(np.zeros(4))
    print(
        f"{'batch':>5}  {'episodes':>8}  {'mean len':>8}  {'min':>4}  "
        f"{'max':>4}  weights (x, x rate, angle, angle rate)"
    )
    for batch in range(1, batches + 1):
        episodes = []
        for i in range(batch_size):
            episode_seed = seed if batch == 1 and i == 0 else None
            episodes.append(
                play_episode(cartpole, policy, rate_from, rng, episode_seed)
            )
        policy.weights += learning_rate * policy_gradient(episodes, gamma)
        if batch % report_every == 0 or batch == 1 or batch == batches:
            lengths = [len(r) for _, _, _, r in episodes]
            weights = " ".join(f"{w:7.3f}" for w in policy.weights)
            print(
                f"{batch:5}  {batch * batch_size:8}  {np.mean(lengths):8.1f}  "
                f"{min(lengths):4}  {max(lengths):4}  {weights}"
            )
    return policy


def evaluate(cartpole, weights, rate_from, episodes):
    """Play episodes with the weights as a controller (no sampling) and
    return their lengths in steps."""
    lengths = []
    for _ in range(episodes):
        controller = LinearController(weights, cartpole.tau, FORCE, rate_from)
        states, _ = run_simulation(cartpole, controller, start_pause=0, end_pause=0)
        lengths.append(len(states) - 1)
    return lengths


def print_comparison(weights):
    """Print the weights next to the hand-tuned PD gains.

    As a controller, the policy only uses the sign of weights · state, so
    multiplying every weight by the same positive number changes nothing.
    To compare like with like, the weights are scaled so the pole-angle
    weight matches Kp.
    """
    scaled = weights * PD_GAINS[2] / weights[2]
    print("Learned weights, scaled so the angle weight matches Kp:")
    print(f"  {'':20}  {'learned':>8}  {'scaled':>8}  {'PD gain':>8}")
    for name, w, s, g in zip(GAIN_NAMES, weights, scaled, PD_GAINS, strict=True):
        print(f"  {name:20}  {w:8.3f}  {s:8.2f}  {g:8.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for training and starting states (default: "
        "different each run)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=200,
        help="number of weight updates (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="episodes played per update (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=10.0,
        help="step size for each update (default: 10)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for future rewards (default: 0.99)",
    )
    parser.add_argument(
        "--rate-from",
        choices=["difference", "velocity"],
        default="difference",
        help="where the policy's rates of change come from: differences of "
        "successive positions and angles, or the true velocities (default: "
        "difference)",
    )
    parser.add_argument(
        "--save",
        default="cartpole/linear-policy.json",
        help="where to save the learned weights (default: cartpole/linear-policy.json)",
    )
    parser.add_argument(
        "--load",
        default=None,
        help="skip training and use weights saved earlier with --save (and the "
        "rate-from they were saved with)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="episodes to test the trained policy on, without sampling (default: 20)",
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
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cartpole = CartPole.make(angle_limit=None)  # standard CartPole

    if args.load:
        weights, rate_from = load_weights(args.load)
        weights = np.array(weights)
        print(f"Loaded weights from {args.load} (rates from {rate_from})")
    else:
        rate_from = args.rate_from
        policy = train(
            cartpole,
            args.batches,
            args.batch_size,
            args.learning_rate,
            args.gamma,
            rate_from,
            rng,
            seed=args.seed,
        )
        weights = policy.weights
        save_weights(args.save, weights, rate_from)
        print(f"Saved weights to {args.save}")

    print()
    print_comparison(weights)
    lengths = evaluate(cartpole, weights, rate_from, args.eval_episodes)
    print()
    print(
        f"Without sampling, over {args.eval_episodes} episodes: mean "
        f"{np.mean(lengths):.1f} steps, shortest {min(lengths)}, longest "
        f"{max(lengths)} (limit {cartpole.max_steps})"
    )
    cartpole.close()

    if not args.no_window:
        cartpole = CartPole.make(render_mode="human", slow=args.slow, angle_limit=None)
        controller = LinearController(weights, cartpole.tau, FORCE, rate_from)
        states, _ = run_simulation(cartpole, controller)
        cartpole.close()
        print_summary(states, cartpole.tau, cartpole.angle_limit, cartpole.track_limit)


if __name__ == "__main__":
    main()
