"""Compute the optimal linear controller for CartPole by LQR.

LQR (the linear-quadratic regulator) finds the gains of the controller

    force = Kx x + Kv x_rate + Kp angle + Kd angle_rate

that minimize the discounted sum of quadratic costs

    sum over steps of gamma^t (x' Q x + R force^2)

for a linear model of the cart and pole. The costs are given the way
train.py's reward gives them, so the answer can be compared with what
training learns for the same reward:

    Q = diag(position_cost / track_limit^2, 0, angle_cost / angle_limit^2, 0)
    R = force_cost / max_force^2

with standard CartPole's limits (2.4 m, 12 degrees, 10 N). The model is
Gymnasium's equations linearized about upright and stepped the way
Gymnasium steps them (explicit Euler, 0.02 s), so it's exact for small
angles. The gains come from iterating the discrete Riccati equation to
its fixed point.

LQR's answer is optimal only for the problem it's told about: linear,
with unlimited force and the true velocities. The real controller is
capped at 10 N, the pole isn't linear far from upright, and the rates
are estimated by differencing readings, which amplifies sensor noise,
so stiff LQR gains can do badly with a noisy sensor.

Settings can come from a train.py config (--config), which supplies the
reward's costs and gamma, overridden by command-line options. A force
cost is required: with free force, the optimal gains are infinite.

Run from the repo root:

    uv run python cartpole/lqr.py --force-cost 0.3 --angle-cost 0.3 \\
        --position-cost 0.2
    uv run python cartpole/lqr.py --force-cost 0.3 \\
        --config cartpole/configs/train/continuous-precise.json
    uv run python cartpole/lqr.py --force-cost 0.3 --angle-cost 0.3 \\
        --position-cost 0.2 --save cartpole/configs/cartpole/lqr-precise.json
"""

import argparse
import cmath
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from cartpole import CartPole
from config import add_setting_options, given_values, load_config, make_config, setting
from model import discretize, linearize
from train import GAIN_NAMES, PD_GAINS, TrainConfig

MAX_FORCE = 10.0  # newtons, the cap the cost is relative to


@dataclasses.dataclass(frozen=True)
class LQRConfig:
    """The costs LQR minimizes, in train.py's terms."""

    description: str = setting("", "what these costs are for")
    angle_cost: float = setting(
        0.0, "cost times (pole angle / 12 degrees) squared, per step"
    )
    position_cost: float = setting(
        0.0, "cost times (cart position / 2.4 m) squared, per step"
    )
    force_cost: float = setting(
        0.0, "cost times (force / 10 N) squared, per step; must be more than 0"
    )
    gamma: float = setting(0.99, "discount factor per step")

    def __post_init__(self):
        for name in ("angle_cost", "position_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} should be at least 0")
        if self.force_cost <= 0:
            raise ValueError(
                "force_cost should be more than 0: with free force, the optimal "
                "gains are infinite"
            )
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma should be more than 0 and at most 1")


def cost_matrices(config, angle_limit, track_limit):
    """Return (Q, R) for config's costs, relative to the limits."""
    q = np.diag(
        [
            config.position_cost / track_limit**2,
            0.0,
            config.angle_cost / angle_limit**2,
            0.0,
        ]
    )
    return q, np.array([[config.force_cost / MAX_FORCE**2]])


def solve_lqr(a, b, q, r, gamma, tolerance=1e-12, max_iterations=1_000_000):
    """Solve discounted discrete-time LQR. Returns (K, P): the optimal
    force is -K state, and P gives the optimal cost to go, state' P state.

    Iterates the Riccati equation
        K = gamma (R + gamma B' P B)^-1 B' P A
        P = Q + gamma A' P (A - B K)
    from P = Q until it stops changing.
    """
    p = q.copy()
    for _ in range(max_iterations):
        k = gamma * np.linalg.solve(r + gamma * b.T @ p @ b, b.T @ p @ a)
        new_p = q + gamma * a.T @ p @ (a - b @ k)
        if np.max(np.abs(new_p - p)) <= tolerance * max(1.0, np.max(np.abs(p))):
            return k, new_p
        p = new_p
    raise RuntimeError("the Riccati iteration didn't converge")


def closed_loop_modes(a, b, gains, tau):
    """The closed loop's modes with force = gains · state, as (decay rate
    per second, oscillation frequency in hertz) pairs, slowest decay
    first, each oscillating pair of poles once. A positive decay rate
    means the mode dies away; it falls to a third (1/e) in 1/rate
    seconds."""
    modes = []
    for z in np.linalg.eigvals(a + b @ np.atleast_2d(gains)):
        if z.imag < 0:
            continue  # the other half of an oscillating pair
        s = cmath.log(z) / tau  # the continuous-time pole it corresponds to
        modes.append((-s.real, abs(s.imag) / (2 * math.pi)))
    return sorted(modes)


def optimal_gains(config, cartpole):
    """Return (gains, discrete A, discrete B): the LQR gains for config on
    cartpole's physics, as Kx, Kv, Kp, Kd with force = gains · state."""
    a, b = discretize(*linearize(cartpole.physics), cartpole.tau)
    q, r = cost_matrices(config, cartpole.angle_limit, cartpole.track_limit)
    k, _ = solve_lqr(a, b, q, r, config.gamma)
    return -k.ravel(), a, b


def pd_command(gains):
    """The cartpole.py command that runs gains as a PD controller."""
    kx, kv, kp, kd = gains
    return (
        f"uv run python cartpole/cartpole.py --controller pd --kp {kp:.3f} "
        f"--kd {kd:.3f} --kx {kx:.3f} --kv {kv:.3f}"
    )


def save_simulation_config(path, gains, config):
    """Write a cartpole.py config that runs gains as a PD controller."""
    kx, kv, kp, kd = (round(float(g), 4) for g in gains)
    description = config.description or (
        f"LQR gains for angle cost {config.angle_cost}, position cost "
        f"{config.position_cost}, force cost {config.force_cost}"
    )
    saved = {
        "description": description,
        "controller": "pd",
        "kp": kp,
        "kd": kd,
        "kx": kx,
        "kv": kv,
    }
    with open(path, "x") as f:  # never overwrite
        json.dump(saved, f, indent=2)
        f.write("\n")


def print_result(config, gains, modes):
    print(
        f"LQR for angle cost {config.angle_cost}, position cost "
        f"{config.position_cost}, force cost {config.force_cost}, gamma "
        f"{config.gamma}:"
    )
    print(f"  {'':20}  {'LQR':>8}  {'gentle PD':>9}")
    for name, g, pd in zip(GAIN_NAMES, gains, PD_GAINS, strict=True):
        print(f"  {name:20}  {g:8.3f}  {pd:9.2f}")
    print(f"  Kd/Kp {gains[3] / gains[2]:.3f} s (gentle PD: 0.150 s)")
    print("Closed-loop modes, slowest first (on the linear model, true velocities):")
    for rate, frequency in modes:
        swing = f", oscillating at {frequency:.2f} Hz" if frequency > 1e-9 else ""
        print(f"  decays at {rate:6.2f}/s (time constant {1 / rate:5.2f} s){swing}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        dest="config_file",
        default=None,
        help="a train.py config, whose costs and gamma to use",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="write a cartpole.py config running the gains as a PD controller",
    )
    add_setting_options(parser, LQRConfig)
    args = parser.parse_args()
    try:
        values = {}
        if args.config_file:
            trained = load_config(args.config_file, TrainConfig)
            names = ("angle_cost", "position_cost", "force_cost", "gamma")
            values = {name: getattr(trained, name) for name in names}
        values.update(given_values(LQRConfig, None, args))
        args.settings = make_config(LQRConfig, values)
    except (OSError, ValueError) as e:
        parser.error(str(e))
    if args.save and Path(args.save).exists():
        parser.error(f"already exists: {args.save}")
    return args


def main():
    args = parse_args()
    config = args.settings
    cartpole = CartPole.make(angle_limit=None)  # standard limits, as in training
    gains, a, b = optimal_gains(config, cartpole)
    cartpole.close()
    print_result(config, gains, closed_loop_modes(a, b, gains, cartpole.tau))
    print()
    print("To run it:")
    print(f"  {pd_command(gains)}")
    if args.save:
        save_simulation_config(args.save, gains, config)
        print(f"Saved a cartpole.py config to {args.save}")


if __name__ == "__main__":
    main()
