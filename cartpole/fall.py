"""Watch the CartPole pole fall over with no control at all.

Uses Gymnasium's CartPole-v1 physics and renderer, with two changes:

- The push force is set to zero, so whichever action we pass, nothing
  acts on the cart except the pole itself. (CartPole has no "do
  nothing" action: action 0 pushes left and 1 pushes right, normally
  with 10 N.)
- The failure angle is raised from 12 degrees to 90 degrees, so the
  episode runs until the pole is lying flat instead of stopping as soon
  as it tilts past 12 degrees.

Run from the repo root:

    uv run python cartpole/fall.py
    uv run python cartpole/fall.py --slow 4 --seed 7
"""

import argparse
import math

import gymnasium as gym

STANDARD_FAIL_DEGREES = 12  # where a normal CartPole episode ends


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for the starting state (default: different each run)",
    )
    parser.add_argument(
        "--slow",
        type=float,
        default=1.0,
        help="slow-motion factor, e.g. 4 plays at quarter speed (default: 1)",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="don't open a window, just print the trace",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.no_window else "human"
    env = gym.make("CartPole-v1", render_mode=render_mode)

    cartpole = env.unwrapped
    cartpole.force_mag = 0.0  # no control: actions push with zero force
    cartpole.theta_threshold_radians = math.pi / 2  # run until the pole is flat
    fps = cartpole.metadata["render_fps"] / args.slow
    cartpole.metadata = {**cartpole.metadata, "render_fps": fps}

    state, _ = env.reset(seed=args.seed)
    hold(env, seconds=1.0, fps=fps)  # show the starting position

    print(f"{'time (s)':>8}  {'cart x (m)':>10}  {'pole angle':>10}")
    print_row(0, cartpole.tau, state)
    passed_standard_limit = False
    step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        # The action doesn't matter, since the force is zero either way.
        state, _, terminated, truncated, _ = env.step(0)
        step += 1
        if step % 5 == 0 or terminated:
            print_row(step, cartpole.tau, state)
        if not passed_standard_limit and abs(state[2]) > math.radians(
            STANDARD_FAIL_DEGREES
        ):
            passed_standard_limit = True
            print(
                f"  -- passed {STANDARD_FAIL_DEGREES} degrees at "
                f"{step * cartpole.tau:.2f} s: a normal episode would end here"
            )

    hold(env, seconds=2.0, fps=fps)  # show where it ended up
    env.close()

    t = step * cartpole.tau
    if abs(state[2]) >= math.pi / 2:
        print(f"Pole lying flat after {t:.2f} s ({step} steps of {cartpole.tau} s).")
    else:
        print(f"Stopped after {t:.2f} s: the cart left the track.")


def print_row(step, tau, state):
    x, _, theta, _ = state
    print(f"{step * tau:8.2f}  {x:10.3f}  {math.degrees(theta):9.1f}°")


def hold(env, seconds, fps):
    """Keep showing the current frame (each render waits one frame)."""
    if env.unwrapped.render_mode == "human":
        for _ in range(round(seconds * fps)):
            env.render()


if __name__ == "__main__":
    main()
