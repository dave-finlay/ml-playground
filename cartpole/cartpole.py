"""Watch the CartPole pole fall over with no control at all.

Uses Gymnasium's CartPole-v1 physics and renderer, with two changes:

- Each step pushes the cart with a force chosen by a controller, here
  one that always chooses 0 N, so nothing acts on the cart except the
  pole itself. (Standard CartPole has no "do nothing" action: action 0
  pushes left and 1 pushes right, always with 10 N. CartPole.step works
  around that.)
- The failure angle is raised from 12 degrees to 90 degrees, so the
  episode runs until the pole is lying flat instead of stopping as soon
  as it tilts past 12 degrees.

Run from the repo root:

    uv run python cartpole/cartpole.py
    uv run python cartpole/cartpole.py --slow 4 --seed 7
"""

import argparse
import math

import gymnasium as gym


class CartPole:
    """A cart with a pole on it, simulated by Gymnasium's CartPole-v1.

    All access to the Gymnasium environment goes through this class, so
    reliance on its internals stays in one place. Create one with
    CartPole.make().
    """

    def __init__(self, env, normal_angle_limit):
        self._env = env
        self._normal_angle_limit = normal_angle_limit

    @staticmethod
    def make(render_mode=None, slow=1.0, angle_limit=math.pi / 2):
        """Create a CartPole whose episodes end when the pole passes angle_limit.

        angle_limit is in radians. The default, 90 degrees, runs until the
        pole is lying flat; standard CartPole stops at 12 degrees.

        render_mode is "human" for a window, "rgb_array" for frames as pixel
        arrays, or None for no drawing. slow is a slow-motion factor for the
        window: 4 plays at quarter speed.
        """
        env = gym.make("CartPole-v1", render_mode=render_mode)
        physics = env.unwrapped
        normal_angle_limit = physics.theta_threshold_radians
        physics.theta_threshold_radians = angle_limit
        fps = physics.metadata["render_fps"] / slow
        physics.metadata = {**physics.metadata, "render_fps": fps}
        return CartPole(env, normal_angle_limit)

    @property
    def tau(self):
        """Seconds between steps."""
        return self._env.unwrapped.tau

    @property
    def angle_limit(self):
        """Pole angle, in radians, at which this CartPole's episodes end."""
        return self._env.unwrapped.theta_threshold_radians

    @property
    def normal_angle_limit(self):
        """Pole angle, in radians, at which a standard CartPole-v1 episode
        ends (12 degrees as of Gymnasium 1.4), read from Gymnasium."""
        return self._normal_angle_limit

    def reset(self, seed=None):
        """Start a new episode and return the starting state.

        The state is (cart position, cart velocity, pole angle, pole angular
        velocity), each random in +-0.05. A seed makes it repeatable.
        """
        state, _ = self._env.reset(seed=seed)
        return state

    def step(self, force):
        """Advance one time step with a push of force newtons on the cart.

        Positive pushes right, negative pushes left, 0 is no push. Standard
        CartPole only allows full pushes of force_mag (10 N) left or right,
        so this sets force_mag before each step: it relies on CartPole's
        internals. Returns (state, reward, terminated, truncated).
        """
        self._env.unwrapped.force_mag = abs(force)
        action = 1 if force >= 0 else 0  # 1 pushes right, 0 pushes left
        state, reward, terminated, truncated, _ = self._env.step(action)
        return state, reward, terminated, truncated

    def pause(self, seconds):
        """Keep showing the current frame in the window, if there is one.

        Each render call waits one frame, so this renders seconds * fps times.
        """
        physics = self._env.unwrapped
        if physics.render_mode == "human":
            for _ in range(round(seconds * physics.metadata["render_fps"])):
                self._env.render()

    def close(self):
        """Close the window, if there is one."""
        self._env.close()


def no_control(state):
    """Controller that never pushes the cart."""
    return 0.0


def run_simulation(cartpole, controller, seed=None, start_pause=1.0, end_pause=2.0):
    """Run one episode under controller and return the states it went through.

    Resets cartpole (with seed, if given) and steps it until the episode
    ends. On each step, controller(state) gives the force in newtons to
    push the cart with (see CartPole.step).

    Returns a list of states, one per time step, starting with the initial
    state. Each state is (cart position, cart velocity, pole angle, pole
    angular velocity).

    If cartpole draws to a window, it shows the starting position for
    start_pause seconds before the pole is released, and the final
    position for end_pause seconds.
    """
    state = cartpole.reset(seed=seed)
    states = [state]
    cartpole.pause(start_pause)

    terminated = truncated = False
    while not (terminated or truncated):
        force = controller(state)
        state, _, terminated, truncated = cartpole.step(force)
        states.append(state)

    cartpole.pause(end_pause)
    return states


def print_trace(states, tau, normal_limit, every=5):
    """Print time, cart position and pole angle every few steps.

    Also marks the step where the pole first passed normal_limit (radians),
    the angle at which a standard episode would have ended. This is only
    for the printout: the simulation itself may have run further.
    """
    print(f"{'time (s)':>8}  {'cart x (m)':>10}  {'pole angle':>10}")
    last = len(states) - 1
    passed_normal_limit = False
    for step, (x, _, theta, _) in enumerate(states):
        if step % every == 0 or step == last:
            print(f"{step * tau:8.2f}  {x:10.3f}  {math.degrees(theta):9.1f}°")
        if not passed_normal_limit and abs(theta) > normal_limit:
            passed_normal_limit = True
            print(
                f"  -- passed {math.degrees(normal_limit):.0f} degrees at "
                f"{step * tau:.2f} s: a normal episode would end here"
            )


def print_summary(states, tau, angle_limit):
    steps = len(states) - 1
    t = steps * tau
    if abs(states[-1][2]) >= angle_limit:
        print(
            f"Pole passed {math.degrees(angle_limit):.0f} degrees after {t:.2f} s "
            f"({steps} steps of {tau} s)."
        )
    else:
        print(f"Stopped after {t:.2f} s: the cart left the track.")


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
    cartpole = CartPole.make(
        render_mode=None if args.no_window else "human", slow=args.slow
    )
    states = run_simulation(cartpole, no_control, seed=args.seed)
    cartpole.close()
    print_trace(states, cartpole.tau, cartpole.normal_angle_limit)
    print_summary(states, cartpole.tau, cartpole.angle_limit)


if __name__ == "__main__":
    main()
