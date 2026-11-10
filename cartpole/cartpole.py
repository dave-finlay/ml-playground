"""Watch a controller try to balance the pole on the cart.

Uses Gymnasium's CartPole-v1 physics and renderer, with two changes:

- Each step pushes the cart with a force chosen by a controller: any
  force, not just CartPole's standard full 10 N left or right
  (CartPole.step works around that). The controllers so far:
  - none: always 0 N, so the pole just falls
  - p: proportional to the pole angle, up to a maximum force
- The failure angle is raised from 12 degrees to 90 degrees, so an
  episode runs until the pole is lying flat instead of stopping as soon
  as it tilts past 12 degrees. It also ends if the cart goes more than
  5 m from the centre, or after 1000 steps (20 s), by default. Standard
  CartPole uses 2.4 m and 500 steps (10 s).

Run from the repo root:

    uv run python cartpole/cartpole.py
    uv run python cartpole/cartpole.py --slow 4 --seed 7
    uv run python cartpole/cartpole.py --controller p --kp 20
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
    def make(
        render_mode=None,
        slow=1.0,
        angle_limit=math.pi / 2,
        track_limit=2.4,
        max_steps=500,
    ):
        """Create a CartPole whose episodes end when the pole passes angle_limit.

        angle_limit is in radians. The default, 90 degrees, runs until the
        pole is lying flat; standard CartPole stops at 12 degrees.

        Episodes also end when the cart goes more than track_limit metres
        from the centre, or after max_steps steps of 0.02 s. The defaults
        are standard CartPole's: 2.4 m and 500 steps (10 s). The window
        widens with the track so the cart and pole stay the same size.

        render_mode is "human" for a window, "rgb_array" for frames as pixel
        arrays, or None for no drawing. slow is a slow-motion factor for the
        window: 4 plays at quarter speed.
        """
        env = gym.make(
            "CartPole-v1", render_mode=render_mode, max_episode_steps=max_steps
        )
        physics = env.unwrapped
        normal_angle_limit = physics.theta_threshold_radians
        physics.theta_threshold_radians = angle_limit
        # The renderer fits the whole track into screen_width pixels.
        physics.screen_width = round(
            physics.screen_width * track_limit / physics.x_threshold
        )
        physics.x_threshold = track_limit
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
    def track_limit(self):
        """Distance in metres from the centre at which the cart leaves the
        track and the episode ends."""
        return self._env.unwrapped.x_threshold

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


class PController:
    """Proportional controller on the pole angle.

    Pushes the cart toward the side the pole is leaning, with a force of
    kp (newtons per radian) times the angle, capped at max_force newtons
    either way. The target angle is 0 (upright), so the angle is the error.
    """

    def __init__(self, kp, max_force=10.0):
        self.kp = kp
        self.max_force = max_force

    def __call__(self, state):
        _, _, theta, _ = state
        force = self.kp * theta
        return max(-self.max_force, min(self.max_force, force))


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


def print_summary(states, tau, angle_limit, track_limit):
    steps = len(states) - 1
    t = steps * tau
    x, _, theta, _ = states[-1]
    if abs(theta) >= angle_limit:
        reason = f"the pole passed {math.degrees(angle_limit):.0f} degrees"
    elif abs(x) > track_limit:
        reason = f"the cart went more than {track_limit} m from the centre"
    else:
        reason = "the time limit, with the pole still up"
    print(f"Stopped after {t:.2f} s ({steps} steps of {tau} s): {reason}.")


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
        "--controller",
        choices=["none", "p"],
        default="none",
        help="none: never push; p: push proportional to the pole angle (default: none)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=20.0,
        help="p controller's gain in newtons per radian of lean (default: 20)",
    )
    parser.add_argument(
        "--max-force",
        type=float,
        default=10.0,
        help="largest push the controller may apply, in newtons (default: 10)",
    )
    parser.add_argument(
        "--track-limit",
        type=float,
        default=5.0,
        help="how far the cart may go from the centre, in metres (default: 5; "
        "standard CartPole: 2.4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="longest episode, in steps of 0.02 s (default: 1000, i.e. 20 s; "
        "standard CartPole: 500)",
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
        render_mode=None if args.no_window else "human",
        slow=args.slow,
        track_limit=args.track_limit,
        max_steps=args.max_steps,
    )
    if args.controller == "p":
        controller = PController(args.kp, max_force=args.max_force)
    else:
        controller = no_control
    states = run_simulation(cartpole, controller, seed=args.seed)
    cartpole.close()
    print_trace(states, cartpole.tau, cartpole.normal_angle_limit)
    print_summary(states, cartpole.tau, cartpole.angle_limit, cartpole.track_limit)


if __name__ == "__main__":
    main()
