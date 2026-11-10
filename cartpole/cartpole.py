"""Watch a controller try to balance the pole on the cart.

Uses Gymnasium's CartPole-v1 physics and renderer, with two changes:

- Each step pushes the cart with a force chosen by a controller: any
  force, not just CartPole's standard full 10 N left or right
  (CartPole.step works around that). The controllers so far:
  - none: always 0 N, so the pole just falls
  - p: proportional to the pole angle, up to a maximum force
  - pd: proportional to the pole angle plus its rate of change, and
    optionally a term proportional to the cart's position
  The controller sees the pole angle through a sensor, which can add
  noise, a scale error and an offset error.
- The failure angle is raised from 12 degrees to 90 degrees, so an
  episode runs until the pole is lying flat instead of stopping as soon
  as it tilts past 12 degrees. It also ends if the cart goes more than
  5 m from the centre, or after 1000 steps (20 s), by default. Standard
  CartPole uses 2.4 m and 500 steps (10 s).

Run from the repo root:

    uv run python cartpole/cartpole.py
    uv run python cartpole/cartpole.py --slow 4 --seed 7
    uv run python cartpole/cartpole.py --controller p --kp 20
    uv run python cartpole/cartpole.py --controller p --angle-noise 1 --angle-offset 1
    uv run python cartpole/cartpole.py --controller pd --kp 20 --kd 3
    uv run python cartpole/cartpole.py --controller pd --kx 0.5 --max-steps 3000
"""

import argparse
import math
import random

import gymnasium as gym


def clamp(value, limit):
    """Return value limited to the range -limit to +limit."""
    return max(-limit, min(limit, value))


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


class AngleSensor:
    """Measures the pole angle the way a real sensor might: imperfectly.

    The reading is scale * (true angle) + offset + noise, where the noise
    is drawn fresh for each reading from a normal distribution with
    standard deviation noise_std. offset and noise_std are in radians. The
    defaults give a perfect sensor. seed makes the noise repeatable.

    The other three state numbers are passed through unchanged, as if
    measured perfectly.
    """

    def __init__(self, scale=1.0, offset=0.0, noise_std=0.0, seed=None):
        self.scale = scale
        self.offset = offset
        self.noise_std = noise_std
        self._random = random.Random(seed)

    def read(self, state):
        """Return state with its pole angle replaced by this sensor's reading."""
        x, x_dot, theta, theta_dot = state
        noise = self._random.gauss(0.0, self.noise_std)
        return (x, x_dot, self.scale * theta + self.offset + noise, theta_dot)


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
        return clamp(self.kp * theta, self.max_force)


class PDController:
    """Proportional-derivative controller on the pole angle.

    Pushes with kp * angle + kd * (rate of change of the angle), capped at
    max_force newtons either way. kp is in newtons per radian, kd in
    newtons per radian per second. The D term resists the pole's motion,
    damping the swings that P control alone leaves.

    rate_from says where the rate of change comes from:
    - "difference": estimated from successive angle readings,
      (angle now - previous angle) / dt, as you would with only an angle
      sensor. It's 0 on the first step, when there's no previous reading.
    - "velocity": the angular velocity in the state, as if from a perfect
      rate sensor (gyroscope).

    kx (newtons per metre, default 0) adds kx * (cart position) to keep
    the cart near the centre. A positive kx is the useful direction, and
    it's counter-intuitive: with the cart right of centre, it pushes the
    cart further right. That tips the pole left, and the angle terms then
    drive the cart back left, toward the centre. A negative kx, pushing
    straight back toward the centre, tips the pole the wrong way and
    makes the cart run away.

    smoothing (0 to just under 1) smooths the rate with an exponential
    moving average before it's used:
        smoothed = smoothing * previous smoothed + (1 - smoothing) * rate
    0 means no smoothing. Higher values cut noise more but respond later,
    and late damping is weaker damping.

    dt is the time between readings in seconds (CartPole.tau). A
    PDController remembers the previous reading and smoothed rate, so use
    a new one for each episode.
    """

    def __init__(
        self,
        kp,
        kd,
        dt,
        max_force=10.0,
        rate_from="difference",
        smoothing=0.0,
        kx=0.0,
    ):
        if rate_from not in ("difference", "velocity"):
            raise ValueError(f"unknown rate_from: {rate_from!r}")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be from 0 to under 1: {smoothing}")
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.max_force = max_force
        self.rate_from = rate_from
        self.smoothing = smoothing
        self.kx = kx
        self._previous_theta = None
        self._smoothed_rate = None

    def __call__(self, state):
        x, _, theta, theta_dot = state
        if self.rate_from == "difference":
            if self._previous_theta is None:
                theta_dot = 0.0
            else:
                theta_dot = (theta - self._previous_theta) / self.dt
            self._previous_theta = theta
        if self._smoothed_rate is None:
            self._smoothed_rate = theta_dot
        else:
            self._smoothed_rate = (
                self.smoothing * self._smoothed_rate
                + (1.0 - self.smoothing) * theta_dot
            )
        force = self.kp * theta + self.kd * self._smoothed_rate + self.kx * x
        return clamp(force, self.max_force)


def run_simulation(
    cartpole,
    controller,
    sensor=None,
    seed=None,
    start_pause=1.0,
    end_pause=2.0,
):
    """Run one episode under controller and return what happened.

    Resets cartpole (with seed, if given) and steps it until the episode
    ends. On each step, sensor reads the state, and controller(reading)
    gives the force in newtons to push the cart with (see CartPole.step).
    With no sensor, the controller sees a perfect reading.

    Returns (states, readings): the true states, one per time step starting
    with the initial state, and what the sensor read for each. Each is
    (cart position, cart velocity, pole angle, pole angular velocity).

    If cartpole draws to a window, it shows the starting position for
    start_pause seconds before the pole is released, and the final
    position for end_pause seconds.
    """
    if sensor is None:
        sensor = AngleSensor()
    state = cartpole.reset(seed=seed)
    states = [state]
    readings = [sensor.read(state)]
    cartpole.pause(start_pause)

    terminated = truncated = False
    while not (terminated or truncated):
        force = controller(readings[-1])
        state, _, terminated, truncated = cartpole.step(force)
        states.append(state)
        readings.append(sensor.read(state))

    cartpole.pause(end_pause)
    return states, readings


def print_trace(states, readings, tau, normal_limit, every=5):
    """Print time, cart position, pole angle and its reading every few steps.

    Also marks the step where the pole first passed normal_limit (radians),
    the angle at which a standard episode would have ended. This is only
    for the printout: the simulation itself may have run further.
    """
    print(f"{'time (s)':>8}  {'cart x (m)':>10}  {'pole angle':>10}  {'reading':>8}")
    last = len(states) - 1
    passed_normal_limit = False
    for step, ((x, _, theta, _), reading) in enumerate(
        zip(states, readings, strict=True)
    ):
        if step % every == 0 or step == last:
            print(
                f"{step * tau:8.2f}  {x:10.3f}  {math.degrees(theta):9.1f}°"
                f"  {math.degrees(reading[2]):7.1f}°"
            )
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
        choices=["none", "p", "pd"],
        default="none",
        help="none: never push; p: push proportional to the pole angle; pd: "
        "proportional to the angle plus its rate of change (default: none)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=20.0,
        help="p and pd controllers' proportional gain, in newtons per radian of "
        "lean (default: 20)",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=3.0,
        help="pd controller's derivative gain, in newtons per radian per second "
        "(default: 3)",
    )
    parser.add_argument(
        "--rate-from",
        choices=["difference", "velocity"],
        default="difference",
        help="where the pd controller gets the angle's rate of change: "
        "difference of successive angle readings, or the true angular "
        "velocity, like a perfect gyroscope (default: difference)",
    )
    parser.add_argument(
        "--d-smoothing",
        type=float,
        default=0.0,
        help="pd controller's smoothing of the rate of change, from 0 (none) to "
        "just under 1 (heavy): the weight an exponential moving average gives "
        "the previous smoothed value (default: 0)",
    )
    parser.add_argument(
        "--kx",
        type=float,
        default=0.0,
        help="pd controller's gain on cart position, in newtons per metre; "
        "positive keeps the cart near the centre (default: 0)",
    )
    parser.add_argument(
        "--max-force",
        type=float,
        default=10.0,
        help="largest push the controller may apply, in newtons (default: 10)",
    )
    parser.add_argument(
        "--angle-noise",
        type=float,
        default=0.0,
        help="angle sensor noise: standard deviation in degrees (default: 0)",
    )
    parser.add_argument(
        "--angle-scale",
        type=float,
        default=1.0,
        help="angle sensor scale error: reading = scale x true angle, e.g. 1.05 "
        "reads 5%% high (default: 1)",
    )
    parser.add_argument(
        "--angle-offset",
        type=float,
        default=0.0,
        help="angle sensor offset error in degrees, added to every reading "
        "(default: 0)",
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
    elif args.controller == "pd":
        controller = PDController(
            args.kp,
            args.kd,
            dt=cartpole.tau,
            max_force=args.max_force,
            rate_from=args.rate_from,
            smoothing=args.d_smoothing,
            kx=args.kx,
        )
    else:
        controller = no_control
    sensor = AngleSensor(
        scale=args.angle_scale,
        offset=math.radians(args.angle_offset),
        noise_std=math.radians(args.angle_noise),
        seed=args.seed,
    )
    states, readings = run_simulation(cartpole, controller, sensor, seed=args.seed)
    cartpole.close()
    print_trace(states, readings, cartpole.tau, cartpole.normal_angle_limit)
    print_summary(states, cartpole.tau, cartpole.angle_limit, cartpole.track_limit)


if __name__ == "__main__":
    main()
