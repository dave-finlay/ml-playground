"""Watch a controller try to balance the pole on the cart.

Uses Gymnasium's CartPole-v1 physics and renderer, with two changes:

- Each step pushes the cart with a force chosen by a controller: any
  force, not just CartPole's standard full 10 N left or right
  (CartPole.step works around that). The controllers so far:
  - none: always 0 N, so the pole just falls
  - p: proportional to the pole angle, up to a maximum force
  - pd: proportional to the pole angle plus its rate of change, and
    optionally terms on the cart's position and velocity
  - learned: full force left or right, by the sign of weights learned by
    train.py
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
    uv run python cartpole/cartpole.py --controller pd --kx 0.5 --kv 1 --max-steps 3000
    uv run python cartpole/cartpole.py --controller learned \
        --weights cartpole/linear-policy.json
"""

import argparse
import json
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
        pole is lying flat; None keeps standard CartPole's limit of 12
        degrees.

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
        if angle_limit is not None:
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
    def max_steps(self):
        """Number of steps after which an episode ends regardless."""
        return self._env.spec.max_episode_steps

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


class Differencer:
    """Estimates the rate of change of a sampled value from successive
    samples: (value now - previous value) / dt.

    Returns 0 for the first sample, when there's no previous one.
    """

    def __init__(self, dt):
        self.dt = dt
        self._previous = None

    def update(self, value):
        """Take the next sample and return the estimated rate of change."""
        rate = 0.0 if self._previous is None else (value - self._previous) / self.dt
        self._previous = value
        return rate


class Smoother:
    """Exponential moving average of a sampled value:
        smoothed = smoothing * previous smoothed + (1 - smoothing) * value

    smoothing runs from 0 (no smoothing) to just under 1 (heavy). Higher
    values cut noise more but respond later. The first sample passes
    through unchanged.
    """

    def __init__(self, smoothing=0.0):
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be from 0 to under 1: {smoothing}")
        self.smoothing = smoothing
        self._smoothed = None

    def update(self, value):
        """Take the next sample and return the smoothed value."""
        if self._smoothed is None:
            self._smoothed = value
        else:
            self._smoothed = (
                self.smoothing * self._smoothed + (1.0 - self.smoothing) * value
            )
        return self._smoothed


class RateEstimator:
    """Turns readings into the four numbers the controllers work from:
    (cart position, its rate of change, pole angle, its rate of change).

    rate_from says where both rates come from:
    - "difference": estimated from successive readings, as you would with
      only angle and position sensors (see Differencer)
    - "velocity": the velocities in the reading, as if from perfect rate
      sensors

    smoothing (0 to just under 1) smooths both rates with an exponential
    moving average (see Smoother). dt is the time between readings in
    seconds (CartPole.tau). A RateEstimator remembers previous readings,
    so use a new one for each episode.
    """

    def __init__(self, dt, rate_from="difference", smoothing=0.0):
        if rate_from not in ("difference", "velocity"):
            raise ValueError(f"unknown rate_from: {rate_from!r}")
        self.rate_from = rate_from
        self._theta_differencer = Differencer(dt)
        self._x_differencer = Differencer(dt)
        self._theta_rate_smoother = Smoother(smoothing)
        self._x_rate_smoother = Smoother(smoothing)

    def update(self, reading):
        """Take the next reading and return (x, x rate, angle, angle rate)."""
        x, x_dot, theta, theta_dot = reading
        if self.rate_from == "difference":
            theta_dot = self._theta_differencer.update(theta)
            x_dot = self._x_differencer.update(x)
        theta_rate = self._theta_rate_smoother.update(theta_dot)
        x_rate = self._x_rate_smoother.update(x_dot)
        return (x, x_rate, theta, theta_rate)


class PDController:
    """Proportional-derivative controller on the pole angle and, optionally,
    the cart position.

    Pushes with
        kp * angle + kd * (angle's rate of change)
        + kx * (cart position) + kv * (cart velocity)
    capped at max_force newtons either way.

    The angle terms balance the pole. kp is in newtons per radian, kd in
    newtons per radian per second. The kd term resists the pole's motion,
    damping the swings that P control alone leaves.

    The cart terms keep the cart near the centre; both default to 0. kx is
    in newtons per metre, kv in newtons per metre per second. Positive
    values are the useful direction, and it's counter-intuitive: with the
    cart right of centre, kx pushes the cart further right. That tips the
    pole left, and the angle terms then drive the cart back left, toward
    the centre. Pushing straight back toward the centre tips the pole the
    wrong way and makes the cart run away. kv damps the cart's motion the
    way kd damps the pole's: without it, kx leaves the cart swinging back
    and forth through the centre, with slowly growing swings. Too much kv
    fails too (above about 2 N·s/m with kp = 20, kd = 3): it undoes the
    damping of the pole's swing.

    dt, rate_from and smoothing say how the rates are estimated (see
    RateEstimator). Late damping is weaker damping, so too much smoothing
    brings swings back. A PDController remembers previous readings, so use
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
        kv=0.0,
    ):
        self.kp = kp
        self.kd = kd
        self.kx = kx
        self.kv = kv
        self.max_force = max_force
        self._rates = RateEstimator(dt, rate_from, smoothing)

    def __call__(self, reading):
        x, x_rate, theta, theta_rate = self._rates.update(reading)
        force = self.kp * theta + self.kd * theta_rate + self.kx * x + self.kv * x_rate
        return clamp(force, self.max_force)


def save_weights(path, weights, rate_from):
    """Save a LinearController's weights, and how the rates they were
    learned with were estimated, as JSON."""
    with open(path, "w") as f:
        json.dump({"weights": list(weights), "rate_from": rate_from}, f, indent=2)
        f.write("\n")


def load_weights(path):
    """Read weights saved by save_weights. Returns (weights, rate_from)."""
    with open(path) as f:
        saved = json.load(f)
    return saved["weights"], saved["rate_from"]


class LinearController:
    """Bang-bang controller with learned weights: pushes max_force newtons
    right when

        weights · (cart position, its rate, pole angle, its rate) >= 0

    and max_force left otherwise. It's what train.py's policy does once
    trained, when it takes the likelier action instead of sampling. The
    weights play the roles of the PD controller's Kx, Kv, Kp and Kd, but
    only their ratios matter, since only the sign of the sum is used.

    dt and rate_from say how the rates are estimated (see RateEstimator).
    Use the same rate_from the weights were learned with. A
    LinearController remembers previous readings, so use a new one for
    each episode.
    """

    def __init__(self, weights, dt, max_force=10.0, rate_from="difference"):
        if len(weights) != 4:
            raise ValueError(f"need 4 weights, got {len(weights)}")
        self.weights = list(weights)
        self.max_force = max_force
        self._rates = RateEstimator(dt, rate_from)

    def __call__(self, reading):
        inputs = self._rates.update(reading)
        total = sum(w * v for w, v in zip(self.weights, inputs, strict=True))
        return self.max_force if total >= 0 else -self.max_force


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


# Options that only some controllers use: option name -> (controllers
# that use it, default). Given with any other controller, it's an error
# rather than being silently ignored.
CONTROLLER_OPTIONS = {
    "kp": (("p", "pd"), 20.0),
    "max_force": (("p", "pd", "learned"), 10.0),
    "kd": (("pd",), 3.0),
    "rate_from": (("pd",), "difference"),  # learned: from the weights file
    "d_smoothing": (("pd",), 0.0),
    "kx": (("pd",), 0.0),
    "kv": (("pd",), 0.0),
}


def apply_controller_options(parser, args):
    """Reject controller options the chosen controller doesn't use, and fill
    in defaults for the ones it does."""
    for name, (controllers, default) in CONTROLLER_OPTIONS.items():
        value = getattr(args, name)
        if value is None:
            setattr(args, name, default)
        elif args.controller not in controllers:
            option = "--" + name.replace("_", "-")
            needs = " or ".join(f"--controller {c}" for c in controllers)
            parser.error(f"{option} needs {needs}")


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
        choices=["none", "p", "pd", "learned"],
        default="none",
        help="none: never push; p: push proportional to the pole angle; pd: "
        "proportional to the angle plus its rate of change; learned: full "
        "force left or right, by weights from --weights (default: none)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="learned controller's weights file, saved by train.py",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=None,  # see CONTROLLER_OPTIONS
        help="p and pd controllers' proportional gain, in newtons per radian of "
        "lean (default: 20)",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=None,  # see CONTROLLER_OPTIONS
        help="pd controller's derivative gain, in newtons per radian per second "
        "(default: 3)",
    )
    parser.add_argument(
        "--rate-from",
        choices=["difference", "velocity"],
        default=None,  # see CONTROLLER_OPTIONS
        help="where the pd controller gets rates of change: differences of "
        "successive readings, or the true velocities, like perfect rate "
        "sensors (default: difference)",
    )
    parser.add_argument(
        "--d-smoothing",
        type=float,
        default=None,  # see CONTROLLER_OPTIONS
        help="pd controller's smoothing of rates of change, from 0 (none) to "
        "just under 1 (heavy): the weight an exponential moving average gives "
        "the previous smoothed value (default: 0)",
    )
    parser.add_argument(
        "--kx",
        type=float,
        default=None,  # see CONTROLLER_OPTIONS
        help="pd controller's gain on cart position, in newtons per metre; "
        "positive keeps the cart near the centre (default: 0)",
    )
    parser.add_argument(
        "--kv",
        type=float,
        default=None,  # see CONTROLLER_OPTIONS
        help="pd controller's gain on cart velocity, in newtons per metre per "
        "second; positive damps the cart's motion (default: 0)",
    )
    parser.add_argument(
        "--max-force",
        type=float,
        default=None,  # see CONTROLLER_OPTIONS
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
    args = parser.parse_args()
    apply_controller_options(parser, args)
    if (args.controller == "learned") != (args.weights is not None):
        parser.error("--controller learned and --weights go together")
    return args


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
            kv=args.kv,
        )
    elif args.controller == "learned":
        weights, rate_from = load_weights(args.weights)
        controller = LinearController(
            weights, dt=cartpole.tau, max_force=args.max_force, rate_from=rate_from
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
