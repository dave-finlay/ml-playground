"""A Kalman filter that estimates the cart and pole's state from noisy
readings of just the cart position and the pole angle.

It estimates five numbers: cart position, cart velocity, pole angle,
pole angular velocity, and the angle sensor's offset (a constant error
added to every angle reading). Each step it

1. predicts the state from its last estimate and the force it knows was
   applied, using the linear model (model.py), and lets its uncertainty
   grow; then
2. corrects the prediction toward the new readings, by a gain chosen
   from how uncertain the prediction is compared with the readings.

The velocities are never measured; the model ties them to how the
positions change. The offset is never measured either, but it shows:
if the pole really leaned by the offset, the cart would have to keep
accelerating to hold it up, and the position readings show whether it
does.

The filter needs to know how noisy the readings are and how far to
trust the model, as variances. These are rough guesses, not the true
values: the gain depends only on their ratios (scale every variance by
the same factor and the estimates don't change), so what's being
chosen is really how far to trust the readings against the model.
"""

import math

import numpy as np

from model import linear_model

# Rough guesses at the noise, used unless others are given. Not the
# simulation's actual noise, which the filter isn't told.
ANGLE_NOISE = math.radians(2.0)  # angle reading, radians
POSITION_NOISE = 0.02  # cart position reading, metres
CART_ACCELERATION_NOISE = 0.5  # unmodelled cart acceleration, m/s^2
POLE_ACCELERATION_NOISE = 0.5  # unmodelled pole acceleration, rad/s^2
OFFSET_DRIFT = math.radians(0.01)  # how fast the offset may wander, rad/s^(1/2)

# How unsure the filter starts out: it begins at the upright, centred,
# at rest, with no offset, and lets the first readings correct it.
INITIAL_STD = (1.0, 1.0, math.radians(10), 1.0, math.radians(5))


class KalmanEstimator:
    """Estimates (cart position, its rate, pole angle, its rate) and the
    angle sensor's offset from readings of position and angle.

    physics are the simulation's constants (CartPole.physics), from which
    the linear model comes. The noise arguments are rough guesses at
    standard deviations (see the module docstring and the constants
    above). With estimate_offset False, the filter assumes the angle
    sensor has no offset and estimates only the four state numbers.

    Call update() with each reading and the force applied since the
    last one. A KalmanEstimator remembers its estimate, so use a new one
    for each episode.
    """

    def __init__(
        self,
        physics,
        angle_noise=ANGLE_NOISE,
        position_noise=POSITION_NOISE,
        cart_acceleration_noise=CART_ACCELERATION_NOISE,
        pole_acceleration_noise=POLE_ACCELERATION_NOISE,
        offset_drift=OFFSET_DRIFT,
        estimate_offset=True,
    ):
        for name, value in (
            ("angle_noise", angle_noise),
            ("position_noise", position_noise),
            ("cart_acceleration_noise", cart_acceleration_noise),
            ("pole_acceleration_noise", pole_acceleration_noise),
            ("offset_drift", offset_drift),
        ):
            if value <= 0:
                raise ValueError(f"{name} should be more than 0")
        a, b = linear_model(physics)
        tau = physics.tau
        size = 5 if estimate_offset else 4
        self.estimate_offset = estimate_offset
        # The model, with the offset (if estimated) carried along unchanged.
        self.a = np.eye(size)
        self.a[:4, :4] = a
        self.b = np.zeros((size, 1))
        self.b[:4] = b
        # Readings: position, and angle plus offset.
        self.h = np.zeros((2, size))
        self.h[0, 0] = 1.0
        self.h[1, 2] = 1.0
        if estimate_offset:
            self.h[1, 4] = 1.0
        # Unmodelled accelerations change the velocities by tau times
        # themselves each step (Gymnasium's Euler step); the offset may
        # drift a little.
        self.q = np.zeros((size, size))
        self.q[1, 1] = (tau * cart_acceleration_noise) ** 2
        self.q[3, 3] = (tau * pole_acceleration_noise) ** 2
        if estimate_offset:
            self.q[4, 4] = offset_drift**2 * tau
        self.r = np.diag([position_noise**2, angle_noise**2])
        self.state = np.zeros(size)
        self.covariance = np.diag(np.square(INITIAL_STD[:size]))
        self._started = False

    @property
    def offset(self):
        """The estimated angle sensor offset, in radians (0 if not
        estimated)."""
        return float(self.state[4]) if self.estimate_offset else 0.0

    def predict(self, force):
        """Step the estimate forward by the model, given the force applied."""
        self.state = self.a @ self.state + self.b[:, 0] * force
        self.covariance = self.a @ self.covariance @ self.a.T + self.q

    def correct(self, position_reading, angle_reading):
        """Move the estimate toward the readings by the Kalman gain."""
        innovation = np.array([position_reading, angle_reading]) - self.h @ self.state
        spread = self.h @ self.covariance @ self.h.T + self.r
        gain = np.linalg.solve(spread, self.h @ self.covariance).T
        self.state = self.state + gain @ innovation
        # Joseph form: the same as (I - KH) P, but stays symmetric and
        # positive despite rounding.
        keep = np.eye(len(self.state)) - gain @ self.h
        self.covariance = keep @ self.covariance @ keep.T + gain @ self.r @ gain.T

    def update(self, reading, force=None):
        """Take the next reading (x, x_rate, angle, angle_rate; only the
        position and angle are used) and the force applied since the last
        one (None for the first reading), and return the estimated
        (x, x_rate, angle, angle_rate)."""
        if self._started:
            self.predict(force)
        self._started = True
        x, _, theta, _ = reading
        self.correct(x, theta)
        return tuple(float(v) for v in self.state[:4])


def observability_rank(estimator):
    """The rank of the observability matrix [H; HA; HA^2; ...]: when it
    equals the number of estimated numbers, the readings over time pin
    them all down (the offset included)."""
    rows, power = [], np.eye(len(estimator.a))
    for _ in range(len(estimator.a)):
        rows.append(estimator.h @ power)
        power = estimator.a @ power
    return int(np.linalg.matrix_rank(np.vstack(rows)))


def estimate_states(estimator, readings, forces):
    """Run estimator over an episode's readings and the forces applied
    between them (one fewer), and return its estimate after each
    reading."""
    estimates = [estimator.update(readings[0])]
    for reading, force in zip(readings[1:], forces, strict=True):
        estimates.append(estimator.update(reading, force))
    return estimates
