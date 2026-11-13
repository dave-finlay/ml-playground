"""A linear model of the cart and pole near upright.

Gymnasium's CartPole equations, linearized about the pole standing
upright and stepped the way Gymnasium steps them, so the model is exact
for small angles. lqr.py designs controllers with it and kalman.py
estimates the state with it.
"""

import collections

import numpy as np

# The constants of the simulation (see CartPole.physics).
Physics = collections.namedtuple(
    "Physics", "gravity cart_mass pole_mass half_length tau"
)


def linearize(physics):
    """Return (A, B) for the cart and pole linearized about upright, in
    continuous time: d/dt state = A state + B force, with state (x,
    x_rate, angle, angle_rate).

    From Gymnasium's equations with small angles (sin = angle, cos = 1)
    and the angle rate's square dropped:
        angle_accel = (g angle - force / M) / (l (4/3 - m / M))
        x_accel = force / M - (m l / M) angle_accel
    where M is the total mass, m the pole's mass and l its half-length.
    """
    total = physics.cart_mass + physics.pole_mass
    lever = physics.pole_mass * physics.half_length / total
    denominator = physics.half_length * (4 / 3 - physics.pole_mass / total)
    angle_from_angle = physics.gravity / denominator
    angle_from_force = -1 / (total * denominator)
    a = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, -lever * angle_from_angle, 0],
            [0, 0, 0, 1],
            [0, 0, angle_from_angle, 0],
        ]
    )
    b = np.array([[0], [1 / total - lever * angle_from_force], [0], [angle_from_force]])
    return a, b


def discretize(a, b, tau):
    """The continuous model stepped by explicit Euler, as Gymnasium steps
    it: state' = (I + tau A) state + tau B force. Exact for the linear
    model, not an approximation of it."""
    return np.eye(len(a)) + tau * a, tau * b


def linear_model(physics):
    """Return (A, B) for one step of the linear model: state' = A state +
    B force, with state (x, x_rate, angle, angle_rate)."""
    return discretize(*linearize(physics), physics.tau)
