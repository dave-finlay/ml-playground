"""Tests for kalman.py: the filter's model, observability and estimates."""

import math

import numpy as np
import pytest

from kalman import KalmanEstimator, estimate_states, observability_rank
from model import Physics, linear_model

PHYSICS = Physics(gravity=9.8, cart_mass=1.0, pole_mass=0.1, half_length=0.5, tau=0.02)


def linear_episode(steps, offset, noise=(0.0, 0.0), seed=0):
    """Simulate the linear model under a stabilizing push, returning the
    true states, readings (with the angle offset and optional noise on
    position and angle) and the forces applied."""
    a, b = linear_model(PHYSICS)
    rng = np.random.default_rng(seed)
    state = np.array([0.1, 0.0, 0.03, 0.0])
    states, readings, forces = [], [], []
    for _ in range(steps):
        states.append(state.copy())
        readings.append(
            (
                state[0] + rng.normal(0, noise[0]) if noise[0] else state[0],
                0.0,
                state[2] + offset + (rng.normal(0, noise[1]) if noise[1] else 0.0),
                0.0,
            )
        )
        force = float(np.array([0.5, 1.0, 20.0, 3.0]) @ state)
        forces.append(force)
        state = a @ state + b[:, 0] * force
    return np.array(states), readings, forces[:-1]


def test_offset_is_observable():
    assert observability_rank(KalmanEstimator(PHYSICS)) == 5
    assert observability_rank(KalmanEstimator(PHYSICS, estimate_offset=False)) == 4


def test_filter_recovers_velocities_and_offset_from_clean_readings():
    states, readings, forces = linear_episode(1500, offset=math.radians(1.5))
    estimator = KalmanEstimator(PHYSICS)
    estimates = np.array(estimate_states(estimator, readings, forces))
    assert estimator.offset == pytest.approx(math.radians(1.5), abs=math.radians(0.05))
    assert estimates[-500:] == pytest.approx(states[-500:], abs=1e-3)


def test_filter_beats_differencing_on_noisy_readings():
    noise = (0.01, math.radians(1.0))
    states, readings, forces = linear_episode(1000, offset=0.0, noise=noise, seed=1)
    estimates = np.array(estimate_states(KalmanEstimator(PHYSICS), readings, forces))
    angle_rate_error = np.sqrt(np.mean((estimates[100:, 3] - states[100:, 3]) ** 2))
    angles = np.array([r[2] for r in readings])
    differenced = np.diff(angles) / PHYSICS.tau
    difference_error = np.sqrt(np.mean((differenced[99:] - states[100:, 3]) ** 2))
    assert angle_rate_error < difference_error / 10


def test_only_the_ratio_of_the_guesses_matters():
    """Scaling every noise guess by the same factor scales the variances
    alike, leaving the gain, and so the estimates once the starting
    uncertainty has worn off, unchanged."""
    noise = (0.01, math.radians(1.0))
    _, readings, forces = linear_episode(1500, offset=0.01, noise=noise, seed=2)

    def final_estimate(scale):
        estimator = KalmanEstimator(
            PHYSICS,
            angle_noise=math.radians(2.0) * scale,
            position_noise=0.02 * scale,
            cart_acceleration_noise=0.5 * scale,
            pole_acceleration_noise=0.5 * scale,
            offset_drift=math.radians(0.01) * scale,
        )
        return estimate_states(estimator, readings, forces)[-1]

    assert final_estimate(3.0) == pytest.approx(final_estimate(1.0), abs=1e-4)


def test_without_an_offset_state_the_angle_is_biased():
    states, readings, forces = linear_episode(1500, offset=math.radians(1.5))
    estimates = np.array(
        estimate_states(
            KalmanEstimator(PHYSICS, estimate_offset=False), readings, forces
        )
    )
    bias = np.mean(estimates[-500:, 2] - states[-500:, 2])
    assert bias > math.radians(0.5)


def test_noise_guesses_must_be_positive():
    with pytest.raises(ValueError):
        KalmanEstimator(PHYSICS, angle_noise=0.0)
