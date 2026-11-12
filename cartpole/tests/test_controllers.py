"""Tests for cartpole.py's rate estimation and controllers."""

import random

import pytest

from cartpole import (
    Differencer,
    LinearController,
    PDController,
    RateEstimator,
    Smoother,
    load_weights,
    save_weights,
)

DT = 0.02


def test_differencer_first_rate_is_zero_then_slope():
    differencer = Differencer(DT)
    rates = [differencer.update(0.1 * k) for k in range(4)]  # rises 0.1 per step
    assert rates[0] == 0.0
    assert rates[1:] == pytest.approx([5.0, 5.0, 5.0])  # 0.1 / 0.02


def test_smoother_passes_first_value_then_averages():
    smoother = Smoother(0.75)
    assert smoother.update(4.0) == 4.0
    assert smoother.update(0.0) == pytest.approx(3.0)  # 0.75 * 4 + 0.25 * 0
    assert smoother.update(0.0) == pytest.approx(2.25)


def test_smoother_with_no_smoothing_passes_values_through():
    smoother = Smoother(0.0)
    assert [smoother.update(v) for v in (1.0, -2.0, 3.0)] == [1.0, -2.0, 3.0]


@pytest.mark.parametrize("smoothing", [-0.1, 1.0, 1.5])
def test_smoother_rejects_out_of_range(smoothing):
    with pytest.raises(ValueError):
        Smoother(smoothing)


def test_rate_estimator_differences_positions_and_angles():
    rates = RateEstimator(DT, "difference")
    # The velocities in the readings are ignored when differencing.
    assert rates.update((1.0, 99.0, 0.10, 99.0)) == (1.0, 0.0, 0.10, 0.0)
    x, x_rate, theta, theta_rate = rates.update((1.02, 99.0, 0.11, 99.0))
    assert (x, theta) == (1.02, 0.11)
    assert x_rate == pytest.approx(1.0)
    assert theta_rate == pytest.approx(0.5)


def test_rate_estimator_velocity_uses_the_readings():
    rates = RateEstimator(DT, "velocity")
    assert rates.update((1.0, 2.0, 0.1, 0.3)) == (1.0, 2.0, 0.1, 0.3)


def test_rate_estimator_smooths_both_rates():
    rates = RateEstimator(DT, "velocity", smoothing=0.5)
    rates.update((0.0, 2.0, 0.0, 4.0))
    _, x_rate, _, theta_rate = rates.update((0.0, 0.0, 0.0, 0.0))
    assert (x_rate, theta_rate) == pytest.approx((1.0, 2.0))


def test_rate_estimator_rejects_unknown_source():
    with pytest.raises(ValueError):
        RateEstimator(DT, "guess")


def test_pd_controller_force_and_cap():
    controller = PDController(
        kp=20, kd=3, dt=DT, max_force=10, rate_from="velocity", kx=0.5, kv=1
    )
    # 20 * 0.1 + 3 * 0.5 + 0.5 * 1.0 + 1 * (-1.0) = 3.0
    assert controller((1.0, -1.0, 0.1, 0.5)) == pytest.approx(3.0)
    assert controller((0.0, 0.0, 1.0, 0.0)) == 10  # 20 N, capped
    assert controller((0.0, 0.0, -1.0, 0.0)) == -10


def test_linear_controller_pushes_by_sign():
    def push(theta):
        return LinearController([0, 0, 1, 0], DT, rate_from="velocity")(
            (0, 0, theta, 0)
        )

    assert push(0.1) == 10.0
    assert push(-0.1) == -10.0
    assert push(0.0) == 10.0  # exactly zero counts as right


def test_linear_controller_ignores_positive_scaling():
    """Multiplying every weight by a positive number gives the same pushes:
    the reason learned weights can be compared after scaling to Kp."""
    rng = random.Random(0)
    weights = [rng.uniform(-2, 2) for _ in range(4)]
    original = LinearController(weights, DT)
    scaled = LinearController([7.5 * w for w in weights], DT)
    for _ in range(200):
        reading = tuple(rng.uniform(-1, 1) for _ in range(4))
        assert original(reading) == scaled(reading)


def test_linear_controller_needs_four_weights():
    with pytest.raises(ValueError):
        LinearController([1, 2, 3], DT)


def test_weights_file_round_trip(tmp_path):
    path = tmp_path / "weights.json"
    save_weights(path, [-0.1, 1.2, 4.3, 6.0], "velocity")
    assert load_weights(path) == ([-0.1, 1.2, 4.3, 6.0], "velocity")
