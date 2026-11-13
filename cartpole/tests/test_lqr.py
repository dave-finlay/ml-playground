"""Tests for lqr.py: the linear model, the LQR solution and its output."""

import json

import numpy as np
import pytest

from cartpole import CartPole, SimulationConfig
from config import load_config
from lqr import (
    LQRConfig,
    closed_loop_modes,
    cost_matrices,
    discretize,
    linearize,
    optimal_gains,
    save_simulation_config,
    solve_lqr,
)


@pytest.fixture(scope="module")
def cartpole():
    cartpole = CartPole.make(angle_limit=None)
    yield cartpole
    cartpole.close()


def test_linear_model_coefficients(cartpole):
    a, b = linearize(cartpole.physics)
    assert a[3, 2] == pytest.approx(15.776, abs=1e-3)  # angle accel per radian
    assert b[3, 0] == pytest.approx(-1.4634, abs=1e-4)  # angle accel per newton
    assert a[1, 2] == pytest.approx(-0.7171, abs=1e-4)  # cart accel per radian
    assert b[1, 0] == pytest.approx(0.9756, abs=1e-4)  # cart accel per newton


def test_linear_model_matches_the_characteristic_polynomial(cartpole):
    """The closed loop with force = Kx x + Kv x_rate + Kp angle + Kd
    angle_rate has the characteristic polynomial worked out by hand:
    s^4 + (1.46 Kd - 0.98 Kv) s^3 + (1.46 Kp - 15.8 - 0.98 Kx) s^2
        + 14.35 Kv s + 14.35 Kx."""
    a, b = linearize(cartpole.physics)
    kx, kv, kp, kd = 0.5, 1.0, 20.0, 3.0
    closed = a + b @ np.array([[kx, kv, kp, kd]])
    by_hand = [
        1,
        1.4634 * kd - 0.9756 * kv,
        1.4634 * kp - 15.776 - 0.9756 * kx,
        14.35 * kv,
        14.35 * kx,
    ]
    assert np.poly(np.linalg.eigvals(closed)) == pytest.approx(by_hand, rel=1e-3)


def test_discretize_is_one_euler_step():
    a, b = np.array([[0.0, 1.0], [2.0, 0.0]]), np.array([[0.0], [3.0]])
    ad, bd = discretize(a, b, 0.1)
    assert ad == pytest.approx(np.array([[1.0, 0.1], [0.2, 1.0]]))
    assert bd == pytest.approx(np.array([[0.0], [0.3]]))


@pytest.mark.parametrize("gamma", [1.0, 0.99])
def test_solution_satisfies_the_riccati_equation(cartpole, gamma):
    a, b = discretize(*linearize(cartpole.physics), cartpole.tau)
    config = LQRConfig(angle_cost=0.3, position_cost=0.2, force_cost=0.3, gamma=gamma)
    q, r = cost_matrices(config, cartpole.angle_limit, cartpole.track_limit)
    k, p = solve_lqr(a, b, q, r, gamma)
    k_again = gamma * np.linalg.solve(r + gamma * b.T @ p @ b, b.T @ p @ a)
    assert k == pytest.approx(k_again, rel=1e-6)
    assert p == pytest.approx(q + gamma * a.T @ p @ (a - b @ k), rel=1e-6)
    assert np.all(np.abs(np.linalg.eigvals(a - b @ k)) < 1)  # stable


def test_optimal_gains_are_stable_and_have_the_right_signs(cartpole):
    gains, a, b = optimal_gains(
        LQRConfig(angle_cost=0.3, position_cost=0.2, force_cost=0.3), cartpole
    )
    assert np.all(gains > 0)  # including the counter-intuitive cart signs
    assert all(rate > 0 for rate, _ in closed_loop_modes(a, b, gains, cartpole.tau))


def test_cheaper_force_gives_stiffer_gains(cartpole):
    def kp(force_cost):
        config = LQRConfig(angle_cost=0.3, position_cost=0.2, force_cost=force_cost)
        return optimal_gains(config, cartpole)[0][2]

    assert kp(0.03) > kp(0.3) > kp(1.0)


def test_force_cost_is_required():
    with pytest.raises(ValueError, match="infinite"):
        LQRConfig(angle_cost=0.3)


def test_saved_config_runs_the_gains_as_pd(tmp_path):
    path = tmp_path / "lqr.json"
    config = LQRConfig(angle_cost=0.3, force_cost=0.3)
    save_simulation_config(path, [1.0, 2.0, 30.0, 4.0], config)
    loaded = load_config(path, SimulationConfig)
    assert loaded.controller == "pd"
    assert (loaded.kx, loaded.kv, loaded.kp, loaded.kd) == (1.0, 2.0, 30.0, 4.0)
    assert "LQR" in json.loads(path.read_text())["description"]
    with pytest.raises(FileExistsError):
        save_simulation_config(path, [1.0, 2.0, 30.0, 4.0], config)
