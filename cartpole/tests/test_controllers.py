"""Tests for cartpole.py's rate estimation and controllers."""

import json
import random
from pathlib import Path

import pytest

import cartpole
from cartpole import (
    Differencer,
    LinearController,
    PDController,
    RateEstimator,
    SimulationConfig,
    Smoother,
    check_controller_settings,
    load_weights,
    locate_weights,
    make_controller,
    save_weights,
)
from config import load_config, load_config_file

DT = 0.02
CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "cartpole"


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


@pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.json")), ids=str)
def test_committed_simulation_configs_load(path):
    config = load_config(path, SimulationConfig)
    check_controller_settings(config, load_config_file(path, SimulationConfig))
    locate_weights(config, from_config_file=True)  # the weights file exists


def test_weights_file_names_its_inputs(tmp_path):
    path = tmp_path / "weights.json"
    save_weights(path, [1, 2, 3, 4], "difference")
    saved = json.loads(path.read_text())
    assert saved["inputs"][2] == "pole angle (rad)"
    assert len(saved["inputs"]) == len(saved["weights"])


@pytest.mark.parametrize(
    "controller, given, message",
    [
        ("p", {"kd": 3.0}, "--kd needs controller pd"),
        ("none", {"max_force": 5.0}, "--max-force needs controller p or"),
        ("pd", {"weights": "w.json"}, "--weights needs controller learned"),
        ("learned", {"rate_from": "velocity"}, "--rate-from needs controller pd"),
        ("learned", {}, "needs --weights"),
    ],
)
def test_settings_the_controller_does_not_use_are_rejected(controller, given, message):
    config = SimulationConfig(controller=controller, **given)
    with pytest.raises(ValueError, match=message):
        check_controller_settings(config, {"controller": controller, **given})


def test_settings_left_at_their_defaults_are_not_rejected():
    # kd has a default, but wasn't asked for, so controller p is fine.
    check_controller_settings(SimulationConfig(controller="p"), {"controller": "p"})


def test_make_controller_uses_the_settings():
    config = SimulationConfig(
        controller="pd", kp=20, kd=3, kx=0.5, kv=1, rate_from="velocity"
    )
    controller = make_controller(config, DT)
    assert controller((1.0, -1.0, 0.1, 0.5)) == pytest.approx(3.0)


def test_make_controller_learned_reads_weights(tmp_path):
    path = tmp_path / "weights.json"
    save_weights(path, [0, 0, 1, 0], "velocity")
    controller = make_controller(
        SimulationConfig(controller="learned", weights=str(path), max_force=4), DT
    )
    assert controller((0, 0, -0.1, 0)) == -4


def learned(weights):
    return SimulationConfig(controller="learned", weights=weights)


def test_config_file_weights_are_relative_to_the_repo_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    weights_file = repo / "cartpole" / "runs" / "r1" / "weights.json"
    weights_file.parent.mkdir(parents=True)
    save_weights(weights_file, [0, 0, 1, 0], "difference")
    monkeypatch.setattr(cartpole, "REPO_ROOT", repo)
    monkeypatch.chdir(tmp_path)  # anywhere but the repo root
    config = locate_weights(
        learned("cartpole/runs/r1/weights.json"), from_config_file=True
    )
    assert config.weights == str(weights_file)


def test_command_line_weights_are_relative_to_the_current_directory(
    tmp_path, monkeypatch
):
    save_weights(tmp_path / "weights.json", [0, 0, 1, 0], "difference")
    monkeypatch.setattr(cartpole, "REPO_ROOT", tmp_path / "elsewhere")
    monkeypatch.chdir(tmp_path)
    config = locate_weights(learned("weights.json"), from_config_file=False)
    assert config.weights == "weights.json"


@pytest.mark.parametrize("from_config_file", [True, False])
def test_missing_weights_file_is_a_clean_error(tmp_path, monkeypatch, from_config_file):
    monkeypatch.setattr(cartpole, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="weights file not found"):
        locate_weights(learned("runs/nope/weights.json"), from_config_file)
