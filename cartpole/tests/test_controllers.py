"""Tests for cartpole.py's rate estimation and controllers."""

import json
import math
import random
from pathlib import Path

import pytest

import cartpole
from cartpole import (
    Differencer,
    LinearController,
    PDController,
    RateEstimator,
    Sensor,
    SimulationConfig,
    Smoother,
    ThreeActionController,
    check_controller_settings,
    control_quality,
    controller_from_weights,
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
    assert load_weights(path) == ([-0.1, 1.2, 4.3, 6.0], "velocity", "two-action")


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


# Push when 10 * angle passes 1 either way, otherwise coast: a dead band
# of 0.1 rad. Rows: push left, don't push, push right; last column: bias.
DEAD_BAND = [[0, 0, -10, 0, 0], [0, 0, 0, 0, 1], [0, 0, 10, 0, 0]]


def three_action_push(weights, theta):
    controller = ThreeActionController(weights, DT, rate_from="velocity")
    return controller((0, 0, theta, 0))


@pytest.mark.parametrize(
    "theta, force", [(0.05, 0.0), (-0.05, 0.0), (0.2, 10.0), (-0.2, -10.0)]
)
def test_three_action_controller_has_a_dead_band(theta, force):
    assert three_action_push(DEAD_BAND, theta) == force


def test_three_action_controller_ignores_positive_scaling():
    scaled = [[3 * w for w in row] for row in DEAD_BAND]
    for theta in (-0.3, -0.08, 0.0, 0.07, 0.25):
        assert three_action_push(scaled, theta) == three_action_push(DEAD_BAND, theta)


def test_without_biases_the_choice_ignores_how_far_from_balance():
    """Why the biases are needed: with none, scaling the state up never
    changes the action, so "don't push" can't mean "nearly balanced"."""
    rng = random.Random(1)
    weights = [[rng.uniform(-2, 2) for _ in range(4)] + [0.0] for _ in range(3)]
    for _ in range(200):
        reading = tuple(rng.uniform(-1, 1) for _ in range(4))
        further = tuple(5 * v for v in reading)
        controller = ThreeActionController(weights, DT, rate_from="velocity")
        same_but_further = ThreeActionController(weights, DT, rate_from="velocity")
        assert controller(reading) == same_but_further(further)


def test_three_action_controller_needs_three_rows_of_five():
    with pytest.raises(ValueError):
        ThreeActionController([[0] * 4] * 3, DT)


def test_controller_from_weights_picks_the_kind():
    assert isinstance(controller_from_weights([0, 0, 1, 0], DT), LinearController)
    assert isinstance(controller_from_weights(DEAD_BAND, DT), ThreeActionController)


def test_three_action_weights_file_round_trip(tmp_path):
    path = tmp_path / "weights.json"
    save_weights(path, DEAD_BAND, "velocity")
    saved = json.loads(path.read_text())
    assert saved["actions"] == ["push left", "don't push", "push right"]
    assert saved["inputs"][-1].startswith("bias")
    assert load_weights(path) == (DEAD_BAND, "velocity", "three-action")


def test_continuous_weights_make_a_pd_controller(tmp_path):
    """Continuous weights are the gains Kx, Kv, Kp, Kd: force = weights ·
    inputs, capped."""
    path = tmp_path / "weights.json"
    save_weights(path, [0.5, 1.0, 20.0, 3.0], "velocity", kind="continuous")
    weights, rate_from, kind = load_weights(path)
    assert kind == "continuous"
    controller = controller_from_weights(weights, DT, 10.0, rate_from, kind)
    assert isinstance(controller, PDController)
    # 0.5 * 1 + 1 * (-1) + 20 * 0.1 + 3 * 0.5 = 3.0
    assert controller((1.0, -1.0, 0.1, 0.5)) == pytest.approx(3.0)
    assert controller((0.0, 0.0, 1.0, 0.0)) == 10.0  # 20 N, capped


def test_weights_saved_without_a_kind_are_two_action(tmp_path):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"weights": [0, 0, 1, 0], "rate_from": "difference"}))
    assert load_weights(path)[2] == "two-action"


def test_control_quality():
    states = [(0.1, 0, math.radians(2), 0), (-0.1, 0, math.radians(-2), 0)]
    quality = control_quality(states, [10.0, 0.0, -10.0, -5.0], tau=0.5)
    assert quality["rms_angle_degrees"] == pytest.approx(2.0)
    assert quality["rms_position"] == pytest.approx(0.1)
    assert quality["mean_force"] == pytest.approx(6.25)
    assert quality["push_fraction"] == pytest.approx(0.75)
    assert quality["reversals_per_second"] == pytest.approx(1 / 2.0)  # 1 in 2 s


STATE = (0.3, -0.4, 0.05, 0.2)


def test_sensor_is_perfect_by_default():
    assert Sensor().read(STATE) == STATE


def test_position_noise_has_the_given_spread():
    sensor = Sensor(position_noise_std=0.02, seed=3)
    readings = [sensor.read(STATE) for _ in range(20000)]
    positions = [r[0] - STATE[0] for r in readings]
    mean = sum(positions) / len(positions)
    spread = math.sqrt(sum((p - mean) ** 2 for p in positions) / len(positions))
    assert abs(mean) < 0.001 and spread == pytest.approx(0.02, rel=0.03)
    assert {r[2] for r in readings} == {STATE[2]}  # the angle stays perfect


def test_position_noise_leaves_the_angle_noise_unchanged():
    """Position noise has its own random stream, so a seed gives the same
    angle noise with or without it."""
    angle_only = Sensor(noise_std=0.01, seed=5)
    both = Sensor(noise_std=0.01, position_noise_std=0.05, seed=5)
    for _ in range(100):
        assert angle_only.read(STATE)[2] == both.read(STATE)[2]


def test_sensor_scale_and_offset_apply_to_the_angle():
    x, _, theta, _ = Sensor(scale=2.0, offset=0.01).read(STATE)
    assert x == STATE[0] and theta == pytest.approx(2 * 0.05 + 0.01)
