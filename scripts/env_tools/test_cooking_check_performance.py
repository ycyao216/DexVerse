"""Check that cooking hot-path reductions preserve live state and status timing."""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

CONFIG = Path(__file__).resolve().parents[2] / "source/dexverse/dexverse/tasks/config"
for suffix, path in (("", CONFIG), (".long_horizon", CONFIG / "long_horizon"),
                     (".long_horizon.cook_foods", CONFIG / "long_horizon/cook_foods"),
                     (".long_horizon.make_coffee", CONFIG / "long_horizon/make_coffee"),
                     (".long_horizon.microwave_retrieve_place", CONFIG / "long_horizon/microwave_retrieve_place")):
    module = ModuleType("cooking_performance_test" + suffix)
    module.__path__ = [str(path)]
    sys.modules[module.__name__] = module
from cooking_performance_test.long_horizon import check_utils
from cooking_performance_test.long_horizon.cook_foods import checks, disposal
from cooking_performance_test.long_horizon.make_coffee import checks as coffee
from cooking_performance_test.long_horizon.microwave_retrieve_place import checks as microwave


def test_status_schedule_preserves_boundary_empty_messages_and_reset(capsys):
    env = SimpleNamespace(episode_length_buf=torch.tensor([0]), step_dt=1 / 60)
    assert check_utils.status_due(env)
    check_utils.print_status(env, "Cook Foods", [])
    assert check_utils.status_due(env)
    check_utils.print_status(env, "Cook Foods", ["water not poured"])
    assert not check_utils.status_due(env)
    env.episode_length_buf.fill_(179)
    assert not check_utils.status_due(env)
    env.episode_length_buf.fill_(180)
    assert check_utils.status_due(env)
    check_utils.print_status(env, "Cook Foods", ["can not poured"])
    env.episode_length_buf.fill_(1)
    assert check_utils.status_due(env)
    check_utils.print_status(env, "Cook Foods", ["water not poured"])
    assert capsys.readouterr().out.splitlines() == [
        "[Cook Foods] Waiting: water not poured.",
        "[Cook Foods] Waiting: can not poured.",
        "[Cook Foods] Waiting: water not poured.",
    ]


def test_cooking_skips_only_diagnostics_that_would_not_be_printed(monkeypatch, capsys):
    env = SimpleNamespace(num_envs=1, episode_length_buf=torch.tensor([0]), step_dt=1 / 60)
    measurements = []

    def measure(*args, **kwargs):
        measurements.append(int(env.episode_length_buf[0]))
        return {key: torch.ones(1, 2, dtype=torch.bool) for key in ("inside", "released", "settled")}

    monkeypatch.setattr(disposal, "disposal_measurements", measure)
    monkeypatch.setattr(checks, "_pot_errors", lambda *a: {key: torch.zeros(1) for key in ("xy", "z", "tilt")})
    monkeypatch.setattr(checks, "ingredients_in_pot", lambda *a: torch.ones(1, dtype=torch.long))
    monkeypatch.setattr(checks, "knob_angle", lambda *a: torch.ones(1))
    params = dict(burner_local=(0., 0., 0.), knob_joint="knob", knob_threshold_rad=.5,
                  ingredient_names=("meat",), zone_center=(0., 0., 0.), zone_radius=.1,
                  zone_half_height=.1, pour_names=("object", "can"), disposal={"fingertip_body_names": ("tip",)})
    # An empty message must not delay a problem that appears on the next call.
    checks.report_status(env, torch.tensor([True]), [], **params)
    assert check_utils.status_due(env)
    checks.report_status(env, torch.tensor([True]), ["water"], **params)
    env.episode_length_buf.fill_(1)
    checks.report_status(env, torch.tensor([True]), ["can"], **params)
    assert measurements == [0, 0]
    env.episode_length_buf.fill_(180)
    checks.report_status(env, torch.tensor([True]), ["can"], **params)
    assert measurements == [0, 0, 180]
    assert capsys.readouterr().out.splitlines() == [
        "[Cook Foods] Waiting: water not poured.", "[Cook Foods] Waiting: can not poured."]


@pytest.mark.parametrize("reporter", ["coffee_place", "coffee_lever", "microwave"])
def test_other_tasks_refresh_diagnostics_when_the_next_message_is_due(reporter, monkeypatch, capsys):
    machine = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 1),
        default_joint_pos=torch.zeros(1, 1)), find_joints=lambda name: ([0], [name]))
    env = SimpleNamespace(num_envs=1, scene={"coffee_machine": machine},
                          episode_length_buf=torch.tensor([0]), step_dt=1 / 60)
    measured_steps = []

    def tray(*args):
        measured_steps.append(int(env.episode_length_buf[0]))
        return [f"mug off tray at step {measured_steps[-1]}"]

    def food(*args):
        measured_steps.append(int(env.episode_length_buf[0]))
        return {key: torch.tensor([.1 if key == "outside" else 0.]) for key in ("outside", "floor", "tilt")}

    monkeypatch.setattr(coffee, "_missing_on_tray", tray)
    monkeypatch.setattr(coffee, "_fingertip_gap", lambda *a: torch.tensor([1.]))
    monkeypatch.setattr(microwave, "_food_errors", food)
    monkeypatch.setattr(microwave, "joint_angle", lambda *a: torch.zeros(1))
    monkeypatch.setattr(microwave, "_dial_angle", lambda *a: torch.ones(1))
    if reporter == "coffee_place":
        report = lambda: coffee.report_place_stage(env, torch.tensor([True]), (0., 0., 0.), "machine", ("tip",))
    elif reporter == "coffee_lever":
        report = lambda: coffee.report_lever_stage(env, torch.tensor([True]), "lever", .5, (0., 0., 0.), "machine")
    else:
        report = lambda: microwave.report_status(env, torch.tensor([True]), {}, "fridge", "door", ("dial",))
    report()
    env.episode_length_buf.fill_(179)
    report()
    assert measured_steps == [0]
    env.episode_length_buf.fill_(180)
    report()
    assert measured_steps == [0, 180]
    messages = capsys.readouterr().out.splitlines()
    assert len(messages) == 2
    if reporter.startswith("coffee"):
        assert "step 0" in messages[0] and "step 180" in messages[1]
    else:
        assert all("food 10.0 cm from inside the microwave" in msg for msg in messages)


def _body(pos):
    return SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([pos]),
        root_quat_w=torch.tensor([[1., 0., 0., 0.]]), root_lin_vel_w=torch.zeros(1, 3),
        root_ang_vel_w=torch.zeros(1, 3)))


def _robot():
    calls = []

    def find_bodies(names, preserve_order):
        calls.append((names, preserve_order))
        return [0 if name == "right_tip" else 1 for name in names], names

    return SimpleNamespace(data=SimpleNamespace(body_pos_w=torch.full((1, 2, 3), 2.)),
                           find_bodies=find_bodies), calls


def test_disposal_lookup_cache_keeps_measurements_live_and_handles_robot_changes():
    robot, calls = _robot()
    env = SimpleNamespace(device="cpu", scene={"trash_bin": _body((0., 0., 0.)),
        "object": _body((0., 0., .1)), "can": _body((0., 0., .1)), "robot": robot},
        _cook_container_points={name: torch.zeros(1, 3) for name in ("object", "can")})
    assert disposal.disposal_measurements(env, ["right_tip"])["ready"].all()
    assert disposal.disposal_measurements(env, ("right_tip",))["ready"].all()
    assert calls == [(["right_tip"], True)]
    # Pose and velocity changes without an environment step are always visible.
    robot.data.body_pos_w[:, 0] = torch.tensor([0., .05, .1])
    assert not disposal.disposal_measurements(env, ("right_tip",))["released"][0, 0]
    env.scene["object"].data.root_lin_vel_w[:, 0] = .2
    assert not disposal.disposal_measurements(env, ("left_tip",))["settled"][0, 0]
    assert calls == [(["right_tip"], True), (["left_tip"], True)]
    replacement, replacement_calls = _robot()
    env.scene["robot"] = replacement
    assert disposal.disposal_measurements(env, ("right_tip",))["released"].all()
    assert replacement_calls == [(["right_tip"], True)]
