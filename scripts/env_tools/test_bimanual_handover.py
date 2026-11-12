# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for wrist boundaries and continuous handover history."""

import __future__
import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/dexverse/dexverse/tasks/config/bimanual/handover.py"


class _JointPositionAction:
    def process_actions(self, actions):
        self._processed_actions = actions * self._scale + self._offset


namespace = {"torch": torch, "math": math, "JointPositionAction": _JointPositionAction}
nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec",
             flags=__future__.annotations.compiler_flag), namespace)
HandoverState = namespace["HandoverState"]
visual_source = SOURCE.with_name("handover_visualization.py")
display_node = next(node for node in ast.parse(visual_source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "handover_display_state")
exec(compile(ast.Module(body=[display_node], type_ignores=[]), str(visual_source), "exec"), namespace)
handover_display_state = namespace["handover_display_state"]


def _state(count=1):
    return HandoverState(count, "cpu", 1 / 60)


def _sample(state, step, *, source=False, receiver=False, lifted=True, clear=True,
            supported=True, stable=True):
    def flags(value):
        return torch.as_tensor(value, dtype=torch.bool).expand_as(state.picked_up)

    return state.update(
        torch.full_like(state.last_step, step), flags(source), flags(receiver), flags(lifted), flags(clear),
        flags(supported), flags(stable),
    ).tolist()


def test_receiver_cannot_pick_up_directly():
    state = _state()
    for step in range(30):
        assert _sample(state, step, receiver=True) == [False]
    assert state.failed.tolist() == [True]


def test_source_touch_on_table_does_not_unlock_receiver_pickup():
    state = _state()
    for step in range(30):
        assert _sample(state, step, source=True, lifted=False, clear=False) == [False]
    assert not state.picked_up.any()
    for step in range(30, 60):
        assert _sample(state, step, receiver=True) == [False]


def test_transfer_requires_live_sustained_receiver_hold():
    state = _state()
    assert _sample(state, 0, source=True, stable=False) == [False]
    assert state.picked_up.tolist() == [True]
    # Both hands may contact during transfer; neither is the exclusive holder.
    assert _sample(state, 1) == [False]
    for step in range(2, 20):
        assert _sample(state, step, receiver=True) == [step == 19]
    # A historical successful hold cannot mask subsequently losing the object.
    assert _sample(state, 20, supported=False) == [False]


@pytest.mark.parametrize("bad_hold", [{"stable": False}, {"lifted": False}])
def test_final_hold_requires_clearance_and_stability(bad_hold):
    state = _state()
    _sample(state, 0, source=True)
    for step in range(1, 30):
        assert _sample(state, step, receiver=True, **bad_hold) == [False]


def test_putting_down_invalidates_attempt_until_reset():
    state = _state()
    _sample(state, 0, source=True)
    _sample(state, 1, source=True, clear=False, lifted=False)
    for step in range(2, 25):
        assert _sample(state, step, receiver=True) == [False]
    assert state.failed.tolist() == [True]
    state.reset()
    _sample(state, 0, source=True)
    for step in range(1, 19):
        result = _sample(state, step, receiver=True)
    assert result == [True]


def test_contact_flicker_is_tolerated_but_free_fall_is_not():
    state = _state()
    _sample(state, 0, source=True)
    for step in range(1, 6):
        _sample(state, step, supported=False)
        assert not state.failed.any()
    _sample(state, 6, source=True)
    assert state.gap_count.tolist() == [0]
    for step in range(7, 13):
        _sample(state, step, supported=False)
    assert state.failed.tolist() == [True]
    for step in range(13, 35):
        assert _sample(state, step, receiver=True) == [False]


def test_spawn_settling_does_not_count_as_dropping():
    state = _state()
    for step in range(30):
        _sample(state, step, lifted=False, clear=False, supported=False)
    assert not state.failed.any()
    assert not state.picked_up.any()


def test_repeated_checks_in_same_step_do_not_advance_timers():
    state = _state()
    _sample(state, 0, source=True)
    for _ in range(30):
        assert _sample(state, 1, receiver=True) == [False]
    assert state.hold_count.tolist() == [1]


def test_visual_feedback_tracks_transfer_and_live_success():
    state = _state()
    assert handover_display_state(state)[0].tolist() == [0]
    _sample(state, 0, source=True)
    assert handover_display_state(state)[0].tolist() == [1]
    _sample(state, 1, receiver=True)
    status, progress = handover_display_state(state)
    assert status.tolist() == [2]
    assert 0 < progress.item() < 1
    for step in range(2, 19):
        _sample(state, step, receiver=True)
    status, progress = handover_display_state(state)
    assert status.tolist() == [3]
    assert progress.item() == 1
    _sample(state, 19, receiver=True, stable=False)
    status, progress = handover_display_state(state)
    assert status.tolist() == [1]
    assert progress.item() == 0


def test_visual_feedback_partial_reset_preserves_other_environment():
    state = _state(2)
    _sample(state, 0, source=True)
    _sample(state, 1, receiver=True, clear=[False, True])
    assert handover_display_state(state)[0].tolist() == [4, 2]
    state.reset(torch.tensor([0]))
    status, progress = handover_display_state(state)
    assert status.tolist() == [0, 2]
    assert progress[0] == 0 and progress[1] > 0


def test_visual_reads_do_not_advance_hold_timer():
    state = _state()
    _sample(state, 0, source=True)
    _sample(state, 1, receiver=True)
    for _ in range(100):
        handover_display_state(state)
    assert state.hold_count.tolist() == [1]
    assert not state.success.any()


def test_previous_success_is_distinct_from_current_episode_success():
    state = _state(2)
    _sample(state, 0, source=True)
    for step in range(1, 19):
        _sample(state, step, receiver=True)
    previous_success = state.success.clone()
    state.reset(torch.tensor([0]))
    status, progress = handover_display_state(state, previous_success)
    assert status.tolist() == [5, 3]  # Previous result versus a live success.
    assert progress.tolist() == [1, 1]
    assert state.success.tolist() == [False, True]
    _sample(state, 19, source=[True, False], receiver=[False, True])
    assert handover_display_state(state, previous_success)[0].tolist() == [1, 3]


def test_partial_reset_and_episode_rollover_clear_history():
    state = _state(2)
    _sample(state, 100, source=True)
    _sample(state, 101, clear=[False, True])
    assert state.failed.tolist() == [True, False]
    state.reset(torch.tensor([0]))
    assert state.picked_up.tolist() == [False, True]
    assert state.last_step.tolist() == [-1, 101]
    _sample(state, 102, source=[True, False], receiver=[False, True])
    assert state.picked_up.tolist() == [True, True]
    _sample(state, 0, lifted=False, clear=False)
    assert state.picked_up.tolist() == [False, False]
    assert not state.failed.any()


@pytest.mark.parametrize("axis,expected", [(1.0, (-0.45, 0.3)), (-1.0, (-0.3, 0.45))])
def test_world_workspace_accounts_for_mount_and_axis_direction(axis, expected):
    limits = namespace["_axis_joint_limits"](-0.3, axis, (-0.75, 0.0))
    assert limits == pytest.approx(expected)
    # Translation of the whole cloned environment must not change joint limits.
    translated = namespace["_axis_joint_limits"](9.7, axis, (9.25, 10.0))
    assert translated == pytest.approx(expected)


def test_workspace_rejects_oblique_translation_axes():
    with pytest.raises(ValueError, match="aligned"):
        namespace["_axis_joint_limits"](0, 0.7, (-0.75, 0))


def test_wrist_clamps_after_action_scale_and_default_joint_offset():
    term = namespace["HandoverWristAction"].__new__(namespace["HandoverWristAction"])
    term._scale = 2.0
    term._offset = torch.tensor([[0.2, -0.2]])
    term._joint_ids = [0, 1]
    term._asset = SimpleNamespace(data=SimpleNamespace(joint_pos_limits=torch.tensor([[[-0.45, 0.3], [-0.3, 0.45]]])))
    term.process_actions(torch.tensor([[5.0, -5.0]]))
    assert term._processed_actions.tolist()[0] == pytest.approx([0.3, -0.3])
    term.process_actions(torch.tensor([[0.025, -0.025]]))
    assert term._processed_actions.tolist()[0] == pytest.approx([0.25, -0.25])


@pytest.fixture
def success_fixture():
    path = SOURCE.with_name("bimanual_handover_cfg.py")
    # Load the task's constants and predicates, without scene construction.
    definitions = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            definitions.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.startswith(("SUCCESS_", "CUBE_SIZE"))
            for target in node.targets
        ):
            definitions.append(node)
    scope = {
        "torch": torch,
        "HandoverState": HandoverState,
        "update_handover_visuals": namespace["update_handover_visuals"],
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
        "quat_apply_inverse": lambda quat, vector: vector,  # Identity object orientation in this fixture.
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)

    def sensor():
        return SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=torch.zeros(1, 1, 6, 3), net_forces_w=torch.zeros(1, 1, 3),
        ))

    sensors = {"handover_object_contact": sensor()}
    obj = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([[10.0, 19.7, 0.63]]), root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_lin_vel_w=torch.zeros(1, 3), root_ang_vel_w=torch.zeros(1, 3),
    ))

    class Scene(dict):
        pass

    robot = SimpleNamespace(body_names=["lh_palm"], data=SimpleNamespace(
        body_com_pos_w=torch.tensor([[[10.0, 20.0, 0.65]]]),
        body_com_lin_vel_w=torch.zeros(1, 1, 3), body_com_ang_vel_w=torch.zeros(1, 1, 3),
    ))
    scene = Scene(object=obj, robot=robot)
    scene.sensors = sensors
    scene.env_origins = torch.tensor([[10.0, 20.0, 0.0]])
    env = SimpleNamespace(
        scene=scene, device="cpu", num_envs=1, step_dt=1 / 60,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        cfg=SimpleNamespace(robot_config=SimpleNamespace(left_palm_body_name="lh_palm"),
                            scene=SimpleNamespace(table=SimpleNamespace(
            init_state=SimpleNamespace(pos=(0, 0, 0.575)), spawn=SimpleNamespace(size=(1.5, 1.5, 0.05)),
        ))),
    )
    check = scope["handover_success"]
    params = {"left_contact_filter_indices": (3, 4, 5), "right_contact_filter_indices": (1, 2)}
    return env, check, params


def test_success_predicate_uses_finger_contacts_and_palm_support_at_midline(success_fixture):
    env, check, params = success_fixture
    obj = env.scene["object"]
    forces = env.scene.sensors["handover_object_contact"].data.force_matrix_w
    # Pinching with two finger segments: opposite forces cancel in net_forces_w.
    forces[:, :, 1, 0] = 2
    forces[:, :, 2, 0] = -2
    assert check(env, **params).tolist() == [False]  # Still on the table.
    assert not env._handover_state.picked_up.any()
    obj.data.root_pos_w[:, 2] = 0.70
    env.episode_length_buf += 1
    check(env, **params)
    assert env._handover_state.picked_up.tolist() == [True]
    forces.zero_()
    forces[:, :, 3, 2] = 3.4  # Only the receiver palm supports the cube.
    obj.data.root_pos_w[:, 1] = 20.0  # Centerline, not 10 cm farther left.
    for _ in range(18):
        env.episode_length_buf += 1
        success = check(env, **params)
    assert success.tolist() == [True]
    forces[:, :, 0, 2] = 1  # Table support invalidates even a high-center cube.
    env.episode_length_buf += 1
    assert check(env, **params).tolist() == [False]
    assert env._handover_state.failed.tolist() == [True]
    namespace["reset_handover_state"](env, torch.tensor([0]))
    assert not env._handover_state.failed.any()
    assert not env._handover_state.picked_up.any()


def test_hold_can_move_with_receiver_but_cannot_slip(success_fixture):
    env, check, params = success_fixture
    obj, robot = env.scene["object"], env.scene["robot"]
    obj.data.root_pos_w[:] = torch.tensor([[10.0, 20.0, 0.70]])
    forces = env.scene.sensors["handover_object_contact"].data.force_matrix_w
    forces[:, :, 1, 2] = 3.4
    check(env, **params)
    forces.zero_()
    forces[:, :, 3, 2] = 3.4
    robot.data.body_com_lin_vel_w[:] = torch.tensor([0.20, 0.10, 0.0])
    robot.data.body_com_ang_vel_w[:] = torch.tensor([0.0, 0.6, 0.0])
    obj.data.root_lin_vel_w[:] = torch.tensor([0.23, 0.10, 0.0])
    obj.data.root_ang_vel_w[:] = torch.tensor([0.0, 0.6, 0.0])
    for _ in range(18):
        env.episode_length_buf += 1
        success = check(env, **params)
    assert success.tolist() == [True]
    obj.data.root_lin_vel_w[:, 0] += 0.10
    env.episode_length_buf += 1
    assert check(env, **params).tolist() == [False]


def test_simulator_reset_reapplies_cached_physical_limits():
    calls = []
    robot = SimpleNamespace(write_joint_position_limit_to_sim=lambda limits, **kwargs: calls.append((limits, kwargs)))
    limits = torch.tensor([[[0.0, 1.5], [-0.3, 0.45]], [[0.0, 1.5], [-0.3, 0.45]]])
    env = SimpleNamespace(scene={"robot": robot}, _handover_wrist_constraints=([0, 2], limits))
    namespace["constrain_handover_wrists"](env)
    assert torch.equal(calls[-1][0], limits)
    namespace["constrain_handover_wrists"](env, torch.tensor([1]))
    assert torch.equal(calls[-1][0], limits[1:2])
    assert calls[-1][1]["env_ids"].tolist() == [1]


def test_shadow_contact_filters_use_colliding_segments_not_empty_tip_markers():
    pytest.importorskip("pxr.Usd")
    path = SOURCE.with_name("bimanual_contact_links.py")
    spec = importlib.util.spec_from_file_location("handover_contact_links_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    usd_path = ROOT / "source/dexverse/dexverse/robot_agents/shadow/floating_shadow_bimanual/floating_shadow_bimanual.usd"
    links = module.resolve_bimanual_collision_links(SimpleNamespace(spawn=SimpleNamespace(usd_path=usd_path)))
    for side, names in (("lh", links.left), ("rh", links.right)):
        assert f"{side}_palm" in names
        assert f"{side}_ffproximal" in names
        assert f"{side}_ffdistal" in names
        assert f"{side}_thdistal" in names
        assert f"{side}_fftip" not in names
        assert f"{side}_thtip" not in names
