"""CPU regressions for corrected long-horizon collection; no simulator startup."""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import random
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "source/dexverse/dexverse"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checks = load_source("long_horizon_check_utils_test", PACKAGE / "tasks/config/long_horizon/check_utils.py")
filters = load_source("long_horizon_wrist_filter_test", PACKAGE / "devices/retargeters/wrist_filter.py")


_SINGLE_WRIST_LINKS = (
    "x_translation_link", "y_translation_link", "z_translation_link",
    "x_rotation_link", "y_rotation_link", "z_rotation_link",
)


@pytest.fixture
def wrist_mass_helpers():
    """Execute the actual helpers while replacing only simulator cfg classes."""
    path = PACKAGE / "tasks/config/long_horizon/handling.py"
    names = {"configure_light_virtual_wrist_links", "set_virtual_wrist_link_mass"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id == "VIRTUAL_WRIST_LINK_MASS" for t in node.targets))]

    class Entity(SimpleNamespace):
        def __init__(self, name, body_names=None, preserve_order=False, body_ids=slice(None)):
            super().__init__(name=name, body_names=body_names, preserve_order=preserve_order, body_ids=body_ids)

    scope = {"deepcopy": deepcopy, "torch": torch, "SceneEntityCfg": Entity,
             "EventTerm": lambda **kwargs: SimpleNamespace(**kwargs)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**scope)


@pytest.mark.parametrize("robot_type,prefixes", [
    ("floating_shadow_right", ("",)), ("floating_shadow_left", ("",)),
    ("floating_shadow_bimanual", ("rh_", "lh_")),
])
def test_virtual_wrist_config_keeps_shared_events_private_and_targets_exact_links(
    wrist_mass_helpers, robot_type, prefixes,
):
    events = SimpleNamespace(existing_reset=SimpleNamespace(mode="reset", params={"nested": [1, 2]}))
    before = deepcopy(events)
    scene = SimpleNamespace(robot=object())
    observations = object()
    cfg = SimpleNamespace(robot_type=robot_type, events=events, scene=scene, observations=observations)
    short_cfg = SimpleNamespace(events=events)
    wrist_mass_helpers.configure_light_virtual_wrist_links(cfg)
    assert cfg.events is not events
    assert short_cfg.events is events and events == before
    assert cfg.events.existing_reset == before.existing_reset
    assert cfg.events.existing_reset.params is not events.existing_reset.params
    assert cfg.scene is scene and cfg.observations is observations
    expected = [prefix + name for prefix in prefixes for name in _SINGLE_WRIST_LINKS]
    assert len(expected) == (12 if len(prefixes) == 2 else 6)
    for mode in ("startup", "reset"):
        event = getattr(cfg.events, "light_virtual_wrist_links_" + mode)
        assert event.mode == mode
        assert event.func is wrist_mass_helpers.set_virtual_wrist_link_mass
        assert event.params["mass"] == .001
        entity = event.params["asset_cfg"]
        assert entity.name == "robot" and entity.preserve_order
        assert entity.body_names == expected
        assert len(set(entity.body_names)) == len(expected)
    assert cfg.events.light_virtual_wrist_links_startup is not cfg.events.light_virtual_wrist_links_reset


@pytest.mark.parametrize("robot_type", ["floating_allegro_right", "floating_leap_bimanual", "ur10e_shadow_right"])
def test_virtual_wrist_config_is_a_noop_for_unsupported_hands(wrist_mass_helpers, robot_type):
    events = SimpleNamespace(existing=object())
    cfg = SimpleNamespace(robot_type=robot_type, events=events)
    before = dict(vars(cfg))
    wrist_mass_helpers.configure_light_virtual_wrist_links(cfg)
    assert vars(cfg) == before
    assert cfg.events is events


def _mass_rig(prefixes=("",)):
    names = ["base", *[prefix + name for prefix in prefixes for name in _SINGLE_WRIST_LINKS],
             "palm", "ffproximal", "ffdistal", "thdistal"]
    virtual_ids = list(range(1, 1 + len(prefixes) * 6))

    class View:
        def __init__(self):
            # Unequal values catch blanket body or environment replacement.
            self.masses = .1 + torch.arange(3 * len(names), dtype=torch.float32).reshape(3, -1) / 100
            self.masses[:, virtual_ids] = 1.
            self.inertias = torch.arange(3 * len(names) * 9, dtype=torch.float32).reshape(3, -1, 9) * 1e-5
            self.coms = torch.arange(3 * len(names) * 7, dtype=torch.float32).reshape(3, -1, 7) * .001
            self.calls = []

        def get_masses(self):
            return self.masses  # Like PhysX, returns an owned buffer, not a fresh copy.

        def set_masses(self, masses, indices):
            assert masses.device.type == indices.device.type == "cpu"
            assert indices.dtype == torch.long
            assert masses.shape == self.masses.shape
            assert masses.data_ptr() != self.masses.data_ptr(), "Setter must clone the PhysX getter buffer"
            self.calls.append((masses.clone(), indices.clone()))
            self.masses[indices] = masses[indices]

        def get_inertias(self):
            return self.inertias

        def get_coms(self):
            return self.coms

        def set_inertias(self, *args):
            pytest.fail("Mass-only override must not call the inertia setter")

        def set_coms(self, *args):
            pytest.fail("Mass-only override must not call the COM setter")

    view = View()
    data = SimpleNamespace(default_mass=view.masses.clone(), default_inertia=view.inertias.clone(),
                           joint_pos=torch.arange(3 * 28, dtype=torch.float32).reshape(3, 28) / 100,
                           joint_vel=torch.full((3, 28), .03))
    robot = SimpleNamespace(body_names=names, root_physx_view=view, data=data)
    # A CUDA-like env label catches an implementation that incorrectly selects
    # mass-property tensor device from env.device; this test allocates no CUDA data.
    env = SimpleNamespace(scene={"robot": robot}, device="cuda:0")
    return SimpleNamespace(env=env, robot=robot, view=view, virtual_ids=virtual_ids)


@pytest.mark.parametrize("prefixes", [("",), ("rh_", "lh_")])
@pytest.mark.parametrize("env_ids", [None, torch.tensor([2, 0]), [2, 0], (1,), torch.tensor([], dtype=torch.long)])
def test_mass_override_is_idempotent_subset_safe_and_preserves_all_other_properties(
    wrist_mass_helpers, prefixes, env_ids,
):
    rig = _mass_rig(prefixes)
    view, data = rig.view, rig.robot.data
    before = {"masses": view.masses.clone(), "inertias": view.inertias.clone(), "coms": view.coms.clone(),
              **{name: value.clone() for name, value in vars(data).items()}}
    entity = wrist_mass_helpers.SceneEntityCfg("robot", body_ids=rig.virtual_ids)
    rng_torch, rng_numpy, rng_python = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    expected = before["masses"].clone()
    selected = torch.arange(3) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long)
    expected[selected[:, None], torch.tensor(rig.virtual_ids)] = .001
    for _ in range(2):
        wrist_mass_helpers.set_virtual_wrist_link_mass(rig.env, env_ids, entity, .001)
        assert torch.equal(view.masses, expected)
        assert torch.equal(view.inertias, before["inertias"])
        assert torch.equal(view.coms, before["coms"])
        for name, value in vars(data).items():
            assert torch.equal(value, before[name]), name
    assert torch.equal(torch.get_rng_state(), rng_torch)
    current_numpy = np.random.get_state()
    assert current_numpy[0] == rng_numpy[0] and current_numpy[2:] == rng_numpy[2:]
    np.testing.assert_array_equal(current_numpy[1], rng_numpy[1])
    assert random.getstate() == rng_python
    for _, sent_ids in view.calls:
        assert torch.equal(sent_ids, selected)


def test_reset_event_reapplies_after_physics_reinitialization(wrist_mass_helpers):
    rig = _mass_rig()
    cfg = SimpleNamespace(robot_type="floating_shadow_right", events=SimpleNamespace())
    wrist_mass_helpers.configure_light_virtual_wrist_links(cfg)
    for mode in ("startup", "reset"):
        event = getattr(cfg.events, "light_virtual_wrist_links_" + mode)
        entity = event.params["asset_cfg"]
        entity.body_ids = [rig.robot.body_names.index(name) for name in entity.body_names]
        event.func(rig.env, None, **event.params)
        assert torch.all(rig.view.masses[:, rig.virtual_ids] == .001)
        # A physics restart reconstructs source masses, before the reset event.
        if mode == "startup":
            rig.view.masses[:] = rig.robot.data.default_mass


@pytest.mark.parametrize("body_ids", [slice(None), [], [0], [7], [1, 7]])
def test_mass_override_rejects_unresolved_or_physical_body_selection(wrist_mass_helpers, body_ids):
    rig = _mass_rig()
    before = rig.view.masses.clone()
    entity = wrist_mass_helpers.SceneEntityCfg("robot", body_ids=body_ids)
    with pytest.raises(ValueError, match="explicitly resolved|floating wrist links"):
        wrist_mass_helpers.set_virtual_wrist_link_mass(rig.env, None, entity, .001)
    assert not rig.view.calls
    assert torch.equal(rig.view.masses, before)


@pytest.mark.parametrize("robot_type", ["floating_shadow_right", "floating_shadow_left", "floating_shadow_bimanual"])
def test_wrist_damping_tuning_changes_only_wrist_damping_on_a_private_config(robot_type):
    path = PACKAGE / "tasks/config/long_horizon/handling.py"
    nodes = [node for node in ast.parse(path.read_text()).body
             if (isinstance(node, ast.FunctionDef) and node.name == "tune_wrist_damping")
             or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WRIST_DAMPING"
                                                     for t in node.targets))]
    scope = {"deepcopy": deepcopy}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    damping = {".*_translation_joint": 400., ".*_rotation_joint": 400., ".*FFJ.*": .1}
    robot = SimpleNamespace(actuators={
        "mixed": SimpleNamespace(damping=damping, stiffness={".*": 2000.}, effort_limit_sim=15.,
                                 velocity_limit_sim=10., armature=None),
        "fingers": SimpleNamespace(damping={".*THJ.*": .1}, stiffness={".*": 10.}, effort_limit_sim=10.),
    }, spawn=SimpleNamespace(rigid_props={"max_depenetration_velocity": 1000.}))
    before = deepcopy(robot)
    cfg = SimpleNamespace(robot_type=robot_type, scene=SimpleNamespace(robot=robot))
    shared_short_task = SimpleNamespace(scene=SimpleNamespace(robot=robot))
    scope["tune_wrist_damping"](cfg)
    assert cfg.scene.robot is not robot
    assert shared_short_task.scene.robot is robot and robot == before
    expected = deepcopy(before)
    expected.actuators["mixed"].damping.update({".*_translation_joint": 200., ".*_rotation_joint": 200.})
    assert cfg.scene.robot == expected
    # Unsupported hands retain even the original configuration identity.
    unsupported = SimpleNamespace(robot_type="floating_allegro_right", scene=SimpleNamespace(robot=robot))
    scope["tune_wrist_damping"](unsupported)
    assert unsupported.scene.robot is robot and robot == before


def test_reset_clears_spill_even_when_next_attempt_reaches_stage_later():
    env = SimpleNamespace(episode_length_buf=torch.tensor([100]), step_dt=0.1)
    _, spilled = checks.track(env, "spill", torch.tensor([True]), 0.1)
    assert spilled.item()
    checks.clear_stage_latches(env, torch.tensor([0]), "coffee")
    env.episode_length_buf[:] = 150
    _, spilled = checks.track(env, "spill", torch.tensor([False]), 0.1)
    assert not spilled.item()


def test_partial_reset_preserves_other_envs_and_refreshes_visuals():
    env = SimpleNamespace(episode_length_buf=torch.tensor([100, 100]), step_dt=0.1)
    checks.track(env, "pour", torch.tensor([True, True]), 0.1)
    stage = SimpleNamespace(latched_flags={"pour": torch.tensor([True, True])},
                            raw_flags={"stale": True}, raw_step_buf=torch.tensor([100, 100]),
                            eval_flags={"stale": True}, eval_step_buf=torch.tensor([100, 100]))
    env._stage_graph_runtime_cache = {"cook": stage}
    env._long_horizon_shown = {"liquid": torch.tensor([3, 3])}
    checks.clear_stage_latches(env, torch.tensor([0]), "cook")
    assert stage.latched_flags["pour"].tolist() == [False, True]
    assert stage.raw_flags is None and stage.eval_flags is None
    assert checks.ever_held(env, "pour", torch.ones(2)).tolist() == [False, True]
    assert env._long_horizon_shown["liquid"].tolist() == [-1, 3]


def test_hold_duration_is_once_per_step_and_reset_starts_fresh():
    env = SimpleNamespace(episode_length_buf=torch.tensor([1]), step_dt=0.1)
    condition = torch.tensor([True])
    for _ in range(20):
        held, _ = checks.track(env, "pour", condition, 0.3)
        assert not held.item()
    for step in (2, 3):
        env.episode_length_buf[:] = step
        held, _ = checks.track(env, "pour", condition, 0.3)
    assert held.item()
    checks.clear_stage_latches(env, None, "coffee")
    held, reached = checks.track(env, "pour", condition, 0.3)
    assert not held.item() and not reached.item()


def test_wrist_filter_rejects_tracking_jump_without_flinging_object():
    now = [0.0]
    filtered = filters.WristPoseFilter(filters.WristFilterCfg(), clock=lambda: now[0])
    home = np.array([0.1, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(filtered.update(home), home, atol=1e-6)
    now[0] += 1 / 60
    bad = home.copy()
    bad[0] += 2
    np.testing.assert_allclose(filtered.update(bad), home, atol=1e-6)
    assert filtered.num_rejected == 1
    now[0] += 1 / 60
    recovered = home.copy()
    recovered[0] += 0.005
    result = filtered.update(recovered)
    assert home[0] < result[0] <= recovered[0]


def test_absolute_wrist_continuity_is_opt_in_and_crosses_pi_smoothly():
    # Compile the actual methods without importing the simulator-only device API.
    methods = []
    for filename, method in [("simple_absolute_retargeting.py", "_assign_hand_wrist_command"),
                              ("simple_relative_retargeting.py", "_unwrap_and_clip_wrist_euler")]:
        tree = ast.parse((PACKAGE / "devices/retargeters" / filename).read_text())
        methods.append(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method))
    namespace = {"np": np, "R": Rotation, "wxyz_to_xyzw": lambda q: np.asarray(q)[[1, 2, 3, 0]]}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "wrist_continuity", "exec"), namespace)
    hand = "right"
    layout = {"wrist_rot_repr": "euler", "wrist_rot_order": "xyz", "wrist_trans_indices": [0, 1, 2],
              "wrist_rot_indices": [3, 4, 5]}
    for enabled in (False, True):
        retargeter = SimpleNamespace(cfg=SimpleNamespace(continuous_wrist_euler=enabled),
            latest_wrist_poses={}, _layout={"hands": {hand: layout}}, _previous_wrist_euler_commands={},
            _wrist_joint_origins={hand: {"pos": [0, 0, 0], "rot": [1, 0, 0, 0]}},
            _assign=lambda action, indices, values: action.__setitem__(indices, values),
            _get_normalized_wrist_rotation=lambda q: Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]))
        retargeter._unwrap_and_clip_wrist_euler = MethodType(namespace["_unwrap_and_clip_wrist_euler"], retargeter)
        commands = []
        for angle in (179, -179):
            quat = Rotation.from_euler("Z", angle, degrees=True).as_quat()[[3, 0, 1, 2]]
            retargeter.latest_wrist_poses[hand] = np.r_[0.1, 0.2, 0.8, quat]
            action = np.zeros(6)
            namespace["_assign_hand_wrist_command"](retargeter, action, hand)
            commands.append(action[5])
        jump = abs(np.rad2deg(commands[1] - commands[0]))
        assert jump == pytest.approx(2 if enabled else 358, abs=0.001)
