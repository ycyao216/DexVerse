"""Pure cooking regressions for disposal, pour feedback and collection migration."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "source/dexverse/dexverse/tasks/config"
for suffix, path in (("", CONFIG), (".long_horizon", CONFIG / "long_horizon"),
                     (".long_horizon.cook_foods", CONFIG / "long_horizon/cook_foods")):
    module = ModuleType("cook_bimanual_test" + suffix)
    module.__path__ = [str(path)]
    sys.modules[module.__name__] = module
from cook_bimanual_test.long_horizon.cook_foods import checks, disposal, pour_feedback
from cook_bimanual_test.long_horizon.check_utils import clear_stage_latches


class Scene(dict):
    pass


def asset(pos, quat=(1., 0., 0., 0.)):
    return SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([pos]), root_quat_w=torch.tensor([quat]),
        root_lin_vel_w=torch.zeros(1, 3), root_ang_vel_w=torch.zeros(1, 3),
        default_root_state=torch.tensor([[0., 0., .8, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])))


def env_at_bin():
    q = 2 ** -.5
    scene = Scene(trash_bin=asset((0., 0., 0.)),
                  object=asset((-.035, -.08, .016), (q, q, 0., 0.)),
                  can=asset((.035, .08, .08), (q, -q, 0., 0.)),
                  pot=asset((0., 0., .8)))
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(body_pos_w=torch.ones(1, 10, 3) * 2),
                                    find_bodies=lambda names, **kwargs: (list(range(10)), names))
    return SimpleNamespace(scene=scene, num_envs=1, device="cpu", step_dt=1/60,
                           episode_length_buf=torch.zeros(1, dtype=torch.long),
                           _cook_container_points={name: mesh_points(name) for name in ("object", "can")})


@lru_cache(maxsize=2)
def mesh_points(name):
    Usd = pytest.importorskip("pxr.Usd")
    assets = ROOT / "source/dexverse/dexverse/assets"
    path = (assets / "long_horizon_extra/cooking/water_bottle_opened/water_bottle_opened.usd"
            if name == "object" else assets / "core_assets/ycb/005_tomato_soup_can/tomato_soup_can.usd")
    if not path.is_file():
        pytest.skip("Local cooking mesh assets are required")
    stage = Usd.Stage.Open(str(path))
    return torch.as_tensor(disposal.mesh_hull_points(stage, stage.GetDefaultPrim()), dtype=torch.float32)


TIPS = tuple(f"{hand}_{finger}tip" for hand in ("rh", "lh") for finger in ("th", "ff", "mf", "rf", "lf"))


def tick_disposal(env, steps):
    for _ in range(steps):
        env.episode_length_buf += 1
        result = disposal.containers_disposed(env, TIPS)
    return result.item()


def test_disposal_needs_continuous_hold_and_remains_live_after_completion():
    env = env_at_bin()
    assert disposal.disposal_measurements(env, TIPS)["ready"].tolist() == [[True, True]]
    assert not tick_disposal(env, 14)
    assert tick_disposal(env, 1)
    # Neither latching nor leaving a root inside permits a protruding bottle.
    env.scene["object"].data.root_pos_w[:, 2] = .15
    assert not tick_disposal(env, 1)
    env.scene["object"].data.root_pos_w[:, 2] = .016
    assert not tick_disposal(env, 14)
    assert tick_disposal(env, 1)
    clear_stage_latches(env, None, "cooking")
    assert not tick_disposal(env, 1)


@pytest.mark.parametrize("failure", ["above_rim", "through_side", "below_floor", "moving", "spinning", "right_hold", "left_hold"])
def test_invalid_disposal_never_counts(failure):
    env = env_at_bin()
    data = env.scene["object"].data
    if failure == "above_rim":
        data.root_pos_w[:, 2] = .15
    elif failure == "through_side":
        data.root_pos_w[:, 0] = .14
    elif failure == "below_floor":
        data.root_pos_w[:, 2] = -.02
    elif failure == "moving":
        data.root_lin_vel_w[:, 0] = .2
    elif failure == "spinning":
        data.root_ang_vel_w[:, 0] = 1.1
    else:
        tip = 0 if failure == "right_hold" else 5
        env.scene["robot"].data.body_pos_w[:, tip] = data.root_pos_w + torch.tensor([0., 0., .1])
    assert not tick_disposal(env, 30)


def test_rotated_bin_uses_local_interior():
    env = env_at_bin()
    from cook_bimanual_test.long_horizon.check_utils import rotate
    quat = torch.tensor([[2 ** -.5, 0., 0., 2 ** -.5]])
    offset = torch.tensor([[.4, .5, .8]])
    env.scene["trash_bin"].data.root_pos_w = offset
    env.scene["trash_bin"].data.root_quat_w = quat
    # Rotate all objects about the vertical axis: world coordinates change but
    # the same geometry must still fit in the bin's frame.
    for name in ("object", "can"):
        data = env.scene[name].data
        data.root_pos_w = rotate(quat, data.root_pos_w) + offset
        w, x, y, z = data.root_quat_w.unbind(-1)
        a = 2 ** -.5
        data.root_quat_w = torch.stack((a*w-a*z, a*x-a*y, a*y+a*x, a*z+a*w), dim=-1)
    assert disposal.disposal_measurements(env, TIPS)["ready"].all()


@pytest.mark.parametrize("roll", [15, 30, 45, 60, 75, 120])
def test_round_bottle_on_its_side_counts_without_imaginary_box_corners(roll):
    import math
    from cook_bimanual_test.long_horizon.check_utils import rotate
    env = env_at_bin()
    data = env.scene["object"].data
    a = math.radians(roll) / 2
    data.root_quat_w[:] = torch.tensor([math.cos(a), 0., math.sin(a), 0.])
    points = mesh_points("object")
    rotated = rotate(data.root_quat_w.expand(len(points), -1), points)
    data.root_pos_w[:] = torch.tensor([-.04, -.1, disposal.BIN_WALL - rotated[:, 2].min()])
    assert disposal.disposal_measurements(env, TIPS)["ready"].all()
    assert tick_disposal(env, 15)
    # A real protrusion, not just an empty box corner, still rejects disposal.
    data.root_pos_w[:, 2] += disposal.BIN_HEIGHT
    assert not disposal.disposal_measurements(env, TIPS)["inside"][0, 0]


def test_pour_cues_share_measurements_without_advancing_hold():
    env = env_at_bin()
    env.scene["object"] = asset((0., -.2, 1.05))
    env.scene["can"] = asset((0., .0475, 1.05))
    sources = {
        "water": dict(pour_name="object", label="Water", tilt_axis_local=(0, 1, 0), spout_local=(0, .2, 0), min_lift=.1),
        "soup": dict(pour_name="can", label="Soup", tilt_axis_local=(0, -1, 0), spout_local=(0, -.0475, 0), min_lift=.1),
    }
    for _ in range(30):
        env.episode_length_buf += 1
        for source in sources.values():
            assert not checks.poured_into_pot(env, **source).any()
    before = {key: tuple(v.clone() for v in value) for key, value in env._long_horizon_check_memory.items()
              if key in ("pour_object", "pour_can")}
    for _ in range(50):
        states = pour_feedback.pour_display_state(env, sources)
        assert all(s["pouring"].all() and s["visible"].all() for s in states)
    for key, values in before.items():
        assert all(torch.equal(a, b) for a, b in zip(values, env._long_horizon_check_memory[key]))
    env.scene["can"].data.root_pos_w[:, 0] = .11
    assert not pour_feedback.pour_display_state(env, sources)[1]["pouring"].any()
    for _ in range(30):
        env.episode_length_buf += 1
        for source in sources.values():
            checks.poured_into_pot(env, **source)
    states = pour_feedback.pour_display_state(env, sources)
    assert states[0]["done"].all() and not states[0]["visible"].any()
    assert not states[1]["done"].any() and states[1]["visible"].all()




def test_recorder_writes_aligned_scene_and_cooking_history_without_aliasing(tmp_path):
    import ast
    import os
    import pickle
    import numpy as np
    sys.path.insert(0, str(ROOT / "source/dexverse"))
    from dexverse.benchmark import task_identity
    from dexverse.teleop_utils.recording_io import write_recording

    path = ROOT / "scripts/record_demos.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "TrajectoryPickleRecorder")
    scope = dict(os=os, np=np, torch=torch, task_identity=task_identity, write_recording=write_recording)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    task = "Dexverse-LongHorizon-CookFoods-v0"
    output = tmp_path / "diagnostic.pkl"
    recorder = scope["TrajectoryPickleRecorder"](str(output), task_name=task, env_name=task,
                                                 robot_type="floating_shadow_bimanual")
    scene = {"rigid_object": {name: {"root_pose": torch.zeros(1, 7), "root_velocity": torch.zeros(1, 6)}
                               for name in ("object", "can", "pot", "trash_bin")},
             "articulation": {"robot": {"joint_position": torch.zeros(1, 56)},
                              "stove": {"joint_position": torch.zeros(1, 1)}}}
    runtime = {"episode_step": torch.tensor(0), "stages": {"pour_water": torch.tensor(False)},
               "timers": {"disposal": (torch.tensor(0), torch.tensor(-1), torch.tensor(False))}}
    recorder.start_reset_attempt()
    recorder.start_episode(scene, task_state={"task_runtime": runtime})
    for i in range(3):
        runtime["episode_step"].fill_(i + 1)
        runtime["stages"]["pour_water"].fill_(True)
        scene["articulation"]["stove"]["joint_position"].fill_(.8)
        recorder.record_action(torch.ones(56) * i)
        recorder.record_state(scene, task_state={"task_runtime": runtime})
    recorder.finalize_episode(success=True)
    with output.open("rb") as stream:
        saved = pickle.load(stream)["episodes"][0]
    assert saved["actions"].shape == (3, 56)
    assert len(saved["states"]) == len(saved["task_states"]) == 4
    assert [int(s["task_runtime"]["episode_step"]) for s in saved["task_states"]] == [0, 1, 2, 3]
    assert not saved["task_state"]["task_runtime"]["stages"]["pour_water"]
    assert not saved["states"][0]["articulation"]["stove"]["joint_position"].any()
    assert saved["states"][-1]["articulation"]["stove"]["joint_position"].item() == pytest.approx(.8)
    assert "trash_bin" in saved["states"][-1]["rigid_object"]


def test_episode_state_calls_task_runtime_hooks_and_reports_missing_restore():
    sys.path.insert(0, str(ROOT / "source/dexverse"))
    from dexverse.teleop_utils.episode_task_state import capture_episode_task_state, restore_episode_task_state
    env = SimpleNamespace(cfg=SimpleNamespace(), progress=torch.tensor([True]))
    env.cfg.capture_task_state = lambda e, i: {"progress": e.progress[i].clone()}
    env.cfg.restore_task_state = lambda e, s, i: e.progress.__setitem__(i, s["progress"])
    snapshot = capture_episode_task_state(env)
    env.progress.zero_()
    assert restore_episode_task_state(env, snapshot) == [] and env.progress.item()
    env.cfg.restore_task_state = None
    assert restore_episode_task_state(env, snapshot) == ["task_runtime"]
