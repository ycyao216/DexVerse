# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for object selection and portable single-environment demos."""

import ast
import contextlib
import itertools
import logging
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))
from dexverse.teleop_utils.environment_cycle import (  # noqa: E402
    ObjectEnvironmentCycle,
    configure_object_cycle,
    configure_replay_object_bank,
    replay_object_env_index,
    select_scene_state,
    single_env_bindings,
)
from dexverse.teleop_utils.episode_task_state import (  # noqa: E402
    capture_episode_task_state,
    restore_episode_task_state,
)


def config(count=100):
    return NS(
        teleop_object_cycle_size=100,
        scene=NS(num_envs=1, object=NS(spawn=NS(
            assets_cfg=[NS(usd_path=f"object_{i}.usd") for i in range(count)], random_choice=True,
        ))),
        events=NS(reset_object=NS(params={
            "per_env_positions": [[i, 0, 0] for i in range(count)],
            "per_env_quats": [[1, 0, 0, 0] for _ in range(count)],
        })),
    )


def test_bank_default_override_and_single_object():
    cfg = config()
    assert configure_object_cycle(cfg) == 100
    assert cfg.scene.num_envs == 100 and not cfg.scene.object.spawn.random_choice
    assert len(cfg.scene.object.spawn.assets_cfg) == 100
    cfg = config(6)
    assert configure_object_cycle(cfg, 20) == 6
    cfg = config()
    assert configure_object_cycle(cfg, 4) == 4 and len(cfg.scene.object.spawn.assets_cfg) == 4
    cfg = config()
    assert configure_object_cycle(cfg, 1) == 1 and cfg.teleop_object_cycle_size == 1
    cfg = config()
    cfg.scene.object.spawn = NS(usd_path="custom.usd")
    assert configure_object_cycle(cfg) == 1 and cfg.scene.num_envs == 1
    with pytest.raises(ValueError, match="at least"):
        configure_object_cycle(config(), 0)


def test_scene_snapshot_is_one_row_and_detached_from_live_state():
    state = {"rigid_object": {"object": {"root_pose": torch.arange(28.).reshape(4, 7)}}}
    selected = select_scene_state(state, 2)
    pose = selected["rigid_object"]["object"]["root_pose"]
    assert pose.shape == (1, 7)
    torch.testing.assert_close(pose, torch.arange(14., 21.).reshape(1, 7))
    pose.zero_()
    assert state["rigid_object"]["object"]["root_pose"][2, 0] == 14


def test_cycle_moves_retarget_origins_without_drift_and_controls_only_selected_hand():
    cfg = config()
    configure_object_cycle(cfg, 4)
    origins = torch.tensor([[3., 2., 0.], [3., -2., 0.], [-3., 2., 0.], [-3., -2., 0.]])
    views = []
    env = NS(
        cfg=cfg, num_envs=4, scene=NS(env_origins=origins),
        action_manager=NS(action=torch.arange(12.).reshape(4, 3)),
        viewport_camera_controller=NS(set_view_env_index=views.append),
    )
    retargeter = NS(
        retarget_base_wrist_poses={"right": np.array([.1, .2, .3, 1., 0., 0., 0.])},
        _wrist_joint_origins={"right": {"pos": np.array([.1, .2, .3])}},
    )
    cycle = ObjectEnvironmentCycle(env, NS(_retargeters=[retargeter]), enabled=True)
    for index in [0, 1, 2, 3, 0, 1]:
        assert cycle.index == index
        expected = origins[index].numpy() + [.1, .2, .3]
        np.testing.assert_allclose(retargeter.retarget_base_wrist_poses["right"][:3], expected)
        np.testing.assert_allclose(retargeter._wrist_joint_origins["right"]["pos"], expected)
        assert cycle.object_asset()["usd_path"] == f"object_{index}.usd"
        actions = cycle.actions(torch.tensor([99., 98., 97.]))
        assert actions[index].tolist() == [99, 98, 97]
        others = [i for i in range(4) if i != index]
        torch.testing.assert_close(actions[others], env.action_manager.action[others])
        cycle.advance()
    assert views[:5] == [0, 1, 2, 3, 0]


def test_full_pool_visits_every_object_and_never_repeats_a_success():
    cfg = config()
    configure_object_cycle(cfg)
    env = NS(cfg=cfg, num_envs=100, scene=NS(env_origins=torch.zeros(100, 3)))
    cycle = ObjectEnvironmentCycle(env, enabled=True)
    visited = []
    successes = []
    # Collect ten successes, then skip/fail the other ninety on the first pass.
    for i in range(100):
        visited.append(cycle.index)
        if i < 10:
            successes.append(cycle.object_asset()["usd_path"])
            cycle.mark_success()
        assert cycle.advance()
    assert len(set(visited)) == 100
    # Retry the unfinished pool; the first ten successful objects stay retired.
    for _ in range(40):
        assert cycle.index not in visited[:10]
        successes.append(cycle.object_asset()["usd_path"])
        cycle.mark_success()
        assert cycle.advance()
    assert len(set(successes)) == 50
    # Finite pools end cleanly instead of collecting duplicate successful demos.
    for _ in range(50):
        cycle.mark_success()
        remaining = cycle.advance()
    assert not remaining
    assert not cycle.advance()


@pytest.mark.parametrize("target,expected", [(50, 50), (0, 100), (1, 1)])
def test_real_recorder_loop_collects_distinct_objects_after_success_resets(tmp_path, target, expected):
    """Exercise the success/reset loop, which previously bypassed cycle.advance."""
    path = ROOT / "scripts/record_demos.py"
    names = {"TrajectoryPickleRecorder", "run_simulation_loop"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    cfg = config()
    configure_object_cycle(cfg)
    if expected == 1:
        cfg.teleop_object_cycle_size = 1
        cfg.collection_object_asset = {"object_id": "cup/001", "usd_path": "object_73.usd", "source_env_index": 0}
    env = NS(
        cfg=cfg, num_envs=1 if expected == 1 else 100, device="cpu",
        scene=NS(env_origins=torch.zeros(100, 3), get_state=lambda **kwargs: {
            "rigid_object": {"object": {"root_pose": torch.zeros(100, 7)}},
        }),
        action_manager=NS(action=torch.zeros(100, 2)),
        sim=NS(reset=lambda: None, render=lambda: None, is_stopped=lambda: False),
        reset=lambda: None,
        step=lambda action: (None, None, torch.zeros(100, dtype=torch.bool), torch.zeros(100, dtype=torch.bool)),
    )
    callbacks = {}

    def advance_device():
        callbacks["START"]()
        return torch.zeros(2)

    device = NS(_retargeters=[], add_callback=callbacks.__setitem__, reset=lambda: None, advance=advance_device)
    iterations, clock = itertools.count(), itertools.count()
    output = str(tmp_path / "demos.pkl")
    from dexverse.teleop_utils.recording_io import write_recording
    scope = {
        "write_recording": write_recording,
        "os": os, "np": np, "torch": torch, "pickle": pickle,
        "task_identity": lambda name: {}, "contextlib": contextlib,
        "time": NS(time=lambda: float(next(clock))), "logger": logging.getLogger(__name__),
        "simulation_app": NS(is_running=lambda: next(iterations) < 210),
        "env_cfg": cfg, "ObjectEnvironmentCycle": ObjectEnvironmentCycle,
        "args_cli": NS(num_demos=target, record_state=True, num_success_steps=1,
                       task="test", teleop_device="test", dataset_file=output),
        "handle_reset": lambda env: env.reset(), "capture_episode_task_state": lambda *args, **kwargs: {},
        "_get_goal_pose_from_task_state": lambda state: None,
        "single_env_bindings": single_env_bindings,
        "_get_active_object_metadata_from_env": lambda *args: None,
        "check_success": lambda *args: (1, True),
    }
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    recorder = scope["TrajectoryPickleRecorder"](output, task_name="test", env_name="test")
    assert scope["run_simulation_loop"](env, device, None, recorder) == expected
    with open(output, "rb") as stream:
        payload = pickle.load(stream)
    assert len({ep["object_asset"]["usd_path"] for ep in payload["episodes"]}) == expected
    assert all(ep["success"] for ep in payload["episodes"])
    if expected == 1:
        assert payload["episodes"][0]["object_asset"]["object_id"] == "cup/001"
    assert len(payload["reset_attempts"]) == expected


def test_replay_reorders_objects_and_reset_placements_by_recorded_identity():
    cfg = config()
    episodes = [{"object_asset": {"usd_path": f"object_{i}.usd"}} for i in [3, 1, 3]]
    configure_replay_object_bank(cfg, episodes)
    assert cfg.scene.num_envs == 2
    assert [asset.usd_path for asset in cfg.scene.object.spawn.assets_cfg] == ["object_3.usd", "object_1.usd"]
    assert cfg.events.reset_object.params["per_env_positions"] == [[3, 0, 0], [1, 0, 0]]
    assert [replay_object_env_index(cfg, ep) for ep in episodes] == [0, 1, 0]
    with pytest.raises(ValueError, match="missing"):
        configure_replay_object_bank(config(), [{"object_asset": {"usd_path": "missing.usd"}}])


def test_trace_bindings_are_filtered_and_rebased():
    bindings = [{"resolved_prim_path": f"/World/envs/env_{i}/Object", "asset_idx": i} for i in range(3)]
    assert single_env_bindings(bindings, 2) == [{"resolved_prim_path": "/World/envs/env_0/Object", "asset_idx": 2}]
    assert bindings[2]["resolved_prim_path"].endswith("env_2/Object")


def test_world_goals_round_trip_between_different_environment_origins():
    origins = torch.tensor([[3., 2., 0.], [-3., -2., 0.]])
    local = torch.tensor([.1, .2, .8, 1., 0., 0., 0.])
    poses = local.repeat(2, 1)
    poses[:, :3] += origins
    term = NS(pose_command_b=poses.clone(), pose_command_w=poses.clone(), cfg=NS(use_world_frame=True))
    env = NS(device="cpu", num_envs=2, scene=NS(env_origins=origins), cfg=NS(events=None),
             command_manager=NS(active_terms=["object_pose"], get_term=lambda name: term))
    snapshot = capture_episode_task_state(env, 1, relative_to_env_origin=True)
    torch.testing.assert_close(snapshot["commands"]["object_pose"]["pose_command_b"], local)
    term.pose_command_b.zero_()
    term.pose_command_w.zero_()
    assert restore_episode_task_state(env, snapshot, 0) == []
    expected = local.clone()
    expected[:3] += origins[0]
    torch.testing.assert_close(term.pose_command_b[0], expected)
    torch.testing.assert_close(term.pose_command_w[0], expected)


def test_recorder_success_checks_selected_environment():
    path = ROOT / "scripts/record_demos.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "check_success")
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    scope = {"args_cli": NS(num_success_steps=2),
             "_resolve_term_callable": lambda env, cfg: lambda env, **kwargs: torch.tensor([False, True])}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    cfg = NS(params={})
    assert scope["check_success"](None, cfg, 1, 0) == (0, False)
    assert scope["check_success"](None, cfg, 1, 1) == (2, True)


def test_single_collection_object_keeps_nonzero_pool_annotation_with_one_initial_env():
    from dexverse.teleop_utils.environment_cycle import configure_single_collection_object, object_pool_paths
    cfg = config(100)
    for i, asset in enumerate(cfg.scene.object.spawn.assets_cfg):
        asset.usd_path = f'/assets/mani_twin_selected/object/{i:03}/base_rescaled_unpacked/output.usdc'
    cfg.object_placements = tuple(NS(position_offset=(i, i+1, i+2), quat=(1,0,0,0)) for i in range(100))
    cfg.scene.object.init_state = NS(pos=(0,0,0), rot=(1,0,0,0))
    # parse_env_cfg(num_envs=1) can build reset arrays only for object 0.
    cfg.events.reset_object.params = {'per_env_positions': [[0,1,2]], 'per_env_quats': [[1,0,0,0]], 'support_z': .601}
    asset = configure_single_collection_object(cfg, 'object/073')
    assert cfg.scene.num_envs == cfg.teleop_object_cycle_size == len(object_pool_paths(cfg)) == 1
    assert asset['object_id'] == 'object/073'
    assert cfg.events.reset_object.params['per_env_positions'] == [[73,74,75]]
    assert cfg.scene.object.init_state.pos == (73,74,75.601)
    cycle = ObjectEnvironmentCycle(NS(cfg=cfg, num_envs=1), enabled=False)
    assert cycle.object_asset() == asset
    assert cycle.advance() and cycle.object_asset() == asset
    with pytest.raises(ValueError, match='Unknown'):
        configure_single_collection_object(config(), '../other')


def test_single_object_recording_replay_uses_full_pool_annotations():
    cfg = config()
    cfg.object_placements = tuple(NS(position_offset=(i,0,.05), quat=(1,0,0,0)) for i in range(100))
    cfg.events.reset_object.params = {'per_env_positions': [[0,0,.05]], 'per_env_quats': [[1,0,0,0]]}
    configure_replay_object_bank(cfg, [{'object_asset': {'usd_path': 'object_73.usd', 'object_id': 'cup/001'}}])
    assert cfg.scene.num_envs == len(cfg.scene.object.spawn.assets_cfg) == 1
    assert cfg.events.reset_object.params['per_env_positions'] == [[73,0,.05]]
