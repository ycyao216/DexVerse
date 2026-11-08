# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for replay-critical data and local procedural assets."""

from __future__ import annotations

import ast
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "dexverse"))

from dexverse.benchmark import (  # noqa: E402
    BASELINE_TASKS,
    BASELINE_V1_TASKS,
    BASELINE_PAIRS,
    V1_CONFIGS,
    LEGACY_UPGRADE_TASKS,
    BENCHMARK_REVISION,
    LEGACY_UPGRADE_REVISION,
    LONG_HORIZON_SIM_FIX_REVISIONS,
    V0_BENCHMARK_REVISION,
    baseline_tasks,
    task_identity,
    validate_replay_identity,
)
from dexverse.teleop_utils.episode_task_state import (  # noqa: E402
    capture_episode_task_state,
    restore_episode_task_state,
    reset_managers_after_state_restore,
)


def _script(name):
    path = ROOT / "scripts" / "demo_tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_replay_updates_last_action_without_processing_actions():
    # Importing the replay entry point launches Isaac Sim. Compile just this
    # pure buffer helper so the regression can run in the shared CPU environment.
    path = ROOT / "scripts" / "demo_tools" / "create_demo_files_sequential.py"
    tree = ast.parse(path.read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_set_last_action")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    update = namespace["_set_last_action"]
    manager = SimpleNamespace(device="cpu", _action=torch.zeros(1, 3), _prev_action=torch.zeros(1, 3))
    env = SimpleNamespace(action_manager=manager)
    current_buffer, previous_buffer = manager._action, manager._prev_action
    first = torch.tensor([[0.2, -0.4, 0.6]])
    second = torch.tensor([[-0.1, 0.3, -0.5]])
    update(env, first)
    torch.testing.assert_close(manager._action, first)
    torch.testing.assert_close(manager._prev_action, torch.zeros_like(first))
    update(env, second)
    torch.testing.assert_close(manager._action, second)
    torch.testing.assert_close(manager._prev_action, first)
    assert manager._action is current_buffer and manager._prev_action is previous_buffer
    second.zero_()
    torch.testing.assert_close(manager._action, torch.tensor([[-0.1, 0.3, -0.5]]))
    update(SimpleNamespace(), first)


@pytest.mark.parametrize(
    "needs_cameras,strip_cameras,expected", [(True, True, True), (False, False, True), (False, True, False)]
)
def test_replay_enables_camera_support_when_sensors_are_retained(needs_cameras, strip_cameras, expected):
    path = ROOT / "scripts" / "demo_tools" / "create_demo_files_sequential.py"
    tree = ast.parse(path.read_text())
    condition = next(
        node for node in tree.body if isinstance(node, ast.If) and "_NEEDS_CAMERAS" in ast.unparse(node.test)
    )
    args = SimpleNamespace(strip_cameras=strip_cameras, enable_cameras=False)
    namespace = {"_NEEDS_CAMERAS": needs_cameras, "args_cli": args}
    exec(compile(ast.Module(body=[condition], type_ignores=[]), str(path), "exec"), namespace)
    assert args.enable_cameras is expected


def test_task_state_restores_goals_and_categorical_choice_without_aliasing():
    command = SimpleNamespace(
        pose_command_b=torch.arange(14.0).reshape(2, 7),
        pose_command_w=torch.arange(14.0).reshape(2, 7).clone(),
        cfg=SimpleNamespace(use_world_frame=True),
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=2,
        cfg=SimpleNamespace(events=SimpleNamespace(face=SimpleNamespace(params={"buffer_name": "face"}))),
        face=torch.tensor([2, 5]),
        command_manager=SimpleNamespace(active_terms=["object_pose"], get_term=lambda _: command),
    )
    snapshot = capture_episode_task_state(env, env_index=1)
    env.face[1] = 0
    command.pose_command_b[1].zero_()
    command.pose_command_w[1].zero_()
    assert snapshot["env_buffers"]["face"].item() == 5
    assert restore_episode_task_state(env, snapshot, env_index=1) == []
    assert env.face.tolist() == [2, 5]
    torch.testing.assert_close(command.pose_command_b[1], torch.arange(7.0, 14.0))
    torch.testing.assert_close(command.pose_command_w[1], command.pose_command_b[1])
    torch.testing.assert_close(command.pose_command_b[0], torch.arange(7.0))


def test_legacy_goal_and_missing_task_terms():
    command = SimpleNamespace(
        pose_command_b=torch.zeros(1, 7), pose_command_w=torch.zeros(1, 7), cfg=SimpleNamespace(use_world_frame=True)
    )

    def get_term(name):
        if name != "object_pose":
            raise KeyError(name)
        return command

    env = SimpleNamespace(device="cpu", num_envs=1, command_manager=SimpleNamespace(get_term=get_term))
    goal = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    assert restore_episode_task_state(env, None, legacy_goal_pose=goal) == []
    torch.testing.assert_close(command.pose_command_w[0], torch.tensor(goal))
    assert restore_episode_task_state(env, {"commands": {"retired": {"command": [1.0]}}}) == ["command:retired"]


def _session(revision):
    return {
        "format": "dexverse_trajectory",
        "schema_version": 5,
        "task": "Dexverse-PushT-v1",
        "env_name": "Dexverse-PushT-v1",
        "robot_type": "floating_shadow_right",
        "benchmark_revision": revision,
        "task_version": 1,
        "task_source_revision": "4f7262f",
        "action_layout": {"dimension": 2},
        "seed": 42,
        "sim_device": "cpu",
        "episodes": [
            {
                "episode_index": 0,
                "actions": np.ones((2, 2), dtype=np.float32),
                "initial_state": {},
                "task_state": {"env_buffers": {"face": 4}},
            }
        ],
    }


@pytest.mark.parametrize("mismatch,value", [("benchmark_revision", None), ("action_layout", None),
                                            ("sim_device", None), ("sim_device", "cuda:0")])
def test_mergers_reject_incompatible_recording_identity(tmp_path, mismatch, value):
    a = _session(BENCHMARK_REVISION)
    b = _session(BENCHMARK_REVISION)
    b[mismatch] = value
    paths = [tmp_path / "a.pkl", tmp_path / "b.pkl"]
    for path, payload in zip(paths, [a, b]):
        with path.open("wb") as stream:
            pickle.dump(payload, stream)
    merger = _script("merge_demos")
    with pytest.raises(ValueError, match="mismatch"):
        merger.merge_trajectory_pickles([str(p) for p in paths], str(tmp_path / "batch.pkl"))
    sessions = _script("merge_session_pickles")
    with pytest.raises(ValueError):
        sessions._merge_one_task(tmp_path, paths, "merged.pkl", False)
    assert not (tmp_path / "batch.pkl").exists()
    assert not (tmp_path / "merged.pkl").exists()
    tree = ast.parse((ROOT / "scripts/demo_tools/create_demo_files_sequential.py").read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_merge_trajectory_payloads")
    namespace = {"Path": Path, "_load_trajectory_pickle": lambda path: a if Path(path) == paths[0] else b}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<replay-merge-test>", "exec"), namespace)
    with pytest.raises(ValueError, match="mismatch"):
        namespace[helper.name](paths)


def test_merge_preserves_task_state_and_revision(tmp_path):
    paths = [tmp_path / "a.pkl", tmp_path / "b.pkl"]
    for path in paths:
        with path.open("wb") as stream:
            pickle.dump(_session(BENCHMARK_REVISION), stream)
    batch = _script("merge_demos").merge_trajectory_pickles([str(p) for p in paths], str(tmp_path / "batch.pkl"))
    assert batch["benchmark_revision"] == BENCHMARK_REVISION
    assert batch["sim_device"] == "cpu"
    assert batch["task_version"] == 1 and batch["task_source_revision"] == "4f7262f"
    assert batch["action_layout"] == {"dimension": 2}
    assert [state["env_buffers"]["face"] for state in batch["task_states"]] == [4, 4]
    output, count, status = _script("merge_session_pickles")._merge_one_task(tmp_path, paths, "merged.pkl", False)
    assert (count, status) == (2, "written")
    with output.open("rb") as stream:
        merged = pickle.load(stream)
    assert merged["benchmark_revision"] == BENCHMARK_REVISION
    assert merged["sim_device"] == "cpu"
    assert merged["task_version"] == 1
    assert merged["episodes"][1]["task_state"]["env_buffers"]["face"] == 4


def test_baseline_manifest_agrees_with_catalog():
    path = ROOT / "source" / "dexverse" / "demonstrations" / "baseline_manifest.txt"
    tasks = [line.split("/")[-1] for line in path.read_text().splitlines() if line and not line.startswith("#")]
    assert tasks == list(BASELINE_TASKS)
    assert len(tasks) == len(set(tasks)) == 20
    path = path.with_name("baseline_manifest_v1.txt")
    tasks = [line.split("/")[-1] for line in path.read_text().splitlines() if line and not line.startswith("#")]
    assert tasks == list(BASELINE_V1_TASKS)
    assert len(BASELINE_PAIRS) == len(V1_CONFIGS) == 20
    assert set(baseline_tasks("all")) == set(BASELINE_TASKS + BASELINE_V1_TASKS)
    assert all(a.removesuffix("-v0") == b.removesuffix("-v1") for a, b in BASELINE_PAIRS.items())
    for module, class_name in V1_CONFIGS.values():
        path = ROOT / "source/dexverse/dexverse/baseline_v1/config" / (module.replace(".", "/") + ".py")
        assert class_name in {n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}


def test_original_task_and_robot_sources_are_preserved():
    # The preservation contract is the official source snapshot, not the dirty
    # main worktree or private recordings labelled v0. Skip outside a git checkout.
    if subprocess.run(["git", "cat-file", "-e", "30cc673^{commit}"], cwd=ROOT, capture_output=True).returncode:
        pytest.skip("Official preservation commit is not available")
    paths = (
        subprocess.check_output(
            [
                "git",
                "ls-tree",
                "-r",
                "--name-only",
                "30cc673",
                "source/dexverse/dexverse/tasks",
                "source/dexverse/dexverse/robot_agents/shadow",
                "source/dexverse/dexverse/visual_purpose.py",
            ],
            cwd=ROOT,
        )
        .decode()
        .splitlines()
    )
    # Corrected in place, and labelled from LONG_HORIZON_SIM_FIX_REVISIONS by task_identity().
    corrected_in_place = {
        "source/dexverse/dexverse/tasks/config/long_horizon/make_coffee/long_horizon_make_coffee_cfg.py",
        "source/dexverse/dexverse/tasks/config/long_horizon/cook_foods/long_horizon_cook_foods_cfg.py",
        "source/dexverse/dexverse/tasks/config/long_horizon/microwave_retrieve_place/"
        "long_horizon_microwave_retrieve_place_cfg.py",
    }
    for path in paths:
        if path in corrected_in_place:
            continue
        original = subprocess.check_output(["git", "show", f"30cc673:{path}"], cwd=ROOT)
        if path == "source/dexverse/dexverse/tasks/__init__.py":
            # The sole discovery change imports the sibling upgrade package;
            # retain every byte of the original task-registration implementation.
            original += (
                b"\n# Keep the established task-discovery entry point registering both versions.\n"
                b"# Only baseline upgrades live in this sibling package; original tasks stay here.\n"
                b"from dexverse import baseline_v1  # noqa: E402, F401\n"
            )
        assert (ROOT / path).read_bytes() == original, path


@pytest.mark.parametrize("version,revision", [(0, V0_BENCHMARK_REVISION), (1, BENCHMARK_REVISION)])
def test_task_metadata_uses_selected_version(version, revision):
    task = f"Dexverse-OpenLaptop-v{version}"
    identity = task_identity(task)
    assert identity["task_version"] == version
    assert identity["benchmark_revision"] == revision
    validate_replay_identity({"task": task, **identity}, task)
    assert task_identity("Dexverse-OpenDrawer-v0")["task_version"] == 0


def test_legacy_upgrade_recordings_require_explicit_mapping():
    task = "Dexverse-OpenLaptop-v0"
    payload = {"task": task, "benchmark_revision": LEGACY_UPGRADE_REVISION}
    with pytest.raises(ValueError, match="provenance"):
        validate_replay_identity(payload, task)
    validate_replay_identity(payload, "Dexverse-OpenLaptop-v1", explicit_override=True)
    # Unlabelled old official recordings retain v0 compatibility.
    validate_replay_identity({"task": task}, task)
    for old, new in LEGACY_UPGRADE_TASKS.items():
        with pytest.raises(ValueError, match="provenance"):
            validate_replay_identity({"task": old}, old)
        validate_replay_identity({"task": old}, new, explicit_override=True)


def test_restoring_initial_laptop_pose_recaptures_only_selected_references():
    path = ROOT / "source/dexverse/dexverse/baseline_v1/mdp/terminations.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "root_pose_displacement_exceeds"
    )

    class ManagerBase:
        def __init__(self, cfg, env):
            self.cfg = cfg

    namespace = {
        "torch": torch,
        "ManagerTermBase": ManagerBase,
        "ManagerBasedRLEnv": object,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3)))
    env = SimpleNamespace(scene={"articulation": asset})
    term = namespace["root_pose_displacement_exceeds"](SimpleNamespace(params={}), env)
    term.reset()
    asset.data.root_pos_w[0, 0] = 0.10
    assert term(env).tolist() == [True, False]
    env.termination_manager = SimpleNamespace(active_terms=["base"], get_term_cfg=lambda _: SimpleNamespace(func=term))
    reset_managers_after_state_restore(env, env_ids=torch.tensor([0]))
    assert term(env).tolist() == [False, False]
    asset.data.root_pos_w[0, 2] = -0.02  # downward settling is not lifting
    assert not term(env, max_upward_z=0.005).any()
    asset.data.root_pos_w[1, 2] = 0.006
    assert term(env, max_upward_z=0.005).tolist() == [False, True]


def test_procedural_assets_build_and_reuse_matching_parameters(tmp_path):
    from pxr import Usd
    from dexverse.assets.rack_builder import ensure_rack_asset
    from dexverse.assets.shelf_builder import ensure_slot_shelf_asset
    from dexverse.assets.small_table_builder import ensure_small_table_asset
    from dexverse.assets.cut_props_builder import ensure_taped_seam_workpiece_asset, ensure_strip_stand_asset
    from dexverse.assets.desk_props_builder import ensure_dispenser_asset

    for index, builder in enumerate(
        (
            ensure_rack_asset,
            ensure_slot_shelf_asset,
            ensure_small_table_asset,
            ensure_taped_seam_workpiece_asset,
            ensure_strip_stand_asset,
            ensure_dispenser_asset,
        )
    ):
        directory = tmp_path / str(index)
        path, manifest = builder(out_dir=directory)
        stage = Usd.Stage.Open(str(path))
        assert stage and stage.GetDefaultPrim()
        before = path.stat().st_mtime_ns
        same_path, same_manifest = builder(out_dir=directory)
        assert same_path == path and same_manifest == manifest
        assert path.stat().st_mtime_ns == before


def test_v1_dispenser_grey_materials_preserve_green_target_and_geometry(tmp_path):
    from pxr import Usd, UsdShade
    from dexverse.assets.desk_props_builder import ensure_dispenser_asset

    config = ROOT / "source/dexverse/dexverse/baseline_v1/config/functional/place_cup_under_dispenser_cfg.py"
    tree = ast.parse(config.read_text())
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DISPENSER_PARAMS" for target in node.targets)
    )
    params = {keyword.arg: ast.literal_eval(keyword.value) for keyword in assignment.value.keywords}
    geometry = {key: value for key, value in params.items() if not key.endswith("_color")}
    _, original = ensure_dispenser_asset(out_dir=tmp_path, **geometry)
    path, updated = ensure_dispenser_asset(out_dir=tmp_path, **params)
    assert {key: value for key, value in original.items() if key != "params"} == {
        key: value for key, value in updated.items() if key != "params"
    }
    stage = Usd.Stage.Open(str(path))
    for material, expected in (
        ("BodyMat", (0.55, 0.55, 0.55)),
        ("NozzleMat", (0.35, 0.35, 0.35)),
        ("TargetMat", (0.15, 0.75, 0.25)),
    ):
        shader = UsdShade.Shader.Get(stage, f"/dispenser/Looks/{material}/PreviewSurface")
        assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx(expected)


@pytest.mark.parametrize("task", sorted(LONG_HORIZON_SIM_FIX_REVISIONS))
def test_fixed_long_horizon_tasks_have_their_own_revision(task):
    identity = task_identity(task)
    assert identity["task_version"] == 0
    assert identity["benchmark_revision"] == LONG_HORIZON_SIM_FIX_REVISIONS[task]
    validate_replay_identity({"task": task, **identity}, task)


def test_long_horizon_revisions_are_per_task():
    assert len(set(LONG_HORIZON_SIM_FIX_REVISIONS.values())) == len(LONG_HORIZON_SIM_FIX_REVISIONS)


def test_recordings_from_before_the_long_horizon_fix_need_an_explicit_override():
    task = "Dexverse-LongHorizon-MakeCoffee-v0"
    old = {"task": task, "benchmark_revision": V0_BENCHMARK_REVISION}
    with pytest.raises(ValueError, match="does not match"):
        validate_replay_identity(old, task)
    validate_replay_identity(old, task, explicit_override=True)


def test_unfixed_long_horizon_tasks_keep_the_official_revision():
    assert task_identity("Dexverse-LongHorizon-OvenBakeSalmon-v0")["benchmark_revision"] == V0_BENCHMARK_REVISION
