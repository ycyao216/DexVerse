"""Simulator-free regressions for launcher seed plumbing and recording metadata."""

from __future__ import annotations

import __future__
import argparse
import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))
from dexverse.benchmark import task_identity  # noqa: E402
from dexverse.teleop_utils.debug_visualization import configure_v1_debug_visualization  # noqa: E402
from dexverse.teleop_utils.seeding import resolve_seed  # noqa: E402
from dexverse.teleop_utils import seeding  # noqa: E402


def tree(script):
    return ast.parse((ROOT / "scripts" / script).read_text())


def test_replay_seed_applied_after_robot_rebuild_before_make():
    tree = ast.parse((ROOT / "scripts/demo_tools/create_demo_files_sequential.py").read_text())
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_env_for_pickle")
    assignments = [n for n in ast.walk(build) if isinstance(n, ast.Assign)]
    rebuild = next(n for n in assignments if ast.unparse(n).startswith("env_cfg = cfg_cls("))
    seed = next(n for n in assignments if ast.unparse(n) == "env_cfg.seed = args_cli.seed")
    make = next(n for n in assignments if ast.unparse(n).startswith("env = gym.make("))
    assert rebuild.lineno < seed.lineno < make.lineno


def load_node(node, namespace):
    # Avoid importing launch scripts, which would start Isaac Sim / XR.
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<seed-test>", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


@pytest.mark.parametrize("script,default", [
    ("record_demos.py", None), ("teleop_agent.py", None), ("zero_agent.py", None),
    ("env_tools/check_task_upgrades.py", 42),
])
def test_seed_cli_accepts_fixed_and_omitted_seed(script, default):
    parser = argparse.ArgumentParser()
    node = next(node for node in tree(script).body if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call) and node.value.args
                and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == "--seed")
    load_node(node, {"parser": parser})
    assert parser.parse_args([]).seed == default
    assert parser.parse_args(["--seed", "4200"]).seed == 4200


@pytest.mark.parametrize("seed,expected", [(1000, 1000), (-1, 73129), (None, 42)])
def test_collection_seed_survives_robot_config_rebuild(seed, expected, monkeypatch):
    monkeypatch.setattr(seeding.secrets, "randbelow", lambda bound: 73129)
    class Config:
        def __init__(self, **kwargs):
            self.seed = 42
            self.robot_type = kwargs.get("robot_type", "old_robot")
            self.sim = SimpleNamespace(device="cpu")
            self.scene = SimpleNamespace(num_envs=5)
            self.terminations = SimpleNamespace(success=object(), time_out=object())
            self.observations = SimpleNamespace(policy=SimpleNamespace(concatenate_terms=True))

    args = SimpleNamespace(task="Dexverse-OpenDoor-v1", device="cuda:0", json_path=None,
                           robot_type="floating_shadow_right", usd_path=None,
                           enable_debug_vis=None, seed=seed, xr=False, cues_in_rgb=False,
                           collection_object_id=None)
    node = next(node for node in tree("record_demos.py").body
                if isinstance(node, ast.FunctionDef) and node.name == "create_environment_config")
    namespace = {"args_cli": args, "parse_env_cfg": lambda *a, **k: Config(),
                 "resolve_seed": resolve_seed,
                 "configure_v1_debug_visualization": configure_v1_debug_visualization}
    load_node(node, namespace)
    cfg, success = namespace[node.name]()
    assert cfg.seed == expected
    assert cfg.robot_type == "floating_shadow_right" and cfg.scene.num_envs == 1
    assert success is not None and cfg.terminations.success is None


def test_random_seed_is_nonnegative_and_each_request_draws_again(monkeypatch):
    draws = iter([0, 2**31 - 1])
    monkeypatch.setattr(seeding.secrets, "randbelow", lambda bound: next(draws))
    assert resolve_seed(0) == 0
    assert resolve_seed(42) == 42
    assert resolve_seed(-1) == 0
    assert resolve_seed(-1) == 2**31 - 1
    with pytest.raises(ValueError, match="nonnegative"):
        resolve_seed(-2)


def test_practice_resolves_cli_and_config_sentinels_before_make(monkeypatch):
    monkeypatch.setattr(seeding.secrets, "randbelow", lambda bound: 73129)
    main = next(n for n in tree("teleop_agent.py").body if isinstance(n, ast.FunctionDef) and n.name == "main")
    seed_if = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args_cli.seed is not None")
    resolve = next(n for n in main.body if isinstance(n, ast.Assign)
                   and ast.unparse(n) == "env_cfg.seed = resolve_seed(env_cfg.seed)")
    make = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) == "gym.make")
    assert seed_if.lineno < resolve.lineno < make.lineno
    for requested, initial, expected in [(-1, 42, 73129), (None, -1, 73129), (0, 42, 0), (None, 42, 42)]:
        cfg = SimpleNamespace(seed=initial)
        namespace = {"args_cli": SimpleNamespace(seed=requested), "env_cfg": cfg, "resolve_seed": resolve_seed}
        load_node(seed_if, namespace)
        load_node(resolve, namespace)
        assert cfg.seed == expected


@pytest.mark.parametrize("script", ["zero_agent.py", "teleop_agent.py"])
def test_preview_seed_is_applied_after_overrides_and_before_gym_make(script):
    main = next(node for node in tree(script).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    seed_write = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                      and ast.unparse(node).startswith("env_cfg.seed = args_cli.seed"))
    override = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "_apply_hydra_env_overrides")
    spawn = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "gym.make")
    assert override.lineno < seed_write.lineno < spawn.lineno


@pytest.mark.parametrize("device", ["cpu", "cuda:0", None])
def test_recorder_keeps_resolved_seed_and_device_in_session_metadata(tmp_path, device):
    node = next(node for node in tree("record_demos.py").body
                if isinstance(node, ast.ClassDef) and node.name == "TrajectoryPickleRecorder")
    namespace = {"os": os, "task_identity": task_identity}
    load_node(node, namespace)
    recorder = namespace[node.name](str(tmp_path / "seeded.pkl"), task_name="Dexverse-OpenDoor-v1",
                                    env_name="Dexverse-OpenDoor-v1", seed=1000, sim_device=device)
    assert recorder._metadata["seed"] == 1000
    assert recorder._metadata["sim_device"] == device
    assert recorder._metadata["benchmark_revision"] == task_identity("Dexverse-OpenDoor-v1")["benchmark_revision"]


def test_collection_defaults_to_cpu_and_records_actual_environment_device():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rendering_mode", default="balanced")
    module = tree("record_demos.py")
    defaults = next(node for node in module.body if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "parser.set_defaults")
    load_node(defaults, {"parser": parser})
    assert parser.parse_args([]).device == "cpu"
    assert parser.parse_args([]).rendering_mode == "performance"
    assert parser.parse_args(["--device", "cuda:0"]).device == "cuda:0"
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    recorder = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "TrajectoryPickleRecorder")
    assert any(kw.arg == "sim_device" and ast.unparse(kw.value) == "str(env.device)" for kw in recorder.keywords)


def test_repeatability_comparison_rejects_different_reset_values():
    node = next(node for node in tree("env_tools/check_task_upgrades.py").body
                if isinstance(node, ast.FunctionDef) and node.name == "compare_state")
    namespace = {"torch": torch}
    load_node(node, namespace)
    compare = namespace[node.name]
    value = {"object": {"root_pose": torch.zeros((2, 7))}}
    compare(value, {"object": {"root_pose": torch.zeros((2, 7))}})
    with pytest.raises(AssertionError):
        compare(value, {"object": {"root_pose": torch.ones((2, 7))}})
