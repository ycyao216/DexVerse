# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for coordinated door opening, without starting Isaac Sim."""

import __future__
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MDP = "source/dexverse/dexverse/tasks/mdp"


def _load_definitions(relative_path, names, namespace):
    """Execute the production definitions while avoiding simulator imports."""
    path = ROOT / relative_path
    nodes = [
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env

    def reset(self, env_ids=None):
        pass


@pytest.fixture
def term_class():
    namespace = {
        "torch": torch,
        "ManagerTermBase": _ManagerTermBase,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name, joint_ids=slice(None), joint_names=None),
    }
    _load_definitions(f"{MDP}/utils.py", ["resolve_joint_ids"], namespace)
    _load_definitions(f"{MDP}/terminations.py", ["_compare", "joint_co_completion"], namespace)
    return namespace["joint_co_completion"]


def _env(num_envs=1, device="cpu"):
    data = SimpleNamespace(
        joint_pos=torch.zeros(num_envs, 2),
        joint_pos_limits=torch.tensor([[[0.0, math.pi / 2]] * 2] * num_envs),
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device=device,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        scene={"articulation": SimpleNamespace(data=data)},
    )


def _pose(env, step, angles):
    env.episode_length_buf[:] = step
    env.scene["articulation"].data.joint_pos[:] = torch.deg2rad(torch.tensor(angles, dtype=torch.float32))


def _sample(term, env, step, angles):
    _pose(env, step, angles)
    return term(env, threshold=0.8, mode="ratio", window_steps=60).tolist()


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
@pytest.mark.parametrize("gap,expected", [(0, True), (30, True), (60, True), (61, False), (180, False)])
def test_crossing_gap_and_held_doors(term_class, device, gap, expected):
    env = _env(device=device)
    term = term_class(None, env)
    assert _sample(term, env, 10, [75, 10]) == [False]
    assert _sample(term, env, 10 + gap, [75, 75]) == [expected]
    # Waiting with both doors open must not make an overdue crossing succeed.
    assert _sample(term, env, 200 + gap, [75, 75]) == [expected]


def test_both_must_remain_open_and_can_retry_after_closing(term_class):
    env = _env()
    term = term_class(None, env)
    assert _sample(term, env, 10, [75, 75]) == [True]
    assert _sample(term, env, 11, [10, 75]) == [False]
    assert _sample(term, env, 100, [10, 10]) == [False]
    assert _sample(term, env, 101, [75, 10]) == [False]
    assert _sample(term, env, 130, [75, 75]) == [True]


def test_recorder_episode_counter_reset_clears_history(term_class):
    env = _env()
    term = term_class(None, env)
    assert _sample(term, env, 100, [75, 10]) == [False]
    assert _sample(term, env, 200, [75, 75]) == [False]
    assert _sample(term, env, 1, [75, 10]) == [False]
    assert _sample(term, env, 31, [75, 75]) == [True]


def test_manager_reset_is_scoped_to_selected_environments(term_class):
    env = _env(num_envs=2)
    term = term_class(None, env)
    term.reset()  # Safe before the first check.
    assert _sample(term, env, 10, [[75, 10], [75, 10]]) == [False, False]
    term.reset(torch.tensor([0]))
    # Explicit reset must work even if a new episode already caught up to the
    # previous step count before this term is checked again.
    assert _sample(term, env, 100, [[75, 75], [75, 75]]) == [True, False]
    term.reset()
    assert _sample(term, env, 101, [[75, 75], [75, 75]]) == [True, True]


@pytest.mark.parametrize("gap,expected", [(30, True), (120, False)])
def test_recorder_consecutive_success_counter(term_class, gap, expected):
    env = _env()
    term = term_class(None, env)
    cfg = SimpleNamespace(params={"threshold": 0.8, "mode": "ratio", "window_steps": 60})
    namespace = {"_resolve_term_callable": lambda env, cfg: term, "args_cli": SimpleNamespace(num_success_steps=10)}
    _load_definitions("scripts/record_demos.py", ["check_success"], namespace)
    check_success = namespace["check_success"]
    _pose(env, 10, [75, 10])
    count, complete = check_success(env, cfg, 0)
    assert (count, complete) == (0, False)
    for offset in range(10):
        _pose(env, 10 + gap + offset, [75, 75])
        count, complete = check_success(env, cfg, count)
        assert complete is (expected and offset == 9)
    assert count == (10 if expected else 0)
