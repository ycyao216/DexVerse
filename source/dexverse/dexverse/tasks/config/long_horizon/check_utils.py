# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for long-horizon success checks.

Free of Isaac Lab imports, so the checks built on them can run in CPU tests.
"""

from __future__ import annotations

import torch

# How often the operator is told what the current stage is still waiting for.
STATUS_PERIOD_S = 3.0

_MEMORY_ATTR = "_long_horizon_check_memory"


def rotate(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by (w, x, y, z) quaternions, row by row."""
    w, xyz = quat_wxyz[:, :1], quat_wxyz[:, 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


def rotate_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the inverse of (w, x, y, z) unit quaternions, row by row."""
    conjugate = torch.cat([quat_wxyz[:, :1], -quat_wxyz[:, 1:]], dim=-1)
    return rotate(conjugate, vec)


def local_point(pos: torch.Tensor, quat_wxyz: torch.Tensor, offset: tuple[float, float, float]) -> torch.Tensor:
    """World position of a point fixed in a body's frame."""
    local = torch.as_tensor(offset, device=pos.device, dtype=pos.dtype).expand(pos.shape[0], -1)
    return pos + rotate(quat_wxyz, local)


def axis_tilt(quat_wxyz: torch.Tensor, axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> torch.Tensor:
    """Angle between a body's local axis and world +Z, in radians."""
    axis = torch.as_tensor(axis_local, device=quat_wxyz.device, dtype=quat_wxyz.dtype).expand(quat_wxyz.shape[0], -1)
    return torch.acos(rotate(quat_wxyz, axis)[:, 2].clamp(-1.0, 1.0))


def check_memory(env) -> dict:
    """Per-env-object storage the long-horizon checks keep between steps."""
    return env.__dict__.setdefault(_MEMORY_ATTR, {})


def track(env, key: str, condition: torch.Tensor, hold_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow ``condition`` over env steps; return (held now, has ever been held).

    Advances once per env step however often it is called, and restarts when the
    episode does.
    """
    memory = check_memory(env)
    steps = env.episode_length_buf
    count, seen, reached = memory.get(
        key, (torch.zeros_like(steps), torch.full_like(steps, -1), torch.zeros_like(condition))
    )
    restarted = steps < seen
    count = torch.where(restarted, torch.zeros_like(count), count)
    reached = reached & ~restarted
    count = torch.where(steps != seen, torch.where(condition, count + 1, torch.zeros_like(count)), count)
    held = count * env.step_dt >= hold_s - 1.0e-6
    reached = reached | held
    memory[key] = (count, steps.clone(), reached)
    return held, reached


def ever_held(env, key: str, like: torch.Tensor) -> torch.Tensor:
    """Whether ``track`` has seen ``key`` held this episode; all False if never tracked."""
    entry = env.__dict__.get(_MEMORY_ATTR, {}).get(key)
    return torch.zeros_like(like, dtype=torch.bool) if entry is None else entry[2]


def announce_once(env, task_label: str, key: str, flag: torch.Tensor, message: str) -> None:
    """Tell the teleoperator, once per episode, when ``flag`` first turns true."""
    if env.num_envs != 1:
        return
    memory = check_memory(env)
    announced, seen = memory.get(f"announce_{key}", (False, -1))
    step = int(env.episode_length_buf[0])
    if step < seen:
        announced = False
    if bool(flag[0]) and not announced:
        print(f"[{task_label}] {message}")
        announced = True
    memory[f"announce_{key}"] = (announced, step)


def print_status(env, task_label: str, missing: list[str]) -> None:
    """Print what the current stage is missing, at most once per ``STATUS_PERIOD_S``."""
    memory = check_memory(env)
    step = int(env.episode_length_buf[0])
    last = memory.get("status_step")
    if not missing or (last is not None and 0 <= step - last < STATUS_PERIOD_S / env.step_dt):
        return
    memory["status_step"] = step
    print(f"[{task_label}] Waiting: {'; '.join(missing)}.")


def clear_stage_latches(env, env_ids, task_key: str) -> None:
    """Reset event: forget which stages of a task's stage graph have latched.

    The stage machine only clears latches if it happens to be evaluated on the exact
    step an episode restarts, which needs an observation of the stage graph. Without
    one, a previous attempt's stages carry over into the next.
    """
    state = getattr(env, "_stage_graph_runtime_cache", {}).get(task_key)
    if state is None or state.latched_flags is None:
        return
    cleared = {}
    for name, flag in state.latched_flags.items():
        flag = flag.clone()
        flag[env_ids] = False
        cleared[name] = flag
    state.latched_flags = cleared
    state.raw_flags = None
    state.raw_step_buf = None
    state.eval_flags = None
    state.eval_step_buf = None

