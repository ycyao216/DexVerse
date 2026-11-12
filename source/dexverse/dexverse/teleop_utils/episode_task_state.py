# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture and restore episode state that is not owned by the scene.

``InteractiveScene.get_state()`` covers articulations and rigid objects, but not
command-manager buffers or task-specific tensors created by reset events.  Goal
markers are often driven from exactly those values, so replaying only the scene
state can produce a different goal (and different goal visuals) from recording.
"""

from __future__ import annotations

from typing import Any

import torch


_COMMAND_BUFFER_NAMES = ("pose_command_b", "pose_command_w")


def _row_copy(value: Any, env_index: int) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] <= env_index:
        return None
    return value[env_index].detach().clone()


def _configured_env_buffer_names(env) -> set[str]:
    """Find reset-event buffers declared through a ``buffer_name`` parameter."""
    names: set[str] = set()
    events_cfg = getattr(getattr(env, "cfg", None), "events", None)
    if events_cfg is None:
        return names
    for term_name in dir(events_cfg):
        if term_name.startswith("_"):
            continue
        try:
            term_cfg = getattr(events_cfg, term_name)
        except Exception:
            continue
        params = getattr(term_cfg, "params", None)
        if isinstance(params, dict) and isinstance(params.get("buffer_name"), str):
            names.add(params["buffer_name"])
    return names


def capture_episode_task_state(env, env_index: int = 0, *, relative_to_env_origin: bool = False) -> dict[str, Any]:
    """Return replay-critical command and task-buffer state for one environment."""
    state: dict[str, Any] = {}

    command_manager = getattr(env, "command_manager", None)
    commands: dict[str, dict[str, torch.Tensor]] = {}
    if command_manager is not None:
        for name in getattr(command_manager, "active_terms", ()):
            try:
                term = command_manager.get_term(name)
            except (AttributeError, KeyError):
                continue
            buffers: dict[str, torch.Tensor] = {}
            for buffer_name in _COMMAND_BUFFER_NAMES:
                row = _row_copy(getattr(term, buffer_name, None), env_index)
                if row is not None:
                    if relative_to_env_origin and (
                        buffer_name == "pose_command_w" or getattr(term.cfg, "use_world_frame", False)
                    ):
                        row[:3] -= env.scene.env_origins[env_index]
                    buffers[buffer_name] = row
            # Preserve non-pose command terms too. Pose terms already expose the
            # writable source buffer above, so duplicating ``command`` is needless.
            if not buffers:
                row = _row_copy(getattr(term, "command", None), env_index)
                if row is not None:
                    buffers["command"] = row
            if buffers:
                commands[str(name)] = buffers
    if commands:
        state["commands"] = commands
        if relative_to_env_origin:
            state["world_pose_frame"] = "env"

    env_buffers: dict[str, torch.Tensor] = {}
    for name in sorted(_configured_env_buffer_names(env)):
        row = _row_copy(getattr(env, name, None), env_index)
        if row is not None:
            env_buffers[name] = row
    if env_buffers:
        state["env_buffers"] = env_buffers

    capture = getattr(getattr(env, "cfg", None), "capture_task_state", None)
    if callable(capture):
        state["task_runtime"] = capture(env, env_index)

    return state


def _restore_tensor_row(target: torch.Tensor, value: Any, env_index: int) -> None:
    source = torch.as_tensor(value, device=target.device, dtype=target.dtype)
    target[env_index].copy_(source.reshape(target[env_index].shape))


def reset_managers_after_state_restore(env, env_ids=None) -> None:
    """Rebase explicitly opted-in managers after restoring an episode's initial scene.

    Call once at the start of an episode, never for intermediate replay frames.
    Ordinary managers and v0 predicates are left untouched.
    """
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        return
    for name in manager.active_terms:
        callback = getattr(manager.get_term_cfg(name).func, "reset_after_state_restore", None)
        if callback is not None:
            callback(env_ids=env_ids)


def restore_episode_task_state(
    env,
    state: Any,
    env_index: int = 0,
    legacy_goal_pose: Any = None,
) -> list[str]:
    """Restore a snapshot from :func:`capture_episode_task_state`.

    Missing terms/buffers are skipped and reported so recordings remain usable
    after benign task configuration changes.
    """
    if not isinstance(state, dict):
        state = {}
    skipped: list[str] = []

    command_manager = getattr(env, "command_manager", None)
    commands = state.get("commands") or {}
    if not commands and legacy_goal_pose is not None:
        # Schema <= 3 recorded only object_pose and replay never applied it.
        commands = {"object_pose": {"pose_command_b": legacy_goal_pose}}
    for name, buffers in commands.items():
        if command_manager is None or not isinstance(buffers, dict):
            skipped.append(f"command:{name}")
            continue
        try:
            term = command_manager.get_term(name)
        except (AttributeError, KeyError):
            skipped.append(f"command:{name}")
            continue
        for buffer_name, value in buffers.items():
            target = getattr(term, buffer_name, None)
            if not isinstance(target, torch.Tensor) or target.shape[0] <= env_index:
                skipped.append(f"command:{name}.{buffer_name}")
                continue
            if state.get("world_pose_frame") == "env" and (
                buffer_name == "pose_command_w"
                or (buffer_name == "pose_command_b" and getattr(term.cfg, "use_world_frame", False))
            ):
                value = torch.as_tensor(value, device=target.device, dtype=target.dtype).clone()
                value[:3] += env.scene.env_origins[env_index]
            _restore_tensor_row(target, value, env_index)
            if buffer_name == "pose_command_b" and getattr(getattr(term, "cfg", None), "use_world_frame", False):
                pose_w = getattr(term, "pose_command_w", None)
                if isinstance(pose_w, torch.Tensor) and pose_w.shape[0] > env_index:
                    _restore_tensor_row(pose_w, value, env_index)

    for name, value in (state.get("env_buffers") or {}).items():
        target = getattr(env, name, None)
        if not isinstance(target, torch.Tensor) or target.shape[0] <= env_index:
            source = torch.as_tensor(value, device=env.device)
            target = torch.zeros((env.num_envs, *source.shape), device=env.device, dtype=source.dtype)
            setattr(env, name, target)
        _restore_tensor_row(target, value, env_index)

    if "task_runtime" in state:
        restore = getattr(getattr(env, "cfg", None), "restore_task_state", None)
        if callable(restore):
            restore(env, state["task_runtime"], env_index)
        else:
            skipped.append("task_runtime")

    return skipped
