# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from isaaclab.envs import ManagerBasedRLEnv


@dataclass(frozen=True)
class StageSpec:
    """Single stage definition in a long-horizon state graph."""

    name: str
    func: Callable[..., torch.Tensor]
    params: Mapping[str, Any] = field(default_factory=dict)
    deps: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageGraphSpec:
    """Ordered stage graph definition."""

    stages: tuple[StageSpec, ...]
    terminal_stage: str | None = None
    ordering_mode: Literal["strict", "free"] = "strict"
    success_mode: Literal["substage", "all"] = "substage"


@dataclass
class StageRuntimeState:
    """Per-task runtime state cached on env for stage graph evaluation."""

    raw_flags: dict[str, torch.Tensor] | None = None
    raw_step_buf: torch.Tensor | None = None
    raw_stage_names: tuple[str, ...] | None = None
    latched_flags: dict[str, torch.Tensor] | None = None
    latched_step_buf: torch.Tensor | None = None
    latched_stage_names: tuple[str, ...] | None = None
    eval_flags: dict[str, torch.Tensor] | None = None
    eval_step_buf: torch.Tensor | None = None
    eval_persistent: bool | None = None
    eval_ordering_mode: Literal["strict", "free"] | None = None
    eval_stage_names: tuple[str, ...] | None = None


_STAGE_GRAPH_REGISTRY: dict[str, StageGraphSpec] = {}


def _get_episode_length_buf(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Return episode-length buffer if the runtime exposes it."""
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if isinstance(episode_length_buf, torch.Tensor) and episode_length_buf.ndim == 1:
        return episode_length_buf
    return None


def _validate_stage_graph(graph: StageGraphSpec) -> None:
    if len(graph.stages) == 0:
        raise ValueError("StageGraphSpec.stages must not be empty.")

    seen: set[str] = set()
    for stage in graph.stages:
        if stage.name in seen:
            raise ValueError(f"Duplicate stage name: {stage.name}")
        for dep in stage.deps:
            if dep not in seen:
                raise ValueError(f"Stage '{stage.name}' depends on unknown or later stage '{dep}'")
        seen.add(stage.name)

    if graph.terminal_stage is not None and graph.terminal_stage not in seen:
        raise ValueError(f"Unknown terminal_stage: {graph.terminal_stage}")
    if graph.ordering_mode not in ("strict", "free"):
        raise ValueError(f"Unsupported ordering_mode: {graph.ordering_mode}")
    if graph.success_mode not in ("substage", "all"):
        raise ValueError(f"Unsupported success_mode: {graph.success_mode}")
    if graph.ordering_mode == "free" and any(stage.deps for stage in graph.stages):
        warnings.warn(
            "StageGraphSpec has deps but ordering_mode='free': deps are ignored during stage gating.",
            stacklevel=2,
        )
    if graph.success_mode == "all" and graph.terminal_stage is not None:
        warnings.warn(
            "StageGraphSpec has terminal_stage but success_mode='all': terminal_stage is ignored for success.",
            stacklevel=2,
        )


def register_stage_graph(task_key: str, graph: StageGraphSpec, *, override: bool = False) -> None:
    """Register stage graph by task key."""
    if not task_key:
        raise ValueError("task_key must be non-empty.")
    _validate_stage_graph(graph)
    if not override and task_key in _STAGE_GRAPH_REGISTRY:
        raise ValueError(f"Stage graph already registered for task_key='{task_key}'")
    _STAGE_GRAPH_REGISTRY[task_key] = graph


def get_stage_graph(task_key: str) -> StageGraphSpec:
    """Fetch a registered stage graph by key."""
    if task_key not in _STAGE_GRAPH_REGISTRY:
        raise KeyError(f"No stage graph registered for task_key='{task_key}'")
    return _STAGE_GRAPH_REGISTRY[task_key]


def _to_stage_flag(value: torch.Tensor, stage_name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Stage '{stage_name}' must return torch.Tensor, got {type(value)}")
    stage_flag = value.bool()
    if stage_flag.ndim == 2 and stage_flag.shape[1] == 1:
        stage_flag = stage_flag[:, 0]
    if stage_flag.ndim != 1:
        raise ValueError(
            f"Stage '{stage_name}' must return shape [num_envs] or [num_envs,1], got {tuple(stage_flag.shape)}"
        )
    return stage_flag


def _evaluate_stage_raw(env: ManagerBasedRLEnv, graph: StageGraphSpec) -> dict[str, torch.Tensor]:
    """Evaluate stage predicates without dependency gating."""
    raw_flags: dict[str, torch.Tensor] = {}
    for stage in graph.stages:
        raw_flags[stage.name] = _to_stage_flag(stage.func(env, **dict(stage.params)), stage.name)
    return raw_flags


def _stage_names(graph: StageGraphSpec) -> tuple[str, ...]:
    return tuple(stage.name for stage in graph.stages)


def _get_or_create_task_runtime_state(env: ManagerBasedRLEnv, task_key: str) -> StageRuntimeState:
    cache = getattr(env, "_stage_graph_runtime_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_stage_graph_runtime_cache", cache)
    state = cache.get(task_key)
    if isinstance(state, StageRuntimeState):
        return state
    if isinstance(state, dict):
        state = StageRuntimeState(
            raw_flags=state.get("raw_flags"),
            raw_step_buf=state.get("raw_step_buf"),
            latched_flags=state.get("latched_flags"),
            latched_step_buf=state.get("latched_step_buf"),
            eval_flags=state.get("eval_flags"),
            eval_step_buf=state.get("eval_step_buf"),
            eval_persistent=state.get("eval_persistent"),
            eval_ordering_mode=state.get("eval_ordering_mode"),
            eval_stage_names=state.get("eval_stage_names"),
        )
    else:
        state = StageRuntimeState()
    cache[task_key] = state
    return state


def _get_raw_flags_cached(
    env: ManagerBasedRLEnv, graph: StageGraphSpec, state: StageRuntimeState
) -> dict[str, torch.Tensor]:
    """Reuse raw stage predicates inside the same environment step."""
    episode_length_buf = _get_episode_length_buf(env)
    stage_names = _stage_names(graph)
    if (
        state.raw_flags is not None
        and state.raw_stage_names == stage_names
        and episode_length_buf is not None
        and isinstance(state.raw_step_buf, torch.Tensor)
        and torch.equal(state.raw_step_buf, episode_length_buf)
    ):
        return state.raw_flags

    raw_flags = _evaluate_stage_raw(env, graph)
    state.raw_flags = raw_flags
    state.raw_stage_names = stage_names
    state.raw_step_buf = episode_length_buf.clone() if episode_length_buf is not None else None
    return raw_flags


def _compute_reset_mask(
    episode_length_buf: torch.Tensor | None,
    previous_step_buf: torch.Tensor | None,
) -> torch.Tensor | None:
    if episode_length_buf is None:
        return None
    if isinstance(previous_step_buf, torch.Tensor) and previous_step_buf.shape == episode_length_buf.shape:
        return (episode_length_buf == 0) & (previous_step_buf != 0)
    return episode_length_buf == 0


def _initialize_latched_flags(
    graph: StageGraphSpec,
    num_envs: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    return {stage.name: torch.zeros(num_envs, dtype=torch.bool, device=device) for stage in graph.stages}


def _get_persistent_base_flags(
    env: ManagerBasedRLEnv,
    graph: StageGraphSpec,
    state: StageRuntimeState,
) -> dict[str, torch.Tensor]:
    episode_length_buf = _get_episode_length_buf(env)
    stage_names = _stage_names(graph)
    latched_flags = state.latched_flags
    num_envs = env.num_envs
    device = env.device
    needs_init = (
        latched_flags is None
        or state.latched_stage_names != stage_names
        or any(latched_flags[name].shape != (num_envs,) for name in stage_names)
    )
    if needs_init:
        latched_flags = _initialize_latched_flags(graph, num_envs, device)
        state.latched_flags = latched_flags
        state.latched_stage_names = stage_names

    reset_mask = _compute_reset_mask(episode_length_buf, state.latched_step_buf)
    if episode_length_buf is not None:
        state.latched_step_buf = episode_length_buf.clone()

    has_reset = reset_mask is not None and bool(torch.any(reset_mask))
    if not has_reset:
        return latched_flags

    reset_base_flags: dict[str, torch.Tensor] = {}
    for stage in graph.stages:
        base_done = latched_flags[stage.name].clone()
        base_done[reset_mask] = False
        reset_base_flags[stage.name] = base_done
    return reset_base_flags


def _evaluate_flags(
    graph: StageGraphSpec,
    raw_flags: dict[str, torch.Tensor],
    ordering_mode: Literal["strict", "free"],
    base_flags: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if ordering_mode not in ("strict", "free"):
        raise ValueError(f"Unsupported ordering_mode: {ordering_mode}")

    flags: dict[str, torch.Tensor] = {}
    for stage in graph.stages:
        stage_flag = raw_flags[stage.name]
        if ordering_mode == "strict" and stage.deps:
            dep_ok = torch.ones_like(stage_flag)
            for dep in stage.deps:
                dep_ok &= flags[dep]
            stage_flag &= dep_ok
        if base_flags is not None:
            stage_flag = base_flags[stage.name] | stage_flag
        flags[stage.name] = stage_flag
    return flags


def _is_linear_chain(graph: StageGraphSpec) -> bool:
    """Whether stages form a simple linear dependency chain."""
    for idx, stage in enumerate(graph.stages):
        if idx == 0:
            if stage.deps:
                return False
            continue
        if stage.deps != (graph.stages[idx - 1].name,):
            return False
    return True


def _evaluate_strict_persistent_active_stage(
    env: ManagerBasedRLEnv,
    graph: StageGraphSpec,
    state: StageRuntimeState,
) -> dict[str, torch.Tensor]:
    """Incremental strict+persistent evaluation: only evaluate current active stage."""
    base_flags = _get_persistent_base_flags(env, graph, state)
    flags = {name: value.clone() for name, value in base_flags.items()}

    ordered_done = torch.stack([base_flags[stage.name] for stage in graph.stages], dim=-1)
    pending = ~ordered_done
    has_pending = pending.any(dim=-1)
    if not bool(torch.any(has_pending)):
        state.latched_flags = flags
        return flags

    pending_idx = pending.to(dtype=torch.int64).argmax(dim=-1)
    for stage_idx, stage in enumerate(graph.stages):
        env_mask = has_pending & (pending_idx == stage_idx)
        if not bool(torch.any(env_mask)):
            continue

        stage_raw = _to_stage_flag(stage.func(env, **dict(stage.params)), stage.name)
        if stage.deps:
            dep_ok = torch.ones_like(stage_raw)
            for dep in stage.deps:
                dep_ok &= flags[dep]
            stage_raw &= dep_ok
        updated = flags[stage.name] | stage_raw
        flags[stage.name] = torch.where(env_mask, updated, flags[stage.name])

    state.latched_flags = flags
    return flags


def _get_cached_eval_flags(
    env: ManagerBasedRLEnv,
    graph: StageGraphSpec,
    state: StageRuntimeState,
    persistent: bool,
    ordering_mode: Literal["strict", "free"],
) -> dict[str, torch.Tensor] | None:
    stage_names = _stage_names(graph)
    episode_length_buf = _get_episode_length_buf(env)
    if (
        state.eval_flags is not None
        and episode_length_buf is not None
        and isinstance(state.eval_step_buf, torch.Tensor)
        and torch.equal(state.eval_step_buf, episode_length_buf)
        and state.eval_persistent == persistent
        and state.eval_ordering_mode == ordering_mode
        and state.eval_stage_names == stage_names
    ):
        return state.eval_flags
    return None


def _set_cached_eval_flags(
    env: ManagerBasedRLEnv,
    graph: StageGraphSpec,
    state: StageRuntimeState,
    flags: dict[str, torch.Tensor],
    persistent: bool,
    ordering_mode: Literal["strict", "free"],
) -> None:
    stage_names = _stage_names(graph)
    episode_length_buf = _get_episode_length_buf(env)
    if episode_length_buf is None:
        state.eval_flags = None
        state.eval_step_buf = None
        state.eval_persistent = None
        state.eval_ordering_mode = None
        state.eval_stage_names = None
        return
    state.eval_flags = flags
    state.eval_step_buf = episode_length_buf.clone()
    state.eval_persistent = persistent
    state.eval_ordering_mode = ordering_mode
    state.eval_stage_names = stage_names


def evaluate_stage_graph(
    env: ManagerBasedRLEnv,
    task_key: str,
    persistent: bool = False,
    ordering_mode: Literal["strict", "free"] | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate registered stage graph and return constrained stage flags.

    Args:
        env: RL environment.
        task_key: Registered stage-graph key.
        persistent: If True, stage completion is latched per env until reset.
        ordering_mode: Optional runtime override, otherwise uses graph.ordering_mode.
    """
    graph = get_stage_graph(task_key)
    state = _get_or_create_task_runtime_state(env, task_key)
    mode = ordering_mode or graph.ordering_mode
    cached_flags = _get_cached_eval_flags(env, graph, state, persistent=persistent, ordering_mode=mode)
    if cached_flags is not None:
        return cached_flags

    if persistent and mode == "strict" and _is_linear_chain(graph):
        flags = _evaluate_strict_persistent_active_stage(env, graph, state)
        _set_cached_eval_flags(env, graph, state, flags, persistent=persistent, ordering_mode=mode)
        return flags

    raw_flags = _get_raw_flags_cached(env, graph, state)
    if not persistent:
        flags = _evaluate_flags(graph, raw_flags, mode)
        _set_cached_eval_flags(env, graph, state, flags, persistent=persistent, ordering_mode=mode)
        return flags

    base_flags = _get_persistent_base_flags(env, graph, state)
    flags = _evaluate_flags(graph, raw_flags, mode, base_flags=base_flags)
    state.latched_flags = flags
    _set_cached_eval_flags(env, graph, state, flags, persistent=persistent, ordering_mode=mode)
    return flags


def stage_signals(
    env: ManagerBasedRLEnv,
    task_key: str,
    persistent: bool = False,
    ordering_mode: Literal["strict", "free"] | None = None,
) -> torch.Tensor:
    """Return stage signals as [s1, s2, ..., psr]."""
    graph = get_stage_graph(task_key)
    flags = evaluate_stage_graph(env, task_key=task_key, persistent=persistent, ordering_mode=ordering_mode)
    ordered = torch.stack([flags[stage.name].float() for stage in graph.stages], dim=-1)
    psr = ordered.mean(dim=-1, keepdim=True)
    return torch.cat((ordered, psr), dim=-1)


def stage_progress(
    env: ManagerBasedRLEnv,
    task_key: str,
    persistent: bool = False,
    ordering_mode: Literal["strict", "free"] | None = None,
) -> torch.Tensor:
    """Return only PSR (stage completion ratio) as shape [num_envs, 1]."""
    graph = get_stage_graph(task_key)
    flags = evaluate_stage_graph(env, task_key=task_key, persistent=persistent, ordering_mode=ordering_mode)
    ordered = torch.stack([flags[stage.name].float() for stage in graph.stages], dim=-1)
    return ordered.mean(dim=-1, keepdim=True)


def stage_success(
    env: ManagerBasedRLEnv,
    task_key: str,
    terminal_stage: str | None = None,
    persistent: bool = False,
    success_mode: Literal["substage", "all"] | None = None,
    ordering_mode: Literal["strict", "free"] | None = None,
) -> torch.Tensor:
    """Return success flag from substage or all-stage completion."""
    graph = get_stage_graph(task_key)
    flags = evaluate_stage_graph(env, task_key=task_key, persistent=persistent, ordering_mode=ordering_mode)
    mode = success_mode or graph.success_mode
    if mode == "all":
        return torch.stack([flags[stage.name] for stage in graph.stages], dim=-1).all(dim=-1)
    if mode == "substage":
        stage_name = terminal_stage or graph.terminal_stage or graph.stages[-1].name
        if stage_name not in flags:
            raise KeyError(f"Unknown terminal stage '{stage_name}' for task_key='{task_key}'")
        return flags[stage_name]
    raise ValueError(f"Unsupported success_mode: {mode}")


__all__ = [
    "StageSpec",
    "StageGraphSpec",
    "register_stage_graph",
    "get_stage_graph",
    "evaluate_stage_graph",
    "stage_signals",
    "stage_progress",
    "stage_success",
]
