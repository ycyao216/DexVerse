"""Cooking history saved alongside scene poses for recording and state replay."""
from __future__ import annotations

import torch


def capture_cooking_state(env, env_index):
    memory = getattr(env, "_long_horizon_check_memory", {})
    timers = {key: tuple(v[env_index].detach().clone() for v in entry)
              for key, entry in memory.items()
              if isinstance(entry, tuple) and len(entry) == 3
              and all(isinstance(v, torch.Tensor) for v in entry)}
    runtime = getattr(env, "_stage_graph_runtime_cache", {}).get(env.cfg.cooking_task_key)
    flags = getattr(runtime, "latched_flags", None) or {}
    return {"version": 1, "episode_step": env.episode_length_buf[env_index].detach().clone(),
            "timers": timers, "stages": {k: v[env_index].detach().clone() for k, v in flags.items()}}


def restore_cooking_state(env, snapshot, env_index):
    from ....mdp.stage_machine import _get_or_create_task_runtime_state, get_stage_graph
    from ..check_utils import clear_stage_latches
    from .long_horizon_cook_foods_cfg import refresh_cooking_visuals

    if snapshot.get("version") != 1:
        raise ValueError("Unsupported cooking task-state version")
    task_key = env.cfg.cooking_task_key
    clear_stage_latches(env, torch.tensor([env_index], device=env.device), task_key)
    step = int(snapshot["episode_step"])
    env.episode_length_buf[env_index] = step
    memory = env.__dict__.setdefault("_long_horizon_check_memory", {})
    for key, values in snapshot["timers"].items():
        entry = memory.get(key)
        if entry is None:
            entry = (torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
                     torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
                     torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        entry = tuple(v.clone() for v in entry)
        for target, source in zip(entry, values):
            target[env_index] = torch.as_tensor(source, device=target.device, dtype=target.dtype)
        memory[key] = entry
    runtime = _get_or_create_task_runtime_state(env, task_key)
    names = tuple(stage.name for stage in get_stage_graph(task_key).stages)
    if runtime.latched_flags is None:
        runtime.latched_flags = {name: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool) for name in names}
    for name in names:
        runtime.latched_flags[name][env_index] = bool(snapshot["stages"].get(name, False))
    runtime.latched_stage_names = names
    if runtime.latched_step_buf is None:
        runtime.latched_step_buf = torch.full_like(env.episode_length_buf, -1)
    runtime.latched_step_buf[env_index] = step
    runtime.raw_flags = runtime.raw_step_buf = runtime.eval_flags = runtime.eval_step_buf = None
    # Draw from restored state without evaluating predicates or advancing timers.
    refresh_cooking_visuals(env, env.cfg.cooking_ready, env.cfg.cooking_sources)
