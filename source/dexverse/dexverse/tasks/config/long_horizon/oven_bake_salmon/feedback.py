"""Baking checklist and recorded stage history; drawing never advances the task."""
from __future__ import annotations

import torch

STAGES = ("salt_poured", "pepper_poured", "salmon_in_oven", "close_door", "rotate_knob")
TASK_KEY = "long_horizon.oven_bake_salmon"


def display_state(env, flags, success):
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done = [flags.get(name, zero) for name in STAGES]
    # Last row reflects the live final condition, not a past knob turn.
    complete = torch.stack((*done[:4], success), dim=1)
    seasoned = done[0] & done[1]
    current = torch.stack((~done[0], ~done[1], seasoned & ~done[2],
                           done[2] & ~done[3], done[3] & ~success), dim=1)
    return torch.where(complete, 2, current.long())


def live_success(env, flags):
    from .long_horizon_oven_bake_salmon_cfg import (
        door_closed_with_salmon_contained, knob_turned_with_salmon_contained,
    )
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (flags.get("rotate_knob", zero) & door_closed_with_salmon_contained(env)
            & knob_turned_with_salmon_contained(env))


def update_visuals(env, flags, success):
    if not getattr(env.cfg, "enable_debug_vis", False):
        return
    visuals = getattr(env, "_salmon_visuals", None)
    if visuals is None:
        from .visualization import SalmonProgress
        visuals = env._salmon_visuals = SalmonProgress(env)
    visuals.update(display_state(env, flags, success), success)


def salmon_success(env, task_key=TASK_KEY):
    from .... import mdp
    flags = mdp.evaluate_stage_graph(env, task_key=task_key, persistent=True)
    success = live_success(env, flags)
    update_visuals(env, flags, success)
    return success


def reset_feedback(env, env_ids, task_key=TASK_KEY):
    from ..check_utils import clear_stage_latches
    clear_stage_latches(env, env_ids, task_key)
    runtime = getattr(env, "_stage_graph_runtime_cache", {}).get(task_key)
    flags = getattr(runtime, "latched_flags", None) or {}
    success = getattr(getattr(env, "_salmon_visuals", None), "last_success", None)
    success = (torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
               if success is None else success.clone())
    success[slice(None) if env_ids is None else env_ids] = False
    update_visuals(env, flags, success)


def capture_baking_state(env, env_index):
    runtime = getattr(env, "_stage_graph_runtime_cache", {}).get(TASK_KEY)
    flags = getattr(runtime, "latched_flags", None) or {}
    return {"version": 1, "episode_step": env.episode_length_buf[env_index].detach().clone(),
            "stages": {name: flag[env_index].detach().clone() for name, flag in flags.items()}}


def restore_baking_state(env, snapshot, env_index):
    from ....mdp.stage_machine import _get_or_create_task_runtime_state
    from ..check_utils import clear_stage_latches
    if snapshot.get("version") != 1:
        raise ValueError("Unsupported baking task-state version")
    clear_stage_latches(env, torch.tensor([env_index], device=env.device), TASK_KEY)
    runtime = _get_or_create_task_runtime_state(env, TASK_KEY)
    if runtime.latched_flags is None:
        runtime.latched_flags = {name: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                                 for name in STAGES}
    for name in STAGES:
        runtime.latched_flags[name][env_index] = bool(snapshot["stages"].get(name, False))
    runtime.latched_stage_names = STAGES
    env.episode_length_buf[env_index] = int(snapshot["episode_step"])
    if runtime.latched_step_buf is None:
        runtime.latched_step_buf = torch.full_like(env.episode_length_buf, -1)
    runtime.latched_step_buf[env_index] = env.episode_length_buf[env_index]
    runtime.raw_flags = runtime.raw_step_buf = runtime.eval_flags = runtime.eval_step_buf = None
    update_visuals(env, runtime.latched_flags, live_success(env, runtime.latched_flags))
