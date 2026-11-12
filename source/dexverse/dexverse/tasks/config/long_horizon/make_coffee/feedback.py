"""Read-only operator feedback; all completion decisions come from the task."""

from __future__ import annotations

import torch

from ..check_utils import clear_stage_latches, ever_held
from .checks import POUR_HOLD_S, milk_pour_measurements


def coffee_display_state(env, flags=None, success=None):
    """Return stage colors, live pour checks and hold fraction without ticking timers.

    Indicator codes: 0 pending, 1 current, 2 complete, 3 invalid/reset required.
    The final row reflects live success, not a historical lever latch. The panel
    reports task conditions, never claims that the recorder has saved a file.
    """
    sample = milk_pour_measurements(env)
    zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    flags = flags or {}
    poured = flags.get("pour_milk", zero)
    placed = flags.get("place_on_machine", zero)
    success = zero if success is None else success
    spilled = ever_held(env, "spill", zero)
    complete = torch.stack((poured, placed, success), dim=-1)
    current = torch.stack((~poured, poured & ~placed, placed & ~success), dim=-1)
    stages = torch.where(complete, 2, current.long())
    stages = torch.where(spilled[:, None], 3, stages)
    entry = getattr(env, "_long_horizon_check_memory", {}).get("pour")
    progress = torch.zeros(env.num_envs, device=env.device)
    if entry is not None:
        progress = (entry[0].float() * env.step_dt / POUR_HOLD_S).clamp(0, 1)
    progress = torch.where(sample["pouring"], progress, 0.0)
    progress = torch.where(poured, 1.0, progress)
    progress = torch.where(spilled, 0.0, progress)
    # Phase codes: pour, transport, lever, task conditions met, reset.
    status = torch.where(poured, 1, 0)
    status = torch.where(placed, 2, status)
    status = torch.where(success, 3, status)
    status = torch.where(spilled, 4, status)
    return {**sample, "stages": stages, "progress": progress, "status": status,
            "poured": poured, "spilled": spilled,
            "flowing": sample["pouring"] & ~poured & ~spilled}


def update_coffee_visuals(env, flags=None, success=None):
    if not getattr(env.cfg, "enable_debug_vis", False):
        return
    visuals = getattr(env, "_coffee_visuals", None)
    if visuals is None:
        from .visualization import CoffeeVisuals

        visuals = env._coffee_visuals = CoffeeVisuals(env)
    visuals.update(env, coffee_display_state(env, flags, success))


def reset_coffee_feedback(env, env_ids, task_key):
    """Clear the task's history and immediately refresh cues on manual/auto reset."""
    clear_stage_latches(env, env_ids, task_key)
    state = getattr(env, "_stage_graph_runtime_cache", {}).get(task_key)
    flags = getattr(state, "latched_flags", None)
    # A reset invalidates current live success. Preserve other environments'
    # display until their next normal success check.
    success = getattr(getattr(env, "_coffee_visuals", None), "last_success", None)
    if success is not None:
        success = success.clone()
        success[slice(None) if env_ids is None else env_ids] = False
    update_coffee_visuals(env, flags, success)
