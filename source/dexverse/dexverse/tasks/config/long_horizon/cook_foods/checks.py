# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success checks for the Cook Foods task.

These decide whether a demonstration earns its success label: both liquids really
went into an upright pot, the pot was not tipped afterwards, it sits on the marked
burner with the meat inside, and the marked knob is turned. They stay free of Isaac
Lab imports so they can run in CPU tests.

Pour and spill thresholds match Make Coffee. The pour template's own 100 degree tilt was
out of reach in teleop: in the two approved Make Coffee demos the bottle peaked at 84 and
89 degrees, and stayed past 70 for 2.0 and 1.7 s. The pour spot is the pot's root, which
is the centre of its 12.6 cm-radius mouth. The burner thresholds come from dropping the
pot around the left burner in the real scene: it sits flat on the grate 10.5 cm above the
stove's base, up to about 8 cm off centre, and slumps off beyond that.

There is deliberately no stillness check on the pot. With meat pieces resting inside it,
the solver reports velocities it does not have: a pot that was not moving read 0.056 m/s
and 0.4 rad/s, or 0.008 m/s if the pieces landed differently. Being flat at grate height
for the consecutive steps success needs is what shows it is resting.
"""

from __future__ import annotations

import math

import torch

from ..check_utils import (
    announce_once,
    axis_tilt,
    check_memory,
    ever_held,
    local_point,
    print_status,
    rotate_inverse,
    status_due,
    track,
)

TASK_LABEL = "Cook Foods"

POUR_TILT_RAD = math.radians(70.0)
POUR_HOLD_S = 1.0
POUR_POT_MAX_TILT_RAD = math.radians(20.0)

# A lifted bottle or can tipped this far is taken as an attempt to pour, so one that
# does not count can be explained once it ends. Shorter tips are ignored.
POUR_ATTEMPT_TILT_RAD = math.radians(45.0)
POUR_ATTEMPT_MIN_S = 0.25

# Tipping the pot this far after a pour counts as spilling it.
SPILL_TILT_RAD = math.radians(50.0)
SPILL_HOLD_S = 0.1

BURNER_XY_TOL = 0.08
POT_ON_GRATE_HEIGHT = 0.105  # pot root above the stove root, sitting on the grate
BURNER_Z_TOL = 0.015
POT_MAX_TILT_RAD = math.radians(15.0)


def _pour_key(pour_name: str) -> str:
    return f"pour_{pour_name}"


def pour_measurements(env, pour_name, tilt_axis_local, spout_local, min_lift,
                      tilt_rad=POUR_TILT_RAD, xy_tol=0.10):
    """Read the same geometry for pour gates and cues without advancing timers."""
    source, pot = env.scene[pour_name].data, env.scene["pot"].data
    spout = local_point(source.root_pos_w, source.root_quat_w, spout_local)
    lifted = source.root_pos_w[:, 2] - source.default_root_state[:, 2] >= min_lift
    tilt = axis_tilt(source.root_quat_w, tilt_axis_local)
    spout_xy = torch.norm(spout[:, :2] - pot.root_pos_w[:, :2], dim=-1)
    pot_tilt = axis_tilt(pot.root_quat_w)
    conditions = torch.stack((lifted, tilt >= tilt_rad, spout_xy <= xy_tol,
                              pot_tilt <= POUR_POT_MAX_TILT_RAD), dim=-1)
    return {"spout": spout, "lifted": lifted, "tilt": tilt, "spout_xy": spout_xy,
            "pot_tilt": pot_tilt, "conditions": conditions, "pouring": conditions.all(dim=-1)}


def poured_into_pot(
    env,
    pour_name: str,
    label: str,
    tilt_axis_local: tuple[float, float, float],
    spout_local: tuple[float, float, float],
    min_lift: float,
    tilt_rad: float = POUR_TILT_RAD,
    xy_tol: float = 0.10,
    hold_s: float = POUR_HOLD_S,
) -> torch.Tensor:
    """True once a lifted, tipped object's spout has stayed over an upright pot's mouth."""
    sample = pour_measurements(env, pour_name, tilt_axis_local, spout_local, min_lift, tilt_rad, xy_tol)
    lifted, tilt, spout_xy, pot_tilt, pouring = (sample[k] for k in
        ("lifted", "tilt", "spout_xy", "pot_tilt", "pouring"))
    key = _pour_key(pour_name)
    poured, _ = track(env, key, pouring, hold_s)
    announce_once(env, TASK_LABEL, key, poured, f"{label} poured.")
    attempting = lifted & (tilt >= POUR_ATTEMPT_TILT_RAD)
    _explain_missed_pour(env, key, label, attempting, pouring, tilt, spout_xy, pot_tilt, tilt_rad, xy_tol, hold_s)
    return poured


def _explain_missed_pour(
    env,
    key: str,
    label: str,
    attempting: torch.Tensor,
    pouring: torch.Tensor,
    tilt: torch.Tensor,
    spout_xy: torch.Tensor,
    pot_tilt: torch.Tensor,
    tilt_rad: float,
    xy_tol: float,
    hold_s: float,
) -> None:
    """When a pour attempt ends without counting, tell the operator what fell short."""
    if env.num_envs != 1:
        return
    memory = check_memory(env)
    step = int(env.episode_length_buf[0])
    attempt = memory.get(f"attempt_{key}")
    if attempt is None or step < attempt["step"]:
        attempt = {"step": -1, "frames": 0}
    if step == attempt["step"]:
        return
    attempt["step"] = step
    memory[f"attempt_{key}"] = attempt
    if bool(attempting[0]):
        tipped = float(tilt[0]) >= tilt_rad
        run = attempt.get("run", 0) + 1 if bool(pouring[0]) else 0
        attempt.update(
            frames=attempt["frames"] + 1,
            run=run,
            longest=max(attempt.get("longest", 0), run),
            max_tilt=max(attempt.get("max_tilt", 0.0), float(tilt[0])),
            spout=min(attempt.get("spout", math.inf), float(spout_xy[0])),
            spout_tipped=min(attempt.get("spout_tipped", math.inf), float(spout_xy[0]) if tipped else math.inf),
            pot_tilt=min(attempt.get("pot_tilt", math.inf), float(pot_tilt[0])),
        )
        return
    if attempt["frames"] == 0:
        return
    ended = attempt
    memory[f"attempt_{key}"] = {"step": step, "frames": 0}
    if ended["frames"] * env.step_dt < POUR_ATTEMPT_MIN_S or bool(ever_held(env, key, attempting)[0]):
        return
    reasons = []
    if ended["max_tilt"] < tilt_rad:
        reasons.append(f"tipped to {math.degrees(ended['max_tilt']):.0f} deg, needs {math.degrees(tilt_rad):.0f}")
        spout = ended["spout"]
    else:
        spout = ended["spout_tipped"]
    if spout > xy_tol:
        reasons.append(f"spout stayed {100 * spout:.1f} cm or more from the pot centre, max {100 * xy_tol:.0f}")
    if ended["pot_tilt"] > POUR_POT_MAX_TILT_RAD:
        reasons.append(
            f"pot tilted {math.degrees(ended['pot_tilt']):.0f} deg, max {math.degrees(POUR_POT_MAX_TILT_RAD):.0f}"
        )
    if not reasons:
        reasons.append(f"all of it held together for {ended['longest'] * env.step_dt:.1f} s, needs {hold_s:.1f}")
    print(f"[{TASK_LABEL}] {label} pour not counted: {'; '.join(reasons)}.")


def pot_spilled(env, pour_names: tuple[str, ...], spill_tilt_rad: float = SPILL_TILT_RAD) -> torch.Tensor:
    """True once the pot has been tipped over after anything was poured into it."""
    tipped = axis_tilt(env.scene["pot"].data.root_quat_w) >= spill_tilt_rad
    poured = torch.zeros_like(tipped)
    for name in pour_names:
        poured = poured | ever_held(env, _pour_key(name), tipped)
    _, spilled = track(env, "spill", tipped & poured, SPILL_HOLD_S)
    message = "The pot tipped over after a pour, so this attempt cannot succeed. Reset."
    announce_once(env, TASK_LABEL, "spill", spilled, message)
    return spilled


def pot_liquid(water: torch.Tensor, can: torch.Tensor, spilled: torch.Tensor) -> torch.Tensor:
    """Which liquid the pot shows, per env: 0 none or spilled, 1 water, 2 the can's soup, 3 both."""
    return torch.where(spilled, torch.zeros_like(water, dtype=torch.long), water.long() + 2 * can.long())


def _pot_errors(env, burner_local: tuple[float, float, float]) -> dict[str, torch.Tensor]:
    """How far the pot is from sitting on the burner, per env."""
    stove, pot = env.scene["stove"].data, env.scene["pot"].data
    burner = local_point(stove.root_pos_w, stove.root_quat_w, burner_local)
    return {
        "xy": torch.norm(pot.root_pos_w[:, :2] - burner[:, :2], dim=-1),
        "z": torch.abs(pot.root_pos_w[:, 2] - stove.root_pos_w[:, 2] - POT_ON_GRATE_HEIGHT),
        "tilt": axis_tilt(pot.root_quat_w),
    }


def pot_on_burner(env, burner_local: tuple[float, float, float]) -> torch.Tensor:
    """True per-env while the pot sits flat on the burner's grate."""
    errors = _pot_errors(env, burner_local)
    return (errors["xy"] <= BURNER_XY_TOL) & (errors["z"] <= BURNER_Z_TOL) & (errors["tilt"] <= POT_MAX_TILT_RAD)


def ingredients_in_pot(
    env,
    ingredient_names: tuple[str, ...],
    zone_center: tuple[float, float, float],
    zone_radius: float,
    zone_half_height: float,
) -> torch.Tensor:
    """How many ingredient pieces are inside a cylinder fixed in the pot's frame, per env."""
    pot = env.scene["pot"].data
    count = torch.zeros(env.num_envs, dtype=torch.long, device=pot.root_pos_w.device)
    center = torch.as_tensor(zone_center, device=pot.root_pos_w.device, dtype=pot.root_pos_w.dtype)
    for name in ingredient_names:
        offset = rotate_inverse(pot.root_quat_w, env.scene[name].data.root_pos_w - pot.root_pos_w) - center
        inside = (offset[:, :2].pow(2).sum(dim=-1) <= zone_radius**2) & (offset[:, 2].abs() <= zone_half_height)
        count = count + inside.long()
    return count


def knob_angle(env, knob_joint: str) -> torch.Tensor:
    """How far the knob has turned from its start, in radians, per env."""
    stove = env.scene["stove"]
    knob_id = stove.find_joints(knob_joint)[0][0]
    return torch.abs(stove.data.joint_pos[:, knob_id] - stove.data.default_joint_pos[:, knob_id])


def stove_on_and_pot_ready(
    env,
    burner_local: tuple[float, float, float],
    knob_joint: str,
    knob_threshold_rad: float,
    ingredient_names: tuple[str, ...],
    zone_center: tuple[float, float, float],
    zone_radius: float,
    zone_half_height: float,
    pour_names: tuple[str, ...],
    disposal: dict | None = None,
) -> torch.Tensor:
    """Live cooking end state, including released containers when disposal is configured."""
    in_pot = ingredients_in_pot(env, ingredient_names, zone_center, zone_radius, zone_half_height)
    ready = (
        (knob_angle(env, knob_joint) >= knob_threshold_rad)
        & pot_on_burner(env, burner_local)
        & (in_pot == len(ingredient_names))
        & ~pot_spilled(env, pour_names)
    )
    if disposal is not None:
        from .disposal import containers_disposed
        ready &= containers_disposed(env, **disposal)
    return ready


def report_status(
    env,
    waiting: torch.Tensor,
    missing_pours: list[str],
    burner_local: tuple[float, float, float],
    knob_joint: str,
    knob_threshold_rad: float,
    ingredient_names: tuple[str, ...],
    zone_center: tuple[float, float, float],
    zone_radius: float,
    zone_half_height: float,
    pour_names: tuple[str, ...],
    disposal: dict | None = None,
) -> None:
    """Tell the teleoperator everything success is still waiting for."""
    if env.num_envs != 1 or not bool(waiting[0]) or not status_due(env):
        return
    if bool(ever_held(env, "spill", waiting)[0]):
        print_status(env, TASK_LABEL, ["the pot tipped over after a pour, so this attempt cannot succeed, reset"])
        return
    missing = [f"{label} not poured" for label in missing_pours]
    if disposal is not None:
        from .disposal import disposal_measurements
        sample = disposal_measurements(env, **disposal)
        for i, label in enumerate(("water bottle", "soup can")):
            if not bool(sample["inside"][0, i]):
                margins = sample["margins"][0, i].detach().cpu().tolist()
                missing.append(f"{label} not fully inside bin (clearance x/y/floor/rim: "
                               + "/".join(f"{1000*m:.0f}" for m in margins) + " mm)")
            if not bool(sample["released"][0, i]):
                missing.append(f"move hands clear of {label} ({100*sample['clearance'][0, i]:.1f} cm, needs 3)")
            if not bool(sample["settled"][0, i]):
                missing.append(f"{label} still moving ({sample['speed'][0, i]:.2f} m/s, "
                               f"{sample['spin'][0, i]:.2f} rad/s)")
    errors = {name: float(value[0]) for name, value in _pot_errors(env, burner_local).items()}
    if errors["xy"] > BURNER_XY_TOL:
        missing.append(f"pot {100 * errors['xy']:.1f} cm from the green burner, max {100 * BURNER_XY_TOL:.0f}")
    if errors["z"] > BURNER_Z_TOL:
        missing.append(f"pot {100 * errors['z']:.1f} cm off grate height, max {100 * BURNER_Z_TOL:.1f}")
    if errors["tilt"] > POT_MAX_TILT_RAD:
        missing.append(f"pot tilted {math.degrees(errors['tilt']):.0f} deg, max {math.degrees(POT_MAX_TILT_RAD):.0f}")
    in_pot = int(ingredients_in_pot(env, ingredient_names, zone_center, zone_radius, zone_half_height)[0])
    if in_pot < len(ingredient_names):
        missing.append(f"{in_pot} of {len(ingredient_names)} meat pieces in the pot")
    angle = float(knob_angle(env, knob_joint)[0])
    if angle < knob_threshold_rad:
        missing.append(f"green knob at {math.degrees(angle):.0f} deg, needs {math.degrees(knob_threshold_rad):.0f}")
    print_status(env, TASK_LABEL, missing)
