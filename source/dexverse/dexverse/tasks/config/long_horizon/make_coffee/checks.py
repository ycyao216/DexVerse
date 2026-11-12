# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success checks for the Make Coffee task.

These decide whether a demonstration earns its success label, so they check what
a person would judge: the milk really went into an upright mug, the mug was
carried without spilling, and it was set down and let go of on the machine's
tray. They stay free of Isaac Lab imports so they can run in CPU tests and against
recorded states.

Thresholds were checked against the first recorded demonstrations, which all set the
mug within about 5 cm of the point under the spouts. A bottle lying
on the table next to the mug has its spout 5.7 cm from the mug axis and within
2 cm of rim height, while real pours reach within 2 cm of the axis, 2-9 cm above
the rim. A mug that has been let go of rests at z = 0.684-0.691 m, tilted under
10 degrees and moving under 0.03 m/s.
"""

from __future__ import annotations

import math

import torch

from ..check_utils import announce_once, axis_tilt, ever_held, local_point, print_status, rotate, status_due, track

TASK_LABEL = "Make Coffee"

# Geometry, from the assets at the task's spawn scale. Both root frames sit at the
# object's base, with local +Z pointing up.
MUG_RIM_HEIGHT = 0.082  # mesh height 9.1 * MUG_SCALE 0.009
MUG_MOUTH_RADIUS = 0.0415
MUG_BASE_RADIUS = 0.0375
MILK_SPOUT_HEIGHT = 0.18

# Pour: spout held inside a column above the mug's mouth, bottle tipped, mug upright.
POUR_TILT_RAD = math.radians(70.0)
POUR_MUG_MAX_TILT_RAD = math.radians(20.0)
POUR_SPOUT_XY_TOL = MUG_MOUTH_RADIUS + 0.003
POUR_SPOUT_MIN_ABOVE_RIM = 0.01
POUR_SPOUT_MAX_ABOVE_RIM = 0.12
POUR_HOLD_S = 1.0

# Transport: tipping this far after the pour counts as spilling the milk.
SPILL_TILT_RAD = math.radians(50.0)
SPILL_HOLD_S = 0.1

# Placement on the machine's tray, with the mug under the spouts. The mark on the tray
# is sized so that the mug's base being fully on it means it is within this tolerance.
PLACE_XY_TOL = 0.05
PLACE_MARK_RADIUS = PLACE_XY_TOL + MUG_BASE_RADIUS
PLACE_Z_TOL = 0.015
PLACE_MAX_TILT_RAD = math.radians(15.0)
PLACE_MAX_SPEED = 0.03
# Fingertip to mug centre. Touching reads about 5 cm; the handle reaches 7.3 cm.
PLACE_FINGER_CLEARANCE = 0.09
PLACE_HOLD_S = 0.25


def _point_above_root(pos: torch.Tensor, quat_wxyz: torch.Tensor, height: float) -> torch.Tensor:
    """World position of the point ``height`` along the body's local +Z."""
    return local_point(pos, quat_wxyz, (0.0, 0.0, height))


def milk_pour_measurements(
    env,
    tilt_rad: float = POUR_TILT_RAD,
    mug_max_tilt_rad: float = POUR_MUG_MAX_TILT_RAD,
    spout_xy_tol: float = POUR_SPOUT_XY_TOL,
    spout_min_above_rim: float = POUR_SPOUT_MIN_ABOVE_RIM,
    spout_max_above_rim: float = POUR_SPOUT_MAX_ABOVE_RIM,
) -> dict[str, torch.Tensor]:
    """Read the exact pour geometry for success checks and operator cues.

    This has no timer/latch side effects, so rendering cannot advance a pour.
    """
    milk, mug = env.scene["milk"].data, env.scene["object"].data
    spout = _point_above_root(milk.root_pos_w, milk.root_quat_w, MILK_SPOUT_HEIGHT)
    rim = _point_above_root(mug.root_pos_w, mug.root_quat_w, MUG_RIM_HEIGHT)
    above_rim = spout[:, 2] - rim[:, 2]
    conditions = torch.stack((
        torch.norm(spout[:, :2] - rim[:, :2], dim=-1) <= spout_xy_tol,
        (above_rim >= spout_min_above_rim) & (above_rim <= spout_max_above_rim),
        axis_tilt(milk.root_quat_w) >= tilt_rad,
        axis_tilt(mug.root_quat_w) <= mug_max_tilt_rad,
    ), dim=-1)
    return {"spout": spout, "rim": rim, "conditions": conditions, "pouring": conditions.all(dim=-1)}


def milk_poured_over_mug(
    env,
    tilt_rad: float = POUR_TILT_RAD,
    mug_max_tilt_rad: float = POUR_MUG_MAX_TILT_RAD,
    spout_xy_tol: float = POUR_SPOUT_XY_TOL,
    spout_min_above_rim: float = POUR_SPOUT_MIN_ABOVE_RIM,
    spout_max_above_rim: float = POUR_SPOUT_MAX_ABOVE_RIM,
    hold_s: float = POUR_HOLD_S,
) -> torch.Tensor:
    """True after one continuous valid pour; a bottle beside the mug never counts."""
    sample = milk_pour_measurements(env, tilt_rad, mug_max_tilt_rad, spout_xy_tol,
                                    spout_min_above_rim, spout_max_above_rim)
    poured, _ = track(env, "pour", sample["pouring"], hold_s)
    announce_once(env, TASK_LABEL, "pour", poured, "Milk poured. Next: carry the mug to the machine's tray and let go.")
    return poured


def _mug_tray_errors(env, place_local: tuple[float, float, float], machine_body: str) -> dict[str, torch.Tensor]:
    """How far the mug is from resting at the tray spot, per env.

    ``place_local`` is the mug's resting root position in the machine body's frame,
    tracked live so it follows the machine.
    """
    machine = env.scene["coffee_machine"]
    mug = env.scene["object"].data
    body_id = machine.find_bodies(machine_body)[0][0]
    body_quat = machine.data.body_quat_w[:, body_id]
    offset = torch.as_tensor(place_local, device=body_quat.device, dtype=body_quat.dtype).expand(env.num_envs, -1)
    target = machine.data.body_pos_w[:, body_id] + rotate(body_quat, offset)
    return {
        "xy": torch.norm(mug.root_pos_w[:, :2] - target[:, :2], dim=-1),
        "z": torch.abs(mug.root_pos_w[:, 2] - target[:, 2]),
        "tilt": axis_tilt(mug.root_quat_w),
        "speed": torch.norm(mug.root_lin_vel_w, dim=-1),
    }


def mug_settled_on_tray(
    env,
    place_local: tuple[float, float, float],
    machine_body: str,
    xy_tol: float = PLACE_XY_TOL,
    z_tol: float = PLACE_Z_TOL,
    max_tilt_rad: float = PLACE_MAX_TILT_RAD,
    max_speed: float = PLACE_MAX_SPEED,
) -> torch.Tensor:
    """True per-env while the mug sits upright and still at the machine's tray spot."""
    errors = _mug_tray_errors(env, place_local, machine_body)
    return (
        (errors["xy"] <= xy_tol)
        & (errors["z"] <= z_tol)
        & (errors["tilt"] <= max_tilt_rad)
        & (errors["speed"] <= max_speed)
    )


def _fingertip_gap(env, fingertip_bodies: tuple[str, ...]) -> torch.Tensor:
    """Distance from the nearest fingertip to the mug's centre, per env."""
    robot = env.scene["robot"]
    mug = env.scene["object"].data
    tip_ids = robot.find_bodies(list(fingertip_bodies), preserve_order=True)[0]
    mug_centre = _point_above_root(mug.root_pos_w, mug.root_quat_w, 0.5 * MUG_RIM_HEIGHT)
    return torch.norm(robot.data.body_pos_w[:, tip_ids] - mug_centre.unsqueeze(1), dim=-1).amin(dim=1)


def mug_placed_and_released(
    env,
    place_local: tuple[float, float, float],
    machine_body: str,
    fingertip_bodies: tuple[str, ...],
    xy_tol: float = PLACE_XY_TOL,
    z_tol: float = PLACE_Z_TOL,
    max_tilt_rad: float = PLACE_MAX_TILT_RAD,
    max_speed: float = PLACE_MAX_SPEED,
    finger_clearance: float = PLACE_FINGER_CLEARANCE,
    hold_s: float = PLACE_HOLD_S,
    spill_tilt_rad: float = SPILL_TILT_RAD,
    spill_hold_s: float = SPILL_HOLD_S,
) -> torch.Tensor:
    """True once the mug has rested on the tray with the fingers off it, unspilled.

    The stage machine evaluates this only after the pour, so tipping the mug past
    ``spill_tilt_rad`` while it is active loses the milk for the rest of the attempt.
    """
    mug = env.scene["object"].data
    _, spilled = track(env, "spill", axis_tilt(mug.root_quat_w) >= spill_tilt_rad, spill_hold_s)
    spill_message = "The mug tipped over after the pour, so this attempt cannot succeed. Reset."
    announce_once(env, TASK_LABEL, "spill", spilled, spill_message)

    settled = mug_settled_on_tray(env, place_local, machine_body, xy_tol, z_tol, max_tilt_rad, max_speed)
    clear = _fingertip_gap(env, fingertip_bodies) >= finger_clearance
    placed, _ = track(env, "place", settled & clear, hold_s)
    placed = placed & ~spilled
    announce_once(env, TASK_LABEL, "place", placed, "Mug placed. Next: turn the machine's switch lever.")
    return placed


def _missing_on_tray(env, place_local: tuple[float, float, float], machine_body: str) -> list[str]:
    """Describe each way the mug is not sitting where success needs it."""
    errors = {name: float(value[0]) for name, value in _mug_tray_errors(env, place_local, machine_body).items()}
    missing = []
    if errors["xy"] > PLACE_XY_TOL:
        missing.append(f"mug {100 * errors['xy']:.1f} cm from the mark's centre, max {100 * PLACE_XY_TOL:.0f}")
    if errors["z"] > PLACE_Z_TOL:
        missing.append(f"mug {100 * errors['z']:.1f} cm off tray height, max {100 * PLACE_Z_TOL:.1f}")
    if errors["tilt"] > PLACE_MAX_TILT_RAD:
        missing.append(f"mug tilted {math.degrees(errors['tilt']):.0f} deg, max {math.degrees(PLACE_MAX_TILT_RAD):.0f}")
    if errors["speed"] > PLACE_MAX_SPEED:
        missing.append(f"mug moving {errors['speed']:.3f} m/s, max {PLACE_MAX_SPEED}")
    return missing


def report_place_stage(
    env,
    waiting: torch.Tensor,
    place_local: tuple[float, float, float],
    machine_body: str,
    fingertip_bodies: tuple[str, ...],
) -> None:
    """Tell the teleoperator which placement condition is missing after the pour."""
    if env.num_envs != 1 or not bool(waiting[0]) or not status_due(env):
        return
    if bool(ever_held(env, "spill", waiting)[0]):
        print_status(env, TASK_LABEL, ["the mug tipped over after the pour, so this attempt cannot succeed, reset"])
        return
    missing = _missing_on_tray(env, place_local, machine_body)
    gap = float(_fingertip_gap(env, fingertip_bodies)[0])
    if gap < PLACE_FINGER_CLEARANCE:
        missing.append(f"fingertips {100 * gap:.0f} cm from the mug, need {100 * PLACE_FINGER_CLEARANCE:.0f}")
    print_status(env, TASK_LABEL, missing)


def report_lever_stage(
    env,
    waiting: torch.Tensor,
    lever_joint: str,
    threshold_rad: float,
    place_local: tuple[float, float, float],
    machine_body: str,
) -> None:
    """Tell the teleoperator which success condition is missing after the mug is placed."""
    if env.num_envs != 1 or not bool(waiting[0]) or not status_due(env):
        return
    machine = env.scene["coffee_machine"]
    lever_id = machine.find_joints(lever_joint)[0][0]
    lever = abs(float(machine.data.joint_pos[0, lever_id] - machine.data.default_joint_pos[0, lever_id]))
    missing = _missing_on_tray(env, place_local, machine_body)
    if lever < threshold_rad:
        missing.insert(0, f"lever at {math.degrees(lever):.0f} deg, needs {math.degrees(threshold_rad):.0f}")
    print_status(env, TASK_LABEL, missing)
