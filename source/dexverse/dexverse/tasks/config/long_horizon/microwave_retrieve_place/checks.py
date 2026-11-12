# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success checks for the Microwave Retrieve Place task.

These decide whether a demonstration earns its success label: the fridge was opened
and shut again, the microwave opened, the dish of food set down upright on the cavity
floor, the door shut on it, and a dial turned. The finished state must still be true
when success is declared, so a dish that was only held inside for a moment and taken
out again does not count. They stay free of Isaac Lab imports so they can run in CPU
tests.

Geometry was measured in the real scene with the microwave at its spawn pose: the
cavity spans x 0.147-0.439, y -0.257-0.081 and z 0.6515-0.886 in the world, and the
dish (root at the centre of its base, see ``dish.py``) rests on its floor at z 0.6515. Both
doors start closed and swing about vertical hinges. A door 18 degrees ajar, the old
"shut" threshold, leaves the microwave's free edge 13 cm open.
"""

from __future__ import annotations

import math

import torch

from ..check_utils import announce_once, axis_tilt, ever_held, print_status, rotate_inverse, status_due, track

TASK_LABEL = "Microwave"

FRIDGE_OPEN_RAD = math.radians(45.0)
FRIDGE_SHUT_RAD = math.radians(10.0)
# In the first successful demo the operator slid the dish in through a door opened
# 38.8 degrees; at 45 the stage only counted after they opened it wider with the dish
# already inside. Placing the dish is what shows the door was open enough.
MICROWAVE_OPEN_RAD = math.radians(30.0)
MICROWAVE_SHUT_RAD = math.radians(5.0)
DIAL_ON_RAD = math.radians(18.0)

# Cavity interior in the microwave's root frame (root at world (0.3, -0.15, 0.77)).
# Local -X is the door side.
# Shared with the asset preparation so the support and success check agree.
MICROWAVE_FLOOR_TOP_LOCAL_Z = -0.3386  # unscaled PartNet asset coordinates
CAVITY_MIN = (-0.153, -0.107, MICROWAVE_FLOOR_TOP_LOCAL_Z * 0.35)
CAVITY_MAX = (0.139, 0.231, 0.116)
# The dish base must be this far behind the door plane. Shutting the door is what
# proves the whole dish is in.
PLACE_FRONT_MARGIN = 0.03
PLACE_FLOOR_TOL = 0.02
PLACE_MAX_TILT_RAD = math.radians(30.0)
PLACE_HOLD_S = 0.5

# The dish's local +Z is its up axis. Tipping it this far spills the stew.
FOOD_UP_AXIS = (0.0, 0.0, 1.0)
SPILL_TILT_RAD = math.radians(50.0)
SPILL_HOLD_S = 0.1


def joint_angle(env, asset_name: str, joint_name: str) -> torch.Tensor:
    """How far a joint has turned from its start, in radians, per env."""
    asset = env.scene[asset_name]
    joint_id = asset.find_joints(joint_name)[0][0]
    return torch.abs(asset.data.joint_pos[:, joint_id] - asset.data.default_joint_pos[:, joint_id])


def _dial_angle(env, dial_joints: tuple[str, ...]) -> torch.Tensor:
    """The furthest either dial has turned, per env."""
    return torch.stack([joint_angle(env, "microwave", joint) for joint in dial_joints], dim=-1).max(dim=-1).values


def door_at_least(env, asset_name: str, joint_name: str, angle_rad: float, key: str, message: str) -> torch.Tensor:
    """True per-env while a door is open at least ``angle_rad``."""
    done = joint_angle(env, asset_name, joint_name) >= angle_rad
    announce_once(env, TASK_LABEL, key, done, message)
    return done


def door_at_most(env, asset_name: str, joint_name: str, angle_rad: float, key: str, message: str) -> torch.Tensor:
    """True per-env while a door is open no more than ``angle_rad``."""
    done = joint_angle(env, asset_name, joint_name) <= angle_rad
    announce_once(env, TASK_LABEL, key, done, message)
    return done


def _food_errors(env) -> dict[str, torch.Tensor]:
    """How far the dish is from resting upright in the cavity, per env."""
    food, microwave = env.scene["object"].data, env.scene["microwave"].data
    local = rotate_inverse(microwave.root_quat_w, food.root_pos_w - microwave.root_pos_w)
    low = torch.as_tensor(CAVITY_MIN, device=local.device, dtype=local.dtype).clone()
    high = torch.as_tensor(CAVITY_MAX, device=local.device, dtype=local.dtype)
    low[0] += PLACE_FRONT_MARGIN
    xy = local[:, :2]
    outside = (torch.clamp(low[:2] - xy, min=0.0) + torch.clamp(xy - high[:2], min=0.0)).norm(dim=-1)
    return {
        "outside": outside,
        "floor": torch.abs(local[:, 2] - CAVITY_MIN[2]),
        "tilt": axis_tilt(food.root_quat_w, FOOD_UP_AXIS),
    }


def food_in_microwave(env) -> torch.Tensor:
    """True per-env while the dish rests upright on the cavity floor."""
    errors = _food_errors(env)
    return (errors["outside"] <= 0.0) & (errors["floor"] <= PLACE_FLOOR_TOL) & (errors["tilt"] <= PLACE_MAX_TILT_RAD)


def food_placed(env, hold_s: float = PLACE_HOLD_S) -> torch.Tensor:
    """True once the dish has rested in the cavity for ``hold_s``."""
    placed, _ = track(env, "place", food_in_microwave(env), hold_s)
    announce_once(env, TASK_LABEL, "place", placed, "Food placed. Next: shut the microwave.")
    return placed


def food_spilled(env) -> torch.Tensor:
    """True once the dish has been tipped over, at any point in the attempt."""
    tipped = axis_tilt(env.scene["object"].data.root_quat_w, FOOD_UP_AXIS) >= SPILL_TILT_RAD
    _, spilled = track(env, "spill", tipped, SPILL_HOLD_S)
    message = "The food tipped over, so this attempt cannot succeed. Reset."
    announce_once(env, TASK_LABEL, "spill", spilled, message)
    return spilled


def dial_turned(env, dial_joints: tuple[str, ...]) -> torch.Tensor:
    """True per-env while either dial is turned at least ``DIAL_ON_RAD`` either way."""
    return _dial_angle(env, dial_joints) >= DIAL_ON_RAD


def microwave_running(env, door_joint: str, dial_joints: tuple[str, ...]) -> torch.Tensor:
    """True per-env while the door is shut and a dial is turned: the microwave is on."""
    return (joint_angle(env, "microwave", door_joint) <= MICROWAVE_SHUT_RAD) & dial_turned(env, dial_joints)


def finished(env, door_joint: str, dial_joints: tuple[str, ...]) -> torch.Tensor:
    """True per-env while the dish is in the shut, running microwave and was never spilled."""
    return food_in_microwave(env) & microwave_running(env, door_joint, dial_joints) & ~food_spilled(env)


def report_status(
    env,
    waiting: torch.Tensor,
    stage_flags: dict[str, torch.Tensor],
    fridge_joint: str,
    door_joint: str,
    dial_joints: tuple[str, ...],
) -> None:
    """Tell the teleoperator what the current stage, or the finished state, is missing."""
    if env.num_envs != 1 or not bool(waiting[0]) or not status_due(env):
        return
    if bool(ever_held(env, "spill", waiting)[0]):
        print_status(env, TASK_LABEL, ["the food tipped over, so this attempt cannot succeed, reset"])
        return
    fridge = math.degrees(float(joint_angle(env, "fridge", fridge_joint)[0]))
    door = math.degrees(float(joint_angle(env, "microwave", door_joint)[0]))
    dial = math.degrees(float(_dial_angle(env, dial_joints)[0]))
    food = {name: float(value[0]) for name, value in _food_errors(env).items()}
    food_missing = []
    if food["outside"] > 0.0:
        food_missing.append(f"food {100 * food['outside']:.1f} cm from inside the microwave")
    if food["floor"] > PLACE_FLOOR_TOL:
        food_missing.append(f"food {100 * food['floor']:.1f} cm off the microwave floor")
    shut, on, max_tilt = math.degrees(MICROWAVE_SHUT_RAD), math.degrees(DIAL_ON_RAD), math.degrees(PLACE_MAX_TILT_RAD)
    if food["tilt"] > PLACE_MAX_TILT_RAD:
        food_missing.append(f"food tilted {math.degrees(food['tilt']):.0f} deg, max {max_tilt:.0f}")
    pending = [name for name, flag in stage_flags.items() if not bool(flag[0])]
    stage = pending[0] if pending else None
    if stage == "open_fridge":
        missing = [f"fridge door at {fridge:.0f} deg, open it to {math.degrees(FRIDGE_OPEN_RAD):.0f}"]
    elif stage == "close_fridge":
        fridge_shut = math.degrees(FRIDGE_SHUT_RAD)
        missing = [f"fridge door at {fridge:.0f} deg, take the food out and shut it to {fridge_shut:.0f}"]
    elif stage == "open":
        missing = [f"microwave door at {door:.0f} deg, open it to {math.degrees(MICROWAVE_OPEN_RAD):.0f}"]
    elif stage == "place":
        missing = food_missing or ["food in place, hold still"]
    elif stage == "close":
        missing = [f"microwave door at {door:.0f} deg, shut it to {shut:.0f}"]
    else:
        # Turning the dial, or every stage done but the finished state lost again.
        missing = list(food_missing)
        if door > shut:
            missing.append(f"microwave door at {door:.0f} deg, shut it to {shut:.0f}")
        if dial < on:
            missing.append(f"dials at {dial:.0f} deg, turn either one to {on:.0f}")
    print_status(env, TASK_LABEL, missing)
