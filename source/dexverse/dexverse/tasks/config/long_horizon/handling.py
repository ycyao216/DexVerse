# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grip and teleop settings shared by the long-horizon tasks.

Long-horizon tasks carry objects for minutes rather than seconds, which exposes
problems short tasks get away with.

Grip: with the default contact material the hand and objects meet at friction 0.5 and
some assets bounce, which operators report as slippery. Friction 1.5 with no bounce is
aimed at picking objects up. Once an object is held it barely matters: on 80 recorded
carrying grasps none dropped when shaken at either value, and slip was about the same.

Hand: the wrist lagged 170-200 ms behind commands, and the finger links are so light
that a wedged finger can send the whole articulation out of control. See
:func:`tune_hand_for_carrying`.

Teleop: hand tracking glitches fling held objects, and an operator's hand keeps going
down after the robot hand has stopped on the table. See :func:`tune_teleop_for_carrying`.
The wrist camera's housing collides but does not show in the headset. See
:func:`tune_wrist_camera_bodies`.
"""

from __future__ import annotations

from copy import deepcopy

import isaaclab.sim as sim_utils
import torch
from dexverse.devices.retargeters import SimpleRelativeRetargeterCfg
from dexverse.devices.retargeters.wrist_filter import WristFilterCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp

GRIP_FRICTION = 1.5

# Wrist translation. The hand carries virtual links, so these joints move 5-7 kg.
# Critically damped at this stiffness, and following commands about 80 ms behind. At
# 15 N the drive saturates and overshoots; at 80 N it jams against contacts.
WRIST_TRANSLATION_STIFFNESS = 4000.0
WRIST_TRANSLATION_DAMPING = 350.0
WRIST_TRANSLATION_EFFORT = 40.0
# Wrist rotation inertia is tiny, so this still overdamps, about 40 ms behind.
WRIST_ROTATION_DAMPING = 100.0

# Stabilizers, chosen by replaying recorded commands with real objects. Without them,
# small changes to a replay wedged a finger and sent joints spinning through their
# limits, in about 1 of 6 runs at the old gains and up to 4 of 6 at stiffer ones. With
# them, none of 66 runs had a body even exceed 8 m/s. Armature gives the light links
# effective inertia, and the extra finger damping cancels the overshoot that adds.
WRIST_ROTATION_ARMATURE = 0.01
FINGER_ARMATURE = 0.002
FINGER_DAMPING = 0.3
MAX_DEPENETRATION_VELOCITY = 5.0

# Recorded sessions never got the wrist closer to the table than this.
MIN_WRIST_HEIGHT_ABOVE_TABLE = 0.04

# Marks showing the operator where a task's success check wants things.
GOAL_MARK_COLOR = (0.1, 0.85, 0.3)
GOAL_MARK_THICKNESS = 0.002


def grip_material_event(asset_name: str) -> EventTerm:
    """Reset event giving every collision shape of an asset grippy, non-bouncy contact.

    PhysX averages the two materials in a contact by default, so both the hand and
    the object it holds need this for their contact to reach ``GRIP_FRICTION``.

    This must run on reset, not at startup: ``record_demos.py`` restarts the physics
    scene on every reset, which silently restores the materials authored in the assets.
    """
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name),
            "static_friction_range": (GRIP_FRICTION, GRIP_FRICTION),
            "dynamic_friction_range": (GRIP_FRICTION, GRIP_FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


def goal_mark_cfg(name: str, radius: float, surface_pos: tuple[float, float, float]) -> AssetBaseCfg:
    """A green disc lying on a surface. Visual only, so nothing touches it.

    Part of the scene like a printed mark, so it also appears in camera observations.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=GOAL_MARK_THICKNESS,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=GOAL_MARK_COLOR, emissive_color=tuple(0.4 * c for c in GOAL_MARK_COLOR)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(surface_pos[0], surface_pos[1], surface_pos[2] + 0.5 * GOAL_MARK_THICKNESS)
        ),
    )


def show_by_state(env, key: str, prims: dict[int, tuple[str, ...]], state: torch.Tensor) -> None:
    """Per env, show the prims listed under its ``state`` and hide the rest.

    For cues the operator needs in the headset, such as liquid in a pot or a burner
    lit. Paths are relative to each env's prim. All start hidden, and only envs whose
    state changed are touched, so this is cheap to call every step.
    """
    from pxr import UsdGeom

    memory = env.__dict__.setdefault("_long_horizon_shown", {})
    shown = memory.get(key, torch.zeros_like(state))
    for env_id in torch.nonzero(state != shown).flatten().tolist():
        for code, paths in prims.items():
            for path in paths:
                prim = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[env_id]}/{path}")
                if prim.IsValid():
                    imageable = UsdGeom.Imageable(prim)
                    imageable.MakeVisible() if code == int(state[env_id]) else imageable.MakeInvisible()
    memory[key] = state.clone()


def tune_hand_for_carrying(env_cfg) -> None:
    """Retune a single-hand floating robot's drives for this task only.

    Call after the base config has set ``scene.robot``. Works on a deep copy, so the
    shared robot config other tasks build from is untouched.
    """
    robot = deepcopy(env_cfg.scene.robot)
    robot.spawn.rigid_props.max_depenetration_velocity = MAX_DEPENETRATION_VELOCITY
    for actuator in robot.actuators.values():
        armature = {}
        for joints in actuator.damping:
            if "translation" in joints:
                actuator.stiffness[joints] = WRIST_TRANSLATION_STIFFNESS
                actuator.damping[joints] = WRIST_TRANSLATION_DAMPING
                actuator.effort_limit_sim[joints] = WRIST_TRANSLATION_EFFORT
                armature[joints] = 0.0
            elif "rotation" in joints:
                actuator.damping[joints] = WRIST_ROTATION_DAMPING
                armature[joints] = WRIST_ROTATION_ARMATURE
            else:
                actuator.damping[joints] = FINGER_DAMPING
                armature[joints] = FINGER_ARMATURE
        actuator.armature = armature
    env_cfg.scene.robot = robot


WRIST_CAMERA_BODY_NAMES = ("wrist_camera_body", "right_wrist_camera_body", "left_wrist_camera_body")


def tune_wrist_camera_bodies(env_cfg) -> None:
    """Stop the wrist cameras' housings from touching anything.

    The base config gives each wrist camera a collidable box housing, about 10 cm behind
    and 7 cm below the palm, tagged so cameras skip it. The headset skips it too, so an
    operator knocks objects over with a part of the hand they cannot see. These tasks
    teleoperate the hand alone, so the housing keeps its viewport visual and loses its
    collider. Works on copies, so the shared config is untouched.
    """
    for name in WRIST_CAMERA_BODY_NAMES:
        body = getattr(env_cfg.scene, name, None)
        if body is None:
            continue
        body = deepcopy(body)
        body.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        setattr(env_cfg.scene, name, body)


def tune_teleop_for_carrying(env_cfg, surface_z: float | None = None) -> None:
    """Filter the tracked wrist and floor its height on a single-hand floating robot.

    Call after the teleop devices are built and the wrist's start pose is final.
    ``surface_z`` is the height of what the hand works on, by default the standard
    table. Replay is unaffected: these act where tracking becomes commands, and
    recordings store the commands.
    """
    robot = env_cfg.scene.robot
    wrist_home_z = robot.init_state.pos[2] + robot.init_state.joint_pos["z_translation_joint"]
    surface_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT if surface_z is None else surface_z
    lowest_wrist_z = surface_z + MIN_WRIST_HEIGHT_ABOVE_TABLE
    for device in env_cfg.teleop_devices.devices.values():
        for retargeter in device.retargeters:
            if isinstance(retargeter, SimpleRelativeRetargeterCfg):
                retargeter.wrist_filter = WristFilterCfg()
                retargeter.continuous_wrist_euler = True
                retargeter.wrist_z_floor = lowest_wrist_z - wrist_home_z
