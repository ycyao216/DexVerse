# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grip and scene helpers shared by the long-horizon tasks.

Hand drives retain the shared floating-hand stiffness, force limits and finger
settings. Wrist damping is reduced to keep tracking responsive in the heavier
long-horizon scenes. Wrist tracking itself uses the shared unfiltered defaults.
Virtual wrist links use the validated 1 g mass-only profile; their inertia
tensors and the physical hand links retain the shared asset properties.

Grip: with the default contact material the hand and objects meet at friction 0.5 and
some assets bounce, which operators report as slippery. Friction 1.5 with no bounce is
aimed at picking objects up. Once an object is held it barely matters: on 80 recorded
carrying grasps none dropped when shaken at either value, and slip was about the same.

The wrist camera's housing collides but does not show in the headset. See
:func:`tune_wrist_camera_bodies`.
"""

from __future__ import annotations

from copy import deepcopy

import isaaclab.sim as sim_utils
import torch
from dexverse.devices.retargeters import SimpleRelativeRetargeterCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from ... import mdp

GRIP_FRICTION = 1.5
WRIST_DAMPING = 200.0
VIRTUAL_WRIST_LINK_MASS = 0.001


def set_virtual_wrist_link_mass(env, env_ids, asset_cfg: SceneEntityCfg, mass: float) -> None:
    """Set selected virtual-link masses without scaling inertia or consuming RNG."""
    robot = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
        raise ValueError("Virtual wrist mass requires explicitly resolved body IDs")
    allowed = {prefix + axis + "_" + kind + "_link"
               for prefix in ("", "rh_", "lh_")
               for kind in ("translation", "rotation") for axis in "xyz"}
    if not asset_cfg.body_ids or any(robot.body_names[i] not in allowed for i in asset_cfg.body_ids):
        raise ValueError("Virtual wrist mass may only target the floating wrist links")
    masses = robot.root_physx_view.get_masses().clone()
    indices = (torch.arange(masses.shape[0], device="cpu") if env_ids is None
               else torch.as_tensor(env_ids, device="cpu", dtype=torch.long))
    body_ids = torch.tensor(asset_cfg.body_ids, device="cpu", dtype=torch.long)
    masses[indices[:, None], body_ids] = mass
    # PhysX's mass setter is independent of inertia/COM, unlike density scaling.
    robot.root_physx_view.set_masses(masses, indices)


def configure_light_virtual_wrist_links(env_cfg) -> None:
    """Apply the same mass-only profile on construction, reset and replay reset_to.

    The recorder restarts physics before resetting the environment, which can
    restore USD masses. A reset event reapplies the profile after that restart.
    Private event configs leave the shared hand assets and short tasks intact.
    """
    if env_cfg.robot_type not in {
        "floating_shadow_right", "floating_shadow_left", "floating_shadow_bimanual",
    }:
        return
    prefixes = ("rh_", "lh_") if env_cfg.robot_type == "floating_shadow_bimanual" else ("",)
    names = [prefix + axis + "_" + kind + "_link"
             for prefix in prefixes for kind in ("translation", "rotation") for axis in "xyz"]
    env_cfg.events = deepcopy(env_cfg.events)
    for mode in ("startup", "reset"):
        setattr(env_cfg.events, "light_virtual_wrist_links_" + mode, EventTerm(
            func=set_virtual_wrist_link_mass, mode=mode,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=names, preserve_order=True),
                    "mass": VIRTUAL_WRIST_LINK_MASS},
        ))


def tune_wrist_damping(env_cfg) -> None:
    """Reduce wrist tracking delay without increasing force or changing fingers.

    Full-observation physics checks compare 400, 200 and 150 damping. At 200,
    a 3 cm command reaches 90% in 233 ms rather than 467 ms; physical wall
    withdrawal overshoots about 1.8 mm versus 13.7 mm at 150. Rotation at 200
    also halves the tested 15-degree step response without overshoot.

    Work on a private robot config so short tasks keep their original drives.
    Physics dt, observation cadence and retargeting are unchanged.
    """
    if env_cfg.robot_type not in {
        "floating_shadow_right", "floating_shadow_left", "floating_shadow_bimanual",
    }:
        return
    robot = deepcopy(env_cfg.scene.robot)
    for actuator in robot.actuators.values():
        actuator.damping = {
            pattern: WRIST_DAMPING if "translation_joint" in pattern or "rotation_joint" in pattern else value
            for pattern, value in actuator.damping.items()
        }
    env_cfg.scene.robot = robot

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


WRIST_CAMERA_BODY_NAMES = ("wrist_camera_body", "right_wrist_camera_body", "left_wrist_camera_body")


def configure_cpu_camera_copies(env_cfg) -> None:
    """Opt these scenes into an exact CPU copy without changing camera settings."""
    from dexverse.cpu_tiled_camera import CpuTiledCamera
    from isaaclab.sensors import TiledCamera, TiledCameraCfg

    for name, camera in vars(env_cfg.scene).items():
        if isinstance(camera, TiledCameraCfg) and camera.class_type is TiledCamera:
            camera = deepcopy(camera)
            camera.class_type = CpuTiledCamera
            setattr(env_cfg.scene, name, camera)


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


def preserve_bimanual_wrist_origins(env_cfg) -> None:
    """Keep both cooking wrist homes when switching relative to absolute control.

    Call after the teleop devices and wrist starting poses are final. This only
    supplies coordinate origins; all motion settings retain their shared defaults.
    """
    from ..robot_init import _compute_retargeter_origin_entry

    origins = _compute_retargeter_origin_entry(env_cfg, "right")
    for device in env_cfg.teleop_devices.devices.values():
        for retargeter in device.retargeters:
            if isinstance(retargeter, SimpleRelativeRetargeterCfg):
                # Relative cfgs do not declare this field. Attach it for the
                # new two-hand setup so swapping to absolute keeps both homes.
                retargeter.wrist_joint_origin = origins
