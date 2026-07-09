# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift-lid base with swappable rigid-object variants."""

from dexverse.tasks.config.articulation import liftlid_cfg
from dexverse.tasks.config.functional import pour_can_cfg as pickup_can_cfg
from dexverse.tasks.config.functional import pour_mug_cfg as pourmug_cfg
from dexverse.tasks.config.grasping import pick_up_stick_cfg as pickup_stick_cfg
from dexverse.tasks.config.grasping import relocate_sphere_cfg as relocate_cfg
from isaaclab.utils import configclass

from .composition import RigidAddonSpec, apply_rigid_addon

_PICKUP_CAN_OBS = pickup_can_cfg.PickUpCanObservationsCfg()
_PICKUP_STICK_OBS = pickup_stick_cfg.PickUpStickObservationsCfg()
_POUR_MUG_OBS = pourmug_cfg.PourMugObservationsCfg()
_RELOCATE_OBS = relocate_cfg.RelocateObservationsCfg()

_LIFTLID_CAN_SPEC = RigidAddonSpec(
    scene_name="can",
    prim_path="{ENV_REGEX_NS}/Can",
    object_cfg=pickup_can_cfg.CAN_CFG,
    half_height_est=pickup_can_cfg.CAN_HALF_HEIGHT_EST,
    init_rot=pickup_can_cfg.CAN_ROT_INIT,
    reset_event_name="reset_can",
    reset_pose_range={
        "x": [0.0, 0.0],
        "y": [-pickup_can_cfg.CENTER_SQUARE_SIZE * 0.5, pickup_can_cfg.CENTER_SQUARE_SIZE * 0.5],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [0.0, 0.0],
    },
    term_map={
        "can_pos_b": "object_pos_b",
        "can_up_b": "object_up_b",
        "can_tilt_angle": "object_tilt_angle",
        "can_lin_vel_b": "object_lin_vel_b",
        "can_ang_vel_b": "object_ang_vel_b",
    },
    contact_obs_name="contact_can",
)

_LIFTLID_STICK_SPEC = RigidAddonSpec(
    scene_name="stick",
    prim_path="{ENV_REGEX_NS}/Stick",
    object_cfg=pickup_stick_cfg.STICK_CFG,
    half_height_est=pickup_stick_cfg.STICK_HEIGHT_OFFSET,
    init_rot=pickup_stick_cfg.HORIZONTAL_QUAT,
    reset_event_name="reset_stick",
    reset_pose_range={
        "x": [0.0, 0.0],
        "y": [-0.1, 0.1],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [0.0, 0.0],
    },
    term_map={
        "stick_pos_b": "object_pos_b",
        "stick_up_b": "object_up_b",
        "stick_tilt_angle": "object_tilt_angle",
        "stick_lin_vel_b": "object_lin_vel_b",
        "stick_ang_vel_b": "object_ang_vel_b",
    },
    contact_obs_name="contact_stick",
)

_LIFTLID_MUG_SPEC = RigidAddonSpec(
    scene_name="mug",
    prim_path="{ENV_REGEX_NS}/Mug",
    object_cfg=pourmug_cfg.MUG_CFG,
    half_height_est=pourmug_cfg.MUG_Z_OFFSET,
    init_rot=pourmug_cfg.MUG_INIT_QUAT,
    reset_event_name="reset_mug",
    reset_pose_range={
        "x": [0.0, 0.0],
        "y": [-pourmug_cfg.CENTER_SQUARE_SIZE * 0.5, pourmug_cfg.CENTER_SQUARE_SIZE * 0.5],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [0.0, 0.0],
    },
    term_map={
        "mug_pos_b": "object_pos_b",
        "mug_up_b": "object_up_b",
        "mug_tilt_angle": "object_tilt_angle",
        "mug_lin_vel_b": "object_lin_vel_b",
        "mug_ang_vel_b": "object_ang_vel_b",
    },
    contact_obs_name="contact_mug",
)

_LIFTLID_RELOCATE_SPEC = RigidAddonSpec(
    scene_name="sphere",
    prim_path="{ENV_REGEX_NS}/Sphere",
    object_cfg=relocate_cfg.OBJECT_CFG,
    half_height_est=relocate_cfg.SPHERE_RADIUS,
    init_rot=None,
    reset_event_name="reset_sphere",
    reset_pose_range={
        "x": [-0.1, 0.1],
        "y": [-relocate_cfg.CENTER_SQUARE_SIZE * 0.5, relocate_cfg.CENTER_SQUARE_SIZE * 0.5],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [0.0, 0.0],
    },
    term_map={
        "sphere_pos_b": "object_pos_b",
        "sphere_lin_vel_b": "object_lin_vel_b",
        "sphere_ang_vel_b": "object_ang_vel_b",
    },
    contact_obs_name="contact_sphere",
)


@configclass
class LiftLidWithRigidAddonEnvCfg(liftlid_cfg.LiftLidEnvCfg):
    """Composable lift-lid env with one rigid addon."""

    rigid_addon_spec: RigidAddonSpec | None = None
    rigid_obs_source = None

    def __post_init__(self):
        super().__post_init__()
        if self.rigid_addon_spec is None or self.rigid_obs_source is None:
            return
        apply_rigid_addon(
            self,
            spec=self.rigid_addon_spec,
            source_obs_cfg=self.rigid_obs_source,
            old_entity_name="object",
        )


@configclass
class LiftLidPickUpCanEnvCfg(LiftLidWithRigidAddonEnvCfg):
    rigid_addon_spec: RigidAddonSpec | None = _LIFTLID_CAN_SPEC
    rigid_obs_source = _PICKUP_CAN_OBS


@configclass
class LiftLidPickUpStickEnvCfg(LiftLidWithRigidAddonEnvCfg):
    rigid_addon_spec: RigidAddonSpec | None = _LIFTLID_STICK_SPEC
    rigid_obs_source = _PICKUP_STICK_OBS


@configclass
class LiftLidPourMugEnvCfg(LiftLidWithRigidAddonEnvCfg):
    rigid_addon_spec: RigidAddonSpec | None = _LIFTLID_MUG_SPEC
    rigid_obs_source = _POUR_MUG_OBS


@configclass
class LiftLidRelocateSphereEnvCfg(LiftLidWithRigidAddonEnvCfg):
    rigid_addon_spec: RigidAddonSpec | None = _LIFTLID_RELOCATE_SPEC
    rigid_obs_source = _RELOCATE_OBS
