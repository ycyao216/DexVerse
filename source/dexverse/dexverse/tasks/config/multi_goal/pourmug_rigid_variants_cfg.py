# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pour-mug base with additional swappable rigid-object variants."""

from dexverse.tasks.config.functional import pour_can_cfg as pickup_can_cfg
from dexverse.tasks.config.functional import pour_mug_cfg as pourmug_cfg
from dexverse.tasks.config.grasping import pick_up_stick_cfg as pickup_stick_cfg
from dexverse.tasks.config.grasping import relocate_sphere_cfg as relocate_cfg
from isaaclab.utils import configclass

from .composition import RigidAddonSpec, apply_rigid_addon

_PICKUP_CAN_OBS = pickup_can_cfg.PickUpCanObservationsCfg()
_PICKUP_STICK_OBS = pickup_stick_cfg.PickUpStickObservationsCfg()
_RELOCATE_OBS = relocate_cfg.RelocateObservationsCfg()

_POURMUG_PLUS_CAN_SPEC = RigidAddonSpec(
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

_POURMUG_PLUS_STICK_SPEC = RigidAddonSpec(
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

_POURMUG_PLUS_SPHERE_SPEC = RigidAddonSpec(
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
class PourMugWithRigidAddonEnvCfg(pourmug_cfg.PourMugEnvCfg):
    """Composable pour-mug env with one extra rigid addon."""

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
class PourMugPickUpCanEnvCfg(PourMugWithRigidAddonEnvCfg):
    rigid_addon_spec: RigidAddonSpec | None = _POURMUG_PLUS_CAN_SPEC
    rigid_obs_source = _PICKUP_CAN_OBS


@configclass
class PourMugPickUpStickEnvCfg(PourMugWithRigidAddonEnvCfg):
    rigid_addon_spec: RigidAddonSpec | None = _POURMUG_PLUS_STICK_SPEC
    rigid_obs_source = _PICKUP_STICK_OBS


@configclass
class PourMugRelocateSphereEnvCfg(PourMugWithRigidAddonEnvCfg):
    rigid_addon_spec: RigidAddonSpec | None = _POURMUG_PLUS_SPHERE_SPEC
    rigid_obs_source = _RELOCATE_OBS
