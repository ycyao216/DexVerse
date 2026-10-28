# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Floating Wuji hand variants and tabletop helpers."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from dexverse.devices.wrist_origin import compute_wrist_joint_origin

from .. import TabletopRobotSetup, dex_retargeting_hand_spec

_ASSET_DIR = Path(__file__).resolve().parent
_RETARGET_DIR = _ASSET_DIR / "retarget"

_WUJI_FINGER_SUFFIXES = tuple(
    f"finger{finger_id}_joint{joint_id}"
    for finger_id in range(1, 6)
    for joint_id in range(1, 5)
)
_WUJI_RIGHT_FINGER_JOINT_NAMES = tuple(f"right_{suffix}" for suffix in _WUJI_FINGER_SUFFIXES)
_WUJI_LEFT_FINGER_JOINT_NAMES = tuple(f"left_{suffix}" for suffix in _WUJI_FINGER_SUFFIXES)

# Wuji and Sharpa are authored with palm normal +X and fingers +Z. Rotate the
# articulation base by +90 degrees about +Y so home becomes palm -Z, fingers +X.
_WUJI_HOME_BASE_ROT = (0.7071067811865476, 0.0, 0.7071067811865475, 0.0)
_WUJI_HOME_TRANSLATION = (-0.3, 0.0, 0.5)

# Official Wuji finger actuator settings from wujihand/body/mjcf/{right,left}.xml and body/urdf.
_WUJI_FINGER_EFFORT_BY_EXPR = {
    ".*_finger1_joint1": 10.0,
    ".*_finger1_joint2": 10.0,
    ".*_finger1_joint3": 10.0,
    ".*_finger1_joint4": 10.0,
    ".*_finger2_joint1": 10.0,
    ".*_finger2_joint2": 10.0,
    ".*_finger2_joint3": 10.0,
    ".*_finger2_joint4": 10.0,
    ".*_finger3_joint1": 10.0,
    ".*_finger3_joint2": 10.0,
    ".*_finger3_joint3": 10.0,
    ".*_finger3_joint4": 10.0,
    ".*_finger4_joint1": 10.0,
    ".*_finger4_joint2": 10.0,
    ".*_finger4_joint3": 10.0,
    ".*_finger4_joint4": 10.0,
    ".*_finger5_joint1": 10.0,
    ".*_finger5_joint2": 10.0,
    ".*_finger5_joint3": 10.0,
    ".*_finger5_joint4": 10.0,
}
_WUJI_FINGER_STIFFNESS_BY_EXPR = {
    ".*_finger1_joint1": 20.0,
    ".*_finger1_joint2": 20.0,
    ".*_finger1_joint3": 20.0,
    ".*_finger1_joint4": 20.0,
    ".*_finger2_joint1": 20.0,
    ".*_finger2_joint2": 20.0,
    ".*_finger2_joint3": 20.0,
    ".*_finger2_joint4": 20.0,
    ".*_finger3_joint1": 20.0,
    ".*_finger3_joint2": 20.0,
    ".*_finger3_joint3": 20.0,
    ".*_finger3_joint4": 20.0,
    ".*_finger4_joint1": 20.0,
    ".*_finger4_joint2": 20.0,
    ".*_finger4_joint3": 20.0,
    ".*_finger4_joint4": 20.0,
    ".*_finger5_joint1": 20.0,
    ".*_finger5_joint2": 20.0,
    ".*_finger5_joint3": 20.0,
    ".*_finger5_joint4": 20.0,
}
_WUJI_FINGER_DAMPING_BY_EXPR = {
    ".*_finger1_joint1": 0.5,
    ".*_finger1_joint2": 0.5,
    ".*_finger1_joint3": 0.5,
    ".*_finger1_joint4": 0.5,
    ".*_finger2_joint1": 0.5,
    ".*_finger2_joint2": 0.5,
    ".*_finger2_joint3": 0.5,
    ".*_finger2_joint4": 0.5,
    ".*_finger3_joint1": 0.5,
    ".*_finger3_joint2": 0.5,
    ".*_finger3_joint3": 0.5,
    ".*_finger3_joint4": 0.5,
    ".*_finger4_joint1": 0.5,
    ".*_finger4_joint2": 0.5,
    ".*_finger4_joint3": 0.5,
    ".*_finger4_joint4": 0.5,
    ".*_finger5_joint1": 0.5,
    ".*_finger5_joint2": 0.5,
    ".*_finger5_joint3": 0.5,
    ".*_finger5_joint4": 0.5,
}
_WUJI_FINGER_VELOCITY_BY_EXPR = {
    ".*_finger1_joint1": 8.587,
    ".*_finger1_joint2": 11.1,
    ".*_finger1_joint3": 12.863,
    ".*_finger1_joint4": 13.578,
    ".*_finger2_joint1": 8.203,
    ".*_finger2_joint2": 8.115,
    ".*_finger2_joint3": 12.863,
    ".*_finger2_joint4": 13.578,
    ".*_finger3_joint1": 8.203,
    ".*_finger3_joint2": 8.115,
    ".*_finger3_joint3": 12.863,
    ".*_finger3_joint4": 13.578,
    ".*_finger4_joint1": 8.203,
    ".*_finger4_joint2": 8.115,
    ".*_finger4_joint3": 12.863,
    ".*_finger4_joint4": 13.578,
    ".*_finger5_joint1": 8.203,
    ".*_finger5_joint2": 8.115,
    ".*_finger5_joint3": 12.863,
    ".*_finger5_joint4": 13.578,
}

_WUJI_RIGHT_RETARGET_URDF_PATH = (
    _RETARGET_DIR / "wuji_hand_right" / "wuji_hand_right_dexrot.urdf"
)
_WUJI_LEFT_RETARGET_URDF_PATH = (
    _RETARGET_DIR / "wuji_hand_left" / "wuji_hand_left_dexrot.urdf"
)

FLOATING_WUJI_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {"right": dex_retargeting_hand_spec(_RETARGET_DIR, "right", _WUJI_RIGHT_RETARGET_URDF_PATH)}
}

FLOATING_WUJI_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {"left": dex_retargeting_hand_spec(_RETARGET_DIR, "left", _WUJI_LEFT_RETARGET_URDF_PATH)}
}

FLOATING_WUJI_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "right": FLOATING_WUJI_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["right"],
        "left": FLOATING_WUJI_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["left"],
    }
}

FLOATING_WUJI_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 26,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "wrist_base_rot": _WUJI_HOME_BASE_ROT,
            "finger_indices": tuple(range(6, 26)),
            "finger_joint_names": _WUJI_RIGHT_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(20)),
        }
    },
}

FLOATING_WUJI_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 26,
    "hands": {
        "left": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "wrist_base_rot": _WUJI_HOME_BASE_ROT,
            "finger_indices": tuple(range(6, 26)),
            "finger_joint_names": _WUJI_LEFT_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(20)),
        }
    },
}

FLOATING_WUJI_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 52,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "wrist_base_rot": _WUJI_HOME_BASE_ROT,
            "finger_indices": tuple(range(12, 32)),
            "finger_joint_names": _WUJI_RIGHT_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(20)),
        },
        "left": {
            "wrist_trans_indices": (6, 7, 8),
            "wrist_rot_indices": (9, 10, 11),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "wrist_base_rot": _WUJI_HOME_BASE_ROT,
            "finger_indices": tuple(range(32, 52)),
            "finger_joint_names": _WUJI_LEFT_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(20)),
        },
    },
}

FLOATING_WUJI_RIGHT_PALM_BODY_NAME = "right_palm_link"
FLOATING_WUJI_RIGHT_FINGERTIP_BODY_NAMES = [f"right_finger{i}_tip_link" for i in range(1, 6)]
FLOATING_WUJI_RIGHT_HAND_TIPS_BODY_NAMES = [
    FLOATING_WUJI_RIGHT_PALM_BODY_NAME,
    *FLOATING_WUJI_RIGHT_FINGERTIP_BODY_NAMES,
]
FLOATING_WUJI_RIGHT_WRIST_JOINT_NAME = "right_(roll|pitch|yaw)_joint"
FLOATING_WUJI_RIGHT_ARM_JOINT_NAMES_EXPR = ["right_(x|y|z)_joint"]
FLOATING_WUJI_RIGHT_BASE_POS = (-0.75, 0.0, 0.5)
FLOATING_WUJI_RIGHT_BASE_ROT = _WUJI_HOME_BASE_ROT
FLOATING_WUJI_RIGHT_WRIST_POSITION_OFFSET = (-0.25, 0.0, 0.8)

FLOATING_WUJI_LEFT_PALM_BODY_NAME = "left_palm_link"
FLOATING_WUJI_LEFT_FINGERTIP_BODY_NAMES = [f"left_finger{i}_tip_link" for i in range(1, 6)]
FLOATING_WUJI_LEFT_HAND_TIPS_BODY_NAMES = [
    FLOATING_WUJI_LEFT_PALM_BODY_NAME,
    *FLOATING_WUJI_LEFT_FINGERTIP_BODY_NAMES,
]
FLOATING_WUJI_LEFT_WRIST_JOINT_NAME = "left_(roll|pitch|yaw)_joint"
FLOATING_WUJI_LEFT_ARM_JOINT_NAMES_EXPR = ["left_(x|y|z)_joint"]
FLOATING_WUJI_LEFT_BASE_POS = FLOATING_WUJI_RIGHT_BASE_POS
FLOATING_WUJI_LEFT_BASE_ROT = FLOATING_WUJI_RIGHT_BASE_ROT
FLOATING_WUJI_LEFT_WRIST_POSITION_OFFSET = FLOATING_WUJI_RIGHT_WRIST_POSITION_OFFSET

FLOATING_WUJI_BIMANUAL_RIGHT_PALM_BODY_NAME = FLOATING_WUJI_RIGHT_PALM_BODY_NAME
FLOATING_WUJI_BIMANUAL_LEFT_PALM_BODY_NAME = FLOATING_WUJI_LEFT_PALM_BODY_NAME
FLOATING_WUJI_BIMANUAL_PALM_BODY_NAME = FLOATING_WUJI_BIMANUAL_RIGHT_PALM_BODY_NAME
FLOATING_WUJI_BIMANUAL_FINGERTIP_BODY_NAMES = [
    *FLOATING_WUJI_RIGHT_FINGERTIP_BODY_NAMES,
    *FLOATING_WUJI_LEFT_FINGERTIP_BODY_NAMES,
]
FLOATING_WUJI_BIMANUAL_HAND_TIPS_BODY_NAMES = [
    FLOATING_WUJI_BIMANUAL_RIGHT_PALM_BODY_NAME,
    FLOATING_WUJI_BIMANUAL_LEFT_PALM_BODY_NAME,
    *FLOATING_WUJI_BIMANUAL_FINGERTIP_BODY_NAMES,
]
FLOATING_WUJI_BIMANUAL_ARM_JOINT_NAMES_EXPR = ["(left|right)_(x|y|z)_joint"]
FLOATING_WUJI_BIMANUAL_BASE_POS = FLOATING_WUJI_RIGHT_BASE_POS
FLOATING_WUJI_BIMANUAL_BASE_ROT = FLOATING_WUJI_RIGHT_BASE_ROT

FLOATING_WUJI_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((_ASSET_DIR / "floating_wuji_hand_right" / "floating_wuji_hand_right.usd").resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.2, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "right_x_joint": _WUJI_HOME_TRANSLATION[0],
            "right_y_joint": _WUJI_HOME_TRANSLATION[1],
            "right_z_joint": _WUJI_HOME_TRANSLATION[2],
            "right_roll_joint": 0.0,
            "right_pitch_joint": 0.0,
            "right_yaw_joint": 0.0,
            "right_finger1_joint1": 0.0475,
            "right_finger1_joint(2|3|4)": 0.0,
            "right_finger(2|3|4|5)_joint(1|2|3|4)": 0.0,
        },
    ),
    actuators={
        "floating_wuji_right_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_(x|y|z)_joint",
                "right_(roll|pitch|yaw)_joint",
                "right_finger(1|2|3|4|5)_joint(1|2|3|4)",
            ],
            effort_limit_sim={
                "right_(x|y|z)_joint": 15.0,
                "right_(roll|pitch|yaw)_joint": 15.0,
                **_WUJI_FINGER_EFFORT_BY_EXPR,
            },
            stiffness={
                "right_(x|y|z)_joint": 2000.0,
                "right_(roll|pitch|yaw)_joint": 2000.0,
                **_WUJI_FINGER_STIFFNESS_BY_EXPR,
            },
            damping={
                "right_(x|y|z)_joint": 400.0,
                "right_(roll|pitch|yaw)_joint": 400.0,
                **_WUJI_FINGER_DAMPING_BY_EXPR,
            },
            velocity_limit_sim={
                "right_(x|y|z)_joint": 10.0,
                "right_(roll|pitch|yaw)_joint": 10.0,
                **_WUJI_FINGER_VELOCITY_BY_EXPR,
            },
            friction={
                "right_(x|y|z)_joint": 0.01,
                "right_(roll|pitch|yaw)_joint": 0.01,
                ".*_finger(1|2|3|4|5)_joint(1|2|3|4)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_WUJI_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((_ASSET_DIR / "floating_wuji_hand_left" / "configuration" / "floating_wuji_hand_left_physics.usd").resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.2, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_x_joint": _WUJI_HOME_TRANSLATION[0],
            "left_y_joint": _WUJI_HOME_TRANSLATION[1],
            "left_z_joint": _WUJI_HOME_TRANSLATION[2],
            "left_roll_joint": 0.0,
            "left_pitch_joint": 0.0,
            "left_yaw_joint": 0.0,
            "left_finger1_joint1": 0.0475,
            "left_finger1_joint(2|3|4)": 0.0,
            "left_finger(2|3|4|5)_joint(1|2|3|4)": 0.0,
        },
    ),
    actuators={
        "floating_wuji_left_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_(x|y|z)_joint",
                "left_(roll|pitch|yaw)_joint",
                "left_finger(1|2|3|4|5)_joint(1|2|3|4)",
            ],
            effort_limit_sim={
                "left_(x|y|z)_joint": 15.0,
                "left_(roll|pitch|yaw)_joint": 15.0,
                **_WUJI_FINGER_EFFORT_BY_EXPR,
            },
            stiffness={
                "left_(x|y|z)_joint": 2000.0,
                "left_(roll|pitch|yaw)_joint": 2000.0,
                **_WUJI_FINGER_STIFFNESS_BY_EXPR,
            },
            damping={
                "left_(x|y|z)_joint": 400.0,
                "left_(roll|pitch|yaw)_joint": 400.0,
                **_WUJI_FINGER_DAMPING_BY_EXPR,
            },
            velocity_limit_sim={
                "left_(x|y|z)_joint": 10.0,
                "left_(roll|pitch|yaw)_joint": 10.0,
                **_WUJI_FINGER_VELOCITY_BY_EXPR,
            },
            friction={
                "left_(x|y|z)_joint": 0.01,
                "left_(roll|pitch|yaw)_joint": 0.01,
                ".*_finger(1|2|3|4|5)_joint(1|2|3|4)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_WUJI_BIMANUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((_ASSET_DIR / "floating_wuji_hand_bimanual" / "configuration" / "floating_wuji_hand_bimanual_physics.usd").resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.2, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "right_x_joint": _WUJI_HOME_TRANSLATION[0],
            "right_y_joint": _WUJI_HOME_TRANSLATION[1],
            "right_z_joint": _WUJI_HOME_TRANSLATION[2],
            "right_roll_joint": 0.0,
            "right_pitch_joint": 0.0,
            "right_yaw_joint": 0.0,
            "left_x_joint": _WUJI_HOME_TRANSLATION[0],
            "left_y_joint": _WUJI_HOME_TRANSLATION[1],
            "left_z_joint": _WUJI_HOME_TRANSLATION[2],
            "left_roll_joint": 0.0,
            "left_pitch_joint": 0.0,
            "left_yaw_joint": 0.0,
            "(left|right)_finger1_joint1": 0.0475,
            "(left|right)_finger1_joint(2|3|4)": 0.0,
            "(left|right)_finger(2|3|4|5)_joint(1|2|3|4)": 0.0,
        },
    ),
    actuators={
        "floating_wuji_bimanual_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "(left|right)_(x|y|z)_joint",
                "(left|right)_(roll|pitch|yaw)_joint",
                "(left|right)_finger(1|2|3|4|5)_joint(1|2|3|4)",
            ],
            effort_limit_sim={
                "(left|right)_(x|y|z)_joint": 15.0,
                "(left|right)_(roll|pitch|yaw)_joint": 15.0,
                **_WUJI_FINGER_EFFORT_BY_EXPR,
            },
            stiffness={
                "(left|right)_(x|y|z)_joint": 2000.0,
                "(left|right)_(roll|pitch|yaw)_joint": 2000.0,
                **_WUJI_FINGER_STIFFNESS_BY_EXPR,
            },
            damping={
                "(left|right)_(x|y|z)_joint": 400.0,
                "(left|right)_(roll|pitch|yaw)_joint": 400.0,
                **_WUJI_FINGER_DAMPING_BY_EXPR,
            },
            velocity_limit_sim={
                "(left|right)_(x|y|z)_joint": 10.0,
                "(left|right)_(roll|pitch|yaw)_joint": 10.0,
                **_WUJI_FINGER_VELOCITY_BY_EXPR,
            },
            friction={
                "(left|right)_(x|y|z)_joint": 0.01,
                "(left|right)_(roll|pitch|yaw)_joint": 0.01,
                ".*_finger(1|2|3|4|5)_joint(1|2|3|4)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_WUJI_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "right": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_WUJI_RIGHT_CFG,
        translation_joint_names=("right_x_joint", "right_y_joint", "right_z_joint"),
        rotation_joint_names=("right_roll_joint", "right_pitch_joint", "right_yaw_joint"),
        rotation_axes="XYZ",
        base_pos=FLOATING_WUJI_RIGHT_BASE_POS,
        base_rot=FLOATING_WUJI_RIGHT_BASE_ROT,
    )
}
FLOATING_WUJI_LEFT_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "left": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_WUJI_LEFT_CFG,
        translation_joint_names=("left_x_joint", "left_y_joint", "left_z_joint"),
        rotation_joint_names=("left_roll_joint", "left_pitch_joint", "left_yaw_joint"),
        rotation_axes="XYZ",
        base_pos=FLOATING_WUJI_LEFT_BASE_POS,
        base_rot=FLOATING_WUJI_LEFT_BASE_ROT,
    )
}

FLOATING_WUJI_BIMANUAL_HAND_MOUNT_OFFSET = {
    "right": (0.0, -0.3, 0.0),
    "left": (0.0, 0.3, 0.0),
}
FLOATING_WUJI_BIMANUAL_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "right": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_WUJI_BIMANUAL_CFG,
        translation_joint_names=("right_x_joint", "right_y_joint", "right_z_joint"),
        rotation_joint_names=("right_roll_joint", "right_pitch_joint", "right_yaw_joint"),
        rotation_axes="XYZ",
        base_pos=FLOATING_WUJI_BIMANUAL_BASE_POS,
        base_rot=FLOATING_WUJI_BIMANUAL_BASE_ROT,
        mount_offset=FLOATING_WUJI_BIMANUAL_HAND_MOUNT_OFFSET["right"],
    ),
    "left": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_WUJI_BIMANUAL_CFG,
        translation_joint_names=("left_x_joint", "left_y_joint", "left_z_joint"),
        rotation_joint_names=("left_roll_joint", "left_pitch_joint", "left_yaw_joint"),
        rotation_axes="XYZ",
        base_pos=FLOATING_WUJI_BIMANUAL_BASE_POS,
        base_rot=FLOATING_WUJI_BIMANUAL_BASE_ROT,
        mount_offset=FLOATING_WUJI_BIMANUAL_HAND_MOUNT_OFFSET["left"],
    ),
}


@configclass
class FloatingWujiRightAbsJointPosActionCfg:
    translation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_(x|y|z)_joint"],
        scale=1.0,
    )
    rotation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_yaw_joint", "right_pitch_joint", "right_roll_joint"],
        scale=1.0,
        preserve_order=True,
    )
    finger_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_WUJI_RIGHT_FINGER_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
    )


@configclass
class FloatingWujiLeftAbsJointPosActionCfg:
    translation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_(x|y|z)_joint"],
        scale=1.0,
    )
    rotation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_yaw_joint", "left_pitch_joint", "left_roll_joint"],
        scale=1.0,
        preserve_order=True,
    )
    finger_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_WUJI_LEFT_FINGER_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
    )


@configclass
class FloatingWujiBimanualAbsJointPosActionCfg:
    right_wrist = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_x_joint",
            "right_y_joint",
            "right_z_joint",
            "right_yaw_joint",
            "right_pitch_joint",
            "right_roll_joint",
        ],
        scale=1.0,
        preserve_order=True,
    )
    left_wrist = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_x_joint",
            "left_y_joint",
            "left_z_joint",
            "left_yaw_joint",
            "left_pitch_joint",
            "left_roll_joint",
        ],
        scale=1.0,
        preserve_order=True,
    )
    right_fingers = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_WUJI_RIGHT_FINGER_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
    )
    left_fingers = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_WUJI_LEFT_FINGER_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
    )


def build_tabletop_floating_wuji_right_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating right Wuji hand."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_WUJI_RIGHT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_WUJI_RIGHT_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_WUJI_RIGHT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_WUJI_RIGHT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_WUJI_RIGHT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_WUJI_RIGHT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_WUJI_RIGHT_CFG.init_state.replace(
                pos=FLOATING_WUJI_RIGHT_BASE_POS,
                rot=FLOATING_WUJI_RIGHT_BASE_ROT,
            ),
        ),
        actions=FloatingWujiRightAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": list(_WUJI_RIGHT_FINGER_JOINT_NAMES),
            "wrist_position_offset": FLOATING_WUJI_RIGHT_WRIST_POSITION_OFFSET,
            "retargeter_config_filename": "floating_wuji_right",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


def build_tabletop_floating_wuji_left_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating left Wuji hand."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_WUJI_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_WUJI_LEFT_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_WUJI_LEFT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_WUJI_LEFT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_WUJI_LEFT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_WUJI_LEFT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_WUJI_LEFT_CFG.init_state.replace(
                pos=FLOATING_WUJI_LEFT_BASE_POS,
                rot=FLOATING_WUJI_LEFT_BASE_ROT,
            ),
        ),
        actions=FloatingWujiLeftAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": list(_WUJI_LEFT_FINGER_JOINT_NAMES),
            "wrist_position_offset": FLOATING_WUJI_LEFT_WRIST_POSITION_OFFSET,
            "retargeter_config_filename": "floating_wuji_left",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


def build_tabletop_floating_wuji_bimanual_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for floating bimanual Wuji hands."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_WUJI_BIMANUAL_PALM_BODY_NAME,
            "right_palm_body_name": FLOATING_WUJI_BIMANUAL_RIGHT_PALM_BODY_NAME,
            "left_palm_body_name": FLOATING_WUJI_BIMANUAL_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_WUJI_BIMANUAL_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_WUJI_BIMANUAL_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": None,
            "arm_joint_names_expr": FLOATING_WUJI_BIMANUAL_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_WUJI_BIMANUAL_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_WUJI_BIMANUAL_CFG.init_state.replace(
                pos=FLOATING_WUJI_BIMANUAL_BASE_POS,
                rot=FLOATING_WUJI_BIMANUAL_BASE_ROT,
            ),
        ),
        actions=FloatingWujiBimanualAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": [],
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "floating_wuji_bimanual",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


__all__ = [
    "FLOATING_WUJI_RIGHT_CFG",
    "FLOATING_WUJI_LEFT_CFG",
    "FLOATING_WUJI_BIMANUAL_CFG",
    "FLOATING_WUJI_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_WUJI_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_WUJI_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_WUJI_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_WUJI_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_WUJI_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_WUJI_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "FLOATING_WUJI_LEFT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "FLOATING_WUJI_BIMANUAL_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "FLOATING_WUJI_BIMANUAL_HAND_MOUNT_OFFSET",
    "FloatingWujiRightAbsJointPosActionCfg",
    "FloatingWujiLeftAbsJointPosActionCfg",
    "FloatingWujiBimanualAbsJointPosActionCfg",
    "build_tabletop_floating_wuji_right_setup",
    "build_tabletop_floating_wuji_left_setup",
    "build_tabletop_floating_wuji_bimanual_setup",
]
