# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Floating Sharpa Wave hand variants and tabletop helpers."""

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

# Sharpa's authored home frame has the palm normal along +X and fingers along +Z.
# Rotate the articulation base by +90 degrees about +Y so home becomes palm -Z,
# fingers +X. The translation defaults below are expressed back in this rotated
# base frame to keep the wrist at the same world position as the Shadow setup.
_SHARPA_HOME_BASE_ROT = (0.7071067811865476, 0.0, 0.7071067811865475, 0.0)
_SHARPA_HOME_TRANSLATION = (-0.3, 0.0, 0.5)

_SHARPA_RIGHT_FINGER_JOINT_NAMES = (
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
)
_SHARPA_LEFT_FINGER_JOINT_NAMES = tuple(
    joint_name.replace("right_", "left_", 1) for joint_name in _SHARPA_RIGHT_FINGER_JOINT_NAMES
)

_SHARPA_RIGHT_RETARGET_URDF_PATH = (
    _RETARGET_DIR / "sharpa_wave_right" / "sharpa_wave_right_dexrot.urdf"
)
_SHARPA_LEFT_RETARGET_URDF_PATH = (
    _RETARGET_DIR / "sharpa_wave_left" / "sharpa_wave_left_dexrot.urdf"
)

FLOATING_SHARPA_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {"right": dex_retargeting_hand_spec(_RETARGET_DIR, "right", _SHARPA_RIGHT_RETARGET_URDF_PATH)}
}

FLOATING_SHARPA_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {"left": dex_retargeting_hand_spec(_RETARGET_DIR, "left", _SHARPA_LEFT_RETARGET_URDF_PATH)}
}

FLOATING_SHARPA_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "right": FLOATING_SHARPA_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["right"],
        "left": FLOATING_SHARPA_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["left"],
    }
}

FLOATING_SHARPA_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 28,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "wrist_base_rot": _SHARPA_HOME_BASE_ROT,
            "finger_indices": tuple(range(6, 28)),
            "finger_joint_names": _SHARPA_RIGHT_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(22)),
        }
    },
}

FLOATING_SHARPA_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 28,
    "hands": {
        "left": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "wrist_base_rot": _SHARPA_HOME_BASE_ROT,
            "finger_indices": tuple(range(6, 28)),
            "finger_joint_names": _SHARPA_LEFT_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(22)),
        }
    },
}

FLOATING_SHARPA_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 56,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "wrist_base_rot": _SHARPA_HOME_BASE_ROT,
            "finger_indices": tuple(range(12, 34)),
            "finger_joint_names": _SHARPA_RIGHT_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(22)),
        },
        "left": {
            "wrist_trans_indices": (6, 7, 8),
            "wrist_rot_indices": (9, 10, 11),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "wrist_base_rot": _SHARPA_HOME_BASE_ROT,
            "finger_indices": tuple(range(34, 56)),
            "finger_joint_names": _SHARPA_LEFT_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(22)),
        },
    },
}

FLOATING_SHARPA_RIGHT_PALM_BODY_NAME = "right_hand_C_MC"
FLOATING_SHARPA_RIGHT_FINGERTIP_BODY_NAMES = [
    "right_thumb_fingertip",
    "right_index_fingertip",
    "right_middle_fingertip",
    "right_ring_fingertip",
    "right_pinky_fingertip",
]
FLOATING_SHARPA_RIGHT_HAND_TIPS_BODY_NAMES = [
    FLOATING_SHARPA_RIGHT_PALM_BODY_NAME,
    *FLOATING_SHARPA_RIGHT_FINGERTIP_BODY_NAMES,
]
FLOATING_SHARPA_RIGHT_WRIST_JOINT_NAME = "right_(roll|pitch|yaw)_joint"
FLOATING_SHARPA_RIGHT_ARM_JOINT_NAMES_EXPR = ["right_(x|y|z)_joint"]
FLOATING_SHARPA_RIGHT_BASE_POS = (-0.75, 0.0, 0.5)
FLOATING_SHARPA_RIGHT_BASE_ROT = _SHARPA_HOME_BASE_ROT
FLOATING_SHARPA_RIGHT_WRIST_POSITION_OFFSET = (-0.25, 0.0, 0.8)

FLOATING_SHARPA_LEFT_PALM_BODY_NAME = "left_hand_C_MC"
FLOATING_SHARPA_LEFT_FINGERTIP_BODY_NAMES = [
    "left_thumb_fingertip",
    "left_index_fingertip",
    "left_middle_fingertip",
    "left_ring_fingertip",
    "left_pinky_fingertip",
]
FLOATING_SHARPA_LEFT_HAND_TIPS_BODY_NAMES = [
    FLOATING_SHARPA_LEFT_PALM_BODY_NAME,
    *FLOATING_SHARPA_LEFT_FINGERTIP_BODY_NAMES,
]
FLOATING_SHARPA_LEFT_WRIST_JOINT_NAME = "left_(roll|pitch|yaw)_joint"
FLOATING_SHARPA_LEFT_ARM_JOINT_NAMES_EXPR = ["left_(x|y|z)_joint"]
FLOATING_SHARPA_LEFT_BASE_POS = FLOATING_SHARPA_RIGHT_BASE_POS
FLOATING_SHARPA_LEFT_BASE_ROT = FLOATING_SHARPA_RIGHT_BASE_ROT
FLOATING_SHARPA_LEFT_WRIST_POSITION_OFFSET = FLOATING_SHARPA_RIGHT_WRIST_POSITION_OFFSET

FLOATING_SHARPA_BIMANUAL_RIGHT_PALM_BODY_NAME = FLOATING_SHARPA_RIGHT_PALM_BODY_NAME
FLOATING_SHARPA_BIMANUAL_LEFT_PALM_BODY_NAME = FLOATING_SHARPA_LEFT_PALM_BODY_NAME
FLOATING_SHARPA_BIMANUAL_PALM_BODY_NAME = FLOATING_SHARPA_BIMANUAL_RIGHT_PALM_BODY_NAME
FLOATING_SHARPA_BIMANUAL_FINGERTIP_BODY_NAMES = [
    *FLOATING_SHARPA_RIGHT_FINGERTIP_BODY_NAMES,
    *FLOATING_SHARPA_LEFT_FINGERTIP_BODY_NAMES,
]
FLOATING_SHARPA_BIMANUAL_HAND_TIPS_BODY_NAMES = [
    FLOATING_SHARPA_BIMANUAL_RIGHT_PALM_BODY_NAME,
    FLOATING_SHARPA_BIMANUAL_LEFT_PALM_BODY_NAME,
    *FLOATING_SHARPA_BIMANUAL_FINGERTIP_BODY_NAMES,
]
FLOATING_SHARPA_BIMANUAL_ARM_JOINT_NAMES_EXPR = ["(left|right)_(x|y|z)_joint"]
FLOATING_SHARPA_BIMANUAL_BASE_POS = FLOATING_SHARPA_RIGHT_BASE_POS
FLOATING_SHARPA_BIMANUAL_BASE_ROT = FLOATING_SHARPA_RIGHT_BASE_ROT

# Floating-base drive values are aligned with Shadow; hand drives inherit MITmode USD values.
FLOATING_SHARPA_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((_ASSET_DIR / "floating_sharpa_wave_right" / "floating_sharpa_wave_right_mitmode.usda").resolve()),
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
            "right_x_joint": _SHARPA_HOME_TRANSLATION[0],
            "right_y_joint": _SHARPA_HOME_TRANSLATION[1],
            "right_z_joint": _SHARPA_HOME_TRANSLATION[2],
            "right_roll_joint": 0.0,
            "right_pitch_joint": 0.0,
            "right_yaw_joint": 0.0,
        },
    ),
    actuators={
        "floating_sharpa_right_base_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_(x|y|z)_joint",
                "right_(roll|pitch|yaw)_joint",
            ],
            effort_limit_sim={
                "right_(x|y|z)_joint": 15.0,
                "right_(roll|pitch|yaw)_joint": 15.0,
            },
            stiffness={
                "right_(x|y|z)_joint": 2000.0,
                "right_(roll|pitch|yaw)_joint": 2000.0,
            },
            damping={
                "right_(x|y|z)_joint": 400.0,
                "right_(roll|pitch|yaw)_joint": 400.0,
            },
            velocity_limit_sim={
                "right_(x|y|z)_joint": 10.0,
                "right_(roll|pitch|yaw)_joint": 10.0,
            },
            friction={
                "right_(x|y|z)_joint": 0.01,
                "right_(roll|pitch|yaw)_joint": 0.01,
            },
        ),
        "floating_sharpa_right_hand_actuators": ImplicitActuatorCfg(
            joint_names_expr=list(_SHARPA_RIGHT_FINGER_JOINT_NAMES),
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
            armature=None,
            friction=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_SHARPA_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((_ASSET_DIR / "floating_sharpa_wave_left" / "floating_sharpa_wave_left_mitmode.usda").resolve()),
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
            "left_x_joint": _SHARPA_HOME_TRANSLATION[0],
            "left_y_joint": _SHARPA_HOME_TRANSLATION[1],
            "left_z_joint": _SHARPA_HOME_TRANSLATION[2],
            "left_roll_joint": 0.0,
            "left_pitch_joint": 0.0,
            "left_yaw_joint": 0.0,
        },
    ),
    actuators={
        "floating_sharpa_left_base_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_(x|y|z)_joint",
                "left_(roll|pitch|yaw)_joint",
            ],
            effort_limit_sim={
                "left_(x|y|z)_joint": 15.0,
                "left_(roll|pitch|yaw)_joint": 15.0,
            },
            stiffness={
                "left_(x|y|z)_joint": 2000.0,
                "left_(roll|pitch|yaw)_joint": 2000.0,
            },
            damping={
                "left_(x|y|z)_joint": 400.0,
                "left_(roll|pitch|yaw)_joint": 400.0,
            },
            velocity_limit_sim={
                "left_(x|y|z)_joint": 10.0,
                "left_(roll|pitch|yaw)_joint": 10.0,
            },
            friction={
                "left_(x|y|z)_joint": 0.01,
                "left_(roll|pitch|yaw)_joint": 0.01,
            },
        ),
        "floating_sharpa_left_hand_actuators": ImplicitActuatorCfg(
            joint_names_expr=list(_SHARPA_LEFT_FINGER_JOINT_NAMES),
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
            armature=None,
            friction=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_SHARPA_BIMANUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((_ASSET_DIR / "floating_sharpa_wave_bimanual" / "floating_sharpa_wave_bimanual_mitmode.usda").resolve()),
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
            "head_x_joint": 0.0,
            "head_y_joint": 0.0,
            "head_z_joint": 0.0,
            "head_roll_joint": 0.0,
            "head_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
            "right_x_joint": _SHARPA_HOME_TRANSLATION[0],
            "right_y_joint": -0.3,
            "right_z_joint": _SHARPA_HOME_TRANSLATION[2],
            "right_roll_joint": 0.0,
            "right_pitch_joint": 0.0,
            "right_yaw_joint": 0.0,
            "left_x_joint": _SHARPA_HOME_TRANSLATION[0],
            "left_y_joint": 0.3,
            "left_z_joint": _SHARPA_HOME_TRANSLATION[2],
            "left_roll_joint": 0.0,
            "left_pitch_joint": 0.0,
            "left_yaw_joint": 0.0,
        },
    ),
    actuators={
        "floating_sharpa_bimanual_base_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_(x|y|z)_joint",
                "head_(roll|pitch|yaw)_joint",
                "(left|right)_(x|y|z)_joint",
                "(left|right)_(roll|pitch|yaw)_joint",
            ],
            effort_limit_sim={
                "(head|left|right)_(x|y|z)_joint": 15.0,
                "(head|left|right)_(roll|pitch|yaw)_joint": 15.0,
            },
            stiffness={
                "(head|left|right)_(x|y|z)_joint": 2000.0,
                "(head|left|right)_(roll|pitch|yaw)_joint": 2000.0,
            },
            damping={
                "(head|left|right)_(x|y|z)_joint": 400.0,
                "(head|left|right)_(roll|pitch|yaw)_joint": 400.0,
            },
            velocity_limit_sim={
                "(head|left|right)_(x|y|z)_joint": 10.0,
                "(head|left|right)_(roll|pitch|yaw)_joint": 10.0,
            },
            friction={
                "(head|left|right)_(x|y|z)_joint": 0.01,
                "(head|left|right)_(roll|pitch|yaw)_joint": 0.01,
            },
        ),
        "floating_sharpa_bimanual_right_hand_actuators": ImplicitActuatorCfg(
            joint_names_expr=list(_SHARPA_RIGHT_FINGER_JOINT_NAMES),
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
            armature=None,
            friction=None,
        ),
        "floating_sharpa_bimanual_left_hand_actuators": ImplicitActuatorCfg(
            joint_names_expr=list(_SHARPA_LEFT_FINGER_JOINT_NAMES),
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
            armature=None,
            friction=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_SHARPA_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "right": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_SHARPA_RIGHT_CFG,
        translation_joint_names=("right_x_joint", "right_y_joint", "right_z_joint"),
        rotation_joint_names=("right_roll_joint", "right_pitch_joint", "right_yaw_joint"),
        rotation_axes="XYZ",
        base_pos=FLOATING_SHARPA_RIGHT_BASE_POS,
        base_rot=FLOATING_SHARPA_RIGHT_BASE_ROT,
    )
}
FLOATING_SHARPA_LEFT_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "left": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_SHARPA_LEFT_CFG,
        translation_joint_names=("left_x_joint", "left_y_joint", "left_z_joint"),
        rotation_joint_names=("left_roll_joint", "left_pitch_joint", "left_yaw_joint"),
        rotation_axes="XYZ",
        base_pos=FLOATING_SHARPA_LEFT_BASE_POS,
        base_rot=FLOATING_SHARPA_LEFT_BASE_ROT,
    )
}

# Official dual_sharpa_wave mounts the right and left wrist chains at y=+0.15
# and y=-0.15 respectively. The bimanual cfg offsets the y joints by -0.3/+0.3
# so the displayed hand sides follow the Shadow convention: right=-Y, left=+Y.
FLOATING_SHARPA_BIMANUAL_HAND_MOUNT_OFFSET = {
    "right": (0.0, 0.15, 0.0),
    "left": (0.0, -0.15, 0.0),
}
FLOATING_SHARPA_BIMANUAL_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "right": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_SHARPA_BIMANUAL_CFG,
        translation_joint_names=("right_x_joint", "right_y_joint", "right_z_joint"),
        rotation_joint_names=("right_roll_joint", "right_pitch_joint", "right_yaw_joint"),
        rotation_axes="XYZ",
        base_pos=FLOATING_SHARPA_BIMANUAL_BASE_POS,
        base_rot=FLOATING_SHARPA_BIMANUAL_BASE_ROT,
        mount_offset=FLOATING_SHARPA_BIMANUAL_HAND_MOUNT_OFFSET["right"],
    ),
    "left": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_SHARPA_BIMANUAL_CFG,
        translation_joint_names=("left_x_joint", "left_y_joint", "left_z_joint"),
        rotation_joint_names=("left_roll_joint", "left_pitch_joint", "left_yaw_joint"),
        rotation_axes="XYZ",
        base_pos=FLOATING_SHARPA_BIMANUAL_BASE_POS,
        base_rot=FLOATING_SHARPA_BIMANUAL_BASE_ROT,
        mount_offset=FLOATING_SHARPA_BIMANUAL_HAND_MOUNT_OFFSET["left"],
    ),
}


@configclass
class FloatingSharpaRightAbsJointPosActionCfg:
    translation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_x_joint", "right_y_joint", "right_z_joint"],
        scale=1.0,
        preserve_order=True,
    )
    rotation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_yaw_joint", "right_pitch_joint", "right_roll_joint"],
        scale=1.0,
        preserve_order=True,
    )
    finger_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_SHARPA_RIGHT_FINGER_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class FloatingSharpaLeftAbsJointPosActionCfg:
    translation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_x_joint", "left_y_joint", "left_z_joint"],
        scale=1.0,
        preserve_order=True,
    )
    rotation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_yaw_joint", "left_pitch_joint", "left_roll_joint"],
        scale=1.0,
        preserve_order=True,
    )
    finger_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_SHARPA_LEFT_FINGER_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class FloatingSharpaBimanualAbsJointPosActionCfg:
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
        joint_names=list(_SHARPA_RIGHT_FINGER_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )
    left_fingers = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_SHARPA_LEFT_FINGER_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )


def build_tabletop_floating_sharpa_right_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating right Sharpa hand."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_SHARPA_RIGHT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_SHARPA_RIGHT_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_SHARPA_RIGHT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_SHARPA_RIGHT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_SHARPA_RIGHT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_SHARPA_RIGHT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_SHARPA_RIGHT_CFG.init_state.replace(
                pos=FLOATING_SHARPA_RIGHT_BASE_POS,
                rot=FLOATING_SHARPA_RIGHT_BASE_ROT,
            ),
        ),
        actions=FloatingSharpaRightAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": list(_SHARPA_RIGHT_FINGER_JOINT_NAMES),
            "wrist_position_offset": FLOATING_SHARPA_RIGHT_WRIST_POSITION_OFFSET,
            "retargeter_config_filename": "floating_sharpa_right",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


def build_tabletop_floating_sharpa_left_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating left Sharpa hand."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_SHARPA_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_SHARPA_LEFT_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_SHARPA_LEFT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_SHARPA_LEFT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_SHARPA_LEFT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_SHARPA_LEFT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_SHARPA_LEFT_CFG.init_state.replace(
                pos=FLOATING_SHARPA_LEFT_BASE_POS,
                rot=FLOATING_SHARPA_LEFT_BASE_ROT,
            ),
        ),
        actions=FloatingSharpaLeftAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": list(_SHARPA_LEFT_FINGER_JOINT_NAMES),
            "wrist_position_offset": FLOATING_SHARPA_LEFT_WRIST_POSITION_OFFSET,
            "retargeter_config_filename": "floating_sharpa_left",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


def build_tabletop_floating_sharpa_bimanual_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for floating bimanual Sharpa hands."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_SHARPA_BIMANUAL_PALM_BODY_NAME,
            "right_palm_body_name": FLOATING_SHARPA_BIMANUAL_RIGHT_PALM_BODY_NAME,
            "left_palm_body_name": FLOATING_SHARPA_BIMANUAL_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_SHARPA_BIMANUAL_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_SHARPA_BIMANUAL_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": None,
            "arm_joint_names_expr": FLOATING_SHARPA_BIMANUAL_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_SHARPA_BIMANUAL_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_SHARPA_BIMANUAL_CFG.init_state.replace(
                pos=FLOATING_SHARPA_BIMANUAL_BASE_POS,
                rot=FLOATING_SHARPA_BIMANUAL_BASE_ROT,
            ),
        ),
        actions=FloatingSharpaBimanualAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": [],
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "floating_sharpa_bimanual",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


__all__ = [
    "FLOATING_SHARPA_RIGHT_CFG",
    "FLOATING_SHARPA_LEFT_CFG",
    "FLOATING_SHARPA_BIMANUAL_CFG",
    "FLOATING_SHARPA_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_SHARPA_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_SHARPA_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_SHARPA_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_SHARPA_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_SHARPA_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_SHARPA_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "FLOATING_SHARPA_LEFT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "FLOATING_SHARPA_BIMANUAL_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "FLOATING_SHARPA_BIMANUAL_HAND_MOUNT_OFFSET",
    "FloatingSharpaRightAbsJointPosActionCfg",
    "FloatingSharpaLeftAbsJointPosActionCfg",
    "FloatingSharpaBimanualAbsJointPosActionCfg",
    "build_tabletop_floating_sharpa_right_setup",
    "build_tabletop_floating_sharpa_left_setup",
    "build_tabletop_floating_sharpa_bimanual_setup",
]
