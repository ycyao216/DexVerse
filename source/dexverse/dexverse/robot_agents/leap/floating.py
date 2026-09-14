# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for floating (virtual-joint) Leap hand robots.

This module consolidates the single-hand (right) and bimanual floating Leap
hand articulations that use virtual prismatic/revolute joints for wrist
translation and rotation.

The following configurations are available:

* :obj:`FLOATING_LEAP_RIGHT_CFG`: Single floating right Leap hand.
* :obj:`BIMANUAL_LEAP_CFG`: Bimanual floating Leap hands (left + right).
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils import configclass

from dexverse.devices.wrist_origin import compute_wrist_joint_origin

from .. import TabletopRobotSetup, dex_retargeting_hand_spec

ASSET_DIR = Path(__file__).resolve().parent / "usd"
_RETARGET_DIR = Path(__file__).resolve().parent / "retarget"


# ---------------------------------------------------------------------------
# Floating right Leap hand
# ---------------------------------------------------------------------------


@configclass
class FloatingLeapRightRelJointPosActionCfg:
    translation_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["(x|y|z)_translation_joint"],
        scale=0.1,
    )
    rotation_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["(x|y|z)_rotation_joint"],
        scale=0.1,
    )
    finger_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["leap_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)"],
        scale=0.1,
    )


@configclass
class FloatingLeapRightAbsJointPosActionCfg:
    """Absolute joint position control for floating Leap teleoperation."""

    translation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["(x|y|z)_translation_joint"],
        scale=1.0,
    )
    rotation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["z_rotation_joint", "y_rotation_joint", "x_rotation_joint"],
        scale=1.0,
        preserve_order=True,
    )
    finger_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "leap_0", "leap_1", "leap_2", "leap_3",
            "leap_4", "leap_5", "leap_6", "leap_7",
            "leap_8", "leap_9", "leap_10", "leap_11",
            "leap_12", "leap_13", "leap_14", "leap_15",
        ],
        scale=1.0,
        preserve_order=True,
    )


FLOATING_LEAP_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 22,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(6, 22)),
            "finger_joint_names": (
                "leap_0", "leap_1", "leap_2", "leap_3",
                "leap_4", "leap_5", "leap_6", "leap_7",
                "leap_8", "leap_9", "leap_10", "leap_11",
                "leap_12", "leap_13", "leap_14", "leap_15",
            ),
            "finger_permutation": tuple(range(16)),
        }
    },
}

FLOATING_LEAP_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "right": dex_retargeting_hand_spec(
            _RETARGET_DIR, "right", _RETARGET_DIR / "leap_hand_right_fixed.urdf"
        )
    },
}

FLOATING_LEAP_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/floating_leap_right/floating_leap_hand.usd",
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
        pos=(-0.2, 0.0, 0.6),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "x_translation_joint": 0.0,
            "y_translation_joint": 0.0,
            "z_translation_joint": 0.3,
            "x_rotation_joint": 0.0,
            "y_rotation_joint": 0.0,
            "z_rotation_joint": 0.0,
            "leap_0": 0.0,
            "leap_1": 0.0,
            "leap_2": 0.0,
            "leap_3": 0.0,
            "leap_4": 0.0,
            "leap_5": 0.0,
            "leap_6": 0.0,
            "leap_7": 0.0,
            "leap_8": 0.0,
            "leap_9": 0.0,
            "leap_10": 0.0,
            "leap_11": 0.0,
            "leap_12": 0.0,
            "leap_13": 0.0,
            "leap_14": 0.0,
            "leap_15": 0.0,
        },
    ),
    actuators={
        "floating_leap_right_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "(x|y|z)_translation_joint",
                "(x|y|z)_rotation_joint",
                "leap_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)",
            ],
            effort_limit_sim={
                "(x|y|z)_translation_joint": 50.0,
                "(x|y|z)_rotation_joint": 50.0,
                "leap_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 3.0,
            },
            stiffness={
                "(x|y|z)_translation_joint": 1000.0,
                "(x|y|z)_rotation_joint": 1000.0,
                "leap_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 20.0,
            },
            damping={
                "(x|y|z)_translation_joint": 200.0,
                "(x|y|z)_rotation_joint": 200.0,
                "leap_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 1.0,
            },
            velocity_limit_sim={
                "(x|y|z)_translation_joint": 1.0,
                "(x|y|z)_rotation_joint": 6.28,
                "leap_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 6.28,
            },
            friction={
                "(x|y|z)_translation_joint": 0.01,
                "(x|y|z)_rotation_joint": 0.01,
                "leap_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_LEAP_RIGHT_PALM_BODY_NAME = "palm_lower"
FLOATING_LEAP_RIGHT_FINGERTIP_BODY_NAMES = [
    "thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3",
]
FLOATING_LEAP_RIGHT_HAND_TIPS_BODY_NAMES = (
    [FLOATING_LEAP_RIGHT_PALM_BODY_NAME] + FLOATING_LEAP_RIGHT_FINGERTIP_BODY_NAMES
)
FLOATING_LEAP_RIGHT_WRIST_JOINT_NAME = "(x|y|z)_rotation_joint"
FLOATING_LEAP_RIGHT_ARM_JOINT_NAMES_EXPR = ["(x|y|z)_translation_joint"]
FLOATING_LEAP_RIGHT_BASE_POS = (-0.1, 0.0, 0.45)
FLOATING_LEAP_RIGHT_BASE_ROT = (1.0, 0.0, 0.0, 0.0)
FLOATING_LEAP_RIGHT_WRIST_POSITION_OFFSET = (
    FLOATING_LEAP_RIGHT_BASE_POS[0] + 0.0,  # base_x + x_translation_joint default
    FLOATING_LEAP_RIGHT_BASE_POS[1] + 0.0,  # base_y + y_translation_joint default
    FLOATING_LEAP_RIGHT_BASE_POS[2] + 0.3,  # base_z + z_translation_joint default
)  # wrist world position at initial state

# World placement of the wrist translation-joint origin, per hand, consumed by
# SimpleAbsoluteRetargeter. Auto-derived from FLOATING_LEAP_RIGHT_CFG's
# init_state and the builder's scene_robot.replace() base pose.
FLOATING_LEAP_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "right": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_LEAP_RIGHT_CFG,
        translation_joint_names=(
            "x_translation_joint",
            "y_translation_joint",
            "z_translation_joint",
        ),
        rotation_joint_names=(
            "x_rotation_joint",
            "y_rotation_joint",
            "z_rotation_joint",
        ),
        rotation_axes="XYZ",
        base_pos=FLOATING_LEAP_RIGHT_BASE_POS,
        base_rot=FLOATING_LEAP_RIGHT_BASE_ROT,
    )
}


# ---------------------------------------------------------------------------
# Bimanual floating Leap hands
# ---------------------------------------------------------------------------


_BIMANUAL_WRIST_JOINT_NAMES = [
    "right_wrist_trans_x", "right_wrist_trans_y", "right_wrist_trans_z",
    "right_wrist_rot_x", "right_wrist_rot_y", "right_wrist_rot_z",
    "left_wrist_trans_x", "left_wrist_trans_y", "left_wrist_trans_z",
    "left_wrist_rot_x", "left_wrist_rot_y", "left_wrist_rot_z",
]

_BIMANUAL_FINGER_JOINT_NAMES = [
    f"{side}_{i}" for side in ("right", "left") for i in range(16)
]

_BIMANUAL_LEAP_FINGER_JOINT_NAMES = tuple(f"leap_{i}" for i in range(16))


@configclass
class BimanualLeapRelJointPosActionCfg:
    wrist = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=_BIMANUAL_WRIST_JOINT_NAMES,
        scale=0.03,
        preserve_order=True,
    )
    fingers = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=_BIMANUAL_FINGER_JOINT_NAMES,
        scale=0.10,
        preserve_order=True,
    )


@configclass
class BimanualLeapAbsJointPosActionCfg:
    """Absolute joint position control for bimanual Leap teleoperation."""

    wrist = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_BIMANUAL_WRIST_JOINT_NAMES,
        scale=1.0,
        preserve_order=True,
    )
    fingers = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_BIMANUAL_FINGER_JOINT_NAMES,
        scale=1.0,
        preserve_order=True,
    )


# Action layout with preserve_order=True:
#   wrist term  [0..11]:  right_trans(0,1,2) right_rot(3,4,5) left_trans(6,7,8) left_rot(9,10,11)
#   finger term [12..43]: right_0..right_15(12..27) left_0..left_15(28..43)
BIMANUAL_LEAP_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 44,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "xyz",
            "wrist_rot_signs": (1.0, 1.0, -1.0),
            "finger_indices": tuple(range(12, 28)),
            "finger_joint_names": _BIMANUAL_LEAP_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(16)),
        },
        "left": {
            "wrist_trans_indices": (6, 7, 8),
            "wrist_rot_indices": (9, 10, 11),
            "wrist_rot_order": "xyz",
            "wrist_rot_signs": (1.0, 1.0, -1.0),
            "finger_indices": tuple(range(28, 44)),
            "finger_joint_names": _BIMANUAL_LEAP_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(16)),
        },
    },
}

BIMANUAL_LEAP_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "left": dex_retargeting_hand_spec(
            _RETARGET_DIR, "left", _RETARGET_DIR / "leap_hand_left_fixed.urdf"
        ),
        "right": dex_retargeting_hand_spec(
            _RETARGET_DIR, "right", _RETARGET_DIR / "leap_hand_right_fixed.urdf"
        ),
    },
}

BIMANUAL_LEAP_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/bimanual_leap/bimanual_leap_hand.usd",
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
            fix_root_link=True,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.1, 0.0, 0.6),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "right_wrist_trans_x": 0.0,
            "right_wrist_trans_y": -0.0,
            "right_wrist_trans_z": 0.3,
            "right_wrist_rot_x": 0.0,
            "right_wrist_rot_y": 0,
            "right_wrist_rot_z": 0.0,
            "left_wrist_trans_x": 0.0,
            "left_wrist_trans_y": 0.0,
            "left_wrist_trans_z": 0.3,
            "left_wrist_rot_x": 0.0,
            "left_wrist_rot_y": 0,
            "left_wrist_rot_z": 0,
            "right_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 0,
            "left_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 0,
        },
    ),
    actuators={
        "bimanual_leap_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                "(right|left)_wrist_(trans|rot)_(x|y|z)",
                "(right|left)_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)",
            ],
            effort_limit_sim={
                "(right|left)_wrist_(trans|rot)_(x|y|z)": 50.0,
                "(right|left)_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 3.0,
            },
            stiffness={
                "(right|left)_wrist_(trans|rot)_(x|y|z)": 1000.0,
                "(right|left)_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 20.0,
            },
            damping={
                "(right|left)_wrist_(trans|rot)_(x|y|z)": 200.0,
                "(right|left)_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 0.1,
            },
            friction={
                "(right|left)_wrist_(trans|rot)_(x|y|z)": 0.01,
                "(right|left)_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 0.01,
            },
            velocity_limit_sim={
                "(right|left)_wrist_trans_(x|y|z)": 1.0,
                "(right|left)_wrist_rot_(x|y|z)": 6.28,
                "(right|left)_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)": 6.28,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

BIMANUAL_LEAP_RIGHT_PALM_BODY_NAME = "right_palm_lower"
BIMANUAL_LEAP_LEFT_PALM_BODY_NAME = "left_palm_lower_left"

BIMANUAL_LEAP_RIGHT_FINGERTIP_BODY_NAMES = [
    "right_thumb_fingertip",
    "right_fingertip",
    "right_fingertip_2",
    "right_fingertip_3",
]
BIMANUAL_LEAP_LEFT_FINGERTIP_BODY_NAMES = [
    "left_thumb_fingertip",
    "left_fingertip",
    "left_fingertip_2",
    "left_fingertip_3",
]
BIMANUAL_LEAP_FINGERTIP_BODY_NAMES = (
    BIMANUAL_LEAP_RIGHT_FINGERTIP_BODY_NAMES + BIMANUAL_LEAP_LEFT_FINGERTIP_BODY_NAMES
)
BIMANUAL_LEAP_HAND_TIPS_BODY_NAMES = [
    BIMANUAL_LEAP_RIGHT_PALM_BODY_NAME,
    BIMANUAL_LEAP_LEFT_PALM_BODY_NAME,
] + BIMANUAL_LEAP_FINGERTIP_BODY_NAMES
BIMANUAL_LEAP_ARM_JOINT_NAMES_EXPR = ["(right|left)_wrist_(trans|rot)_(x|y|z)"]


def build_tabletop_floating_leap_right_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating right Leap hand."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_LEAP_RIGHT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_LEAP_RIGHT_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_LEAP_RIGHT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_LEAP_RIGHT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_LEAP_RIGHT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_LEAP_RIGHT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_LEAP_RIGHT_CFG.init_state.replace(
                pos=FLOATING_LEAP_RIGHT_BASE_POS,
                rot=FLOATING_LEAP_RIGHT_BASE_ROT,
            ),
        ),
        actions=FloatingLeapRightAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": [f"leap_{i}" for i in range(16)],
            "wrist_position_offset": FLOATING_LEAP_RIGHT_WRIST_POSITION_OFFSET,
            "retargeter_config_filename": "leap_hand_right_dexpilot.yml",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


def build_tabletop_bimanual_leap_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating bimanual Leap hand."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": BIMANUAL_LEAP_RIGHT_PALM_BODY_NAME,
            "right_palm_body_name": BIMANUAL_LEAP_RIGHT_PALM_BODY_NAME,
            "left_palm_body_name": BIMANUAL_LEAP_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": BIMANUAL_LEAP_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": BIMANUAL_LEAP_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": None,
            "arm_joint_names_expr": BIMANUAL_LEAP_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=BIMANUAL_LEAP_CFG.replace(prim_path=prim_path),
        actions=BimanualLeapAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": [],
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "bimanual_leap",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


# The bimanual builder/layout double as the "floating_leap_bimanual" robot type.
FLOATING_LEAP_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT = (
    BIMANUAL_LEAP_SIMPLE_RELATIVE_RETARGETER_LAYOUT
)
FLOATING_LEAP_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING = (
    BIMANUAL_LEAP_SIMPLE_RELATIVE_DEX_RETARGETING
)
build_tabletop_floating_leap_bimanual_setup = build_tabletop_bimanual_leap_setup
