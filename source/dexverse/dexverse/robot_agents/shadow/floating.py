# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Floating Shadow hand variants and tabletop helpers."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from dexverse.devices.wrist_origin import compute_wrist_joint_origin
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils import configclass

from .. import TabletopRobotSetup, dex_retargeting_hand_spec

_ASSET_DIR = Path(__file__).resolve().parent
_RETARGET_DIR = _ASSET_DIR / "retarget"
_SHADOW_FINGER_JOINT_NAMES = (
    "FFJ1",
    "FFJ2",
    "FFJ3",
    "FFJ4",
    "MFJ1",
    "MFJ2",
    "MFJ3",
    "MFJ4",
    "RFJ1",
    "RFJ2",
    "RFJ3",
    "RFJ4",
    "LFJ1",
    "LFJ2",
    "LFJ3",
    "LFJ4",
    "LFJ5",
    "THJ1",
    "THJ2",
    "THJ3",
    "THJ4",
    "THJ5",
)


# ---------------------------------------------------------------------------
# Single floating right Shadow hand
# ---------------------------------------------------------------------------

FLOATING_SHADOW_RIGHT_BASE_POS = (-0.75, 0.0, 0.5)
FLOATING_SHADOW_RIGHT_BASE_ROT = (1.0, 0.0, 0.0, 0.0)
FLOATING_SHADOW_RIGHT_WRIST_POSITION_OFFSET = (
    FLOATING_SHADOW_RIGHT_BASE_POS[0] + 0.5,  # base_x + x_translation_joint
    FLOATING_SHADOW_RIGHT_BASE_POS[1] + 0.0,  # base_y + y_translation_joint
    FLOATING_SHADOW_RIGHT_BASE_POS[2] + 0.3,  # base_z + z_translation_joint
)  # = (-0.25, 0.0, 0.8) wrist world position at initial state

FLOATING_SHADOW_RIGHT_PALM_BODY_NAME = "palm"
FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES = ["thtip", "fftip", "mftip", "rftip", "lftip"]
FLOATING_SHADOW_RIGHT_HAND_TIPS_BODY_NAMES = [
    FLOATING_SHADOW_RIGHT_PALM_BODY_NAME,
    *FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES,
]
FLOATING_SHADOW_RIGHT_WRIST_JOINT_NAME = "(x|y|z)_rotation_joint"
FLOATING_SHADOW_RIGHT_ARM_JOINT_NAMES_EXPR = ["(x|y|z)_translation_joint"]


@configclass
class FloatingShadowRightRelJointPosActionCfg:
    """Relative joint-position control for the floating Shadow hand."""

    translation_action = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=["(x|y|z)_translation_joint"], scale=0.1
    )
    rotation_action = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=["(x|y|z)_rotation_joint"], scale=0.1
    )
    finger_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["WRJ(1|2)", "FFJ(1|2|3|4)", "MFJ(1|2|3|4)", "RFJ(1|2|3|4)", "LFJ(1|2|3|4|5)", "THJ(1|2|3|4|5)"],
        scale=0.1,
    )


@configclass
class FloatingShadowRightAbsJointPosActionCfg:
    """Absolute joint-position control for the floating Shadow hand (teleop).

    Rotation joints are ordered [z, y, x] = [yaw, pitch, roll] to match the
    retargeter output.
    """

    translation_action = JointPositionActionCfg(
        asset_name="robot", joint_names=["(x|y|z)_translation_joint"], scale=1.0
    )
    rotation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["z_rotation_joint", "y_rotation_joint", "x_rotation_joint"],
        scale=1.0,
        preserve_order=True,
    )
    finger_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_SHADOW_FINGER_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
    )


FLOATING_SHADOW_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 28,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(6, 28)),
            "finger_joint_names": _SHADOW_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(22)),
        }
    },
}


FLOATING_SHADOW_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSET_DIR}/floating_shadow_right/floating_shadow_right.usd",
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
            "y_translation_joint": 0.0,
            "x_translation_joint": 0.5,
            "z_translation_joint": 0.3,
            "x_rotation_joint": 0.0,
            "y_rotation_joint": 0.0,
            "z_rotation_joint": 0.0,
            "FFJ(1|2|3|4)": 0.0,
            "MFJ(1|2|3|4)": 0.0,
            "RFJ(1|2|3|4)": 0.0,
            "LFJ(1|2|3|4|5)": 0.0,
            "THJ(1|2|3|4|5)": 0.0,
        },
    ),
    actuators={
        "floating_shadow_right_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "(x|y|z)_translation_joint",
                "(x|y|z)_rotation_joint",
                "FFJ(1|2|3|4)",
                "MFJ(1|2|3|4)",
                "RFJ(1|2|3|4)",
                "LFJ(1|2|3|4|5)",
                "THJ(1|2|3|4|5)",
            ],
            effort_limit_sim={
                "(x|y|z)_translation_joint": 15.0,
                "(x|y|z)_rotation_joint": 15.0,
                "FFJ(1|2|3|4)": 10.0,
                "MFJ(1|2|3|4)": 10.0,
                "RFJ(1|2|3|4)": 10.0,
                "LFJ(1|2|3|4|5)": 10.0,
                "THJ(1|2|3|4|5)": 10.0,
            },
            stiffness={
                "(x|y|z)_translation_joint": 2000.0,
                "(x|y|z)_rotation_joint": 2000.0,
                "FFJ(1|2|3|4)": 10.0,
                "MFJ(1|2|3|4)": 10.0,
                "RFJ(1|2|3|4)": 10.0,
                "LFJ(1|2|3|4|5)": 10.0,
                "THJ(1|2|3|4|5)": 10.0,
            },
            damping={
                "(x|y|z)_translation_joint": 400.0,
                "(x|y|z)_rotation_joint": 400.0,
                "FFJ(1|2|3|4)": 0.1,
                "MFJ(1|2|3|4)": 0.1,
                "RFJ(1|2|3|4)": 0.1,
                "LFJ(1|2|3|4|5)": 0.1,
                "THJ(1|2|3|4|5)": 0.1,
            },
            velocity_limit_sim={
                "(x|y|z)_translation_joint": 10.0,
                "(x|y|z)_rotation_joint": 5.0,
                "FFJ(1|2|3|4)": 5.0,
                "MFJ(1|2|3|4)": 5.0,
                "RFJ(1|2|3|4)": 5.0,
                "LFJ(1|2|3|4|5)": 5.0,
                "THJ(1|2|3|4|5)": 5.0,
            },
            friction={
                "(x|y|z)_translation_joint": 0.01,
                "(x|y|z)_rotation_joint": 0.01,
                "FFJ(1|2|3|4)": 0.01,
                "MFJ(1|2|3|4)": 0.01,
                "RFJ(1|2|3|4)": 0.01,
                "LFJ(1|2|3|4|5)": 0.01,
                "THJ(1|2|3|4|5)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# World placement of the wrist translation-joint origin, consumed by
# SimpleAbsoluteRetargeter. Auto-derived from the right-hand cfg's init_state
# and the builder's scene_robot.replace() base pose.
FLOATING_SHADOW_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "right": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_SHADOW_RIGHT_CFG,
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
        base_pos=FLOATING_SHADOW_RIGHT_BASE_POS,
        base_rot=FLOATING_SHADOW_RIGHT_BASE_ROT,
    )
}


FLOATING_SHADOW_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {"right": dex_retargeting_hand_spec(_RETARGET_DIR, "right", _RETARGET_DIR / "floating_shadow_right.urdf")}
}


FLOATING_SHADOW_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 28,
    "hands": {
        "left": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(6, 28)),
            "finger_joint_names": _SHADOW_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(22)),
        }
    },
}


FLOATING_SHADOW_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {"left": dex_retargeting_hand_spec(_RETARGET_DIR, "left", _RETARGET_DIR / "floating_shadow_left.urdf")}
}


FLOATING_SHADOW_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 56,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(12, 34)),
            "finger_joint_names": _SHADOW_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(22)),
        },
        "left": {
            "wrist_trans_indices": (6, 7, 8),
            "wrist_rot_indices": (9, 10, 11),
            "wrist_rot_order": "yaw_pitch_roll",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(34, 56)),
            "finger_joint_names": _SHADOW_FINGER_JOINT_NAMES,
            "finger_permutation": tuple(range(22)),
        },
    },
}


FLOATING_SHADOW_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "right": FLOATING_SHADOW_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["right"],
        "left": FLOATING_SHADOW_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["left"],
    }
}


FLOATING_SHADOW_LEFT_PALM_BODY_NAME = "palm"
FLOATING_SHADOW_LEFT_FINGERTIP_BODY_NAMES = ["thtip", "fftip", "mftip", "rftip", "lftip"]
FLOATING_SHADOW_LEFT_HAND_TIPS_BODY_NAMES = [
    FLOATING_SHADOW_LEFT_PALM_BODY_NAME,
    *FLOATING_SHADOW_LEFT_FINGERTIP_BODY_NAMES,
]
FLOATING_SHADOW_LEFT_WRIST_JOINT_NAME = "(x|y|z)_rotation_joint"
FLOATING_SHADOW_LEFT_ARM_JOINT_NAMES_EXPR = ["(x|y|z)_translation_joint"]
FLOATING_SHADOW_LEFT_BASE_POS = FLOATING_SHADOW_RIGHT_BASE_POS
FLOATING_SHADOW_LEFT_BASE_ROT = FLOATING_SHADOW_RIGHT_BASE_ROT
FLOATING_SHADOW_LEFT_WRIST_POSITION_OFFSET = FLOATING_SHADOW_RIGHT_WRIST_POSITION_OFFSET

FLOATING_SHADOW_BIMANUAL_RIGHT_PALM_BODY_NAME = "rh_palm"
FLOATING_SHADOW_BIMANUAL_LEFT_PALM_BODY_NAME = "lh_palm"
FLOATING_SHADOW_BIMANUAL_PALM_BODY_NAME = FLOATING_SHADOW_BIMANUAL_RIGHT_PALM_BODY_NAME
FLOATING_SHADOW_BIMANUAL_FINGERTIP_BODY_NAMES = [
    "rh_thtip",
    "rh_fftip",
    "rh_mftip",
    "rh_rftip",
    "rh_lftip",
    "lh_thtip",
    "lh_fftip",
    "lh_mftip",
    "lh_rftip",
    "lh_lftip",
]
FLOATING_SHADOW_BIMANUAL_HAND_TIPS_BODY_NAMES = [
    "rh_palm",
    "lh_palm",
    *FLOATING_SHADOW_BIMANUAL_FINGERTIP_BODY_NAMES,
]
FLOATING_SHADOW_BIMANUAL_ARM_JOINT_NAMES_EXPR = ["(lh|rh)_(x|y|z)_translation_joint"]
FLOATING_SHADOW_BIMANUAL_BASE_POS = FLOATING_SHADOW_RIGHT_BASE_POS
FLOATING_SHADOW_BIMANUAL_BASE_ROT = FLOATING_SHADOW_RIGHT_BASE_ROT


FLOATING_SHADOW_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSET_DIR}/floating_shadow_left/floating_shadow_left.usd",
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
            "y_translation_joint": 0.0,
            "x_translation_joint": 0.5,
            "z_translation_joint": 0.3,
            "x_rotation_joint": 0.0,
            "y_rotation_joint": 0.0,
            "z_rotation_joint": 0.0,
            "FFJ(1|2|3|4)": 0.0,
            "MFJ(1|2|3|4)": 0.0,
            "RFJ(1|2|3|4)": 0.0,
            "LFJ(1|2|3|4|5)": 0.0,
            "THJ(1|2|3|4|5)": 0.0,
        },
    ),
    actuators={
        "floating_shadow_left_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "(x|y|z)_translation_joint",
                "(x|y|z)_rotation_joint",
                "FFJ(1|2|3|4)",
                "MFJ(1|2|3|4)",
                "RFJ(1|2|3|4)",
                "LFJ(1|2|3|4|5)",
                "THJ(1|2|3|4|5)",
            ],
            effort_limit_sim={
                "(x|y|z)_translation_joint": 15.0,
                "(x|y|z)_rotation_joint": 15.0,
                "FFJ(1|2|3|4)": 10.0,
                "MFJ(1|2|3|4)": 10.0,
                "RFJ(1|2|3|4)": 10.0,
                "LFJ(1|2|3|4|5)": 10.0,
                "THJ(1|2|3|4|5)": 10.0,
            },
            stiffness={
                "(x|y|z)_translation_joint": 2000.0,
                "(x|y|z)_rotation_joint": 2000.0,
                "FFJ(1|2|3|4)": 10.0,
                "MFJ(1|2|3|4)": 10.0,
                "RFJ(1|2|3|4)": 10.0,
                "LFJ(1|2|3|4|5)": 10.0,
                "THJ(1|2|3|4|5)": 10.0,
            },
            damping={
                "(x|y|z)_translation_joint": 400.0,
                "(x|y|z)_rotation_joint": 400.0,
                "FFJ(1|2|3|4)": 0.1,
                "MFJ(1|2|3|4)": 0.1,
                "RFJ(1|2|3|4)": 0.1,
                "LFJ(1|2|3|4|5)": 0.1,
                "THJ(1|2|3|4|5)": 0.1,
            },
            velocity_limit_sim={
                "(x|y|z)_translation_joint": 10.0,
                "(x|y|z)_rotation_joint": 5.0,
                "FFJ(1|2|3|4)": 5.0,
                "MFJ(1|2|3|4)": 5.0,
                "RFJ(1|2|3|4)": 5.0,
                "LFJ(1|2|3|4|5)": 5.0,
                "THJ(1|2|3|4|5)": 5.0,
            },
            friction={
                "(x|y|z)_translation_joint": 0.01,
                "(x|y|z)_rotation_joint": 0.01,
                "FFJ(1|2|3|4)": 0.01,
                "MFJ(1|2|3|4)": 0.01,
                "RFJ(1|2|3|4)": 0.01,
                "LFJ(1|2|3|4|5)": 0.01,
                "THJ(1|2|3|4|5)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# World placement of the wrist translation-joint origin per hand, consumed by
# SimpleAbsoluteRetargeter. Auto-derived from the left-hand cfg's init_state
# (base pose + home translation-joint values) and the builder's
# scene_robot.replace() base pose, so this stays in sync automatically if
# FLOATING_SHADOW_LEFT_CFG changes.
FLOATING_SHADOW_LEFT_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "left": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_SHADOW_LEFT_CFG,
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
        base_pos=FLOATING_SHADOW_LEFT_BASE_POS,
        base_rot=FLOATING_SHADOW_LEFT_BASE_ROT,
    )
}


FLOATING_SHADOW_BIMANUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSET_DIR}/floating_shadow_bimanual/floating_shadow_bimanual.usd",
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
            "rh_y_translation_joint": 0.0,
            "rh_x_translation_joint": 0.5,
            "rh_z_translation_joint": 0.3,
            "rh_x_rotation_joint": 0.0,
            "rh_y_rotation_joint": 0.0,
            "rh_z_rotation_joint": 0.0,
            "lh_y_translation_joint": 0.0,
            "lh_x_translation_joint": 0.5,
            "lh_z_translation_joint": 0.3,
            "lh_x_rotation_joint": 0.0,
            "lh_y_rotation_joint": 0.0,
            "lh_z_rotation_joint": 0.0,
            "rh_FFJ(1|2|3|4)": 0.0,
            "rh_MFJ(1|2|3|4)": 0.0,
            "rh_RFJ(1|2|3|4)": 0.0,
            "rh_LFJ(1|2|3|4|5)": 0.0,
            "rh_THJ(1|2|3|4|5)": 0.0,
            "lh_FFJ(1|2|3|4)": 0.0,
            "lh_MFJ(1|2|3|4)": 0.0,
            "lh_RFJ(1|2|3|4)": 0.0,
            "lh_LFJ(1|2|3|4|5)": 0.0,
            "lh_THJ(1|2|3|4|5)": 0.0,
        },
    ),
    actuators={
        "floating_shadow_bimanual_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "(lh|rh)_(x|y|z)_translation_joint",
                "(lh|rh)_(x|y|z)_rotation_joint",
                "(lh|rh)_FFJ(1|2|3|4)",
                "(lh|rh)_MFJ(1|2|3|4)",
                "(lh|rh)_RFJ(1|2|3|4)",
                "(lh|rh)_LFJ(1|2|3|4|5)",
                "(lh|rh)_THJ(1|2|3|4|5)",
            ],
            effort_limit_sim={
                "(lh|rh)_(x|y|z)_translation_joint": 15.0,
                "(lh|rh)_(x|y|z)_rotation_joint": 15.0,
                "(lh|rh)_FFJ(1|2|3|4)": 10.0,
                "(lh|rh)_MFJ(1|2|3|4)": 10.0,
                "(lh|rh)_RFJ(1|2|3|4)": 10.0,
                "(lh|rh)_LFJ(1|2|3|4|5)": 10.0,
                "(lh|rh)_THJ(1|2|3|4|5)": 10.0,
            },
            stiffness={
                "(lh|rh)_(x|y|z)_translation_joint": 2000.0,
                "(lh|rh)_(x|y|z)_rotation_joint": 2000.0,
                "(lh|rh)_FFJ(1|2|3|4)": 10.0,
                "(lh|rh)_MFJ(1|2|3|4)": 10.0,
                "(lh|rh)_RFJ(1|2|3|4)": 10.0,
                "(lh|rh)_LFJ(1|2|3|4|5)": 10.0,
                "(lh|rh)_THJ(1|2|3|4|5)": 10.0,
            },
            damping={
                "(lh|rh)_(x|y|z)_translation_joint": 400.0,
                "(lh|rh)_(x|y|z)_rotation_joint": 400.0,
                "(lh|rh)_FFJ(1|2|3|4)": 0.1,
                "(lh|rh)_MFJ(1|2|3|4)": 0.1,
                "(lh|rh)_RFJ(1|2|3|4)": 0.1,
                "(lh|rh)_LFJ(1|2|3|4|5)": 0.1,
                "(lh|rh)_THJ(1|2|3|4|5)": 0.1,
            },
            velocity_limit_sim={
                "(lh|rh)_(x|y|z)_translation_joint": 10.0,
                "(lh|rh)_(x|y|z)_rotation_joint": 5.0,
                "(lh|rh)_FFJ(1|2|3|4)": 5.0,
                "(lh|rh)_MFJ(1|2|3|4)": 5.0,
                "(lh|rh)_RFJ(1|2|3|4)": 5.0,
                "(lh|rh)_LFJ(1|2|3|4|5)": 5.0,
                "(lh|rh)_THJ(1|2|3|4|5)": 5.0,
            },
            friction={
                "(lh|rh)_(x|y|z)_translation_joint": 0.01,
                "(lh|rh)_(x|y|z)_rotation_joint": 0.01,
                "(lh|rh)_FFJ(1|2|3|4)": 0.01,
                "(lh|rh)_MFJ(1|2|3|4)": 0.01,
                "(lh|rh)_RFJ(1|2|3|4)": 0.01,
                "(lh|rh)_LFJ(1|2|3|4|5)": 0.01,
                "(lh|rh)_THJ(1|2|3|4|5)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# Fixed per-hand mount offset of each hand's translation-joint chain from the
# articulation root, in the robot base frame. Read from
# floating_shadow_bimanual.usd: rh_base/lh_base sit at y = -0.3 / +0.3 with
# identity rotation. Without this, both hands' origins collapse onto the
# centerline and the absolute retargeter spreads the hands outward by 0.3 m each.
FLOATING_SHADOW_BIMANUAL_HAND_MOUNT_OFFSET = {
    "right": (0.0, -0.3, 0.0),
    "left": (0.0, 0.3, 0.0),
}


# World placement of each wrist translation-joint origin, consumed by
# SimpleAbsoluteRetargeter for robot_type="floating_shadow_bimanual". Auto-derived
# per hand from the bimanual cfg's init_state (base pose + home translation-joint
# values), the per-hand mount offset, and the builder's scene_robot.replace() base
# pose, so this stays in sync automatically if FLOATING_SHADOW_BIMANUAL_CFG changes.
# The joint names are the rh_/lh_-prefixed bimanual joints.
FLOATING_SHADOW_BIMANUAL_SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
    "right": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_SHADOW_BIMANUAL_CFG,
        translation_joint_names=(
            "rh_x_translation_joint",
            "rh_y_translation_joint",
            "rh_z_translation_joint",
        ),
        rotation_joint_names=(
            "rh_x_rotation_joint",
            "rh_y_rotation_joint",
            "rh_z_rotation_joint",
        ),
        rotation_axes="XYZ",
        base_pos=FLOATING_SHADOW_BIMANUAL_BASE_POS,
        base_rot=FLOATING_SHADOW_BIMANUAL_BASE_ROT,
        mount_offset=FLOATING_SHADOW_BIMANUAL_HAND_MOUNT_OFFSET["right"],
    ),
    "left": compute_wrist_joint_origin(
        articulation_cfg=FLOATING_SHADOW_BIMANUAL_CFG,
        translation_joint_names=(
            "lh_x_translation_joint",
            "lh_y_translation_joint",
            "lh_z_translation_joint",
        ),
        rotation_joint_names=(
            "lh_x_rotation_joint",
            "lh_y_rotation_joint",
            "lh_z_rotation_joint",
        ),
        rotation_axes="XYZ",
        base_pos=FLOATING_SHADOW_BIMANUAL_BASE_POS,
        base_rot=FLOATING_SHADOW_BIMANUAL_BASE_ROT,
        mount_offset=FLOATING_SHADOW_BIMANUAL_HAND_MOUNT_OFFSET["left"],
    ),
}


@configclass
class FloatingShadowBimanualAbsJointPosActionCfg:
    right_wrist = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "rh_x_translation_joint",
            "rh_y_translation_joint",
            "rh_z_translation_joint",
            "rh_z_rotation_joint",
            "rh_y_rotation_joint",
            "rh_x_rotation_joint",
        ],
        scale=1.0,
        preserve_order=True,
    )
    left_wrist = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "lh_x_translation_joint",
            "lh_y_translation_joint",
            "lh_z_translation_joint",
            "lh_z_rotation_joint",
            "lh_y_rotation_joint",
            "lh_x_rotation_joint",
        ],
        scale=1.0,
        preserve_order=True,
    )
    right_fingers = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"rh_{joint_name}" for joint_name in _SHADOW_FINGER_JOINT_NAMES],
        scale=1.0,
        preserve_order=True,
    )
    left_fingers = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"lh_{joint_name}" for joint_name in _SHADOW_FINGER_JOINT_NAMES],
        scale=1.0,
        preserve_order=True,
    )


def build_tabletop_floating_shadow_left_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating left Shadow hand."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_SHADOW_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_SHADOW_LEFT_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_SHADOW_LEFT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_SHADOW_LEFT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_SHADOW_LEFT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_SHADOW_LEFT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_SHADOW_LEFT_CFG.init_state.replace(
                pos=FLOATING_SHADOW_LEFT_BASE_POS,
                rot=FLOATING_SHADOW_LEFT_BASE_ROT,
            ),
        ),
        actions=FloatingShadowRightAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": list(_SHADOW_FINGER_JOINT_NAMES),
            "wrist_position_offset": FLOATING_SHADOW_LEFT_WRIST_POSITION_OFFSET,
            "retargeter_config_filename": "floating_shadow_left",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": True,
        },
    )


def build_tabletop_floating_shadow_bimanual_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating bimanual Shadow hand."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_SHADOW_BIMANUAL_PALM_BODY_NAME,
            "right_palm_body_name": FLOATING_SHADOW_BIMANUAL_RIGHT_PALM_BODY_NAME,
            "left_palm_body_name": FLOATING_SHADOW_BIMANUAL_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_SHADOW_BIMANUAL_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_SHADOW_BIMANUAL_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": None,
            "arm_joint_names_expr": FLOATING_SHADOW_BIMANUAL_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_SHADOW_BIMANUAL_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_SHADOW_BIMANUAL_CFG.init_state.replace(
                pos=FLOATING_SHADOW_BIMANUAL_BASE_POS,
                rot=FLOATING_SHADOW_BIMANUAL_BASE_ROT,
            ),
        ),
        actions=FloatingShadowBimanualAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": [],
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "floating_shadow_bimanual",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": True,
        },
    )


def build_tabletop_floating_shadow_right_setup(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
) -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating right Shadow hand."""

    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_SHADOW_RIGHT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_SHADOW_RIGHT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_SHADOW_RIGHT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_SHADOW_RIGHT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_SHADOW_RIGHT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_SHADOW_RIGHT_CFG.init_state.replace(
                pos=FLOATING_SHADOW_RIGHT_BASE_POS,
                rot=FLOATING_SHADOW_RIGHT_BASE_ROT,
            ),
        ),
        actions=FloatingShadowRightAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": list(_SHADOW_FINGER_JOINT_NAMES),
            "wrist_position_offset": FLOATING_SHADOW_RIGHT_WRIST_POSITION_OFFSET,
            "retargeter_config_filename": "shadow_hand_right_dexpilot.yml",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": True,
        },
    )
