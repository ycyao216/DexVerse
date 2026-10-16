# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Floating (virtual-joint) Allegro hand variants and tabletop helpers.

Single-hand (right / left) and bimanual floating Allegro hands that use virtual
prismatic/revolute joints for wrist translation and rotation. The palm-down home
orientation and the wrist-centered pivot are baked into the converted USDs
(``floating_allegro_hand_*/``), so the CFG base rotations stay identity.

Configurations:

* :obj:`FLOATING_ALLEGRO_RIGHT_CFG`   - single floating right Allegro hand.
* :obj:`FLOATING_ALLEGRO_LEFT_CFG`    - single floating left Allegro hand.
* :obj:`FLOATING_ALLEGRO_BIMANUAL_CFG`- bimanual floating Allegro hands.
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

from .. import TabletopRobotSetup, dex_retargeting_hand_spec

_ASSET_DIR = Path(__file__).resolve().parent
_RETARGET_DIR = _ASSET_DIR / "retarget"

# Finger joints in numeric order (index 0-3, middle 4-7, ring 8-11, thumb 12-15).
# The allegro thumb's joint_12_0 has a URDF lower limit > 0, so its home is
# clamped to that limit; everything else homes at 0.
_FINGER_IDS = list(range(16))
_THUMB_ABDUCTION_HOME = {"joint_12_0": 0.263}
# Fingertip rigid-body names (verified against the converted USD); the arm
# assembly uses a "hand_" prefix, the standalone floating hand drops it.
_RIGHT_FINGERTIP_SUFFIXES = ["link_3_0_tip", "link_7_0_tip", "link_11_0_tip", "link_15_0_tip"]

_SINGLE_TRANS_EXPR = "(x|y|z)_translation_joint"
_SINGLE_ROT_EXPR = "(x|y|z)_rotation_joint"
_SINGLE_FINGER_EXPR = "joint_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)_0"


def _finger_joint_names(prefix: str = "") -> list[str]:
    return [f"{prefix}joint_{i}_0" for i in _FINGER_IDS]


def _finger_home(prefix: str = "") -> dict[str, float]:
    home = {f"{prefix}joint_{i}_0": 0.0 for i in _FINGER_IDS}
    for name, val in _THUMB_ABDUCTION_HOME.items():
        home[f"{prefix}{name}"] = val
    return home


# ---------------------------------------------------------------------------
# Single floating right / left Allegro hand
# ---------------------------------------------------------------------------

FLOATING_ALLEGRO_RIGHT_BASE_POS = (-0.1, 0.0, 0.45)
FLOATING_ALLEGRO_RIGHT_BASE_ROT = (1.0, 0.0, 0.0, 0.0)  # orientation baked in URDF
FLOATING_ALLEGRO_LEFT_BASE_POS = FLOATING_ALLEGRO_RIGHT_BASE_POS
FLOATING_ALLEGRO_LEFT_BASE_ROT = FLOATING_ALLEGRO_RIGHT_BASE_ROT

_SINGLE_HOME_JOINTS = {
    "x_translation_joint": 0.0,
    "y_translation_joint": 0.0,
    "z_translation_joint": 0.3,
    "x_rotation_joint": 0.0,
    "y_rotation_joint": 0.0,
    "z_rotation_joint": 0.0,
    **_finger_home(),
}

FLOATING_ALLEGRO_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((_ASSET_DIR / "floating_allegro_hand_right" / "floating_allegro_hand_right.usd").resolve()),
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
        pos=FLOATING_ALLEGRO_RIGHT_BASE_POS,
        rot=FLOATING_ALLEGRO_RIGHT_BASE_ROT,
        joint_pos=dict(_SINGLE_HOME_JOINTS),
    ),
    actuators={
        "floating_base": ImplicitActuatorCfg(
            joint_names_expr=[_SINGLE_TRANS_EXPR, _SINGLE_ROT_EXPR],
            effort_limit_sim={_SINGLE_TRANS_EXPR: 50.0, _SINGLE_ROT_EXPR: 50.0},
            stiffness={_SINGLE_TRANS_EXPR: 1000.0, _SINGLE_ROT_EXPR: 1000.0},
            damping={_SINGLE_TRANS_EXPR: 200.0, _SINGLE_ROT_EXPR: 200.0},
            velocity_limit_sim={_SINGLE_TRANS_EXPR: 1.0, _SINGLE_ROT_EXPR: 6.28},
            friction={_SINGLE_TRANS_EXPR: 0.01, _SINGLE_ROT_EXPR: 0.01},
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[_SINGLE_FINGER_EXPR],
            effort_limit_sim=30.0,
            stiffness=50.0,
            damping=0.1,
            velocity_limit_sim=6.0,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_ALLEGRO_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((_ASSET_DIR / "floating_allegro_hand_left" / "floating_allegro_hand_left.usd").resolve()),
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
        pos=FLOATING_ALLEGRO_LEFT_BASE_POS,
        rot=FLOATING_ALLEGRO_LEFT_BASE_ROT,
        joint_pos=dict(_SINGLE_HOME_JOINTS),
    ),
    actuators={
        "floating_base": ImplicitActuatorCfg(
            joint_names_expr=[_SINGLE_TRANS_EXPR, _SINGLE_ROT_EXPR],
            effort_limit_sim={_SINGLE_TRANS_EXPR: 50.0, _SINGLE_ROT_EXPR: 50.0},
            stiffness={_SINGLE_TRANS_EXPR: 1000.0, _SINGLE_ROT_EXPR: 1000.0},
            damping={_SINGLE_TRANS_EXPR: 200.0, _SINGLE_ROT_EXPR: 200.0},
            velocity_limit_sim={_SINGLE_TRANS_EXPR: 1.0, _SINGLE_ROT_EXPR: 6.28},
            friction={_SINGLE_TRANS_EXPR: 0.01, _SINGLE_ROT_EXPR: 0.01},
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[_SINGLE_FINGER_EXPR],
            effort_limit_sim=30.0,
            stiffness=50.0,
            damping=0.1,
            velocity_limit_sim=6.0,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_ALLEGRO_RIGHT_PALM_BODY_NAME = "palm"
FLOATING_ALLEGRO_RIGHT_FINGERTIP_BODY_NAMES = list(_RIGHT_FINGERTIP_SUFFIXES)
FLOATING_ALLEGRO_RIGHT_HAND_TIPS_BODY_NAMES = (
    [FLOATING_ALLEGRO_RIGHT_PALM_BODY_NAME] + FLOATING_ALLEGRO_RIGHT_FINGERTIP_BODY_NAMES
)
FLOATING_ALLEGRO_RIGHT_WRIST_JOINT_NAME = _SINGLE_ROT_EXPR
FLOATING_ALLEGRO_RIGHT_ARM_JOINT_NAMES_EXPR = [_SINGLE_TRANS_EXPR]

FLOATING_ALLEGRO_LEFT_PALM_BODY_NAME = FLOATING_ALLEGRO_RIGHT_PALM_BODY_NAME
FLOATING_ALLEGRO_LEFT_FINGERTIP_BODY_NAMES = FLOATING_ALLEGRO_RIGHT_FINGERTIP_BODY_NAMES
FLOATING_ALLEGRO_LEFT_HAND_TIPS_BODY_NAMES = FLOATING_ALLEGRO_RIGHT_HAND_TIPS_BODY_NAMES
FLOATING_ALLEGRO_LEFT_WRIST_JOINT_NAME = FLOATING_ALLEGRO_RIGHT_WRIST_JOINT_NAME
FLOATING_ALLEGRO_LEFT_ARM_JOINT_NAMES_EXPR = FLOATING_ALLEGRO_RIGHT_ARM_JOINT_NAMES_EXPR


@configclass
class FloatingAllegroSingleAbsJointPosActionCfg:
    """Absolute joint-position control for a single floating Allegro hand."""

    translation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["x_translation_joint", "y_translation_joint", "z_translation_joint"],
        scale=1.0,
        preserve_order=True,
    )
    rotation_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["x_rotation_joint", "y_rotation_joint", "z_rotation_joint"],
        scale=1.0,
        preserve_order=True,
    )
    finger_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_finger_joint_names(),
        scale=1.0,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class FloatingAllegroSingleRelJointPosActionCfg:
    """Relative joint-position control for a single floating Allegro hand."""

    translation_action = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[_SINGLE_TRANS_EXPR], scale=0.1
    )
    rotation_action = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[_SINGLE_ROT_EXPR], scale=0.1
    )
    finger_action = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[_SINGLE_FINGER_EXPR], scale=0.1
    )


# ---------------------------------------------------------------------------
# Bimanual floating Allegro hands
# ---------------------------------------------------------------------------

FLOATING_ALLEGRO_BIMANUAL_BASE_POS = (-0.1, 0.0, 0.6)
FLOATING_ALLEGRO_BIMANUAL_BASE_ROT = (1.0, 0.0, 0.0, 0.0)

_BIMANUAL_TRANS_EXPR = "(right|left)_wrist_trans_(x|y|z)"
_BIMANUAL_ROT_EXPR = "(right|left)_wrist_rot_(x|y|z)"
_BIMANUAL_FINGER_EXPR = "(right|left)_joint_(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15)_0"

_BIMANUAL_WRIST_JOINT_NAMES = [
    f"{side}_wrist_{kind}_{axis}"
    for side in ("right", "left")
    for kind in ("trans", "rot")
    for axis in ("x", "y", "z")
]
_BIMANUAL_FINGER_JOINT_NAMES = _finger_joint_names("right_") + _finger_joint_names("left_")

_BIMANUAL_HOME_JOINTS: dict[str, float] = {}
for _side in ("right", "left"):
    _BIMANUAL_HOME_JOINTS.update(
        {
            f"{_side}_wrist_trans_x": 0.0,
            f"{_side}_wrist_trans_y": 0.0,
            f"{_side}_wrist_trans_z": 0.3,
            f"{_side}_wrist_rot_x": 0.0,
            f"{_side}_wrist_rot_y": 0.0,
            f"{_side}_wrist_rot_z": 0.0,
        }
    )
    _BIMANUAL_HOME_JOINTS.update(_finger_home(f"{_side}_"))

FLOATING_ALLEGRO_BIMANUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str((_ASSET_DIR / "floating_allegro_hand_bimanual" / "floating_allegro_hand_bimanual.usd").resolve()),
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
        pos=FLOATING_ALLEGRO_BIMANUAL_BASE_POS,
        rot=FLOATING_ALLEGRO_BIMANUAL_BASE_ROT,
        joint_pos=dict(_BIMANUAL_HOME_JOINTS),
    ),
    actuators={
        "floating_base": ImplicitActuatorCfg(
            joint_names_expr=[_BIMANUAL_TRANS_EXPR, _BIMANUAL_ROT_EXPR],
            effort_limit_sim={_BIMANUAL_TRANS_EXPR: 50.0, _BIMANUAL_ROT_EXPR: 50.0},
            stiffness={_BIMANUAL_TRANS_EXPR: 1000.0, _BIMANUAL_ROT_EXPR: 1000.0},
            damping={_BIMANUAL_TRANS_EXPR: 200.0, _BIMANUAL_ROT_EXPR: 200.0},
            velocity_limit_sim={_BIMANUAL_TRANS_EXPR: 1.0, _BIMANUAL_ROT_EXPR: 6.28},
            friction={_BIMANUAL_TRANS_EXPR: 0.01, _BIMANUAL_ROT_EXPR: 0.01},
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[_BIMANUAL_FINGER_EXPR],
            effort_limit_sim=30.0,
            stiffness=50.0,
            damping=0.1,
            velocity_limit_sim=6.0,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_ALLEGRO_BIMANUAL_RIGHT_PALM_BODY_NAME = "right_palm"
FLOATING_ALLEGRO_BIMANUAL_LEFT_PALM_BODY_NAME = "left_palm"
FLOATING_ALLEGRO_BIMANUAL_PALM_BODY_NAME = FLOATING_ALLEGRO_BIMANUAL_RIGHT_PALM_BODY_NAME
FLOATING_ALLEGRO_BIMANUAL_FINGERTIP_BODY_NAMES = [
    f"right_{s}" for s in _RIGHT_FINGERTIP_SUFFIXES
] + [f"left_{s}" for s in _RIGHT_FINGERTIP_SUFFIXES]
FLOATING_ALLEGRO_BIMANUAL_HAND_TIPS_BODY_NAMES = [
    FLOATING_ALLEGRO_BIMANUAL_RIGHT_PALM_BODY_NAME,
    FLOATING_ALLEGRO_BIMANUAL_LEFT_PALM_BODY_NAME,
] + FLOATING_ALLEGRO_BIMANUAL_FINGERTIP_BODY_NAMES
FLOATING_ALLEGRO_BIMANUAL_ARM_JOINT_NAMES_EXPR = ["(right|left)_wrist_trans_(x|y|z)"]


@configclass
class FloatingAllegroBimanualAbsJointPosActionCfg:
    """Absolute joint-position control for bimanual floating Allegro hands."""

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
        use_default_offset=False,
    )


# ---------------------------------------------------------------------------
# Retargeter layouts + dex-retargeting (teleop). Reuse the allegro DexPilot
# configs and the dexrot hand URDFs already in the package.
# ---------------------------------------------------------------------------

# DexPilot output DOF names are the dexrot-URDF joints (joint_N.0, dots restored).
_DEX_FINGER_NAMES = tuple(f"joint_{i}.0" for i in _FINGER_IDS)

FLOATING_ALLEGRO_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 22,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "xyz",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(6, 22)),
            "finger_joint_names": _DEX_FINGER_NAMES,
            "finger_permutation": tuple(range(16)),
        }
    },
}
FLOATING_ALLEGRO_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 22,
    "hands": {
        "left": dict(FLOATING_ALLEGRO_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT["hands"]["right"]),
    },
}
FLOATING_ALLEGRO_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT = {
    "output_dim": 44,
    "hands": {
        "right": {
            "wrist_trans_indices": (0, 1, 2),
            "wrist_rot_indices": (3, 4, 5),
            "wrist_rot_order": "xyz",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(12, 28)),
            "finger_joint_names": _DEX_FINGER_NAMES,
            "finger_permutation": tuple(range(16)),
        },
        "left": {
            "wrist_trans_indices": (6, 7, 8),
            "wrist_rot_indices": (9, 10, 11),
            "wrist_rot_order": "xyz",
            "wrist_rot_signs": (1.0, 1.0, 1.0),
            "finger_indices": tuple(range(28, 44)),
            "finger_joint_names": _DEX_FINGER_NAMES,
            "finger_permutation": tuple(range(16)),
        },
    },
}

FLOATING_ALLEGRO_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "right": dex_retargeting_hand_spec(
            _RETARGET_DIR, "right", _RETARGET_DIR / "allegro_hand_right" / "allegro_hand_right_dexrot.urdf"
        )
    }
}
FLOATING_ALLEGRO_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "left": dex_retargeting_hand_spec(
            _RETARGET_DIR, "left", _RETARGET_DIR / "allegro_hand_left" / "allegro_hand_left_dexrot.urdf"
        )
    }
}
FLOATING_ALLEGRO_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING = {
    "hands": {
        "right": FLOATING_ALLEGRO_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["right"],
        "left": FLOATING_ALLEGRO_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING["hands"]["left"],
    }
}


# ---------------------------------------------------------------------------
# Tabletop env-setup builders
# ---------------------------------------------------------------------------


def build_tabletop_floating_allegro_right_setup(prim_path: str = "{ENV_REGEX_NS}/Robot") -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating right Allegro hand."""
    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_ALLEGRO_RIGHT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_ALLEGRO_RIGHT_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_ALLEGRO_RIGHT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_ALLEGRO_RIGHT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_ALLEGRO_RIGHT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_ALLEGRO_RIGHT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_ALLEGRO_RIGHT_CFG.init_state.replace(
                pos=FLOATING_ALLEGRO_RIGHT_BASE_POS,
                rot=FLOATING_ALLEGRO_RIGHT_BASE_ROT,
            ),
        ),
        actions=FloatingAllegroSingleAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": _finger_joint_names(),
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "floating_allegro_right",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


def build_tabletop_floating_allegro_left_setup(prim_path: str = "{ENV_REGEX_NS}/Robot") -> TabletopRobotSetup:
    """Return the tabletop env setup for the floating left Allegro hand."""
    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_ALLEGRO_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_ALLEGRO_LEFT_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_ALLEGRO_LEFT_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": FLOATING_ALLEGRO_LEFT_WRIST_JOINT_NAME,
            "arm_joint_names_expr": FLOATING_ALLEGRO_LEFT_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_ALLEGRO_LEFT_CFG.replace(
            prim_path=prim_path,
            init_state=FLOATING_ALLEGRO_LEFT_CFG.init_state.replace(
                pos=FLOATING_ALLEGRO_LEFT_BASE_POS,
                rot=FLOATING_ALLEGRO_LEFT_BASE_ROT,
            ),
        ),
        actions=FloatingAllegroSingleAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": _finger_joint_names(),
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "floating_allegro_left",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


def build_tabletop_floating_allegro_bimanual_setup(prim_path: str = "{ENV_REGEX_NS}/Robot") -> TabletopRobotSetup:
    """Return the tabletop env setup for floating bimanual Allegro hands."""
    return TabletopRobotSetup(
        robot_config_kwargs={
            "palm_body_name": FLOATING_ALLEGRO_BIMANUAL_PALM_BODY_NAME,
            "right_palm_body_name": FLOATING_ALLEGRO_BIMANUAL_RIGHT_PALM_BODY_NAME,
            "left_palm_body_name": FLOATING_ALLEGRO_BIMANUAL_LEFT_PALM_BODY_NAME,
            "fingertip_body_names": FLOATING_ALLEGRO_BIMANUAL_FINGERTIP_BODY_NAMES,
            "hand_tips_body_names": FLOATING_ALLEGRO_BIMANUAL_HAND_TIPS_BODY_NAMES,
            "wrist_joint_name": None,
            "arm_joint_names_expr": FLOATING_ALLEGRO_BIMANUAL_ARM_JOINT_NAMES_EXPR,
            "setup_contact_sensors": True,
        },
        scene_robot=FLOATING_ALLEGRO_BIMANUAL_CFG.replace(prim_path=prim_path),
        actions=FloatingAllegroBimanualAbsJointPosActionCfg(),
        controller_mode="joint",
        teleop_config={
            "hand_joint_names": [],
            "wrist_position_offset": (0.0, 0.0, 0.0),
            "retargeter_config_filename": "floating_allegro_bimanual",
            "retargeter_urdf_path": None,
            "apply_shadow_specific_postprocess": False,
        },
    )


__all__ = [
    "FLOATING_ALLEGRO_RIGHT_CFG",
    "FLOATING_ALLEGRO_LEFT_CFG",
    "FLOATING_ALLEGRO_BIMANUAL_CFG",
    "FLOATING_ALLEGRO_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_ALLEGRO_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_ALLEGRO_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    "FLOATING_ALLEGRO_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_ALLEGRO_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FLOATING_ALLEGRO_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
    "FloatingAllegroSingleAbsJointPosActionCfg",
    "FloatingAllegroSingleRelJointPosActionCfg",
    "FloatingAllegroBimanualAbsJointPosActionCfg",
    "build_tabletop_floating_allegro_right_setup",
    "build_tabletop_floating_allegro_left_setup",
    "build_tabletop_floating_allegro_bimanual_setup",
]
