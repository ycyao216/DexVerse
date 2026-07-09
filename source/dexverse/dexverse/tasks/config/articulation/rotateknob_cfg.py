# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rotate-knob task: rotate a wall-mounted knob to ±π/2.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`. The scene
adds two leaf-specific entities: a kinematic wall ``board`` behind the
knob, and a slim ``marker`` cuboid that visually rides with the knob's
rotating shaft. The knob has a soft restoring actuator and its joint
limits are tightened at startup.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from dexverse.assets import PARTNET_MOBILITY_ARTICULATIONS_DIR
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg
from .articulation_base.articulation_base_cfg import (
    ARTICULATION_KEY,
    ArticulationBaseEventCfg,
    ArticulationBaseSceneCfg,
)

ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "knob"
KNOB_USD_PATH = str(ASSET_DIR / "100866.usd")
KNOB_SCALE = (0.3, 0.3, 0.2)
KNOB_HALF_HEIGHT_EST = 0.15
KNOB_LIMIT_ABS = math.pi / 2.0
KNOB_SUCCESS_THRESHOLD = 1.57

BOARD_SIZE = (0.02, 0.4, 0.45)  # thin in x, faces +x
BOARD_COLOR = (0.5, 0.3, 0.15)
BOARD_Z_OFFSET = 0.0

MARKER_SIZE = (0.12, 0.003, 0.003)
MARKER_COLOR = (0.9, 0.1, 0.1)
MARKER_LOCAL_POS = (-0.06, 0.0, 0.04)


BOARD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Board",
    spawn=sim_utils.CuboidCfg(
        size=BOARD_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOARD_COLOR),
        visible=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, 0.1)),
)

MARKER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/KnobMarker",
    spawn=sim_utils.CuboidCfg(
        size=MARKER_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=MARKER_COLOR,
            emissive_color=(0.3, 0.0, 0.0),
            roughness=1.0,
            metallic=0.0,
        ),
        visible=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=MARKER_LOCAL_POS),
)


@configclass
class RotateKnobSceneCfg(ArticulationBaseSceneCfg):
    """Articulation-base scene plus the wall board and the rotating marker."""

    board: RigidObjectCfg = BOARD_CFG
    marker: RigidObjectCfg = MARKER_CFG


@configclass
class RotateKnobEventCfg(ArticulationBaseEventCfg):
    """Replace the per-articulation reset with a paired board+knob reset,
    tighten joint limits at startup, and keep the marker glued to the shaft.
    """

    # Wipe the base's single-asset reset — paired reset below handles both.
    reset_articulation = None

    set_knob_joint_limit = EventTerm(
        func=mdp.events.set_joint_position_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[".*"]),
            "lower": -KNOB_LIMIT_ABS,
            "upper": KNOB_LIMIT_ABS,
        },
    )

    reset_board_and_knob = EventTerm(
        func=mdp.events.reset_board_and_switch_xy,
        mode="reset",
        params={
            "board_cfg": SceneEntityCfg("board"),
            "switch_cfg": SceneEntityCfg(ARTICULATION_KEY),
            "switch_joint_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=".*"),
            "x_range": (0.0, 0.0),
            "y_range": (-0.2, 0.2),
        },
    )

    update_marker_pose = EventTerm(
        func=mdp.events.update_marker_from_body,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "marker_cfg": SceneEntityCfg("marker"),
            "body_cfg": SceneEntityCfg(ARTICULATION_KEY, body_names=["link_0"]),
            "offset": MARKER_LOCAL_POS,
        },
    )


@configclass
class RotateKnobRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward proportional to absolute knob joint progress."""

    knob_progress = RewTerm(
        func=mdp.joint_range_progress,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[".*"]),
        },
    )


@configclass
class RotateKnobTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when |knob_angle| reaches ``KNOB_SUCCESS_THRESHOLD`` (~π/2)."""

    success = DoneTerm(
        func=mdp.joint_reach_threshold,
        params={
            "threshold": KNOB_SUCCESS_THRESHOLD,
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[".*"]),
            "ref": "value",
            "tol": 1e-2,
        },
    )


@configclass
class RotateKnobEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Rotate-knob env (floating dex-hand variant)."""

    articulation_usd_path: str = KNOB_USD_PATH
    articulation_scale: tuple[float, float, float] = KNOB_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.2, 0.0, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = KNOB_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True

    success_joint_names: list[str] = [".*"]
    # Wired directly into the leaf's termination above (joint_reach_threshold
    # with explicit threshold and ref="value"). Leaving success_threshold None
    # here so the base doesn't double-patch.

    rewards: RotateKnobRewardsCfg = RotateKnobRewardsCfg()
    terminations: RotateKnobTerminationsCfg = RotateKnobTerminationsCfg()
    events: RotateKnobEventCfg = RotateKnobEventCfg()
    scene: RotateKnobSceneCfg = RotateKnobSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
    )

    def __post_init__(self):
        super().__post_init__()
        # Brushed-metal visual finish on the knob.
        self.scene.articulation.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.7, 0.7, 0.72),
            metallic=0.85,
            roughness=0.2,
        )
        # Soft restoring actuator so the knob falls back without the policy's grip.
        self.scene.articulation.actuators = {
            "knob_position_actuator": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.05,
                damping=0.05,
                effort_limit_sim=1.0,
            ),
        }
        # Seat the board upright on the table.
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        board_z = table_top_z + BOARD_SIZE[2] * 0.5 + BOARD_Z_OFFSET
        board_pos = self.scene.board.init_state.pos
        self.scene.board.init_state.pos = (board_pos[0], board_pos[1], board_z)


# Backward-compat alias.
RotateKnobEnvCfg = RotateKnobEnvFloatingDexHandRightCfg
