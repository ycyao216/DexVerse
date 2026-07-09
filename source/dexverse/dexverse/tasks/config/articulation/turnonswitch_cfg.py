# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn-on-switch task: flip a wall-mounted switch past 80% of its travel.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`. The switch
is mounted on a kinematic board behind it (a thin upright cuboid); both
are reset together via :func:`mdp.events.reset_board_and_switch_xy`,
which replaces the base's per-articulation reset event.

Reward is ``joint_range_progress_from_init`` (signed progress from
init); success is a progress-mode ``joint_relative_move``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from dexverse.assets import PARTNET_MOBILITY_ARTICULATIONS_DIR
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

ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "switch"
SWITCH_USD_PATH = str(ASSET_DIR / "100849.usd")
SWITCH_SCALE = (0.1, 0.1, 0.1)
SWITCH_HALF_HEIGHT_EST = 0.15
SWITCH_JOINT_NAME = "joint_0"
SWITCH_ON_PROGRESS = 0.8

BOARD_SIZE = (0.02, 0.35, 0.45)  # thin in x, faces +x
BOARD_COLOR = (0.5, 0.3, 0.15)
BOARD_Z_OFFSET = 0.0


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
    # x moved forward (toward the robot) from 0.6 so the wall sits just behind
    # the switch's back face (switch back ~x=0.48 at SWITCH_SCALE; board is
    # ~0.02 thick, so front face lands ~x=0.49). This closes the gap that
    # previously exposed the switch internals. z is overridden in __post_init__.
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.1)),
)


@configclass
class TurnOnSwitchSceneCfg(ArticulationBaseSceneCfg):
    """Articulation-base scene plus the wall-mounted board behind the switch."""

    board: RigidObjectCfg = BOARD_CFG


@configclass
class TurnOnSwitchEventCfg(ArticulationBaseEventCfg):
    """Replace the per-articulation reset event with a paired board+switch reset."""

    # Wipe the base's single-asset reset — we use a paired reset below.
    reset_articulation = None

    reset_board_and_switch = EventTerm(
        func=mdp.events.reset_board_and_switch_xy,
        mode="reset",
        params={
            "board_cfg": SceneEntityCfg("board"),
            "switch_cfg": SceneEntityCfg(ARTICULATION_KEY),
            "x_range": (0.0, 0.0),
            "y_range": (-0.2, 0.2),
        },
    )


@configclass
class TurnOnSwitchRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward proportional to signed switch-joint progress from init."""

    switch_on = RewTerm(
        func=mdp.joint_range_progress_from_init,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[SWITCH_JOINT_NAME]),
        },
    )


@configclass
class TurnOnSwitchTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when the switch has progressed ``SWITCH_ON_PROGRESS`` from init."""

    success = DoneTerm(
        func=mdp.joint_relative_move,
        params={
            "threshold": SWITCH_ON_PROGRESS,
            "mode": "progress",
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[SWITCH_JOINT_NAME]),
        },
    )


@configclass
class TurnOnSwitchEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Turn-on-switch env (floating dex-hand variant)."""

    articulation_usd_path: str = SWITCH_USD_PATH
    articulation_scale: tuple[float, float, float] = SWITCH_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.4, 0.0, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = SWITCH_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True

    success_joint_names: list[str] = [SWITCH_JOINT_NAME]
    # success_threshold left at None: progress-mode ``joint_relative_move``
    # doesn't read the base threshold.

    rewards: TurnOnSwitchRewardsCfg = TurnOnSwitchRewardsCfg()
    terminations: TurnOnSwitchTerminationsCfg = TurnOnSwitchTerminationsCfg()
    events: TurnOnSwitchEventCfg = TurnOnSwitchEventCfg()
    scene: TurnOnSwitchSceneCfg = TurnOnSwitchSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
    )

    def __post_init__(self):
        super().__post_init__()
        # Seat the board upright on the table (z is the half-height of the board).
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        board_z = table_top_z + BOARD_SIZE[2] * 0.5 + BOARD_Z_OFFSET
        board_pos = self.scene.board.init_state.pos
        self.scene.board.init_state.pos = (board_pos[0], board_pos[1], board_z)


# Backward-compat alias.
TurnOnSwitchEnvCfg = TurnOnSwitchEnvFloatingDexHandRightCfg
