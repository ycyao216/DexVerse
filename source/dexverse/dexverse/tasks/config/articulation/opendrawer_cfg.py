# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-drawer task: slide the drawer past 30% of its travel range.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`. The drawer
slide has a soft stabilising actuator (so the drawer doesn't slam open
or closed). Reward is the base's ``joint_open_reward`` with an absolute
travel threshold; success is a ratio-mode ``joint_relative_move``.
"""

from __future__ import annotations

from dexverse.assets import PARTNET_MOBILITY_ARTICULATIONS_DIR
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY

ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "storage_unit"
DRAWER_USD_PATH = str(ASSET_DIR / "45575.usd")
DRAWER_SCALE = (0.35, 0.35, 0.35)
DRAWER_HALF_HEIGHT_EST = 0.32
DRAWER_JOINT_NAME = "joint_1"
INITIAL_OPEN_FRACTION = 0.05
CLOSE_POS = 0.0
OPEN_POS = 1.0
OPEN_SUCCESS_FRACTION = 0.3


@configclass
class OpenDrawerRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward when the drawer crosses an absolute travel threshold."""

    open_amount = RewTerm(
        func=mdp.joint_open_reward,
        weight=5.0,
        params={
            "threshold_rad": CLOSE_POS + (OPEN_POS - CLOSE_POS) * OPEN_SUCCESS_FRACTION,
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[DRAWER_JOINT_NAME]),
        },
    )


@configclass
class OpenDrawerTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success once the drawer has slid ``OPEN_SUCCESS_FRACTION`` of its range."""

    success = DoneTerm(
        func=mdp.joint_relative_move,
        params={
            "threshold": OPEN_SUCCESS_FRACTION,
            "mode": "ratio",
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[DRAWER_JOINT_NAME]),
        },
    )


@configclass
class OpenDrawerEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Open-drawer env (floating dex-hand variant)."""

    articulation_usd_path: str = DRAWER_USD_PATH
    articulation_scale: tuple[float, float, float] = DRAWER_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.3, 0.0, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = DRAWER_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True

    success_joint_names: list[str] = [DRAWER_JOINT_NAME]
    # success_threshold left at None: ratio-mode ``joint_relative_move``
    # doesn't read the base threshold.

    rewards: OpenDrawerRewardsCfg = OpenDrawerRewardsCfg()
    terminations: OpenDrawerTerminationsCfg = OpenDrawerTerminationsCfg()

    def __post_init__(self):
        # Pre-open the drawer slightly so contact starts with the slide engaged.
        self.articulation_init_joint_pos = {DRAWER_JOINT_NAME: INITIAL_OPEN_FRACTION}
        super().__post_init__()
        # Stabilising actuator on the drawer slide so the drawer doesn't
        # slam closed under contact.
        self.scene.articulation.actuators = {
            "drawer_target_actuator": ImplicitActuatorCfg(
                joint_names_expr=[DRAWER_JOINT_NAME],
                effort_limit_sim={DRAWER_JOINT_NAME: 2.0},
                stiffness={DRAWER_JOINT_NAME: 0.1},
                damping={DRAWER_JOINT_NAME: 0.15},
                friction={DRAWER_JOINT_NAME: 0.05},
            ),
        }


# Backward-compat alias.
OpenDrawerEnvCfg = OpenDrawerEnvFloatingDexHandRightCfg
