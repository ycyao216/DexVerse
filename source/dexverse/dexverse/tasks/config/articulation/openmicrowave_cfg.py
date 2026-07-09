# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-microwave task: rotate the microwave door past 80% of its range.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`; only the
task-specific reward (``joint_open_reward`` toward 90°) and termination
(ratio-mode ``joint_relative_move``) are overridden.
"""

from __future__ import annotations

from dexverse.assets import PARTNET_MOBILITY_ARTICULATIONS_DIR
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY

ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "microwave"
MICROWAVE_USD_PATH = str(ASSET_DIR / "7167.usd")
MICROWAVE_SCALE = (0.35, 0.35, 0.35)
MICROWAVE_HALF_HEIGHT_EST = 0.17
MICROWAVE_JOINT_NAME = "joint_0"
INIT_ANGLE_RAD = 0.2
OPEN_ANGLE_RAD = 1.5707963  # 90°
SUCCESS_OPEN_RATIO = 0.8


@configclass
class OpenMicrowaveRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward proportional to door opening past the configured threshold."""

    open_angle = RewTerm(
        func=mdp.joint_open_reward,
        weight=5.0,
        params={
            "threshold_rad": OPEN_ANGLE_RAD,
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[MICROWAVE_JOINT_NAME]),
        },
    )


@configclass
class OpenMicrowaveTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when the door has covered ``SUCCESS_OPEN_RATIO`` of its range."""

    success = DoneTerm(
        func=mdp.joint_relative_move,
        params={
            "threshold": SUCCESS_OPEN_RATIO,
            "mode": "ratio",
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[MICROWAVE_JOINT_NAME]),
        },
    )


@configclass
class OpenMicrowaveEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Open-microwave env (floating dex-hand variant)."""

    articulation_usd_path: str = MICROWAVE_USD_PATH
    articulation_scale: tuple[float, float, float] = MICROWAVE_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.4, 0.0, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = MICROWAVE_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True

    success_joint_names: list[str] = [MICROWAVE_JOINT_NAME]
    # success_threshold left at None: this task swaps in a ratio-mode
    # ``joint_relative_move`` termination that doesn't use the base threshold.

    rewards: OpenMicrowaveRewardsCfg = OpenMicrowaveRewardsCfg()
    terminations: OpenMicrowaveTerminationsCfg = OpenMicrowaveTerminationsCfg()

    def __post_init__(self):
        # Start the door slightly ajar to avoid a hard-contact transient at t=0.
        self.articulation_init_joint_pos = {MICROWAVE_JOINT_NAME: INIT_ANGLE_RAD}
        super().__post_init__()


# Backward-compat alias for any importer that referenced the old base class.
OpenMicrowaveEnvCfg = OpenMicrowaveEnvFloatingDexHandRightCfg
