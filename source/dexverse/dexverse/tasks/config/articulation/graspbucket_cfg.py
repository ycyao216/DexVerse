# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp-bucket task: rotate the bucket's handle past 20% of its range.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`. The bucket
starts with the handle at its lower joint limit. Reward is
``joint_range_progress``; success is a progress-mode ``joint_relative_move``.
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

ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "bucket" / "100431"
BUCKET_USD_PATH = str(ASSET_DIR / "100431.usd")
BUCKET_SCALE = (0.225, 0.225, 0.225)
BUCKET_HALF_HEIGHT_EST = 0.118
BUCKET_JOINT_NAME = "joint_0"
BUCKET_SUCCESS_RATIO = 0.2


@configclass
class GraspBucketRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward proportional to bucket-handle joint progress."""

    handle_progress = RewTerm(
        func=mdp.joint_range_progress,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[".*"])},
    )


@configclass
class GraspBucketTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when the handle has progressed ``BUCKET_SUCCESS_RATIO`` from init."""

    success = DoneTerm(
        func=mdp.joint_relative_move,
        params={
            "threshold": BUCKET_SUCCESS_RATIO,
            "mode": "progress",
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[BUCKET_JOINT_NAME]),
        },
    )


@configclass
class GraspBucketEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Grasp-bucket env (floating dex-hand variant)."""

    articulation_usd_path: str = BUCKET_USD_PATH
    articulation_scale: tuple[float, float, float] = BUCKET_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.2, 0.2, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # USD lower limit for joint_0: -151.19998168945312 deg.
    articulation_init_joint_pos: dict[str, float] = {BUCKET_JOINT_NAME: -2.6389375094360954}
    articulation_half_height_est: float = BUCKET_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True

    success_joint_names: list[str] = [BUCKET_JOINT_NAME]
    # success_threshold left at None: progress-mode ``joint_relative_move``
    # doesn't read the base threshold.

    rewards: GraspBucketRewardsCfg = GraspBucketRewardsCfg()
    terminations: GraspBucketTerminationsCfg = GraspBucketTerminationsCfg()


# Backward-compat alias.
GraspBucketEnvCfg = GraspBucketEnvFloatingDexHandRightCfg
