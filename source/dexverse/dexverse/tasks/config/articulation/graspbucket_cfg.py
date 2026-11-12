# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp-bucket task: lift the bucket handle to within 10 degrees of upright.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`. The bucket
starts with the handle at its lower joint limit. Reward is
``joint_range_progress``; recording requires 10 consecutive steps near upright.
"""

from __future__ import annotations

import math

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
# In this USD the handle's arch is tilted 56.115 degrees at joint zero.
# Its upright pose is therefore -56.115 degrees, not the upper joint limit.
# Check a bounded interval so folding it down the opposite side cannot succeed.
BUCKET_UPRIGHT_ANGLE = math.radians(-56.115)
BUCKET_UPRIGHT_TOLERANCE = math.radians(10.0)
# Bucket radius (~12 cm) plus a conservative margin around the hand links.
BUCKET_HAND_CLEARANCE = 0.16


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
    """Success only while the handle is nearly vertical above the bucket."""

    success = DoneTerm(
        func=mdp.joint_reach_threshold,
        params={
            "threshold": BUCKET_UPRIGHT_ANGLE,
            "op": "==",
            "tol": BUCKET_UPRIGHT_TOLERANCE,
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[BUCKET_JOINT_NAME]),
        },
    )


@configclass
class GraspBucketEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Grasp-bucket env (floating dex-hand variant)."""

    articulation_usd_path: str = BUCKET_USD_PATH
    articulation_scale: tuple[float, float, float] = BUCKET_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.0, 0.1, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # USD lower limit for joint_0: -151.19998168945312 deg.
    articulation_init_joint_pos: dict[str, float] = {BUCKET_JOINT_NAME: -2.6389375094360954}
    articulation_half_height_est: float = BUCKET_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True
    # Sample a patch near the hand instead of a fixed-x line. The 16 cm
    # hand-clearance rejection below still applies to every sampled pose.
    # Both axes vary by +/-20 cm. Lateral offsets are relative to y=0.10 m
    # (absolute range -0.10--0.30 m); x is centered on 0.0 m.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.2, 0.2],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-math.pi / 6, math.pi / 6],
    }

    success_joint_names: list[str] = [BUCKET_JOINT_NAME]
    # Keep success_threshold unset so the base preserves the upright interval.

    rewards: GraspBucketRewardsCfg = GraspBucketRewardsCfg()
    terminations: GraspBucketTerminationsCfg = GraspBucketTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Robot reset terms run first, so body poses describe the initial hand,
        # even if the operator ended the previous episode somewhere else.
        self.events.reset_articulation.func = mdp.reset_root_pose_uniform_excluding
        self.events.reset_articulation.params.update(
            reference_asset_cfg=SceneEntityCfg("robot", body_names=[".*"]),
            use_reference_bodies=True,
            min_xy_distance=BUCKET_HAND_CLEARANCE,
            max_attempts=512,
            fail_on_exhaustion=True,
        )


# Backward-compat alias.
GraspBucketEnvCfg = GraspBucketEnvFloatingDexHandRightCfg
