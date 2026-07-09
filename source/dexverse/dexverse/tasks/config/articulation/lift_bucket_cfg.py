# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift-bucket task: pick the whole bucket up by its handle.

A variant of ``graspbucket_cfg`` (which only rotates the handle in place). Key
differences:

* **Free-floating bucket** (``fix_root_link=False``): the bucket is no longer
  pinned to the table, so the robot can actually lift it off the surface.
* **Smaller** bucket: ``BUCKET_SCALE`` is 0.75x the grasp-bucket scale (0.3 ->
  0.225) so a single floating hand can pick it up.
* **Real mass**: the PartNet bucket authors mass = 0 on every link (PhysX would
  give it a near-zero, ill-conditioned inertia). We set a concrete
  ``BUCKET_MASS`` via the spawn so it is a stable, liftable dynamic body.
* **Passive handle joint**: ``stiffness = 0`` (no restoring spring -- unlike the
  grasp task) with a little Coulomb ``friction`` so the handle is a free pivot
  the bucket hangs from when lifted. ``armature`` conditions the otherwise
  near-zero-inertia friction/limit solve so the hinge behaves (see the
  articulation-hinge-armature note).
* **Goal = lift height**: success is the bucket root lifted ``BUCKET_LIFT_TARGET``
  above its resting height (object-lift reward + sparse bonus), instead of
  handle joint progress.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
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

ASSET_DIR = PARTNET_MOBILITY_ARTICULATIONS_DIR / "bucket" / "100431"
BUCKET_USD_PATH = str(ASSET_DIR / "100431.usd")

# 0.75x the grasp-bucket scale (0.3) -> a smaller, hand-liftable bucket.
BUCKET_SCALE = (0.225, 0.225, 0.225)
# Origin->bottom of the bucket at this scale (~0.465 * 0.225); seats it on the
# table with a few mm of drop. The free body settles from here.
BUCKET_HALF_HEIGHT_EST = 0.11
# PartNet bucket authors mass = 0 on every link; give it a concrete, stable mass.
BUCKET_MASS = 0.3

BUCKET_JOINT_NAME = "joint_0"
# Lift the bucket root this far (m) above its resting height to succeed.
BUCKET_LIFT_TARGET = 0.12

# Passive handle hinge: stiffness 0 (no spring-back), light Coulomb friction so
# it holds where left, armature to condition the near-zero-inertia solve.
# TUNE BUCKET_JOINT_FRICTION if the handle sags/flops (raise) or is too stiff to
# grasp-and-tilt (lower); raise BUCKET_JOINT_ARMATURE toward 0.01 if it jitters.
BUCKET_JOINT_FRICTION = 0.1
BUCKET_JOINT_ARMATURE = 0.005
BUCKET_JOINT_DAMPING = 0.0
BUCKET_JOINT_EFFORT = 5.0


def bucket_lifted_reward(
    env, min_height: float = BUCKET_LIFT_TARGET, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Sparse success bonus: 1.0 once the bucket is lifted past the target."""
    asset_cfg = asset_cfg or SceneEntityCfg(ARTICULATION_KEY)
    return mdp.object_lifted(env, asset_cfg=asset_cfg, min_height=min_height).float()


@configclass
class LiftBucketRewardsCfg(dexverse_base_env.RewardsCfg):
    """Approach the bucket, then lift it; sparse bonus on reaching the target."""

    fingers_to_bucket = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            # body_names wired to fingertips in __post_init__.
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg(ARTICULATION_KEY),
        },
    )
    lift_progress = RewTerm(
        func=mdp.object_lift_height,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY), "min_height": BUCKET_LIFT_TARGET},
    )
    lifted_success = RewTerm(
        func=bucket_lifted_reward,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY), "min_height": BUCKET_LIFT_TARGET},
    )


@configclass
class LiftBucketTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when the bucket root is lifted ``BUCKET_LIFT_TARGET`` above rest."""

    success = DoneTerm(
        func=mdp.object_lifted,
        params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY), "min_height": BUCKET_LIFT_TARGET},
    )


@configclass
class LiftBucketEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Lift-bucket env (floating dex-hand variant): pick the bucket up by the handle."""

    articulation_usd_path: str = BUCKET_USD_PATH
    articulation_scale: tuple[float, float, float] = BUCKET_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.2, 0.1, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = BUCKET_HALF_HEIGHT_EST
    # Start (and reset) the handle hinge at 30 degrees. The hinge stays passive
    # (see actuators in __post_init__); this only sets its initial angle.
    articulation_init_joint_pos: dict[str, float] = {BUCKET_JOINT_NAME: math.radians(40.0)}
    # Free-floating so the bucket can be lifted off the table.
    articulation_fix_root_link: bool | None = False
    # Modest per-reset jitter; keep the bucket in the hand's workspace.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.05, 0.15],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.3, 0.3],
    }

    success_joint_names: list[str] = [BUCKET_JOINT_NAME]
    # success_threshold left None: success is lift-height, not a joint threshold.

    rewards: LiftBucketRewardsCfg = LiftBucketRewardsCfg()
    terminations: LiftBucketTerminationsCfg = LiftBucketTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Give the zero-mass PartNet bucket a concrete, stable mass.
        self.scene.articulation.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=BUCKET_MASS)

        # Fully passive handle hinge with light friction (see module docstring).
        self.scene.articulation.actuators = {
            "bucket_passive_joint": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=BUCKET_JOINT_DAMPING,
                friction=BUCKET_JOINT_FRICTION,
                armature=BUCKET_JOINT_ARMATURE,
                effort_limit_sim=BUCKET_JOINT_EFFORT,
            ),
        }

        # Point the approach-distance reward at the fingertips.
        self.rewards.fingers_to_bucket.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


# Backward-compat / convenience alias.
LiftBucketEnvCfg = LiftBucketEnvFloatingDexHandRightCfg
