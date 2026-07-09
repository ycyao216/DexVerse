# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional grasping: kettle (handle grasp) and pour over a mug."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .base_cfg import (
    SYNTHESIS_DIR,
    ForbiddenZone,
    FunctionalPourEnvFloatingDexHandRightCfg,
)

KETTLE_USD_PATH = str(SYNTHESIS_DIR / "teapot" / "model_Teapot_B09T313HT9_WhiteColorfulFloral_TU_69323.usd")
KETTLE_ROT_INIT = (1.0, 0.0, 0.0, 0.0)

MUG_USD_PATH = str(SYNTHESIS_DIR / "tea_cup" / "model_Cup_B0CYL5PSR3_Orange_69323.usd")
MUG_SCALE = (1.0, 1.0, 1.0)
# Approximate distance from the cup's base to the table-resting plane; tune
# after eyeballing the spawned asset.
MUG_HALF_HEIGHT_EST = 0.0
MUG_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


def _build_mug_cfg(
    *,
    scale: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
    init_pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Kinematic mug prop. Re-synced under the pour goal each reset."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Mug",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=MUG_USD_PATH,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
            ),
            collision_props=None,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
    )


@configclass
class GraspKettleEnvFloatingDexHandRightCfg(FunctionalPourEnvFloatingDexHandRightCfg):
    """Functional kettle task: grasp by the handle, then lift and pour over a mug.

    Success requires the kettle's xy to match the mug's xy (via the
    ``success_marker`` and ``pour_goal_xy_threshold``), the kettle to be
    lifted by ``pour_lift_height``, and the kettle to be tilted past
    ``pour_angle_rad`` about ``pour_axis_local``.
    """

    usd_path: str = KETTLE_USD_PATH
    object_mass: float = 0.5
    object_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    object_half_height: float = 0.04
    object_static_friction: float | None = 2.0
    object_dynamic_friction: float | None = 2.0
    object_friction_combine_mode: str = "average"
    table_clearance: float = 0.0
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    object_init_rot: tuple[float, float, float, float] = KETTLE_ROT_INIT
    object_reset_x_range: tuple[float, float] = (-0.05, 0.15)
    object_reset_y_range: tuple[float, float] = (-0.3, -0.1)
    object_reset_yaw_range: tuple[float, float] = (3.14 - 0.3, 3.14 + 0.3)

    pour_lift_height: float = 0.15
    pour_angle_rad: float = math.radians(45)
    pour_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    pour_tilt_ge: bool = True
    # Max horizontal distance (m) between the kettle and the mug for the
    # pour to count. Inherited default from FunctionalPourEnvCfg is 0.10.
    pour_goal_xy_threshold: float | None = 0.05

    # Tilt success uses only the directional primary gate (``pour_angle_rad``
    # about ``pour_axis_local`` = local +X vs world +Z). The secondary
    # plane-angle gate is left disabled (base default ``None``): being
    # undirected, its 45° threshold accepted the upright kettle (local +Z near
    # world +Z), letting a lift-without-pour pass. Pouring is now required.

    # Shifts the xy gate measurement point from the object's mass centre
    # to ``object.root_pos + R(object.root_quat) * pour_goal_object_local_offset``
    # (object-local coordinates). Useful when the relevant point is the
    # spout / lip rather than the bounding-box centre.
    pour_goal_object_local_offset: tuple[float, float, float] = (0.0, -0.13, 0.12)  # TODO: tune to the kettle's spout

    # Goal indicator: sphere at the goal that animates red -> green as the
    # object tilts toward the threshold (green = tilt criterion met). Visual
    # only; disable for headless / faster rendering.
    pour_show_progress_marker: bool = False

    forbidden_zones: tuple[ForbiddenZone, ...] = (ForbiddenZone(kind="sphere", center=(0.0, -0.13, 0.12), radius=0.03),)

    mug_usd_path: str = MUG_USD_PATH
    mug_scale: tuple[float, float, float] = MUG_SCALE
    mug_half_height: float = MUG_HALF_HEIGHT_EST
    mug_init_rot: tuple[float, float, float, float] = MUG_INIT_ROT
    # Pour-goal randomization range (offsets from the table center). Defaults
    # place the goal in front of the kettle's reset zone so it does not
    # overlap the kettle on spawn.
    mug_reset_x_range: tuple[float, float] = (-0.15, 0.15)
    mug_reset_y_range: tuple[float, float] = (0.05, 0.3)

    # Minimum xy distance (m) the pour goal/mug must keep from the kettle
    # spawn. Enforced via rejection sampling at reset time.
    min_object_goal_xy_distance: float = 0.10

    def __post_init__(self):
        super().__post_init__()

        # Re-tilt the goal marker rod to match this cfg's pour_angle_rad. The
        # base SUCCESS_MARKER_QUAT is computed from the module-level
        # POUR_ANGLE_RAD constant (100°) at import time, so any subclass that
        # overrides pour_angle_rad must rebuild the visual rotation here.
        # Convention matches base_cfg.SUCCESS_MARKER_QUAT: rotation about -x.
        half = self.pour_angle_rad * 0.5
        self.scene.success_marker.init_state.rot = (
            math.cos(half),
            -math.sin(half),
            0.0,
            0.0,
        )
        self.scene.success_marker.spawn.visible = False

        obj_pos = self.scene.object.init_state.pos
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT

        # Spawn the mug with a placeholder pose; reset events below relocate
        # it under the randomized pour goal each episode.
        mug_cfg = _build_mug_cfg(
            scale=self.mug_scale,
            init_rot=self.mug_init_rot,
            init_pos=(obj_pos[0], obj_pos[1], table_top_z + self.mug_half_height),
        )
        self.scene.mug = mug_cfg

        # Replace the base sync (marker tied to the kettle) with a uniform xy
        # randomization for the pour goal indicator. roll/pitch/yaw deltas
        # default to zero so the marker keeps its tilted SUCCESS_MARKER_QUAT.
        # The excluding variant rejection-samples xy so the marker (and the
        # mug it drives via reset_mug) stays at least
        # ``min_object_goal_xy_distance`` away from the kettle spawn.
        self.events.reset_success_marker = EventTerm(
            func=mdp.reset_root_pose_uniform_excluding,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(self.mug_reset_x_range),
                    "y": list(self.mug_reset_y_range),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": [0.0, 0.0],
                },
                "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
                "asset_cfg": SceneEntityCfg("success_marker"),
                "reference_asset_cfg": SceneEntityCfg("object"),
                "min_xy_distance": self.min_object_goal_xy_distance,
            },
        )

        # Slave the mug to the randomized marker xy and drop it onto the
        # tabletop. This event is appended after reset_success_marker, so it
        # runs after the marker has been resampled.
        marker_z = self.scene.success_marker.init_state.pos[2]
        mug_z_offset = table_top_z + self.mug_half_height - marker_z
        self.events.reset_mug = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("mug"),
                "source_cfg": SceneEntityCfg("success_marker"),
                "z_offset": mug_z_offset,
                "quat": self.mug_init_rot,
            },
        )
