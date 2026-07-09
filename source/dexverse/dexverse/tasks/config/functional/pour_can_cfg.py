# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional grasping: pour can into a bowl."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from dexverse.assets import SYNTHESIS_DIR, YCB_DIR
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .base_cfg import (
    POUR_ANGLE_RAD,
    ForbiddenZone,
    FunctionalPourEnvFloatingDexHandRightCfg,
    FunctionalPourObservationsCfg,
    build_object_cfg_from_usd,
)

CAN_USD_PATH = str(YCB_DIR / "005_tomato_soup_can" / "tomato_soup_can.usd")
CAN_SCALE = (1.0, 1.0, 1.0)
CAN_MASS = 0.3
CAN_HALF_HEIGHT_EST = 0.05
CAN_ROT_INIT = (0.707107, -0.707107, 0.0, 0.0)
CAN_FORBIDDEN_ZONE_ROT_OFFSET = (
    math.cos(math.pi / 4.0),
    math.sin(math.pi / 4.0),
    0.0,
    0.0,
)
CENTER_SQUARE_SIZE = 0.45

BOWL_USD_PATH = str(
    SYNTHESIS_DIR / "Bowl_B0888F8FR2_SpiralCenter_1_TU" / "model_Bowl_B0888F8FR2_SpiralCenter_1_TU_69323.usd"
)
BOWL_SCALE = (1.5, 1.5, 1.5)
BOWL_HALF_HEIGHT_EST = 0.001

CAN_CFG = build_object_cfg_from_usd(
    CAN_USD_PATH,
    mass=CAN_MASS,
    scale=CAN_SCALE,
    init_rot=CAN_ROT_INIT,
    collision_enabled=True,
)
PickUpCanObservationsCfg = FunctionalPourObservationsCfg


def _build_bowl_cfg(
    *,
    scale: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
    init_pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Kinematic bowl prop. Re-synced under the pour goal each reset."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=BOWL_USD_PATH,
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
class PourCanEnvFloatingDexHandRightCfg(FunctionalPourEnvFloatingDexHandRightCfg):
    """Lift and tilt the tomato soup can over a bowl prop."""

    usd_path: str = CAN_USD_PATH
    object_mass: float = CAN_MASS
    object_scale: tuple[float, float, float] = CAN_SCALE
    object_half_height: float = CAN_HALF_HEIGHT_EST
    object_static_friction: float | None = 2.5
    object_dynamic_friction: float | None = 2.5
    object_friction_combine_mode: str = "average"
    table_clearance: float = 0.0
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    object_init_rot: tuple[float, float, float, float] = CAN_ROT_INIT
    object_collision_enabled: bool = False

    object_reset_x_range: tuple[float, float] = (-0.15, 0.0)
    object_reset_y_range: tuple[float, float] = (-0.15, 0.15)
    object_reset_yaw_range: tuple[float, float] = (0.0, 0.0)

    pour_lift_height: float = 0.2
    pour_angle_rad: float = POUR_ANGLE_RAD
    pour_axis_local: tuple[float, float, float] = (0.0, -1.0, 0.0)
    pour_tilt_ge: bool = True

    # Tilt success uses only the directional primary gate (``pour_angle_rad``
    # about ``pour_axis_local`` vs world +Z). The secondary plane-angle gate is
    # left disabled (base default ``None``): it's undirected (can't tell a
    # tipped-down pour from an upright lift) and at a 90° threshold never fired.

    # Shifts the xy gate measurement point from the object's mass centre
    # to ``object.root_pos + R(object.root_quat) * pour_goal_object_local_offset``
    # (object-local coordinates). Useful when the relevant point is the
    # spout / lip rather than the bounding-box centre.
    pour_goal_object_local_offset: tuple[float, float, float] = (0.0, -0.0475, 0.0)  # TODO: tune to the can's lip

    # Goal indicator: sphere at the goal that animates red -> green as the
    # object tilts toward the threshold (green = tilt criterion met). Visual
    # only; disable for headless / faster rendering.
    pour_show_progress_marker: bool = False

    forbidden_zones: tuple[ForbiddenZone, ...] = (
        ForbiddenZone(
            kind="cylinder",
            center=(0.0, -0.048, 0.0),
            radius=0.03,
            half_height=0.005,
            rotation_offset=CAN_FORBIDDEN_ZONE_ROT_OFFSET,
        ),
    )

    bowl_usd_path: str = BOWL_USD_PATH
    bowl_scale: tuple[float, float, float] = BOWL_SCALE
    bowl_half_height: float = BOWL_HALF_HEIGHT_EST
    bowl_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Pour-goal randomization (offsets from the table centre). The bowl is
    # slaved to this same xy via ``reset_bowl`` below, so the goal
    # indicator and the bowl prop always share their location. Defaults
    # place the goal in front of the table centre so it never overlaps
    # the can's reset zone (x ∈ [-0.15, 0.0]).
    goal_reset_x_range: tuple[float, float] = (0.0, 0.15)
    goal_reset_y_range: tuple[float, float] = (-0.15, 0.15)

    # Minimum xy distance (m) the pour goal/bowl must keep from the can
    # spawn. Enforced via rejection sampling at reset time.
    min_object_goal_xy_distance: float = 0.1

    def __post_init__(self):
        super().__post_init__()

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        marker_pos = self.scene.success_marker.init_state.pos

        # Spawn the bowl directly under the success marker so the goal
        # indicator and the bowl prop are visually paired even at frame 0
        # (before the first reset event runs). Both x/y/z come from the
        # marker pose so they can never disagree by accident; only z is
        # dropped to the tabletop.
        bowl_cfg = _build_bowl_cfg(
            scale=self.bowl_scale,
            init_rot=self.bowl_init_rot,
            init_pos=(marker_pos[0], marker_pos[1], table_top_z + self.bowl_half_height),
        )
        self.scene.bowl = bowl_cfg

        # Replace the base sync (marker tied to can) with a uniform xy
        # randomization for the pour goal indicator. roll/pitch/yaw deltas
        # default to zero so the marker keeps its tilted SUCCESS_MARKER_QUAT.
        # The excluding variant rejection-samples xy so the marker (and the
        # bowl it drives via reset_bowl) stays at least
        # ``min_object_goal_xy_distance`` away from the can spawn.
        self.events.reset_success_marker = EventTerm(
            func=mdp.reset_root_pose_uniform_excluding,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(self.goal_reset_x_range),
                    "y": list(self.goal_reset_y_range),
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

        # Slave the bowl to the (now-randomized) marker xy and drop it
        # onto the tabletop. This event is appended after
        # ``reset_success_marker``, so on every reset the marker pose is
        # resampled first and the bowl then snaps under it. The
        # marker-driven xy is the single source of truth -- bowl and goal
        # cannot diverge.
        marker_z = self.scene.success_marker.init_state.pos[2]
        bowl_z_offset = table_top_z + self.bowl_half_height - marker_z
        self.events.reset_bowl = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("bowl"),
                "source_cfg": SceneEntityCfg("success_marker"),
                "z_offset": bowl_z_offset,
                "quat": self.bowl_init_rot,
            },
        )
