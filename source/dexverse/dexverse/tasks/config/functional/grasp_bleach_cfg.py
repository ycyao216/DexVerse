# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional grasping: bleach bottle."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation as R

from ... import mdp
from .base_cfg import (
    POUR_ANGLE_RAD,
    SYNTHESIS_DIR,
    ForbiddenZone,
    FunctionalPourEnvFloatingDexHandRightCfg,
)

BLEACH_USD_PATH = str(SYNTHESIS_DIR / "Bleach_1" / "model_Bleach_1_69323.usd")
BLEACH_ROT_INIT = tuple((R.from_euler("z", -90, degrees=True)).as_quat().tolist())


@configclass
class GraspBleachEnvFloatingDexHandRightCfg(FunctionalPourEnvFloatingDexHandRightCfg):
    """Functional manipulation of a bleach bottle with pour-style success.

    Forbidden zone is a placeholder around the trigger / nozzle area at the
    top of the bottle in the object's local frame. Tune ``center`` / ``radius``
    once you can preview the spawned asset.
    """

    usd_path: str = BLEACH_USD_PATH
    object_mass: float = 1.0
    object_scale: tuple[float, float, float] = (1.7, 1.7, 1.7)
    object_half_height: float = 0.01
    object_static_friction: float | None = 2.5
    object_dynamic_friction: float | None = 2.5
    object_friction_combine_mode: str = "average"
    table_clearance: float = 0.0
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    # 180 degrees around y-axis, then -90 degrees around z-axis, w,x,y,z.
    object_init_rot: tuple[float, float, float, float] = BLEACH_ROT_INIT
    object_reset_x_range: tuple[float, float] = (0.1, 0.3)
    object_reset_y_range: tuple[float, float] = (-0.4, 0.0)
    object_reset_pitch_range: tuple[float, float] = (math.radians(150) - 0.3, math.radians(150) + 0.3)
    pour_lift_height: float = 0.3
    pour_angle_rad: float = POUR_ANGLE_RAD
    pour_axis_local: tuple[float, float, float] = (0.0, -1.0, 0.0)
    pour_tilt_ge: bool = True
    pour_goal_xy_threshold: float | None = 0.02

    # Tilt success uses only the directional primary gate (``pour_angle_rad``
    # about ``pour_axis_local`` vs world +Z). The secondary plane-angle gate is
    # left disabled (base default ``None``): it's undirected (can't tell a
    # tipped-down pour from an upright lift) and at a 90° threshold never fired.

    # Pour-goal randomization: xy offsets from the table centre. Defaults bias
    # the goal toward the front-left of the table so it doesn't sit on top of
    # the bottle's reset zone (x in [0.1, 0.3]).
    goal_reset_x_range: tuple[float, float] = (0.0, 0.4)
    goal_reset_y_range: tuple[float, float] = (0.0, 0.4)

    # The xy gate measures distance from this point on the bottle (in its
    # local frame) to the success marker, instead of from the bottle's
    # mass centre. The placeholder matches the existing forbidden-zone
    # centre at the trigger/nozzle area; tune to the actual spout location.
    pour_goal_object_local_offset: tuple[float, float, float] = (0.0, -0.4, -0.03)

    # Goal indicator: hide the cuboid pole and show a sphere at the goal that
    # tracks the success marker and animates red -> green as the bottle tilts
    # toward the threshold (turns green when the tilt criterion is met). Purely
    # visual; does not affect the success criterion. Disable for headless /
    # faster rendering.
    pour_show_progress_marker: bool = False

    forbidden_zones: tuple[ForbiddenZone, ...] = (
        ForbiddenZone(kind="sphere", center=(0.0, -0.4, -0.03), radius=0.035),
    )
    object_collision_enabled: bool = False

    def __post_init__(self):
        super().__post_init__()

        # The base FunctionalPourEnvCfg syncs the success marker to the
        # object's reset pose (so the goal always floats directly above the
        # bottle). Replace it with uniform xy randomization in the table
        # frame; the marker's init z (set by the base) keeps the goal at
        # ``pour_lift_height`` above the tabletop.
        # The marker is a kinematic body, so use the pose-only reset
        # (``reset_root_pose_uniform``); ``reset_root_state_uniform`` also writes
        # velocity, which PhysX rejects on a kinematic body ("Body must be
        # non-kinematic!").
        self.events.reset_success_marker = EventTerm(
            func=mdp.reset_root_pose_uniform,
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
                "asset_cfg": SceneEntityCfg("success_marker"),
            },
        )
