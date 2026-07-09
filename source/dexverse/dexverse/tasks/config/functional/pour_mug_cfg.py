# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional grasping: pour mug."""

from __future__ import annotations

import numpy as np
from dexverse.assets import DEXVERSE_AUTHORED_ASSETS_DIR
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation as R

from ... import mdp
from .base_cfg import (
    DEFAULT_FLOATING_SHADOW_XR_CFG,
    POUR_ANGLE_RAD,
    ForbiddenZone,
    FunctionalPourEnvCfg,
    FunctionalPourEnvFloatingDexHandRightCfg,
    FunctionalPourObservationsCfg,
    build_object_cfg_from_usd,
)

MUG_USD_PATH = str(DEXVERSE_AUTHORED_ASSETS_DIR / "mug" / "SM_Mug_A2.usd")
MUG_SCALE = (0.015, 0.015, 0.015)
MUG_MASS = 0.3
MUG_Z_OFFSET = 0.05

# rotate 90 degrees around z axis
rot_quat = R.from_euler("x", np.pi / 2).as_quat()

MUG_INIT_QUAT = tuple(rot_quat.tolist())
CENTER_SQUARE_SIZE = 0.45

MUG_CFG = build_object_cfg_from_usd(
    MUG_USD_PATH,
    mass=MUG_MASS,
    scale=MUG_SCALE,
    init_rot=MUG_INIT_QUAT,
    collision_enabled=False,
)
PourMugObservationsCfg = FunctionalPourObservationsCfg


@configclass
class PourMugEnvCfg(FunctionalPourEnvCfg):
    """Lift and tilt a mug into a pouring pose."""

    usd_path: str = MUG_USD_PATH
    object_mass: float = MUG_MASS
    object_scale: tuple[float, float, float] = MUG_SCALE
    object_half_height: float = MUG_Z_OFFSET
    object_static_friction: float | None = 2.0
    object_dynamic_friction: float | None = 2.0
    object_friction_combine_mode: str = "average"
    table_clearance: float = 0.0
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    object_init_rot: tuple[float, float, float, float] = MUG_INIT_QUAT
    object_collision_enabled: bool = False

    object_reset_x_range: tuple[float, float] = (-0.1, 0.2)
    object_reset_y_range: tuple[float, float] = (-0.35, 0.35)
    object_reset_yaw_range: tuple[float, float] = (-0.3, 0.3)

    pour_lift_height: float = 0.2
    pour_angle_rad: float = POUR_ANGLE_RAD
    pour_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pour_tilt_ge: bool = True

    # Tilt success uses only the directional primary gate (``pour_angle_rad``
    # about ``pour_axis_local`` vs world +Z). The secondary plane-angle gate is
    # left disabled (base default ``None``): it's undirected (can't tell a
    # tipped-down pour from an upright lift) and at a 90° threshold never fired.

    # Shifts the xy gate measurement point from the object's mass centre
    # to ``object.root_pos + R(object.root_quat) * pour_goal_object_local_offset``
    # (object-local coordinates). Useful when the relevant point is the
    # spout / lip rather than the bounding-box centre.
    pour_goal_object_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.14)  # TODO: tune to the mug's rim

    # Pour-goal randomization: xy offsets from the table centre. Defaults
    # bias the goal away from the mug's reset zone (x ∈ (-0.1, 0.2)) so the
    # pour target rarely overlaps the spawn region. Tune as needed.
    goal_reset_x_range: tuple[float, float] = (0.2, 0.35)
    goal_reset_y_range: tuple[float, float] = (-0.25, 0.25)

    # Minimum xy distance (m) the pour goal must keep from the mug spawn.
    # Enforced via rejection sampling at reset time.
    min_object_goal_xy_distance: float = 0.10

    # Goal indicator: sphere at the goal that animates red -> green as the
    # object tilts toward the threshold (green = tilt criterion met). Visual
    # only; disable for headless / faster rendering.
    pour_show_progress_marker: bool = False

    forbidden_zones: tuple[ForbiddenZone, ...] = (
        ForbiddenZone(kind="cylinder", center=(0.0, 0.0, 0.14), radius=0.065, half_height=0.02),
    )

    def __post_init__(self):
        super().__post_init__()

        # The base FunctionalPourEnvCfg syncs the success marker to the
        # mug's reset pose (so the pour goal floats directly above the
        # mug). Replace that with uniform xy randomization so the goal
        # actually moves to a different location each episode -- matches
        # the grasp_bleach / pour_can pattern. The marker's init z (set
        # by the base) keeps the goal at ``pour_lift_height`` above the
        # tabletop.
        # Rejection-sample the goal xy so it keeps
        # ``min_object_goal_xy_distance`` from the mug's reset xy.
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


@configclass
class PourMugEnvFloatingDexHandRightCfg(PourMugEnvCfg, FunctionalPourEnvFloatingDexHandRightCfg):
    """Floating-hand version of the functional pour-mug task."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG
