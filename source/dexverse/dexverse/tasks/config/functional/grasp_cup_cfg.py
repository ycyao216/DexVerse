# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional grasping: cup (rim avoidance)."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import (
    POUR_ANGLE_RAD,
    SYNTHESIS_DIR,
    ForbiddenZone,
    FunctionalPourEnvFloatingDexHandRightCfg,
)

CUP_USD_PATH = str(SYNTHESIS_DIR / "Cup_B0CMD4LX4D_ForestGreen" / "model_Cup_B0CMD4LX4D_ForestGreen_69323.usd")
CUP_ROT_INIT = (1, 0, 0, 0)


@configclass
class GraspCupEnvFloatingDexHandRightCfg(FunctionalPourEnvFloatingDexHandRightCfg):
    """Functional cup task: avoid the rim, then lift and pour.

    Forbidden zone is a placeholder around the rim / drinking edge in the
    object's local frame. Tune ``center`` / ``radius`` once you can preview
    the spawned asset.
    """

    usd_path: str = CUP_USD_PATH
    object_mass: float = 0.2
    object_scale: tuple[float, float, float] = (0.9, 0.9, 0.9)
    object_half_height: float = 0.025
    object_static_friction: float | None = 2.5
    object_dynamic_friction: float | None = 2.5
    object_friction_combine_mode: str = "average"
    table_clearance: float = 0.0
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    object_init_rot: tuple[float, float, float, float] = CUP_ROT_INIT
    object_reset_x_range: tuple[float, float] = (-0.15, 0.15)
    object_reset_y_range: tuple[float, float] = (-0.15, 0.15)
    object_reset_yaw_range: tuple[float, float] = (0.0, 0.0)
    target_height_range: tuple[float, float] = (0.10, 0.20)

    pour_lift_height: float = 0.3
    pour_angle_rad: float = POUR_ANGLE_RAD
    pour_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pour_tilt_ge: bool = True

    # Tilt success uses only the directional primary gate (``pour_angle_rad``
    # about ``pour_axis_local`` vs world +Z). The secondary plane-angle gate is
    # left disabled (base default ``None``): being undirected it can't tell a
    # tipped-down pour from an upright lift, so the previous 80° threshold let
    # a near-upright cup pass the tilt check.

    # Shifts the xy gate measurement point from the object's mass centre
    # to ``object.root_pos + R(object.root_quat) * pour_goal_object_local_offset``
    # (object-local coordinates). Useful when the relevant point is the
    # spout / lip rather than the bounding-box centre.
    pour_goal_object_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.1)  # TODO: tune to the cup's rim

    # Pour-goal randomization: xy offsets from the table centre. Defaults
    # bias the goal toward the front-right of the table so it doesn't sit
    # inside the cup's reset zone (x ∈ (-0.15, 0.15)). Tune as needed.
    goal_reset_x_range: tuple[float, float] = (0.0, 0.35)
    goal_reset_y_range: tuple[float, float] = (-0.25, 0.25)

    # Goal indicator: sphere at the goal that animates red -> green as the
    # object tilts toward the threshold (green = tilt criterion met). Visual
    # only; disable for headless / faster rendering.
    pour_show_progress_marker: bool = False

    forbidden_zones: tuple[ForbiddenZone, ...] = (
        ForbiddenZone(kind="cylinder", center=(0.0, 0.0, 0.11), radius=0.035, half_height=0.005),
    )

    def __post_init__(self):
        super().__post_init__()

        # The base FunctionalPourEnvCfg syncs the success marker to the
        # cup's reset pose (so the goal floats directly above the cup).
        # Replace it with uniform xy randomization so the goal lives
        # somewhere on the table independent of the cup -- matches the
        # grasp_bleach / pour_can / pour_mug pattern. The marker's init z
        # (set by the base) keeps the goal at ``pour_lift_height`` above
        # the tabletop.
        self.events.reset_success_marker = EventTerm(
            func=mdp.reset_root_state_uniform,
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
            },
        )
