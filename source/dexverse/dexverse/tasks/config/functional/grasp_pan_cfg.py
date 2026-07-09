# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional grasping: pan (handle grasp) and place flat on a stove top."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from dexverse.assets import SYNTHESIS_DIR
from dexverse.tasks.config.floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG
from dexverse.tasks.config.robot_init import (
    align_retargeter_wrist_origin_to_init,
    set_robot_wrist_init_world_pos,
)
from isaaclab.assets import ArticulationCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation as R

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..articulation.articulation_base.usd_helpers import ensure_single_articulation_root
from .base_cfg import ForbiddenZone, FunctionalPourEnvFloatingDexHandRightCfg

PAN_USD_PATH = str(SYNTHESIS_DIR / "FryingPan" / "model_FryingPan_69323.usd")
PAN_ROT_INIT = (1.0, 0.0, 0.0, 0.0)

STOVE_USD_PATH = str(SYNTHESIS_DIR / "cooker007" / "model_cooker_007.usd")
STOVE_SCALE = (1.0, 1.0, 1.0)
# Placement of the stove relative to the table centre (x, y, z). The z entry
# is the clearance above the tabletop where the stove base rests. Tune.
STOVE_INIT_OFFSET = (0.0, 0.0, 0.0)
# 90 degrees around z-axis
STOVE_INIT_ROT = tuple((R.from_euler("X", -90, degrees=True)).as_quat().tolist())
# Offset (in the stove's local frame) from the stove origin to the place-goal,
# i.e. the burner centre. Tune so the goal sits over the cooking surface.
GOAL_OFFSET_FROM_STOVE = (0.21, 0.025, 0.12)


def _build_stove_cfg(
    *,
    scale: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
    init_pos: tuple[float, float, float],
) -> ArticulationCfg:
    """Articulated stove prop, anchored to the world.

    The cooker USD is an articulation: its body is the articulation root and
    the knobs are sibling rigid bodies linked by revolute joints. Wrapping it
    as a single RigidObjectCfg triggers ``Rigid Body of <knob> missing
    xformstack reset when child of another enabled rigid body`` errors in
    PhysX. Spawning it as an ArticulationCfg with ``fix_root_link=True``
    handles the hierarchy correctly and keeps the stove anchored on the
    table.
    """
    cleaned_usd = ensure_single_articulation_root(STOVE_USD_PATH)
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Stove",
        spawn=sim_utils.UsdFileCfg(
            usd_path=cleaned_usd,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(
                max_effort=0.0,
                stiffness=0.0,
                damping=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
        actuators={},
    )


@configclass
class GraspPanEnvFloatingDexHandRightCfg(FunctionalPourEnvFloatingDexHandRightCfg):
    """Grasp the pan and place it flat onto the stove burner.

    Uses the same ``lift_and_tilt_with_contact_zones`` success template as
    the pour tasks but inverted: success requires the pan to be near
    horizontal (tilt ≤ a small threshold) AND the pan's xy to land within
    a small radius of the success-marker xy. The success marker is
    synced each reset to ``stove.root_pos + R(stove.root_quat) *
    goal_offset_from_stove`` so the goal automatically follows the
    randomized stove pose.
    """

    usd_path: str = PAN_USD_PATH
    object_mass: float = 0.5
    object_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    object_half_height: float = 0.025
    object_static_friction: float | None = 2.0
    object_dynamic_friction: float | None = 2.0
    object_friction_combine_mode: str = "average"
    table_clearance: float = 0.0
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    object_init_rot: tuple[float, float, float, float] = PAN_ROT_INIT
    object_collision_enabled: bool = False
    object_reset_x_range: tuple[float, float] = (-0.15, -0.05)
    object_reset_y_range: tuple[float, float] = (-0.3, 0.3)
    object_reset_yaw_range: tuple[float, float] = (3.14 - 0.3, 3.14 + 0.3)

    forbidden_zones: tuple[ForbiddenZone, ...] = (
        ForbiddenZone(
            kind="cylinder",
            center=(0.0, 0.0, 0.015),
            radius=0.15,
            half_height=0.01,
        ),
    )

    # ---- Pour-style success criterion (inverted: pan should stay flat) ----
    # Lift gate: 0.0 = "pan hasn't fallen below spawn z" (always true while
    # held above the table). The pan can rest *on* the stove top, so no
    # actual lift is required.
    pour_lift_height: float = 0.0
    # Primary tilt gate: angle between pan local +Z and world +Z must be
    # ≤ 20° (with ``pour_tilt_ge=False``). Pan local +Z is the cooking-
    # surface normal at rest; tighten the threshold for a stricter "flat"
    # requirement.
    pour_angle_rad: float = math.radians(10)
    pour_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pour_world_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pour_tilt_ge: bool = False
    # xy gate: pan centre within 5 cm of the success marker (= burner xy).
    pour_goal_xy_threshold: float | None = 0.025
    # Measure xy from the pan's mass centre. Set to e.g. ``(0.1, 0, 0)``
    # if you want to anchor on the cooking-surface centre instead.
    pour_goal_object_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Disable the secondary plane-angle gate: it's redundant with the
    # primary "axis vs +Z" gate for a flat-placement task. Re-enable by
    # setting both fields if you want an OR fallback.
    pour_plane_axis_local: tuple[float, float, float] | None = None
    pour_plane_angle_threshold_rad: float | None = None

    # Goal indicator: sphere at the goal that animates red -> green as the
    # pan approaches flat (green = within the tilt threshold). Visual only;
    # disable for headless / faster rendering.
    pour_show_progress_marker: bool = False

    # ---- Stove prop ----
    stove_usd_path: str = STOVE_USD_PATH
    stove_scale: tuple[float, float, float] = STOVE_SCALE
    stove_init_offset: tuple[float, float, float] = STOVE_INIT_OFFSET
    stove_init_rot: tuple[float, float, float, float] = STOVE_INIT_ROT
    # Goal-position offset from the stove origin in the stove's local frame.
    goal_offset_from_stove: tuple[float, float, float] = GOAL_OFFSET_FROM_STOVE

    # Per-reset stove randomization, expressed as offsets from its spawn pose.
    stove_reset_x_range: tuple[float, float] = (0.2, 0.4)
    stove_reset_y_range: tuple[float, float] = (-0.3, 0.3)
    stove_reset_yaw_range: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        table_pos = self.scene.table.init_state.pos

        stove_pos = (
            table_pos[0] + self.stove_init_offset[0],
            table_pos[1] + self.stove_init_offset[1],
            table_top_z + self.stove_init_offset[2],
        )
        self.scene.stove = _build_stove_cfg(
            scale=self.stove_scale,
            init_rot=self.stove_init_rot,
            init_pos=stove_pos,
        )

        # Reorder reset events so ``reset_stove`` runs *before*
        # ``reset_success_marker`` (the marker is sync'd to the stove's
        # randomized pose, so it has to read the post-randomization pose).
        # The event manager iterates ``cfg.__dict__.items()`` in insertion
        # order, and re-assignment doesn't change position, so we
        # explicitly delete the inherited entry and add the new one after
        # ``reset_stove``.
        if "reset_success_marker" in self.events.__dict__:
            del self.events.reset_success_marker
        self.events.reset_stove = EventTerm(
            func=mdp.reset_root_pose_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(self.stove_reset_x_range),
                    "y": list(self.stove_reset_y_range),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": list(self.stove_reset_yaw_range),
                },
                "asset_cfg": SceneEntityCfg("stove"),
            },
        )
        # Sync the success marker to ``stove.root_pos + R(stove.root_quat)
        # * goal_offset_from_stove``. The pour base hides the marker's
        # cuboid pole (opacity 0) and the colored progress sphere becomes
        # the visible goal indicator.
        self.events.reset_success_marker = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("success_marker"),
                "source_cfg": SceneEntityCfg("stove"),
                "source_local_offset": tuple(self.goal_offset_from_stove),
                "z_offset": -0.05,
                "quat": (1.0, 0.0, 0.0, 0.0),
            },
        )

        # Pull the wrist ~15 cm further from the table and shift the XR
        # anchor to match. World coords so the same intent works for armed
        # robots (UR10e re-IKs the arm to land the palm at the same world pose).
        # x = floating_shadow base x (-0.75) + 0.35.
        set_robot_wrist_init_world_pos(self, x=-0.40)
        self.xr = XrCfg(
            anchor_pos=[-0.65, 0.0, 0.1],
            anchor_rot=DEFAULT_FLOATING_SHADOW_XR_CFG.anchor_rot,
        )
        align_retargeter_wrist_origin_to_init(self)
