# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional manipulation: power-drill application.

The robot must grasp the power drill and place its bit on a designated drill
spot — implemented as a kinematic cuboid marker with a configurable world
orientation. Success requires:

- Forbidden-zone clearance on the drill (keeps fingertips off the chuck/bit).
- Affinity-zone overlap between the drill bit and the target spot.
- Axis alignment: a chosen local axis on the drill (typically the bit axis)
  must align with a chosen local axis on the target (typically the spot
  normal) within a cosine threshold.

Tune affinity centers/radii, axis vectors, and the target init rotation
once you can preview the spawned drill. The placeholder values below are
sized for a typical YCB power-drill USD.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..grasping.forbidden_zones import ForbiddenZone, split_zones
from .base_cfg import (
    YCB_DIR,
    FunctionalGraspingEnvCfg,
    FunctionalGraspingEnvFloatingDexHandRightCfg,
    FunctionalPourObservationsCfg,
)

DRILL_USD_PATH = str(YCB_DIR / "035_power_drill_usd" / "035_power_drill.usd")
DRILL_ROT_INIT = (1.0, 0.0, 0.0, 0.0)
DRILL_MASS = 0.6

# Scene field name for the drill spot target marker.
TARGET_KEY = "drill_target"

DRILL_TARGET_DEFAULT_SIZE = (0.05, 0.05, 0.01)

# Flat wood board placed under the drill spot (the surface to be drilled on).
# A full MDL wood material (with normals + roughness), matching how the table
# legs are textured. Oak is known-present in this scene (the table legs use the
# same file); swap for any other ``Materials/Base/Wood/*.mdl`` such as
# ``Plywood.mdl``, ``Birch.mdl``, or ``Walnut.mdl``.
DRILL_BOARD_MATERIAL_PATH = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl"
DRILL_BOARD_DEFAULT_SIZE = (0.15, 0.15, 0.01)


def _make_drill_target_cfg(
    *,
    init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """Kinematic cuboid marker representing the drill spot pose (free of physics)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DrillTarget",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                emissive_color=(0.0, 0.0, 0.0),
                roughness=0.7,
                metallic=0.0,
            ),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
    )


def _make_drill_board_cfg(
    *,
    init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
    size: tuple[float, float, float],
    material_path: str,
    texture_scale: tuple[float, float] = (0.4, 0.4),
) -> RigidObjectCfg:
    """Kinematic flat wood board under the drill spot (visual workpiece).

    Collision is disabled — the board is a visual cue only (the drill object
    itself spawns with collisions off), so it never perturbs the physics. The
    wood look comes from a Nucleus MDL material, the same scheme the table legs
    use.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DrillBoard",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=material_path,
                project_uvw=True,
                texture_scale=texture_scale,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
    )


@configclass
class DrillApplySceneCfg(FunctionalGraspingEnvCfg.FunctionalGraspingSceneCfg):
    """Scene with the drill ``object``, the kinematic drill-spot target, and the
    flat wood board (workpiece) spawned beneath it."""

    drill_target: RigidObjectCfg = _make_drill_target_cfg(
        init_pos=(0.0, 0.0, 0.0),
        init_rot=(1.0, 0.0, 0.0, 0.0),
        size=DRILL_TARGET_DEFAULT_SIZE,
        color=(0.85, 0.15, 0.15),
    )

    drill_board: RigidObjectCfg = _make_drill_board_cfg(
        init_pos=(0.0, 0.0, 0.0),
        init_rot=(1.0, 0.0, 0.0, 0.0),
        size=DRILL_BOARD_DEFAULT_SIZE,
        material_path=DRILL_BOARD_MATERIAL_PATH,
    )


@configclass
class DrillApplyObservationsCfg(FunctionalPourObservationsCfg):
    """Observation layout for drill-apply (no ``object_pose`` command).

    Reuses the functional layout from ``FunctionalPourObservationsCfg``:
    ``state`` carries the object pose plus the functional point + tilt-axis
    (the latter set per-asset in ``__post_init__``), ``privileged`` carries
    the object velocities. Only the ``goal`` differs — it points at the drill
    target asset instead of the pour success marker.
    """

    @configclass
    class GoalObsCfg(ObsGroup):
        goal_pos_b = ObsTerm(
            func=mdp.asset_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg(TARGET_KEY)},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    goal: GoalObsCfg = GoalObsCfg()


@configclass
class DrillApplyEnvFloatingDexHandRightCfg(FunctionalGraspingEnvFloatingDexHandRightCfg):
    """Drill apply: grasp + affinity overlap + drill-axis-vs-target-normal alignment."""

    usd_path: str = DRILL_USD_PATH
    object_mass: float = DRILL_MASS
    object_scale: tuple[float, float, float] = (1.3, 1.3, 1.3)
    object_half_height: float = 0.12
    object_static_friction: float | None = 2.0
    object_dynamic_friction: float | None = 2.0
    object_friction_combine_mode: str = "average"
    object_init_rot: tuple[float, float, float, float] = DRILL_ROT_INIT
    object_collision_enabled: bool = False

    # Drill spawns toward one side of the table; tune once previewed.
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    # object_init_z_offset: float = 0.05
    object_reset_x_range: tuple[float, float] = (0.05, 0.25)
    object_reset_y_range: tuple[float, float] = (-0.2, 0.2)
    object_reset_roll_range: tuple[float, float] = (1.57, 1.57)
    object_reset_yaw_range: tuple[float, float] = (3.14 - 0.7, 3.14 + 0.7)

    # Drill-spot target placement (offsets from the table centre). Init rot
    # encodes the surface normal — by default the marker is upright so its
    # +Z axis is the world-frame "drill direction".
    target_init_x_offset: float = 0.0
    # Wood/drill-spot region sits on the +y side, away from the drill's own
    # spawn region (centred near the table centre), so the board doesn't spawn
    # on top of the drill. Tune together with ``min_object_target_xy_distance``.
    target_init_y_offset: float = 0.3
    # Clearance of the drill spot above the wood board's top face — the goal
    # sphere hovers this far over the surface that gets drilled.
    target_init_z_offset: float = 0.005
    target_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    target_size: tuple[float, float, float] = DRILL_TARGET_DEFAULT_SIZE
    target_color: tuple[float, float, float] = (0.85, 0.15, 0.15)
    target_reset_x_range: tuple[float, float] = (0.05, 0.25)
    target_reset_y_range: tuple[float, float] = (-0.2, 0.2)
    # Minimum xy distance (m) the drill spot (and thus the wood board, which
    # tracks it) is kept from the drill's reset pose, via rejection sampling.
    # Keeps the workpiece from spawning too close to the drill.
    min_object_target_xy_distance: float = 0.25

    # Flat wood "workpiece" spawned under the drill-spot goal sphere — the
    # surface to be drilled on. Sits flush on the tabletop and tracks the
    # drill spot's x/y on reset. ``drill_board_material_path`` is a full MDL
    # wood material (same scheme as the table legs); swap for any other
    # ``Materials/Base/Wood/*.mdl``. ``drill_board_size`` is (x, y, thickness).
    drill_board_size: tuple[float, float, float] = DRILL_BOARD_DEFAULT_SIZE
    drill_board_material_path: str = DRILL_BOARD_MATERIAL_PATH
    drill_board_texture_scale: tuple[float, float] = (0.4, 0.4)
    drill_board_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Default no-touch zone keeps fingertips off the drill chuck/bit area.
    forbidden_zones: tuple[ForbiddenZone, ...] = (
        ForbiddenZone(kind="sphere", center=(-0.17, 0.1, 0.035), radius=0.01),
    )

    # Axis alignment requirement.
    # ``drill_axis_local`` is the drill bit direction in the drill's frame.
    # Defaults to +Z; tune once the drill USD orientation is confirmed.
    drill_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # --- Pour-style success criterion (template; tune per asset) ---
    # Replaces the legacy cosine-alignment + affinity-overlap success with
    # the unified ``lift_and_tilt_with_contact_zones`` gate. Captures the
    # same intent — drill aligned vertically AND its bit close to the
    # target spot — using the same template the pour configs use.
    #
    # Lift gate: typically not needed for drill apply, since the bit
    # presses *down* onto the target. Default 0.0 means "at or above
    # spawn height", which is essentially always true while held.
    drill_min_height: float = 0.0
    # Primary tilt gate: max angle (rad) between ``drill_axis_local``
    # (rotated by drill quat) and ``drill_world_axis``. With
    # ``drill_tilt_ge=False`` success requires the angle to be ≤ this
    # threshold, i.e. drill near-vertical.
    drill_tilt_threshold_rad: float = math.radians(80)  # TODO: tune
    drill_world_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    drill_tilt_ge: bool = False
    # xy gate: max horizontal distance between the drill's bit tip
    # (computed via ``drill_goal_object_local_offset`` in drill-local
    # coords) and the drill_target marker's xy. Set to ``None`` to
    # disable the xy gate.
    drill_goal_xy_threshold: float | None = 0.025  # TODO: tune
    # Drill-local offset to the bit tip (the "functional region" we want
    # to bring close to the target). Defaults match the affinity zone
    # source_local; tune to the actual bit-tip position.
    drill_goal_object_local_offset: tuple[float, float, float] = (-0.17, 0.1, 0.035)  # TODO: tune
    # Optional secondary plane-angle gate. The base ``lift_and_tilt_*``
    # ORs this with the primary tilt threshold — satisfying *either*
    # counts as enough tilt. Angle is between
    # ``drill_axis_local`` (rotated to world) and the ground plane
    # (normal = world +Z); π/2 means perfectly perpendicular to the
    # ground, i.e. drill straight up. Set to ``None`` to disable.
    drill_plane_angle_threshold_rad: float | None = math.radians(90)  # TODO: tune

    # Whether to show the drill-progress sphere (colored ball at the drill
    # target that animates red -> green with success-angle progress).
    # Independent of ``enable_debug_vis`` — this is a goal-progress
    # indicator, not a debug aid (mirrors ``pour_show_progress_marker``).
    drill_show_progress_marker: bool = False

    # Override the scene to include the drill spot marker.
    scene: DrillApplySceneCfg = DrillApplySceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
    )
    observations: DrillApplyObservationsCfg = DrillApplyObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Position the drill target on top of the table at the configured offset.
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        table_pos = self.scene.table.init_state.pos

        # The flat wood board sits flush on the tabletop, directly under the
        # drill spot. The spot (and its goal sphere) then rests
        # ``target_init_z_offset`` above the board's top face, so the marker
        # hovers just over the wood surface that gets drilled.
        board_thickness = self.drill_board_size[2]
        board_center_z = table_top_z + board_thickness / 2.0
        spot_z = table_top_z + board_thickness + self.target_init_z_offset

        self.scene.drill_target = _make_drill_target_cfg(
            init_pos=(
                table_pos[0] + self.target_init_x_offset,
                table_pos[1] + self.target_init_y_offset,
                spot_z,
            ),
            init_rot=self.target_init_rot,
            size=self.target_size,
            color=self.target_color,
        )

        # Flat wood piece spawned beneath the goal sphere (the surface to drill
        # on). Centred on the drill spot's x/y, flush on the tabletop.
        self.scene.drill_board = _make_drill_board_cfg(
            init_pos=(
                table_pos[0] + self.target_init_x_offset,
                table_pos[1] + self.target_init_y_offset,
                board_center_z,
            ),
            init_rot=self.drill_board_init_rot,
            size=self.drill_board_size,
            material_path=self.drill_board_material_path,
            texture_scale=self.drill_board_texture_scale,
        )

        # Reset event for the drill spot target (kinematic, no joints). The
        # excluding variant rejection-samples the spot's xy so it stays at least
        # ``min_object_target_xy_distance`` from the drill's already-reset xy,
        # so the wood board (which tracks the spot below) never spawns on top of
        # the drill. Runs after the base ``reset_object`` event, so it reads the
        # drill's post-reset pose.
        self.events.reset_drill_target = EventTerm(
            func=mdp.reset_root_pose_uniform_excluding,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(self.target_reset_x_range),
                    "y": list(self.target_reset_y_range),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": [0.0, 0.0],
                },
                "asset_cfg": SceneEntityCfg(TARGET_KEY),
                "reference_asset_cfg": SceneEntityCfg("object"),
                "min_xy_distance": self.min_object_target_xy_distance,
                "max_attempts": 50,
            },
        )

        # Keep the wood board under the drill spot after it is re-randomized:
        # copy the target's x/y, drop back down by a fixed z-offset so the board
        # stays flush on the table, and hold it upright. Registered after
        # ``reset_drill_target`` so it reads the post-reset spot pose.
        self.events.reset_drill_board = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("drill_board"),
                "source_cfg": SceneEntityCfg(TARGET_KEY),
                "z_offset": board_center_z - spot_z,
                "quat": self.drill_board_init_rot,
            },
        )

        # No object-pose goal command for this task.
        self.commands.object_pose = None
        self.rewards.position_tracking = None
        self.rewards.success = None
        # Functional point + tilt-axis live in the (non-privileged) ``state``
        # group. ``goal_pos_b``'s ``asset_cfg`` is fixed to ``TARGET_KEY`` at
        # the class level (see ``DrillApplyObservationsCfg.GoalObsCfg``). The
        # scalar tilt / plane angles were dropped — the tilt-axis vector
        # carries the current tilt.
        self.observations.state.object_functional_point_pos_b.params["local_offset"] = (
            self.drill_goal_object_local_offset
        )
        self.observations.state.object_functional_axis_b.params["axis_local"] = self.drill_axis_local

        # Replace success termination with the pour-style lift-and-tilt
        # gate. Drill must be near-vertical (primary tilt or secondary
        # plane gate) AND its bit-tip xy must be within
        # ``drill_goal_xy_threshold`` of the drill_target marker. The
        # forbidden zones still gate fingertip clearance on the chuck/bit.
        # Affinity zones are kept for visualization but no longer enforced
        # by the success criterion (the xy gate replaces that role).
        sphere_zones, box_zones, cylinder_zones = split_zones(self.forbidden_zones)
        self.terminations.success = DoneTerm(
            func=mdp.lift_and_tilt_with_contact_zones,
            params={
                "min_height": self.drill_min_height,
                "threshold_rad": self.drill_tilt_threshold_rad,
                "axis_local": self.drill_axis_local,
                "world_axis": self.drill_world_axis,
                "tilt_ge": self.drill_tilt_ge,
                "sphere_zones": sphere_zones,
                "box_zones": box_zones,
                "cylinder_zones": cylinder_zones,
                "asset_cfg": SceneEntityCfg("robot", body_names=self.robot_config.fingertip_body_names),
                "object_cfg": SceneEntityCfg("object"),
                "goal_asset_cfg": SceneEntityCfg(TARGET_KEY),
                "goal_xy_threshold": self.drill_goal_xy_threshold,
                "goal_object_local_offset": self.drill_goal_object_local_offset,
                "plane_axis_local": self.drill_axis_local,
                "plane_angle_threshold_rad": self.drill_plane_angle_threshold_rad,
            },
        )

        # Drill progress sphere: colored ball at the drill target whose color
        # animates red -> green with success-angle progress. Lives in
        # ``scene_vis`` (always-on, preset-preserved) so the marker renders
        # regardless of which observation preset is active. Trained policies
        # ignore scene_vis; the obs manager still ticks the term so the USD
        # marker actually updates each step. Gated by
        # ``drill_show_progress_marker`` (not ``enable_debug_vis`` — this is
        # a goal-progress indicator, not a debug aid).
        if self.drill_show_progress_marker:
            if self.observations.scene_vis is None:
                self.observations.scene_vis = dexverse_base_env.ObservationsCfg.SceneVisObsCfg()
            self.observations.scene_vis.drill_progress_marker_vis = ObsTerm(
                func=mdp.pour_progress_marker_vis,
                params={
                    "goal_asset_cfg": SceneEntityCfg(TARGET_KEY),
                    "object_cfg": SceneEntityCfg("object"),
                    "primary_threshold_rad": self.drill_tilt_threshold_rad,
                    "primary_axis_local": self.drill_axis_local,
                    "primary_world_axis": self.drill_world_axis,
                    "primary_tilt_ge": self.drill_tilt_ge,
                    "plane_threshold_rad": self.drill_plane_angle_threshold_rad,
                    "plane_axis_local": self.drill_axis_local,
                    "plane_normal": (0.0, 0.0, 1.0),
                    "radius": 0.04,
                    "num_color_steps": 11,
                    "prim_path_prefix": "/Visuals/DrillProgressMarker",
                },
            )
