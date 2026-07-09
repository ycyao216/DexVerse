# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base for functional manipulation tasks.

Each functional-grasping task pins a single USD asset (one object per task)
because the eventual success criterion is object-specific (e.g. correct
power-grasp on a bleach bottle, correct handle-grasp on a pan). This module
contains both lift-to-goal and pouring variants, all built on the same
single-object setup.

Forbidden and designated contact zones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each task can declare a list of :class:`ForbiddenZone` entries in the
object's local frame. Success requires reaching the goal AND keeping every
fingertip outside every zone. Zones are also rendered as red,
semi-transparent markers that track the object pose for inspection.

Tasks can also declare :class:`DesignatedContactZone` entries. Each contact
zone is satisfied when at least one configured body, by default a fingertip,
lies inside that object-local region.
"""

from __future__ import annotations

import math
from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from ..grasping.forbidden_zones import DesignatedContactZone, ForbiddenZone, split_zones

__all__ = [
    "FUNCTIONAL_OBJECTS_DIR",
    "SYNTHESIS_DIR",
    "YCB_DIR",
    "DesignatedContactZone",
    "ForbiddenZone",
    "FunctionalGraspingEnvCfg",
    "FunctionalGraspingEnvFloatingDexHandRightCfg",
    "FunctionalPourEnvCfg",
    "FunctionalPourEnvFloatingDexHandRightCfg",
    "FunctionalPourEventCfg",
    "FunctionalPourObservationsCfg",
    "FunctionalPourRewardsCfg",
    "FunctionalPourTerminationsCfg",
    "POUR_ANGLE_RAD",
    "build_object_cfg_from_usd",
]


# Per-object USD asset roots used by this task family. Objects are sourced from
# the synthesis pool, except the YCB-derived drill/hammer which live under ycb/.
from dexverse.assets import FUNCTIONAL_OBJECTS_DIR, SYNTHESIS_DIR, YCB_DIR

DEFAULT_OBJECT_ROT = (0.707107, -0.707107, 0.0, 0.0)
DEFAULT_OBJECT_MASS = 0.3
DEFAULT_OBJECT_HALF_HEIGHT = 0.05
CENTER_SQUARE_SIZE = 0.45
BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.5

POUR_ANGLE_RAD = math.radians(100.0)
POUR_LIFT_HEIGHT_M = 0.2
SUCCESS_MARKER_SIZE = (0.01, 0.01, 0.12)
SUCCESS_MARKER_COLOR = (0.1, 0.9, 0.1)
SUCCESS_MARKER_QUAT = (
    math.cos(POUR_ANGLE_RAD * 0.5),
    -math.sin(POUR_ANGLE_RAD * 0.5),
    0.0,
    0.0,
)


SUCCESS_MARKER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/SuccessMarker",
    spawn=sim_utils.CuboidCfg(
        size=SUCCESS_MARKER_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=SUCCESS_MARKER_COLOR,
            emissive_color=(0.0, 0.3, 0.0),
            roughness=1.0,
            metallic=0.0,
        ),
        visible=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=SUCCESS_MARKER_QUAT),
)


def build_object_cfg_from_usd(
    usd_path: str,
    *,
    mass: float = DEFAULT_OBJECT_MASS,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    init_rot: tuple[float, float, float, float] = DEFAULT_OBJECT_ROT,
    collision_enabled: bool = False,
    static_friction: float | None = None,
    dynamic_friction: float | None = None,
    restitution: float = 0.0,
    friction_combine_mode: str = "average",
    prim_name: str = "Object",
) -> RigidObjectCfg:
    """Build a single-asset :class:`RigidObjectCfg` from a USD file path.

    Note: ``collision_props`` is intentionally left ``None``. Applying
    :class:`CollisionPropertiesCfg` here would cause the spawn pipeline to
    apply ``UsdPhysics.CollisionAPI`` to the root Xform, and the
    ``@apply_nested`` walk in :func:`modify_collision_properties` then stops
    at the root before visiting the child mesh. That overrides the source
    USD's authored :class:`MeshCollisionAPI` (e.g. ``convexDecomposition``)
    with PhysX's default ``convexHull`` fallback. We rely on the asset USD
    to already author CollisionAPI / MeshCollisionAPI on the mesh prim.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=usd_path,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=(sim_utils.CollisionPropertiesCfg(collision_enabled=True)) if collision_enabled else None,
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=init_rot),
    )


# ---------------------------------------------------------------------------
# Forbidden zones
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class _GoalObsCfg(ObsGroup):
    """Goal pose pulled from the ``object_pose`` command (non-privileged)."""

    target_object_pose_b = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "object_pose"},
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class _FunctionalTiltGoalObsCfg(ObsGroup):
    """Goal position for asset-goal functional tasks.

    Position of the per-task ``success_marker`` (a scene asset, so this is
    reward-independent). The object's *current* tilt is observed via the
    tilt-axis vector in the ``state`` group; the target tilt threshold is a
    per-task constant kept out of observations on purpose (this is an
    imitation-learning benchmark — the demonstrated motion teaches the tilt).
    """

    goal_pos_b = ObsTerm(
        func=mdp.asset_pos_b,
        noise=Unoise(n_min=-0.0, n_max=0.0),
        params={"asset_cfg": SceneEntityCfg("success_marker")},
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class FunctionalGraspingObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for single-object functional grasping.

    Extends the root layout by adding:
      - ``state``: object pose (observable, deployable; no velocities).
      - ``privileged``: object linear / angular velocities (sim-only).
      - ``goal``: ``target_object_pose_b`` from the ``object_pose`` command.

    Debug markers (forbidden / designated-contact / affinity zone
    visualizers) are populated into ``scene_vis`` by ``configure_debug_vis``
    (see ``DexVerseBaseEnvCfg.enable_debug_vis``), not a dedicated
    group here.

    ``proprio`` stays as the base's joint-pos-only group.
    """

    @configclass
    class StateObsCfg(ObsGroup):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))

    state: StateObsCfg = StateObsCfg()
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: _GoalObsCfg = _GoalObsCfg()


@configclass
class FunctionalPourObservationsCfg(FunctionalGraspingObservationsCfg):
    """Observation layout for pouring tasks.

    Inherits the grasping layout, then:
      - Adds task-specific functional point + tilt-axis terms to ``state``.
        These are pure functions of the object pose plus per-asset constants
        (pour_local_offset, pour_axis_local, …) baked in at cfg time, so they
        are observable, deployable state. The object's *current* tilt is read
        off ``object_functional_axis_b`` (a direction vector); the redundant
        scalar tilt / plane angles are intentionally dropped.
      - Replaces the command-pose goal with the success-marker goal position
        in ``goal``.

    Note: ``object_up_b`` and ``object_tilt_angle`` are intentionally NOT
    added — they are pure trigonometric derivations of ``object_quat_b``
    (already in ``state``) and bloat the observation vector without adding
    information.
    """

    @configclass
    class StateObsCfg(FunctionalGraspingObservationsCfg.StateObsCfg):
        object_functional_point_pos_b = ObsTerm(
            func=mdp.object_local_point_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )
        object_functional_axis_b = ObsTerm(
            func=mdp.object_rot_axis_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )

    state: StateObsCfg = StateObsCfg()
    goal: _FunctionalTiltGoalObsCfg = _FunctionalTiltGoalObsCfg()


# ---------------------------------------------------------------------------
# Commands / Rewards / Terminations
# ---------------------------------------------------------------------------


@configclass
class FunctionalGraspingCommandsCfg(dexverse_base_env.CommandsCfg):
    """Goal-position command (sphere visualizer), held fixed for the episode."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.10, 0.20),  # overwritten in __post_init__
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="object",
        position_only=True,
    )


@configclass
class FunctionalGraspingRewardsCfg(dexverse_base_env.RewardsCfg):
    """Pickup-style shaping + goal tracking and success bonus."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
        },
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=2.0,
        params={"std": 0.15, "command_name": "object_pose"},
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=8.0,
        params={"pos_std": 0.05, "rot_std": None, "command_name": "object_pose"},
    )


@configclass
class FunctionalGraspingTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Out-of-bound + goal-proximity success (replaced in __post_init__ when zones are set)."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.object_at_goal_position,
        params={"command_name": "object_pose", "threshold": 0.03},
    )


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------


@configclass
class FunctionalGraspingEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Robot-agnostic base for single-object functional grasping.

    Subclasses must set :attr:`usd_path` to a single USD file and may
    override :attr:`object_half_height` to match the asset's geometry.
    Subclasses also populate :attr:`forbidden_zones` with object-specific
    regions the hand must not touch.
    """

    supports_object_pose_command: bool = True

    usd_path: str | None = None
    object_half_height: float = DEFAULT_OBJECT_HALF_HEIGHT
    object_mass: float = DEFAULT_OBJECT_MASS
    object_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    object_init_rot: tuple[float, float, float, float] = DEFAULT_OBJECT_ROT
    object_collision_enabled: bool = False
    object_static_friction: float | None = None
    object_dynamic_friction: float | None = None
    object_restitution: float = 0.0
    object_friction_combine_mode: str = "average"

    # ---- Object initial spawn pose (offsets from table centre) ----
    # The z spawn coordinate is always derived from
    # ``DEFAULT_TABLE_TOP_HEIGHT + object_half_height + table_clearance`` so
    # the object sits flush on the tabletop unless a clearance is requested.
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    table_clearance: float = 0.0

    # ---- Object reset randomization (per-axis ranges in object-local frame) ----
    # ``object_reset_y_range`` defaults to ``None``, which means "derive from
    # ``center_square_size``" so the historical centre-square reset behaviour
    # still applies when no explicit y-range is given. All other axes default
    # to ``(0.0, 0.0)`` (no randomization).
    center_square_size: float = CENTER_SQUARE_SIZE
    object_reset_x_range: tuple[float, float] = (0.0, 0.0)
    object_reset_y_range: tuple[float, float] | None = None
    object_reset_z_range: tuple[float, float] = (0.0, 0.0)
    object_reset_roll_range: tuple[float, float] = (0.0, 0.0)
    object_reset_pitch_range: tuple[float, float] = (0.0, 0.0)
    object_reset_yaw_range: tuple[float, float] = (0.0, 0.0)

    # ---- Goal-pose randomization ----
    # x/y are interpreted as offsets from the table centre, z as offset from
    # the tabletop surface. Defaults of ``(0.0, 0.0)`` for x/y reproduce the
    # original behaviour (goal directly above the table centre).
    target_x_range: tuple[float, float] = (0.0, 0.0)
    target_y_range: tuple[float, float] = (0.0, 0.0)
    target_height_range: tuple[float, float] = (0.10, 0.20)

    # ---- Forbidden zones ----
    forbidden_zones: tuple[ForbiddenZone, ...] = field(default_factory=tuple)
    success_position_threshold: float = 0.03
    forbidden_zone_color: tuple[float, float, float] = (0.9, 0.1, 0.1)
    forbidden_zone_opacity: float = 0.4

    # ---- Designated contact zones ----
    designated_contact_zones: tuple[DesignatedContactZone, ...] = field(default_factory=tuple)
    designated_contact_asset_name: str = "robot"
    designated_contact_body_names: tuple[str, ...] | None = None
    designated_contact_object_name: str = "object"
    designated_contact_zone_color: tuple[float, float, float] = (0.1, 0.8, 0.2)
    designated_contact_zone_opacity: float = 0.45

    @configclass
    class FunctionalGraspingSceneCfg(dexverse_base_env.SceneCfg):
        # Placeholder; replaced from ``usd_path`` in __post_init__.
        object: RigidObjectCfg = build_object_cfg_from_usd(
            usd_path=str(FUNCTIONAL_OBJECTS_DIR / "Bleach_1" / "model_Bleach_1_69323.usd"),
            mass=DEFAULT_OBJECT_MASS,
            init_rot=DEFAULT_OBJECT_ROT,
        )

    scene: FunctionalGraspingSceneCfg = FunctionalGraspingSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
    )
    observations: FunctionalGraspingObservationsCfg = FunctionalGraspingObservationsCfg()
    commands: FunctionalGraspingCommandsCfg = FunctionalGraspingCommandsCfg()
    rewards: FunctionalGraspingRewardsCfg = FunctionalGraspingRewardsCfg()
    terminations: FunctionalGraspingTerminationsCfg = FunctionalGraspingTerminationsCfg()

    def __post_init__(self):
        # Resolve single-USD object before scene materialization so super()
        # sees the correct ``scene.object``.
        if self.usd_path is not None:
            self.scene.object = build_object_cfg_from_usd(
                self.usd_path,
                mass=self.object_mass,
                scale=self.object_scale,
                init_rot=self.object_init_rot,
                collision_enabled=self.object_collision_enabled,
                static_friction=self.object_static_friction,
                dynamic_friction=self.object_dynamic_friction,
                restitution=self.object_restitution,
                friction_combine_mode=self.object_friction_combine_mode,
            )

        super().__post_init__()

        self.episode_length_s = 20.0

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        table_pos = self.scene.table.init_state.pos

        # Place object flush on the table surface, applying x/y offsets.
        self.scene.object.init_state.pos = (
            table_pos[0] + self.object_init_x_offset,
            table_pos[1] + self.object_init_y_offset,
            table_top_z + self.object_half_height + self.table_clearance,
        )

        # Randomize object reset on the tabletop. ``object_reset_y_range``
        # falls back to a centre-square (±center_square_size / 2) when left
        # as ``None`` so older configs continue to behave the same.
        if self.object_reset_y_range is None:
            half_side = self.center_square_size * 0.5
            y_range = (-half_side, half_side)
        else:
            y_range = self.object_reset_y_range
        if self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": list(self.object_reset_x_range),
                "y": list(y_range),
                "z": list(self.object_reset_z_range),
                "roll": list(self.object_reset_roll_range),
                "pitch": list(self.object_reset_pitch_range),
                "yaw": list(self.object_reset_yaw_range),
            }

        # Clamp out-of-bound to table footprint.
        if self.terminations.object_out_of_bound is not None:
            table_size = self.scene.table.spawn.size
            self.terminations.object_out_of_bound.params["in_bound_range"] = {
                "x": (-table_size[0] * 0.5, table_size[0] * 0.5),
                "y": (-table_size[1] * 0.5, table_size[1] * 0.5),
                "z": (BOUND_Z_MIN, BOUND_Z_MAX),
            }

        # Configure goal command: fixed within episode, world-frame, table-anchored.
        self.commands.object_pose.body_name = self.robot_config.palm_body_name
        self.commands.object_pose.resampling_time_range = (
            self.episode_length_s + 1.0,
            self.episode_length_s + 1.0,
        )
        self.commands.object_pose.use_world_frame = True
        self.commands.object_pose.ranges.pos_x = (
            table_pos[0] + self.target_x_range[0],
            table_pos[0] + self.target_x_range[1],
        )
        self.commands.object_pose.ranges.pos_y = (
            table_pos[1] + self.target_y_range[0],
            table_pos[1] + self.target_y_range[1],
        )
        self.commands.object_pose.ranges.pos_z = (
            table_top_z + self.target_height_range[0],
            table_top_z + self.target_height_range[1],
        )

        # Goal visualizers.
        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9), opacity=0.35),
                )
            },
        )
        self.commands.object_pose.success_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": sim_utils.SphereCfg(
                    radius=0.03,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
                ),
                "success": sim_utils.SphereCfg(
                    radius=0.03,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
                ),
            },
        )

        # Contact sensors per fingertip (filtered to the object).
        mdp.setup_fingertip_contact_observation(self)
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names

        # Replace success termination with the zone-gated version.
        sphere_zones, box_zones, cylinder_zones = split_zones(self.forbidden_zones)
        contact_sphere_zones, contact_box_zones, contact_cylinder_zones = split_zones(self.designated_contact_zones)
        contact_body_names = self.designated_contact_body_names
        if contact_body_names is None and self.designated_contact_asset_name == "robot":
            contact_body_names = self.robot_config.fingertip_body_names

        success_func = (
            mdp.success_with_contact_zones
            if (contact_sphere_zones or contact_box_zones or contact_cylinder_zones)
            else mdp.success_no_forbidden_contact
        )
        success_params = {
            "command_name": "object_pose",
            "threshold": self.success_position_threshold,
            "sphere_zones": sphere_zones,
            "box_zones": box_zones,
            "cylinder_zones": cylinder_zones,
            "asset_cfg": SceneEntityCfg("robot", body_names=self.robot_config.fingertip_body_names),
            "object_cfg": SceneEntityCfg("object"),
        }
        if contact_sphere_zones or contact_box_zones or contact_cylinder_zones:
            success_params.update({
                "contact_sphere_zones": contact_sphere_zones,
                "contact_box_zones": contact_box_zones,
                "contact_cylinder_zones": contact_cylinder_zones,
                "contact_asset_cfg": SceneEntityCfg(
                    self.designated_contact_asset_name,
                    body_names=contact_body_names,
                ),
                "contact_object_cfg": SceneEntityCfg(self.designated_contact_object_name),
            })
        self.terminations.success = DoneTerm(
            func=success_func,
            params=success_params,
        )

        self.configure_debug_vis()

    def configure_debug_vis(self) -> None:
        """Populate the forbidden-zone / designated-contact-zone markers in
        ``observations.scene_vis`` when ``self.enable_debug_vis`` is True.

        Zone data comes straight from the ``forbidden_zones`` /
        ``designated_contact_zones`` fields (set at construction time), so
        this can be safely called at the end of ``__post_init__``.
        """
        if not self.enable_debug_vis:
            return
        sphere_zones, box_zones, cylinder_zones = split_zones(self.forbidden_zones)
        contact_sphere_zones, contact_box_zones, contact_cylinder_zones = split_zones(self.designated_contact_zones)
        contact_body_names = self.designated_contact_body_names
        if contact_body_names is None and self.designated_contact_asset_name == "robot":
            contact_body_names = self.robot_config.fingertip_body_names

        want_forbidden = bool(sphere_zones or box_zones or cylinder_zones)
        want_contact_zones = bool(contact_sphere_zones or contact_box_zones or contact_cylinder_zones)
        if not (want_forbidden or want_contact_zones):
            return

        if self.observations.scene_vis is None:
            self.observations.scene_vis = dexverse_base_env.ObservationsCfg.SceneVisObsCfg()
        if want_forbidden:
            self.observations.scene_vis.forbidden_zones_vis = ObsTerm(
                func=mdp.forbidden_zones_vis,
                params={
                    "sphere_zones": sphere_zones,
                    "box_zones": box_zones,
                    "cylinder_zones": cylinder_zones,
                    "object_cfg": SceneEntityCfg("object"),
                    "color": self.forbidden_zone_color,
                    "opacity": self.forbidden_zone_opacity,
                },
            )
        if want_contact_zones:
            self.observations.scene_vis.designated_contact_zones_vis = ObsTerm(
                func=mdp.contact_zones_vis,
                params={
                    "sphere_zones": contact_sphere_zones,
                    "box_zones": contact_box_zones,
                    "cylinder_zones": contact_cylinder_zones,
                    "object_cfg": SceneEntityCfg(self.designated_contact_object_name),
                    "color": self.designated_contact_zone_color,
                    "opacity": self.designated_contact_zone_opacity,
                    "prim_path_prefix": "/Visuals/DesignatedContactZone",
                },
            )


@configclass
class FunctionalPourRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reach, grasp, and tilt shaping for pouring."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
        },
    )

    tilt_reward = RewTerm(
        func=mdp.tilt_angle_reward,
        weight=5.0,
        params={
            "threshold_rad": POUR_ANGLE_RAD,
            "axis_local": (0.0, 0.0, 1.0),
            "world_axis": (0.0, 0.0, 1.0),
            "tilt_ge": True,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class FunctionalPourTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Out-of-bound plus lift-and-tilt success for pouring."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.lift_and_tilt,
        params={
            "min_height": POUR_LIFT_HEIGHT_M,
            "threshold_rad": POUR_ANGLE_RAD,
            "axis_local": (0.0, 0.0, 1.0),
            "world_axis": (0.0, 0.0, 1.0),
            "tilt_ge": True,
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class FunctionalPourEventCfg(dexverse_base_env.EventCfg):
    """Keep the pour success marker synced to the object reset pose."""

    reset_success_marker = EventTerm(
        func=mdp.sync_object,
        mode="reset",
        params={
            "target_cfg": SceneEntityCfg("success_marker"),
            "source_cfg": SceneEntityCfg("object"),
            "z_offset": POUR_LIFT_HEIGHT_M,
            "quat": SUCCESS_MARKER_QUAT,
        },
    )


@configclass
class FunctionalPourEnvCfg(FunctionalGraspingEnvCfg):
    """Robot-agnostic base for single-object functional pouring."""

    pour_lift_height: float = POUR_LIFT_HEIGHT_M
    pour_angle_rad: float = POUR_ANGLE_RAD
    pour_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pour_world_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pour_tilt_ge: bool = True
    # Max horizontal distance (in metres) from the success marker for the
    # pour to count. Set to ``None`` to disable the xy gate and fall back to
    # the lift-and-tilt-only criterion.
    pour_goal_xy_threshold: float | None = 0.10
    # Optional offset (in the object's local frame) for the xy-gate. When
    # set, the xy distance is measured between
    # ``object.root_pos + R(object.root_quat) * pour_goal_object_local_offset``
    # and the success marker's xy. Use this to anchor the gate at e.g. the
    # bottle's spout instead of the object's mass centre. ``None`` (the
    # default) reproduces the previous "object root vs. goal" behaviour.
    pour_goal_object_local_offset: tuple[float, float, float] | None = None
    # Optional secondary tilt criterion: angle between the rotated
    # ``pour_plane_axis_local`` of the object and the plane with normal
    # ``pour_plane_normal`` (default = the ground plane). When both
    # ``pour_plane_axis_local`` and ``pour_plane_angle_threshold_rad`` are set,
    # the tilt requirement is satisfied if *either* the primary axis-vs-axis
    # criterion (``pour_angle_rad``) or this plane-angle criterion is met. The
    # plane-angle metric is in ``[0, π/2]`` (sign-agnostic about the plane).
    pour_plane_axis_local: tuple[float, float, float] | None = None
    pour_plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pour_plane_angle_threshold_rad: float | None = None
    # When True, hide the success-marker cuboid pole and add a sphere that
    # tracks the goal pose with a red→green tilt-progress colour. Visual
    # only; doesn't affect the success criterion.
    pour_show_progress_marker: bool = False
    pour_progress_marker_radius: float = 0.04
    pour_progress_marker_opacity: float = 0.7
    pour_progress_marker_num_color_steps: int = 11

    observations: FunctionalPourObservationsCfg = FunctionalPourObservationsCfg()
    rewards: FunctionalPourRewardsCfg = FunctionalPourRewardsCfg()
    terminations: FunctionalPourTerminationsCfg = FunctionalPourTerminationsCfg()
    events: FunctionalPourEventCfg = FunctionalPourEventCfg()

    @configclass
    class FunctionalPourSceneCfg(FunctionalGraspingEnvCfg.FunctionalGraspingSceneCfg):
        object: RigidObjectCfg = build_object_cfg_from_usd(str(SYNTHESIS_DIR / "Bleach_1" / "model_Bleach_1_69323.usd"))
        success_marker: RigidObjectCfg = SUCCESS_MARKER_CFG

    scene: FunctionalPourSceneCfg = FunctionalPourSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        success_marker=SUCCESS_MARKER_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        # Pouring uses lift-and-tilt success, while keeping the shared
        # forbidden/contact zone gates from the functional manipulation base.
        self.commands.object_pose = None
        sphere_zones, box_zones, cylinder_zones = split_zones(self.forbidden_zones)
        contact_sphere_zones, contact_box_zones, contact_cylinder_zones = split_zones(self.designated_contact_zones)
        contact_body_names = self.designated_contact_body_names
        if contact_body_names is None and self.designated_contact_asset_name == "robot":
            contact_body_names = self.robot_config.fingertip_body_names
        self.terminations.success = DoneTerm(
            func=mdp.lift_and_tilt_with_contact_zones,
            params={
                "min_height": self.pour_lift_height,
                "threshold_rad": self.pour_angle_rad,
                "axis_local": self.pour_axis_local,
                "world_axis": self.pour_world_axis,
                "tilt_ge": self.pour_tilt_ge,
                "sphere_zones": sphere_zones,
                "box_zones": box_zones,
                "cylinder_zones": cylinder_zones,
                "contact_sphere_zones": contact_sphere_zones,
                "contact_box_zones": contact_box_zones,
                "contact_cylinder_zones": contact_cylinder_zones,
                "asset_cfg": SceneEntityCfg("robot", body_names=self.robot_config.fingertip_body_names),
                "contact_asset_cfg": SceneEntityCfg(
                    self.designated_contact_asset_name,
                    body_names=contact_body_names,
                ),
                "object_cfg": SceneEntityCfg("object"),
                "contact_object_cfg": SceneEntityCfg(self.designated_contact_object_name),
                "goal_asset_cfg": SceneEntityCfg("success_marker"),
                "goal_xy_threshold": self.pour_goal_xy_threshold,
                "goal_object_local_offset": self.pour_goal_object_local_offset,
                "plane_axis_local": self.pour_plane_axis_local,
                "plane_normal": self.pour_plane_normal,
                "plane_angle_threshold_rad": self.pour_plane_angle_threshold_rad,
            },
        )
        self.rewards.tilt_reward.params.update({
            "threshold_rad": self.pour_angle_rad,
            "axis_local": self.pour_axis_local,
            "world_axis": self.pour_world_axis,
            "tilt_ge": self.pour_tilt_ge,
        })
        # Functional point + tilt-axis now live in the (non-privileged)
        # ``state`` group. ``goal_pos_b``'s ``asset_cfg`` is fixed to
        # ``"success_marker"`` at the class level (see
        # ``_FunctionalTiltGoalObsCfg``). The scalar tilt / plane angles were
        # dropped — the tilt-axis vector carries the current tilt, and the
        # target angle is a per-task constant kept out of observations.
        self.observations.state.object_functional_point_pos_b.params["local_offset"] = (
            self.pour_goal_object_local_offset
        )
        self.observations.state.object_functional_axis_b.params["axis_local"] = self.pour_axis_local

        obj_pos = self.scene.object.init_state.pos
        self.scene.success_marker.init_state.pos = (
            obj_pos[0],
            obj_pos[1],
            obj_pos[2] + self.pour_lift_height,
        )
        self.scene.success_marker.init_state.rot = SUCCESS_MARKER_QUAT
        self.events.reset_success_marker.params["z_offset"] = self.pour_lift_height

        # Pour-progress sphere: debug-only visual cue. The legacy cuboid-pole
        # marker stays as a hidden scene entity because success/reset logic
        # still references it. Default poster renders should stay marker-free.
        if self.pour_show_progress_marker and self.enable_debug_vis:
            self.scene.success_marker.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=SUCCESS_MARKER_COLOR,
                emissive_color=(0.0, 0.0, 0.0),
                roughness=1.0,
                metallic=0.0,
                opacity=0.0,
            )
            if self.observations.scene_vis is None:
                self.observations.scene_vis = dexverse_base_env.ObservationsCfg.SceneVisObsCfg()
            self.observations.scene_vis.pour_progress_marker_vis = ObsTerm(
                func=mdp.pour_progress_marker_vis,
                params={
                    "goal_asset_cfg": SceneEntityCfg("success_marker"),
                    "object_cfg": SceneEntityCfg("object"),
                    "primary_threshold_rad": self.pour_angle_rad,
                    "primary_axis_local": self.pour_axis_local,
                    "primary_world_axis": self.pour_world_axis,
                    "primary_tilt_ge": self.pour_tilt_ge,
                    "plane_threshold_rad": self.pour_plane_angle_threshold_rad,
                    "plane_axis_local": self.pour_plane_axis_local,
                    "plane_normal": self.pour_plane_normal,
                    "radius": self.pour_progress_marker_radius,
                    "opacity": self.pour_progress_marker_opacity,
                    "num_color_steps": self.pour_progress_marker_num_color_steps,
                    "prim_path_prefix": "/Visuals/PourProgressMarker",
                },
            )


@configclass
class FunctionalGraspingEnvFloatingDexHandRightCfg(FunctionalGraspingEnvCfg):
    """Functional-grasping config for floating dexterous hands (Shadow / Leap)."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)


@configclass
class FunctionalPourEnvFloatingDexHandRightCfg(FunctionalPourEnvCfg, FunctionalGraspingEnvFloatingDexHandRightCfg):
    """Functional pouring with a floating right dexterous hand."""

    # Repeat these fields here because IsaacLab's configclass field collection
    # does not reliably include fields that only come from the second base in
    # this multiple-inheritance mixin.
    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG
