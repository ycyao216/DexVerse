# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional manipulation: hammer strike.

The robot must grasp the hammer by its handle and drive a nail into a board.
The target is a nail-on-a-board articulation (a single prismatic joint along
``-Z`` so positive joint values mean the nail is being pushed *into* the
board) authored at ``DEXVERSE_AUTHORED_ARTICULATIONS_DIR/nail_board``.

Success requires:

- Forbidden-zone clearance on the hammer (fingertips off the hammer head).
- Affinity-zone overlap between the hammer head and the nail.
- Compression of the nail's prismatic joint past a tunable threshold.

Nail dynamics:
    Gravity is disabled for the nail and the prismatic joint has zero
    stiffness, only a small damping term. A strike imparts a velocity
    impulse along the joint axis; damping bleeds the velocity off and the
    nail settles at whatever displacement was reached. Tune
    ``nail_board_damping`` if the nail moves too easily (raise) or strikes
    barely register (lower). The previous high-stiffness +
    ``velocity_limit_sim`` "break-through" mechanic has been removed.

Hammer stand:
    A single kinematic cylinder (``hammer_stand``) is spawned beneath the
    hammer's spawn pose so the head is lifted off the table while the
    handle hangs over the stand edge. The cylinder is rotationally
    symmetric about its z-axis, so any yaw applied to the hammer still
    leaves the head fully supported. ``table_clearance`` is automatically
    set to the cylinder's height in ``__post_init__`` so the hammer rests
    on top of the stand. At each reset the stand is snapped to the
    hammer's reset pose via ``mdp.sync_object``: ``hammer_stand_xy_offset``
    is interpreted in the hammer's local frame, so the offset rotates with
    the hammer's yaw and the per-reset xy randomization is tracked
    automatically.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from dexverse.assets import DEXVERSE_AUTHORED_ARTICULATIONS_DIR
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..grasping.forbidden_zones import ForbiddenZone, split_zones
from .base_cfg import (
    YCB_DIR,
    FunctionalGraspingEnvCfg,
    FunctionalGraspingEnvFloatingDexHandRightCfg,
    FunctionalGraspingObservationsCfg,
)

HAMMER_USD_PATH = str(YCB_DIR / "048_hammer_usd" / "048_hammer.usd")

# Head-down vertical pose: -90° about world +y rotates the asset's local +x
# (handle axis) onto world -z, putting the head face flat on the tabletop and
# the handle pointing straight up. Quaternion is (w, x, y, z).

HAMMER_ROT_INIT = (1, 0, 0, 0)
HAMMER_MASS = 0.6

NAIL_BOARD_USD_PATH = str(DEXVERSE_AUTHORED_ARTICULATIONS_DIR / "nail_board" / "nail_board.usd")
NAIL_BOARD_DEFAULT_SCALE = (1.0, 1.0, 1.0)
# Half thickness of the board in metres at unit scale (URDF box is 0.10 m
# tall, frame at centre); apply the asset scale to recover the actual
# half-height. The URDF was scaled 2x at source rather than via USD
# scale because USD scaling did not also scale the prismatic joint range.
NAIL_BOARD_UNIT_HALF_HEIGHT = 0.05
NAIL_BOARD_HALF_HEIGHT = NAIL_BOARD_UNIT_HALF_HEIGHT * NAIL_BOARD_DEFAULT_SCALE[2]
NAIL_PRISMATIC_JOINT = "nail_prismatic"
# Nail head centre, at q=0, expressed in the (unscaled) board root frame
# (URDF: board top at z=+0.05, nail head visual centred at +0.084 above
# the nail link origin which sits on the board top).
NAIL_HEAD_UNIT_Z = 0.05 + 0.084
# Default forbidden-zone sphere radius around the nail head (in metres,
# applied in the *scaled* board frame). Tuned so a fingertip touching the
# head is rejected but the hammer head can still strike from above.
NAIL_HEAD_FORBIDDEN_RADIUS = 0.025

# Scene field name for the nail-on-a-board target.
TARGET_KEY = "hammer_target"

# Default geometry for the cylindrical hammer stand (single kinematic prop).
# Radius is sized slightly larger than the old 0.05x0.05 cube's half-diagonal
# (~0.0354 m) so the hammer head stays fully supported at any yaw angle.
HAMMER_STAND_DEFAULT_RADIUS = 0.1
HAMMER_STAND_DEFAULT_HEIGHT = 0.05
HAMMER_STAND_DEFAULT_XY_OFFSET = (-0.07, 0.08)
HAMMER_STAND_DEFAULT_COLOR = (0.35, 0.22, 0.12)


def _make_nail_board_articulation_cfg(
    *,
    init_pos: tuple[float, float, float],
    scale: tuple[float, float, float],
    stiffness: float,
    damping: float,
    effort_limit_sim: float,
    velocity_limit_sim: float,
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/HammerTarget",
        spawn=sim_utils.UsdFileCfg(
            usd_path=NAIL_BOARD_USD_PATH,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={NAIL_PRISMATIC_JOINT: 0.0},
        ),
        actuators={
            "nail_spring": ImplicitActuatorCfg(
                joint_names_expr=[NAIL_PRISMATIC_JOINT],
                stiffness=stiffness,
                damping=damping,
                effort_limit_sim=effort_limit_sim,
                velocity_limit_sim=velocity_limit_sim,
            ),
        },
    )


@configclass
class HammerStrikeSceneCfg(FunctionalGraspingEnvCfg.FunctionalGraspingSceneCfg):
    """Scene with the hammer ``object``, the nail-on-a-board target, and two
    cylindrical pedestals lifting the hammer off the table."""

    # Placeholder; the actual init pose / actuator gains are written in
    # ``HammerStrikeEnvCfg.__post_init__`` so subclasses can tweak them.
    hammer_target: ArticulationCfg = _make_nail_board_articulation_cfg(
        init_pos=(0.0, 0.0, 0.0),
        scale=NAIL_BOARD_DEFAULT_SCALE,
        stiffness=0.0,
        damping=0.1,
        effort_limit_sim=1.0,
        velocity_limit_sim=1000.0,
    )


@configclass
class HammerStrikeObservationsCfg(FunctionalGraspingObservationsCfg):
    """Observation layout for hammer strike without an ``object_pose`` command."""

    @configclass
    class StateObsCfg(FunctionalGraspingObservationsCfg.StateObsCfg):
        # Current nail press depth (a joint position) — observable state.
        nail_press = ObsTerm(
            func=mdp.max_joint_pos_signed,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "asset_cfg": SceneEntityCfg(TARGET_KEY, joint_names=[NAIL_PRISMATIC_JOINT]),
            },
        )

    @configclass
    class GoalObsCfg(ObsGroup):
        goal_pos_b = ObsTerm(
            func=mdp.asset_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg(TARGET_KEY)},
        )
        required_press_distance = ObsTerm(func=mdp.scalar_obs, params={"value": 0.0})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    state: StateObsCfg = StateObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class HammerStrikeEnvFloatingDexHandRightCfg(FunctionalGraspingEnvFloatingDexHandRightCfg):
    """Hammer strike: grasp + affinity overlap + nail compression success."""

    usd_path: str = HAMMER_USD_PATH
    object_mass: float = HAMMER_MASS
    object_scale: tuple[float, float, float] = (1.2, 1.2, 1.2)
    # Vertical (head-down) pose: distance from the asset origin to the head's
    # bottom face is roughly half the hammer's total length. Tune in the
    # viewer if the hammer floats above or sinks into the table.
    object_half_height: float = 0.02
    object_static_friction: float | None = 2.0
    object_dynamic_friction: float | None = 2.0
    object_friction_combine_mode: str = "average"
    object_init_rot: tuple[float, float, float, float] = HAMMER_ROT_INIT
    object_collision_enabled: bool = False

    # Place the hammer toward one side of the table; tune once previewed.
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = -0.2
    object_reset_x_range: tuple[float, float] = (-0.05, 0.2)
    object_reset_y_range: tuple[float, float] = (-0.2, -0.05)
    # Vertical hammer is unstable; keep yaw fixed so it doesn't tip on spawn.
    object_reset_yaw_range: tuple[float, float] = (-0.4 - 1.0, -0.4 - 0.5)
    # object_reset_roll_range: tuple[float, float] = (-1.57,-1.57)
    # object_reset_pitch_range: tuple[float, float] = (0.3,0.3)
    # Nail-on-a-board target placement (offsets from the table centre).
    nail_board_init_x_offset: float = 0.0
    nail_board_init_y_offset: float = 0.2
    nail_board_reset_x_range: tuple[float, float] = (-0.05, 0.2)
    nail_board_reset_y_range: tuple[float, float] = (0.0, 0.1)
    nail_board_scale: tuple[float, float, float] = NAIL_BOARD_DEFAULT_SCALE
    nail_board_half_height: float = NAIL_BOARD_HALF_HEIGHT

    # Damping-only actuator: zero stiffness (no restoring spring), small
    # damping bleeds off post-strike velocity, no velocity cap so a strike
    # can actually move the nail. See module docstring.
    nail_board_stiffness: float = 0.0
    nail_board_damping: float = 0.1
    nail_board_effort_limit_sim: float = 1.0
    nail_board_velocity_limit_sim: float = 10.0

    # (Hammer-frame) forbidden zones intentionally empty: success is gated on
    # the nail-prismatic joint compression plus the target-frame forbidden
    # zone below. Re-populate ``forbidden_zones`` to add a
    # hand-stay-off-the-hammer-head requirement.
    forbidden_zones: tuple[ForbiddenZone, ...] = ()

    # Forbidden zones in the *target asset* (nail+board) local frame. Default
    # is a sphere around the nail head so the hand cannot just press the
    # nail in directly -- pressing must come via the hammer. Centre uses the
    # scaled nail head height in the board root frame.
    target_forbidden_zones: tuple[ForbiddenZone, ...] = (
        ForbiddenZone(
            kind="sphere",
            center=(0.0, 0.0, NAIL_HEAD_UNIT_Z * NAIL_BOARD_DEFAULT_SCALE[2]),
            radius=NAIL_HEAD_FORBIDDEN_RADIUS,
        ),
    )

    # Single kinematic cylindrical stand the hammer rests on.
    # ``hammer_stand_radius`` is the cylinder's radius (rotational symmetry
    # keeps the hammer supported at any yaw); ``hammer_stand_height`` is the
    # cylinder's height and doubles as the hammer lift (set into
    # ``table_clearance`` in __post_init__ so the hammer's bottom face lands
    # on top of the cylinder). ``hammer_stand_xy_offset`` is interpreted in
    # the *hammer's local frame* by the per-reset sync, so e.g. a positive
    # x offset places the stand under the head end and rotates with the
    # hammer's yaw -- the head stays supported regardless of yaw.
    hammer_stand_radius: float = HAMMER_STAND_DEFAULT_RADIUS
    hammer_stand_height: float = HAMMER_STAND_DEFAULT_HEIGHT
    hammer_stand_xy_offset: tuple[float, float] = HAMMER_STAND_DEFAULT_XY_OFFSET

    # Press distance (m) the nail must travel relative to its init pose.
    press_distance_m: float = 0.06
    # ``displacement`` measures abs(q - q_init); ``ratio`` uses joint range.
    press_mode: str = "displacement"

    # Minimum xy distance (m) the nail-board target must keep from the
    # hammer spawn. Enforced via rejection sampling at reset time.
    min_object_goal_xy_distance: float = 0.10

    # Override the scene to include the nail-board target articulation.
    scene: HammerStrikeSceneCfg = HammerStrikeSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
    )
    observations: HammerStrikeObservationsCfg = HammerStrikeObservationsCfg()

    def __post_init__(self):
        # Lift the hammer's spawn z by the cylinder's height so it rests on
        # top of the stand (read by the base ``__post_init__``).
        self.table_clearance = self.hammer_stand_height
        super().__post_init__()

        # Position the nail-board on top of the table at the configured offset.
        # Board frame is at the *centre* of the board, so add half the board
        # thickness to land its bottom face on the table top.
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        table_pos = self.scene.table.init_state.pos

        # Spawn the kinematic cylindrical stand under the hammer. Initial
        # placement uses the *nominal* hammer xy at config time; the
        # ``reset_hammer_stand`` event below snaps the stand to the actual
        # post-reset hammer pose at every reset. Centre z is set so the
        # stand's bottom face sits on the table top. Collision-enabled so
        # the hammer actually rests on it; gravity disabled + kinematic so
        # the stand never falls. Cylinder axis is Z so its top is a flat disc.
        stand_radius = self.hammer_stand_radius
        stand_height = self.hammer_stand_height
        stand_world_z = table_top_z + stand_height * 0.5
        stand_x = table_pos[0] + self.object_init_x_offset + self.hammer_stand_xy_offset[0]
        stand_y = table_pos[1] + self.object_init_y_offset + self.hammer_stand_xy_offset[1]
        self.scene.hammer_stand = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/HammerStand",
            spawn=sim_utils.CylinderCfg(
                radius=stand_radius,
                height=stand_height,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=HAMMER_STAND_DEFAULT_COLOR,
                    roughness=1.0,
                    metallic=0.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(stand_x, stand_y, stand_world_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.scene.hammer_target = _make_nail_board_articulation_cfg(
            init_pos=(
                table_pos[0] + self.nail_board_init_x_offset,
                table_pos[1] + self.nail_board_init_y_offset,
                table_top_z + self.nail_board_half_height,
            ),
            scale=self.nail_board_scale,
            stiffness=self.nail_board_stiffness,
            damping=self.nail_board_damping,
            effort_limit_sim=self.nail_board_effort_limit_sim,
            velocity_limit_sim=self.nail_board_velocity_limit_sim,
        )

        # Attach reset events for the nail-board (root pose + prismatic joint).
        # The excluding variant rejection-samples the nail-board xy so it
        # stays at least ``min_object_goal_xy_distance`` from the hammer's
        # already-reset xy.
        self.events.reset_hammer_target = EventTerm(
            func=mdp.reset_root_pose_uniform_excluding,
            mode="reset",
            params={
                "pose_range": {
                    "x": list(self.nail_board_reset_x_range),
                    "y": list(self.nail_board_reset_y_range),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": [0.0, 0.0],
                },
                "asset_cfg": SceneEntityCfg(TARGET_KEY),
                "reference_asset_cfg": SceneEntityCfg("object"),
                "min_xy_distance": self.min_object_goal_xy_distance,
            },
        )
        self.events.reset_hammer_target_joints = EventTerm(
            func=mdp.reset_joints_to_init,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(TARGET_KEY, joint_names=[NAIL_PRISMATIC_JOINT]),
            },
        )

        # Snap the kinematic stand onto the hammer's reset pose. The xy
        # offset is interpreted in the hammer's local frame, so it rotates
        # with the hammer's yaw (the cylinder is yaw-symmetric, so we don't
        # need to rotate the stand mesh itself). z_offset cancels the
        # hammer's lift (object_half_height + stand_height) and re-adds
        # half the stand height, landing the stand's bottom face on the
        # table top regardless of where the hammer was sampled in xy.
        stand_z_offset = -(self.object_half_height + self.hammer_stand_height * 0.5)
        self.events.reset_hammer_stand = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("hammer_stand"),
                "source_cfg": SceneEntityCfg("object"),
                "source_local_offset": (
                    self.hammer_stand_xy_offset[0],
                    self.hammer_stand_xy_offset[1],
                    0.0,
                ),
                "z_offset": stand_z_offset,
            },
        )

        # No object-pose goal command: the goal is encoded by affinity overlap
        # and nail compression instead. Drop reward terms that depended on
        # the goal command so the RewardManager doesn't try to evaluate them.
        self.commands.object_pose = None
        self.rewards.position_tracking = None
        self.rewards.success = None
        # ``goal_pos_b.asset_cfg`` and ``nail_press.asset_cfg`` are fixed to
        # ``TARGET_KEY`` at the class level (see ``HammerStrikeObservationsCfg``).
        # Only the leaf-tunable ``press_distance_m`` needs propagation.
        self.observations.goal.required_press_distance.params["value"] = self.press_distance_m

        # Replace success termination with the hammer-strike-specific success.
        sphere_zones, box_zones, cylinder_zones = split_zones(self.forbidden_zones)
        target_sphere_zones, target_box_zones, target_cylinder_zones = split_zones(self.target_forbidden_zones)
        self.terminations.success = DoneTerm(
            func=mdp.hammer_strike_success,
            params={
                "press_threshold_m": self.press_distance_m,
                "press_mode": self.press_mode,
                "sphere_zones": sphere_zones,
                "box_zones": box_zones,
                "cylinder_zones": cylinder_zones,
                "target_sphere_zones": target_sphere_zones,
                "target_box_zones": target_box_zones,
                "target_cylinder_zones": target_cylinder_zones,
                "asset_cfg": SceneEntityCfg("robot", body_names=self.robot_config.fingertip_body_names),
                "object_cfg": SceneEntityCfg("object"),
                "target_asset_cfg": SceneEntityCfg(TARGET_KEY),
                "target_joint_cfg": SceneEntityCfg(TARGET_KEY, joint_names=[NAIL_PRISMATIC_JOINT]),
            },
        )
