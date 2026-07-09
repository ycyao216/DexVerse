# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for relocate task with tabletop manipulation."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .base_cfg import PickupObjectObservationsCfg

SPHERE_RADIUS = 0.035
TARGET_OPACITY = 0.35
CENTER_SQUARE_SIZE = 0.45
TARGET_RANGE_Z = (0.10, 0.20)
LINE_THICKNESS = 0.005
LINE_HEIGHT = 0.002
LINE_Z_OFFSET = 0.001

OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.SphereCfg(
        radius=SPHERE_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)

LINE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/RelocateArea_Line",
    spawn=sim_utils.CuboidCfg(
        size=(0.1, LINE_THICKNESS, LINE_HEIGHT),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        visible=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)


@configclass
class RelocateCommandsCfg(dexverse_base_env.CommandsCfg):
    """Command terms for relocate task."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=TARGET_RANGE_Z,
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="object",
        position_only=True,
    )


@configclass
class RelocateObservationsCfg(PickupObjectObservationsCfg):
    """Observation layout for relocate-sphere.

    Inherits the pickup-object split (object pose in ``state``, linear & angular
    velocities in ``privileged``) and adds the commanded goal position in the
    ``goal`` group (``target_object_pose_b``).
    """

    @configclass
    class GoalObsCfg(ObsGroup):
        target_object_pose_b = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    goal: GoalObsCfg = GoalObsCfg()


@configclass
class RelocateRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for relocate task."""

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
        params={
            "std": 0.15,
            "command_name": "object_pose",
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=8.0,
        params={
            "pos_std": 0.05,
            "rot_std": None,
            "command_name": "object_pose",
        },
    )


@configclass
class RelocateEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for relocate task."""

    # Disable per-env reset debug logging in training path.
    debug_reset = None


@configclass
class RelocateTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for relocate task."""

    success = DoneTerm(
        func=mdp.object_at_goal_position,
        params={
            "command_name": "object_pose",
            "threshold": 0.03,
        },
    )


@configclass
class RelocateEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Relocate task configuration (base, robot-agnostic)."""

    supports_object_pose_command: bool = True

    commands: RelocateCommandsCfg = RelocateCommandsCfg()
    observations: RelocateObservationsCfg = RelocateObservationsCfg()
    rewards: RelocateRewardsCfg = RelocateRewardsCfg()
    events: RelocateEventCfg = RelocateEventCfg()
    terminations: RelocateTerminationsCfg = RelocateTerminationsCfg()
    scene: dexverse_base_env.SceneCfg = dexverse_base_env.SceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=OBJECT_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        # Keep a single target per episode (no mid-episode resampling) and position-only tracking.
        self.episode_length_s = 20.0
        self.commands.object_pose.resampling_time_range = (self.episode_length_s + 1.0, self.episode_length_s + 1.0)
        self.commands.object_pose.position_only = True

        # Visual target sphere (no collision) at the commanded goal position.
        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.6, 0.9),
                        opacity=TARGET_OPACITY,
                    ),
                )
            },
        )

        # Replace success markers with spheres (avoid table geometry markers).
        self.commands.object_pose.success_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
                ),
                "success": sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    visible=False,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
                ),
            },
        )

        # Randomize object reset on the tabletop.
        if self.events.reset_object is not None:
            table_pos = self.scene.table.init_state.pos
            table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
            object_z = table_top_z + SPHERE_RADIUS
            half_side = CENTER_SQUARE_SIZE * 0.5
            x_min = table_pos[0] - half_side
            x_max = table_pos[0] + half_side
            y_min = table_pos[1] - half_side
            y_max = table_pos[1] + half_side
            # Ensure initial object z sits on the tabletop.
            self.scene.object.init_state.pos = (
                table_pos[0],
                table_pos[1],
                object_z,
            )
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [-half_side, half_side],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-3.14, 3.14],
            }
            self.commands.object_pose.use_world_frame = True
            self.commands.object_pose.ranges.pos_x = (table_pos[0], table_pos[0])
            self.commands.object_pose.ranges.pos_y = (table_pos[1], table_pos[1])
            self.commands.object_pose.ranges.pos_z = (
                table_top_z + TARGET_RANGE_Z[0],
                table_top_z + TARGET_RANGE_Z[1],
            )

            # Add red outline of the center square on the tabletop.
            z_line = table_top_z + LINE_HEIGHT * 0.5 + LINE_Z_OFFSET
            line_specs = [
                ("line_y_pos", (CENTER_SQUARE_SIZE, LINE_THICKNESS, LINE_HEIGHT), (table_pos[0], y_max, z_line)),
                ("line_y_neg", (CENTER_SQUARE_SIZE, LINE_THICKNESS, LINE_HEIGHT), (table_pos[0], y_min, z_line)),
                ("line_x_pos", (LINE_THICKNESS, CENTER_SQUARE_SIZE, LINE_HEIGHT), (x_max, table_pos[1], z_line)),
                ("line_x_neg", (LINE_THICKNESS, CENTER_SQUARE_SIZE, LINE_HEIGHT), (x_min, table_pos[1], z_line)),
            ]
            for name, size, pos in line_specs:
                line_cfg = LINE_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/RelocateArea_{name}")
                line_cfg.spawn.size = size
                line_pos = line_cfg.init_state.pos
                line_cfg.init_state.pos = (line_pos[0], line_pos[1], pos[2])
                setattr(self.scene, name, line_cfg)

        # Setup contact sensors if enabled
        mdp.setup_fingertip_contact_observation(self)

        # Override reward body names with robot-specific values
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


# Unified robot configuration (supports all robot types via robot_type argument)
@configclass
class RelocateEnvFloatingDexHandRightCfg(RelocateEnvCfg):
    """Relocate environment configuration for Floating DexHand (supports Shadow and Leap).

    Robot configuration is handled by the base class registry. This config only needs to:
    1. Set the default robot_type
    2. Configure teleoperation devices if needed
    """

    # Set default robot_type for this config
    robot_type: str = "floating_shadow_right"
    # XR configuration (needed for teleop)
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        """Post initialization."""
        # Call parent __post_init__ which will configure the robot based on robot_type
        super().__post_init__()
        setup_floating_teleop(self)
