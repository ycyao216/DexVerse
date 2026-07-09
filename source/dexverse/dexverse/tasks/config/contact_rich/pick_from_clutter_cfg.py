# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for pick-from-clutter tabletop manipulation."""

from __future__ import annotations

import colorsys
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from ..robot_init import set_robot_wrist_init_world_pos

TARGET_COLOR = (0.10, 0.85, 0.20)
NUM_DISTRACTORS = 19
DISTRACTOR_COLORS = tuple(
    colorsys.hsv_to_rgb((0.03 + 0.61803398875 * i) % 1.0, 0.72, 0.95) for i in range(NUM_DISTRACTORS)
)

BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.5
LIFT_HEIGHT_M = 0.20
OBJECT_MASS_KG = 0.12
# Hard-coded max safety radius across clutter primitives.
OBJECT_SAFE_RADIUS_M = 0.087801
OBJECT_SPAWN_CENTER_Z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT + OBJECT_SAFE_RADIUS_M + 0.004
CLUTTER_LAYER_SPACING_M = 2.0 * OBJECT_SAFE_RADIUS_M + 0.006

# Keep the original corral size and place objects in four XY slots per layer.
CLUTTER_LAYER_SLOT_OFFSETS = (
    (-0.10, -0.10),
    (-0.10, 0.10),
    (0.10, -0.10),
    (0.10, 0.10),
)
POSITION_JITTER_X_M = 0.004
POSITION_JITTER_Y_M = 0.004
POSITION_JITTER_Z_RANGE_M = (0.0, 0.003)
GOAL_MARKER_RADIUS_M = 0.02
WALL_HEIGHT_M = 0.20
WALL_THICKNESS_M = 0.04
CORRAL_HALF_X_M = 0.24
CORRAL_HALF_Y_M = 0.24
OBJECT_ROLL_PITCH_RANGE_RAD = (-math.pi, math.pi)
OBJECT_YAW_RANGE_RAD = (-math.pi, math.pi)
ROBOT_INIT_X_TRANSLATION = 0.13
ROBOT_INIT_Y_TRANSLATION = 0.0
ROBOT_INIT_Z_TRANSLATION = 0.38

CLUTTER_OBJECT_NAMES = ("object",) + tuple(f"distractor_{i}" for i in range(NUM_DISTRACTORS))
CLUTTER_SLOT_OFFSETS_BY_ASSET = tuple(
    CLUTTER_LAYER_SLOT_OFFSETS[i % len(CLUTTER_LAYER_SLOT_OFFSETS)] for i in range(len(CLUTTER_OBJECT_NAMES))
)
CLUTTER_BASE_HEIGHTS_BY_ASSET = tuple(
    OBJECT_SPAWN_CENTER_Z + (i // len(CLUTTER_LAYER_SLOT_OFFSETS)) * CLUTTER_LAYER_SPACING_M
    for i in range(len(CLUTTER_OBJECT_NAMES))
)


def _shape_assets(color: tuple[float, float, float]) -> list[sim_utils.SpawnerCfg]:
    material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        roughness=0.55,
        metallic=0.0,
    )
    return [
        sim_utils.CuboidCfg(
            size=(0.062, 0.062, 0.062),
            visual_material=material,
            visible=True,
        ),
        sim_utils.CylinderCfg(
            radius=0.022,
            height=0.17,
            visual_material=material,
            visible=True,
        ),
        sim_utils.SphereCfg(
            radius=0.038,
            visual_material=material,
            visible=True,
        ),
        sim_utils.CuboidCfg(
            size=(0.112, 0.056, 0.062),
            visual_material=material,
            visible=True,
        ),
    ]


def _make_clutter_object_cfg(name: str, color: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=_shape_assets(color),
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS_KG),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_CENTER_Z)),
    )


def _make_goal_marker_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.SphereCfg(
            radius=GOAL_MARKER_RADIUS_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.85, 0.95),
                emissive_color=(0.00, 0.25, 0.30),
                roughness=0.8,
                metallic=0.0,
            ),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_CENTER_Z + LIFT_HEIGHT_M)),
    )


def _make_wall_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, WALL_HEIGHT_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.35, 0.38),
                roughness=0.9,
                metallic=0.0,
            ),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


TARGET_OBJECT_CFG = _make_clutter_object_cfg("Object", TARGET_COLOR)
GOAL_MARKER_CFG = _make_goal_marker_cfg()
DISTRACTOR_CFGS = [_make_clutter_object_cfg(f"Distractor{i}", DISTRACTOR_COLORS[i]) for i in range(NUM_DISTRACTORS)]
WALL_X_NEG_CFG = _make_wall_cfg("WallXNeg")
WALL_X_POS_CFG = _make_wall_cfg("WallXPos")
WALL_Y_NEG_CFG = _make_wall_cfg("WallYNeg")
WALL_Y_POS_CFG = _make_wall_cfg("WallYPos")


@configclass
class PickFromClutterObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for pick-from-clutter.

    Target object position / orientation / up-axis / tilt in ``proprio``;
    velocities in ``privileged``. Distractor positions are not observed
    (the policy must localise via cameras).
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=0.0, n_max=0.0))
        object_up_b = ObsTerm(func=mdp.object_up_b, noise=Unoise(n_min=0.0, n_max=0.0))
        object_tilt_angle = ObsTerm(func=mdp.object_tilt_angle, noise=Unoise(n_min=0.0, n_max=0.0))
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, noise=Unoise(n_min=0.0, n_max=0.0))
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, noise=Unoise(n_min=0.0, n_max=0.0))

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class PickFromClutterRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward terms for pick-from-clutter."""

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

    lift_height = RewTerm(
        func=mdp.object_lift_height,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "min_height": LIFT_HEIGHT_M,
        },
    )


@configclass
class PickFromClutterTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for pick-from-clutter."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (BOUND_Z_MIN, BOUND_Z_MAX)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.object_lifted,
        params={
            "min_height": LIFT_HEIGHT_M,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickFromClutterEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for pick-from-clutter."""

    reset_goal_marker = EventTerm(
        func=mdp.sync_object,
        mode="reset",
        params={
            "target_cfg": SceneEntityCfg("goal_marker"),
            "source_cfg": SceneEntityCfg("object"),
            "z_offset": LIFT_HEIGHT_M,
        },
    )


@configclass
class PickFromClutterEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Pick-from-clutter task configuration with one green target and clutter distractors."""

    observations: PickFromClutterObservationsCfg = PickFromClutterObservationsCfg()
    rewards: PickFromClutterRewardsCfg = PickFromClutterRewardsCfg()
    terminations: PickFromClutterTerminationsCfg = PickFromClutterTerminationsCfg()
    events: PickFromClutterEventCfg = PickFromClutterEventCfg()

    @configclass
    class PickFromClutterSceneCfg(dexverse_base_env.SceneCfg):
        object: RigidObjectCfg = TARGET_OBJECT_CFG
        goal_marker: RigidObjectCfg = GOAL_MARKER_CFG
        distractor_0: RigidObjectCfg = DISTRACTOR_CFGS[0]
        distractor_1: RigidObjectCfg = DISTRACTOR_CFGS[1]
        distractor_2: RigidObjectCfg = DISTRACTOR_CFGS[2]
        distractor_3: RigidObjectCfg = DISTRACTOR_CFGS[3]
        distractor_4: RigidObjectCfg = DISTRACTOR_CFGS[4]
        distractor_5: RigidObjectCfg = DISTRACTOR_CFGS[5]
        distractor_6: RigidObjectCfg = DISTRACTOR_CFGS[6]
        distractor_7: RigidObjectCfg = DISTRACTOR_CFGS[7]
        distractor_8: RigidObjectCfg = DISTRACTOR_CFGS[8]
        distractor_9: RigidObjectCfg = DISTRACTOR_CFGS[9]
        distractor_10: RigidObjectCfg = DISTRACTOR_CFGS[10]
        distractor_11: RigidObjectCfg = DISTRACTOR_CFGS[11]
        distractor_12: RigidObjectCfg = DISTRACTOR_CFGS[12]
        distractor_13: RigidObjectCfg = DISTRACTOR_CFGS[13]
        distractor_14: RigidObjectCfg = DISTRACTOR_CFGS[14]
        distractor_15: RigidObjectCfg = DISTRACTOR_CFGS[15]
        distractor_16: RigidObjectCfg = DISTRACTOR_CFGS[16]
        distractor_17: RigidObjectCfg = DISTRACTOR_CFGS[17]
        distractor_18: RigidObjectCfg = DISTRACTOR_CFGS[18]
        wall_x_neg: RigidObjectCfg = WALL_X_NEG_CFG
        wall_x_pos: RigidObjectCfg = WALL_X_POS_CFG
        wall_y_neg: RigidObjectCfg = WALL_Y_NEG_CFG
        wall_y_pos: RigidObjectCfg = WALL_Y_POS_CFG

    scene: PickFromClutterSceneCfg = PickFromClutterSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=TARGET_OBJECT_CFG,
        goal_marker=GOAL_MARKER_CFG,
        distractor_0=DISTRACTOR_CFGS[0],
        distractor_1=DISTRACTOR_CFGS[1],
        distractor_2=DISTRACTOR_CFGS[2],
        distractor_3=DISTRACTOR_CFGS[3],
        distractor_4=DISTRACTOR_CFGS[4],
        distractor_5=DISTRACTOR_CFGS[5],
        distractor_6=DISTRACTOR_CFGS[6],
        distractor_7=DISTRACTOR_CFGS[7],
        distractor_8=DISTRACTOR_CFGS[8],
        distractor_9=DISTRACTOR_CFGS[9],
        distractor_10=DISTRACTOR_CFGS[10],
        distractor_11=DISTRACTOR_CFGS[11],
        distractor_12=DISTRACTOR_CFGS[12],
        distractor_13=DISTRACTOR_CFGS[13],
        distractor_14=DISTRACTOR_CFGS[14],
        distractor_15=DISTRACTOR_CFGS[15],
        distractor_16=DISTRACTOR_CFGS[16],
        distractor_17=DISTRACTOR_CFGS[17],
        distractor_18=DISTRACTOR_CFGS[18],
        wall_x_neg=WALL_X_NEG_CFG,
        wall_x_pos=WALL_X_POS_CFG,
        wall_y_neg=WALL_Y_NEG_CFG,
        wall_y_pos=WALL_Y_POS_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0

        # Lift the hand above the corral walls for cleaner resets.
        set_robot_wrist_init_world_pos(
            self,
            x=-0.5,
            y=0,
            z=0.7,
        )

        for object_idx, name in enumerate(CLUTTER_OBJECT_NAMES):
            obj_cfg = getattr(self.scene, name)
            slot_x, slot_y = CLUTTER_SLOT_OFFSETS_BY_ASSET[object_idx]
            obj_cfg.init_state.pos = (slot_x, slot_y, CLUTTER_BASE_HEIGHTS_BY_ASSET[object_idx])
            obj_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.events.reset_object = EventTerm(
            func=mdp.reset_clutter_objects,
            mode="reset",
            params={
                "asset_names": CLUTTER_OBJECT_NAMES,
                "slot_offsets": CLUTTER_LAYER_SLOT_OFFSETS,
                "slot_offsets_by_asset": CLUTTER_SLOT_OFFSETS_BY_ASSET,
                "base_height": OBJECT_SPAWN_CENTER_Z,
                "base_height_by_asset": CLUTTER_BASE_HEIGHTS_BY_ASSET,
                "position_jitter_x": POSITION_JITTER_X_M,
                "position_jitter_y": POSITION_JITTER_Y_M,
                "position_jitter_z": POSITION_JITTER_Z_RANGE_M,
                "roll_range": OBJECT_ROLL_PITCH_RANGE_RAD,
                "pitch_range": OBJECT_ROLL_PITCH_RANGE_RAD,
                "yaw_range": OBJECT_YAW_RANGE_RAD,
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "unique_slots": False,
            },
        )

        self.scene.goal_marker.init_state.pos = (
            CLUTTER_SLOT_OFFSETS_BY_ASSET[0][0],
            CLUTTER_SLOT_OFFSETS_BY_ASSET[0][1],
            CLUTTER_BASE_HEIGHTS_BY_ASSET[0] + LIFT_HEIGHT_M,
        )

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        wall_inner_half_x = CORRAL_HALF_X_M
        wall_inner_half_y = CORRAL_HALF_Y_M
        wall_center_z = table_top_z + WALL_HEIGHT_M * 0.5

        self.scene.wall_x_neg.spawn.size = (
            WALL_THICKNESS_M,
            2 * wall_inner_half_y + 2 * WALL_THICKNESS_M,
            WALL_HEIGHT_M,
        )
        self.scene.wall_x_neg.init_state.pos = (-wall_inner_half_x - WALL_THICKNESS_M * 0.5, 0.0, wall_center_z)

        self.scene.wall_x_pos.spawn.size = (
            WALL_THICKNESS_M,
            2 * wall_inner_half_y + 2 * WALL_THICKNESS_M,
            WALL_HEIGHT_M,
        )
        self.scene.wall_x_pos.init_state.pos = (wall_inner_half_x + WALL_THICKNESS_M * 0.5, 0.0, wall_center_z)

        self.scene.wall_y_neg.spawn.size = (2 * wall_inner_half_x, WALL_THICKNESS_M, WALL_HEIGHT_M)
        self.scene.wall_y_neg.init_state.pos = (0.0, -wall_inner_half_y - WALL_THICKNESS_M * 0.5, wall_center_z)

        self.scene.wall_y_pos.spawn.size = (2 * wall_inner_half_x, WALL_THICKNESS_M, WALL_HEIGHT_M)
        self.scene.wall_y_pos.init_state.pos = (0.0, wall_inner_half_y + WALL_THICKNESS_M * 0.5, wall_center_z)

        if self.terminations.object_out_of_bound is not None:
            self.terminations.object_out_of_bound.params["in_bound_range"] = {
                "x": (-wall_inner_half_x, wall_inner_half_x),
                "y": (-wall_inner_half_y, wall_inner_half_y),
                "z": (BOUND_Z_MIN, BOUND_Z_MAX),
            }

        mdp.setup_fingertip_contact_observation(self)
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names


@configclass
class PickFromClutterEnvFloatingDexHandRightCfg(PickFromClutterEnvCfg):
    """Pick-from-clutter environment configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
