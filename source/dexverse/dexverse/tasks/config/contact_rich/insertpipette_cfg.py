# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configuration for inserting a pipette into a glassware opening.

Mirrors the ``insertpeg`` / ``plugcharger`` factory-insertion family, but uses
two ``synthesis`` assets:

* a **pipette** (held object) that starts lying on the tabletop, and
* a **glassware** flask (fixed receptacle, kinematic) standing upright.

The robot must pick up the pipette, bring it vertical, and seat its tip into the
neck of the glassware ("close the entrance"). Success is the factory-style
insertion test: the pipette tip is centered over the neck opening (XY) and
pushed below the rim (Z).

Geometry (from the authored USDs, metersPerUnit=1.0, Z-up):
* pipette: ~0.115 m long along local +X, tip at the local origin, bulb at the
  far end; lies flat on the table.
* glassware005: ~0.136 m tall along local +Z, bottom at local z=0, neck opening
  at the top with an inner radius of ~0.020 m.
"""

import math

import isaaclab.sim as sim_utils
from dexverse.assets import SYNTHESIS_DIR
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .base_cfg import ContactRichEnvCfg
from .synthesis_insertion_assets import prepare_insertion_usd, prepare_visible_usd

# Collapse the multi-body synthesis USDs to a single rigid body. For the
# kinematic glassware we also re-author the colliders to an exact triangle mesh
# so the neck cavity stays hollow and the pipette can actually be inserted.
PIPETTE_USD_PATH = prepare_visible_usd(
    prepare_insertion_usd(SYNTHESIS_DIR / "pipette" / "model_pipette.usd"),
    material_name="Visible_Pipette_Component13",
    diffuse_color=(0.2, 0.75, 1.0),
    opacity=1.0,
    roughness=0.35,
    metallic=0.0,
    cache_tag="visible_pipette_component13_1",
    target_prim_paths=("/root/E_Component13_1",),
)
GLASSWARE_USD_PATH = prepare_visible_usd(
    prepare_insertion_usd(
        SYNTHESIS_DIR / "glassware005" / "model_glassware005.usd",
        collision_approximation="none",
    ),
    material_name="Visible_Glassware",
    diffuse_color=(0.2, 0.75, 1.0),
    opacity=1.0,
    roughness=0.35,
    metallic=0.0,
    cache_tag="visible_glassware",
)

ASSET_SCALE = 1.0
# Mesh check:
# - glassware neck min inner radius is ~0.01975 m
# - pipette max mesh radius is ~0.00558 m before scaling
# Scale 3.0 keeps the pipette below the neck radius with ~3 mm radial clearance.
PIPETTE_ASSET_SCALE = 3.0

# --- Task geometry (meters, after per-asset scale where applicable) ---
PIPETTE_LENGTH = 0.1146 * PIPETTE_ASSET_SCALE
PIPETTE_RADIUS = 0.0054 * PIPETTE_ASSET_SCALE
GLASSWARE_HEIGHT = 0.1358 * ASSET_SCALE - 0.02
GLASSWARE_OPENING_INNER_RADIUS = 0.020 * ASSET_SCALE
TABLE_CLEARANCE = 0.002

# Root offsets from the table top.
PIPETTE_ROOT_OFFSET = PIPETTE_RADIUS + TABLE_CLEARANCE  # lies flat on its side
GLASSWARE_ROOT_OFFSET = TABLE_CLEARANCE  # stands on its flat bottom (local z=0)

# Reference points in each asset's local frame. Pipette USD root is identity and
PIPETTE_TIP_LOCAL_OFFSET = (0.0, 0.0, 0.0)
GLASSWARE_OPENING_LOCAL_OFFSET = (0.0, 0.0, GLASSWARE_HEIGHT)  # neck rim center

# Engaged (shaping) tolerances: tip near/above the opening and roughly centered.
ENGAGE_CENTER_DIST_THRESH = GLASSWARE_OPENING_INNER_RADIUS  # ~0.020 m
ENGAGE_Z_THRESHOLD = 0.02  # tip within 2 cm above the rim counts as engaged

# Success tolerances: tip clearly inside the neck and pushed below the rim.
SUCCESS_CENTER_DIST_THRESH = 0.02
SUCCESS_Z_THRESHOLD = -0.01  # tip at least 1 cm below the rim

PIPETTE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Pipette",
    spawn=sim_utils.UsdFileCfg(
        func=dexverse_base_env.spawn_usd_with_rigid_properties,
        usd_path=PIPETTE_USD_PATH,
        scale=(PIPETTE_ASSET_SCALE, PIPETTE_ASSET_SCALE, PIPETTE_ASSET_SCALE),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
            max_contact_impulse=1.0e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        # Collision is authored in the (collapsed) USD; keep convexDecomposition.
        collision_props=None,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.15, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

GLASSWARE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Glassware",
    spawn=sim_utils.UsdFileCfg(
        func=dexverse_base_env.spawn_usd_with_rigid_properties,
        usd_path=GLASSWARE_USD_PATH,
        scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Keep the glassware fixed during simulation (still resettable on
            # episode reset).
            kinematic_enabled=True,
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=4,
            max_contact_impulse=1.0e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        # Exact triangle-mesh collision authored on the cleaned USD; do not
        # override it with a convex hull here.
        collision_props=None,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.10, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


@configclass
class InsertPipetteObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for the insert-pipette task.

    Two-part assembly: pipette (held) and glassware (fixed). Absolute and
    relative body states live in ``privileged``; the goal group exposes the
    glassware opening position in the robot base frame.
    """

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        pipette_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("pipette"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        glassware_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("glassware"), "base_asset_cfg": SceneEntityCfg("table")},
        )
        pipette_state_rel_glassware = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"body_asset_cfg": SceneEntityCfg("pipette"), "base_asset_cfg": SceneEntityCfg("glassware")},
        )

    @configclass
    class GoalObsCfg(ObsGroup):
        """Insertion target: glassware position in the robot base frame."""

        goal_pos_b = ObsTerm(
            func=mdp.asset_pos_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg("glassware")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: GoalObsCfg = GoalObsCfg()


@configclass
class InsertPipetteRewardsCfg(dexverse_base_env.RewardsCfg):
    """Rewards for the insert-pipette task."""

    engaged = RewTerm(
        func=mdp.factory_insert_engaged_reward,
        weight=2.0,
        params={
            "held_cfg": SceneEntityCfg("pipette"),
            "fixed_cfg": SceneEntityCfg("glassware"),
            "held_base_local_offset": PIPETTE_TIP_LOCAL_OFFSET,
            "target_local_offset": GLASSWARE_OPENING_LOCAL_OFFSET,
            "center_dist_thresh": ENGAGE_CENTER_DIST_THRESH,
            "z_threshold": ENGAGE_Z_THRESHOLD,
        },
    )

    success = RewTerm(
        func=mdp.factory_insert_success_reward,
        weight=10.0,
        params={
            "held_cfg": SceneEntityCfg("pipette"),
            "fixed_cfg": SceneEntityCfg("glassware"),
            "held_base_local_offset": PIPETTE_TIP_LOCAL_OFFSET,
            "target_local_offset": GLASSWARE_OPENING_LOCAL_OFFSET,
            "center_dist_thresh": SUCCESS_CENTER_DIST_THRESH,
            "z_threshold": SUCCESS_Z_THRESHOLD,
        },
    )


@configclass
class InsertPipetteTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Termination terms for the insert-pipette task."""

    success = DoneTerm(
        func=mdp.factory_insert_success,
        params={
            "held_cfg": SceneEntityCfg("pipette"),
            "fixed_cfg": SceneEntityCfg("glassware"),
            "held_base_local_offset": PIPETTE_TIP_LOCAL_OFFSET,
            "target_local_offset": GLASSWARE_OPENING_LOCAL_OFFSET,
            "center_dist_thresh": SUCCESS_CENTER_DIST_THRESH,
            "z_threshold": SUCCESS_Z_THRESHOLD,
        },
    )


@configclass
class InsertPipetteEventCfg(dexverse_base_env.EventCfg):
    """Event configuration for the insert-pipette task."""

    reset_glassware = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("glassware"),
            "pose_range": {
                "x": [0.05, 0.15],
                "y": [-0.08, 0.08],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.radians(15.0), math.radians(15.0)],
            },
        },
    )

    reset_pipette_on_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("pipette"),
            "pose_range": {
                "x": [-0.2, -0.05],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-math.pi / 3.0, math.pi / 3.0],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        },
    )


@configclass
class InsertPipetteSceneCfg(dexverse_base_env.SceneCfg):
    pipette: RigidObjectCfg = PIPETTE_CFG
    glassware: RigidObjectCfg = GLASSWARE_CFG
    object = None


@configclass
class InsertPipetteEnvCfg(ContactRichEnvCfg):
    """Insert-pipette task configuration (base, robot-agnostic)."""

    observations: InsertPipetteObservationsCfg = InsertPipetteObservationsCfg()
    rewards: InsertPipetteRewardsCfg = InsertPipetteRewardsCfg()
    terminations: InsertPipetteTerminationsCfg = InsertPipetteTerminationsCfg()
    events: InsertPipetteEventCfg = InsertPipetteEventCfg()
    scene: InsertPipetteSceneCfg = InsertPipetteSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        pipette=PIPETTE_CFG,
        glassware=GLASSWARE_CFG,
    )

    contact_object_prim: str = "Pipette"

    def __post_init__(self):
        super().__post_init__()

        table_size = self.scene.table.spawn.size
        table_pos = self.scene.table.init_state.pos
        table_top_z = table_pos[2] + table_size[2] * 0.5

        self.scene.glassware.init_state.pos = (0.10, 0.0, table_top_z + GLASSWARE_ROOT_OFFSET)
        self.scene.pipette.init_state.pos = (-0.15, 0.0, table_top_z + PIPETTE_ROOT_OFFSET)

        self.episode_length_s = 25.0

        self.configure_debug_vis()

    def configure_debug_vis(self) -> None:
        """Populate the held/target insertion-reference markers in
        ``observations.scene_vis`` when ``self.enable_debug_vis`` is True.
        """
        super().configure_debug_vis()
        if not self.enable_debug_vis:
            return
        if self.observations.scene_vis is None:
            self.observations.scene_vis = dexverse_base_env.ObservationsCfg.SceneVisObsCfg()
        self.observations.scene_vis.insertion_reference_points = ObsTerm(
            func=mdp.insertion_reference_points_vis,
            params={
                "held_cfg": SceneEntityCfg("pipette"),
                "fixed_cfg": SceneEntityCfg("glassware"),
                "held_local_offset": PIPETTE_TIP_LOCAL_OFFSET,
                "target_local_offset": GLASSWARE_OPENING_LOCAL_OFFSET,
                "held_color": (0.95, 0.15, 0.15),
                "target_color": (0.10, 0.85, 0.20),
                "radius": 0.01,
                "prim_path": "/Visuals/Debug/InsertPipetteReferencePoints",
                "show_frames": True,
                "frame_scale": (0.05, 0.05, 0.05),
                "held_frame_prim_path": "/Visuals/Debug/InsertPipetteTipFrame",
                "target_frame_prim_path": "/Visuals/Debug/InsertPipetteOpeningFrame",
            },
        )


@configclass
class InsertPipetteEnvFloatingDexHandRightCfg(InsertPipetteEnvCfg):
    """Insert-pipette environment configuration for floating dexterous hands."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
