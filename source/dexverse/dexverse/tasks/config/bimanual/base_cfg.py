# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base for bimanual coordinated lift tasks.

Success is height-based: the object root must rise by ``lift_height`` above
its reset pose.  A green cube visualises the target height.

Subclass ``BimanualLiftObjectEnvCfg`` (or the concrete Shadow-bimanual
variant below) and override the class attributes that describe the object:

    usd_path, scale, mass, object_half_height, table_clearance,
    object_init_rot, object_init_x_offset, lift_height, obj_y_range,
    obj_yaw_range, episode_length_s_override.

The rewards skip the single-hand ``lift_when_grasping`` term (which assumes
four fingertips) and rely on ``object_ee_distance`` (summed over all
fingertips from both hands) plus the continuous ``object_lift_height``
signal.  Both-hand contact naturally drives the reach term down to zero.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .usd_helpers import ensure_single_rigid_body

SUCCESS_MARKER_SIZE = (0.03, 0.03, 0.03)
SUCCESS_MARKER_COLOR = (0.1, 0.9, 0.1)
SUCCESS_MARKER_QUAT = (1.0, 0.0, 0.0, 0.0)


def make_lift_object_cfg(
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mass: float = 0.5,
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> RigidObjectCfg:
    """Build a ``RigidObjectCfg`` for a lift target from a USD file."""
    usd_path = ensure_single_rigid_body(usd_path)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
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
            # None since the USD already was processed to have collision properties.
            collision_props=None,
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=init_rot),
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


@configclass
class BimanualLiftObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for bimanual coordinated lift tasks.

    ``state`` (observable, no velocities): object pose + ``object_lift_height``
    (the height the object has risen above its spawn — the height-based success
    signal). ``privileged``: object linear / angular velocities (+ inherited
    robot ``joint_vel`` / ``hand_tips``). ``proprio`` stays joint-pos-only.

    No ``goal`` group: success is "rise by a per-task constant ``lift_height``"
    (uninformative as a constant for imitation learning, and would otherwise need
    a reward-side value); the current lift height in ``state`` plus the
    demonstrated motion carry the objective.
    """

    @configclass
    class StateObsCfg(ObsGroup):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_lift_height = ObsTerm(func=mdp.object_height_delta, noise=Unoise(n_min=-0.0, n_max=0.0))

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


@configclass
class BimanualLiftRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reach + height-based lift signal (works with any fingertip count)."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            # body_names wired to fingertip_body_names in __post_init__.
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    object_lift_height = RewTerm(
        func=mdp.object_lift_height,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": 0.0},
    )


@configclass
class BimanualLiftTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Bounding-box termination + height-based success."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-0.2, 1.5)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = DoneTerm(
        func=mdp.object_lifted,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            # overwritten from lift_height in __post_init__
            "min_height": 0.2,
        },
    )


@configclass
class BimanualLiftEventCfg(dexverse_base_env.EventCfg):
    """Syncs the success marker to the object spawn at each reset."""

    reset_success_marker = EventTerm(
        func=mdp.sync_object,
        mode="reset",
        params={
            "target_cfg": SceneEntityCfg("success_marker"),
            "source_cfg": SceneEntityCfg("object"),
            # overwritten from lift_height in __post_init__
            "z_offset": 0.2,
            "quat": SUCCESS_MARKER_QUAT,
        },
    )


@configclass
class BimanualLiftObjectEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Robot-agnostic base for bimanual coordinated lift tasks.

    Concrete tasks set ``usd_path`` and tune scale / mass / geometry.
    """

    # Object parameters -- subclasses should override.
    usd_path: str = ""
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mass: float = 0.5
    object_half_height: float = 0.05
    """Distance from the mesh root to the geometric bottom (meters).

    Used together with ``table_clearance`` to place the object flush on the
    tabletop.  For USDs whose root is at the geometric centre, set this to
    half the object height; otherwise tune until the object sits cleanly.
    """
    table_clearance: float = 0.0
    """Extra lift above the estimated bottom (meters)."""
    object_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0

    # Task parameters.
    lift_height: float = 0.15
    """Height above the reset pose that marks success (meters)."""

    # ---- Object reset randomization (per-axis ranges) ----
    # ``obj_y_range`` and ``obj_yaw_range`` keep their historical defaults so
    # existing configs are unaffected. The remaining axes default to no
    # randomization; subclasses can override.
    obj_x_range: tuple[float, float] = (0.0, 0.0)
    obj_y_range: tuple[float, float] = (-0.1, 0.1)
    obj_z_range: tuple[float, float] = (0.0, 0.0)
    obj_roll_range: tuple[float, float] = (0.0, 0.0)
    obj_pitch_range: tuple[float, float] = (0.0, 0.0)
    obj_yaw_range: tuple[float, float] = (-0.2, 0.2)

    episode_length_s_override: float = 10.0

    observations: BimanualLiftObservationsCfg = BimanualLiftObservationsCfg()
    rewards: BimanualLiftRewardsCfg = BimanualLiftRewardsCfg()
    terminations: BimanualLiftTerminationsCfg = BimanualLiftTerminationsCfg()
    events: BimanualLiftEventCfg = BimanualLiftEventCfg()

    @configclass
    class BimanualLiftSceneCfg(dexverse_base_env.SceneCfg):
        object: RigidObjectCfg | None = None
        success_marker: RigidObjectCfg = SUCCESS_MARKER_CFG
        wrist_camera: TiledCameraCfg | None = None
        right_wrist_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rh_palm/wrist_cam",
            update_period=0.0,
            width=256,
            height=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, -0.1355, -0.1875),
                rot=(0.0, 0.0, 0.04997916927067833, 0.9987502603949663),
                convention="ros",
            ),
        )
        left_wrist_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/lh_palm/wrist_cam",
            update_period=0.0,
            width=256,
            height=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, -0.1355, -0.1875),
                rot=(0.0, 0.0, 0.04997916927067833, 0.9987502603949663),
                convention="ros",
            ),
        )

    scene: BimanualLiftSceneCfg = BimanualLiftSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        success_marker=SUCCESS_MARKER_CFG,
    )

    def __post_init__(self):
        if not self.usd_path:
            raise ValueError(f"{type(self).__name__}: `usd_path` must be set before __post_init__.")

        # Build the object scene entity so the base class sees it.
        self.scene.object = make_lift_object_cfg(
            usd_path=self.usd_path,
            scale=self.scale,
            mass=self.mass,
            init_rot=self.object_init_rot,
        )

        super().__post_init__()

        self.episode_length_s = self.episode_length_s_override
        if hasattr(self.commands, "object_pose"):
            self.commands.object_pose = None

        # Some imported USDs do not support dynamic material randomisation.
        self.events.object_physics_material = None

        # Place the object flush on the table surface (apply xy offsets).
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        target_z = table_top_z + self.object_half_height + self.table_clearance
        ox, oy, _ = self.scene.object.init_state.pos
        spawn_x = ox + self.object_init_x_offset
        spawn_y = oy + self.object_init_y_offset
        self.scene.object.init_state.pos = (spawn_x, spawn_y, target_z)
        self.scene.success_marker.init_state.pos = (
            spawn_x,
            spawn_y,
            target_z + self.lift_height,
        )
        self.scene.success_marker.init_state.rot = SUCCESS_MARKER_QUAT

        # Clamp out-of-bound range to the table footprint.
        if self.terminations.object_out_of_bound is not None:
            table_size = self.scene.table.spawn.size
            self.terminations.object_out_of_bound.params["in_bound_range"] = {
                "x": (-table_size[0] * 0.5, table_size[0] * 0.5),
                "y": (-table_size[1] * 0.5, table_size[1] * 0.5),
                "z": (-0.2, 1.5),
            }

        # Propagate lift height into success termination and marker reset.
        self.terminations.success.params["min_height"] = self.lift_height
        self.events.reset_success_marker.params["z_offset"] = self.lift_height

        # Reset-time pose randomisation for the object.
        if self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": list(self.obj_x_range),
                "y": list(self.obj_y_range),
                "z": list(self.obj_z_range),
                "roll": list(self.obj_roll_range),
                "pitch": list(self.obj_pitch_range),
                "yaw": list(self.obj_yaw_range),
            }

        # Contact sensors covering all fingertips of both hands.
        mdp.setup_fingertip_contact_observation(self)
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names

        # Success remains pure height-based for bimanual lift tasks.


@configclass
class BimanualLiftObjectEnvFloatingShadowBimanualCfg(BimanualLiftObjectEnvCfg):
    """Bimanual lift concrete config using the floating Shadow bimanual hand."""

    robot_type: str = "floating_shadow_bimanual"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
