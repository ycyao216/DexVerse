# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-faucet task: turn a fixed faucet handle past a target angle.

A static sink basin is parented in front of and below the faucet (re-synced to
the faucet's pose every reset) so the scene reads like a real sink. The faucet
joint is passive — zero drive stiffness (no spring-back) plus elevated joint
friction and a small armature, so the handle holds whatever angle it is turned
to, like a real valve.
"""

import math

import isaaclab.sim as sim_utils
from dexverse.assets import SYNTHESIS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation as R

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg

FAUCET_USD_PATH = str(SYNTHESIS_DIR / "faucet001" / "model_faucet_3.usd")
SINK_USD_PATH = str(SYNTHESIS_DIR / "sink" / "sink.usd")


def _build_sink_cfg(
    *,
    scale: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
    init_pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Static (kinematic) sink-basin prop.

    The sink is a plain converted mesh (no joints), so it spawns as a kinematic
    rigid object anchored in place; the ``reset_sink`` event re-syncs it under
    the faucet each reset. The asset's authored convex-decomposition colliders
    are kept (``collision_props=None``) so the basin is solid.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Sink",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=SINK_USD_PATH,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
            ),
            collision_props=None,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
    )


def _build_support_cfg(
    *,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    init_pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Purely cosmetic pedestal under the (elevated) faucet.

    A plain primitive cuboid (no USD): a kinematic, gravity-free rigid body
    with **collision disabled**, so it never touches the hand, faucet, or sink
    — it only fills the visible gap between the tabletop and the lifted faucet
    base. It is still a rigid object (not a bare visual prim) so the
    ``reset_support`` event can re-seat it under the faucet each reset via
    ``mdp.sync_object``. The box is axis-aligned, so it spawns/syncs at the
    identity orientation.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FaucetSupport",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class OpenFaucetEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Faucet anchored on the tabletop; success = handle joint reaches target angle."""

    articulation_usd_path: str = FAUCET_USD_PATH
    # Slightly larger than the original 0.6 so the handle is easier to grasp.
    articulation_scale: tuple = (0.75, 0.75, 0.75)
    articulation_init_pos: tuple = (0.3, 0.0, 0.0)
    # -90 degrees about +z: (cos(45 deg), 0, 0, -sin(45 deg)).
    articulation_init_rot: tuple = (math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5))
    # Elevate the faucet off the tabletop so it stands above the sink basin.
    # The base seats the articulation at ``table_top_z + this offset`` (was 0.0,
    # i.e. the base sat flush on the table). Tune so the spout clears the rim.
    articulation_half_height_est: float = 0.1
    articulation_fix_root_link: bool | None = True

    # Faucet reset pose randomization (world-frame offsets from its init pose).
    # The sink/support are re-synced to the faucet each reset, so this also sets
    # how far the whole scene jitters. Replaces the base default (y only); x is
    # added here while the base y range is kept.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [0.0, 0.2],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [0.0, 0.0],
    }

    success_joint_names: str = ".*"
    success_threshold: float = math.radians(80)

    # ---- Passive faucet joint (valve-like) -------------------------------
    # Zero drive stiffness (handled in __post_init__) means no spring-back;
    # elevated joint friction makes the handle hold its angle, and a small
    # armature keeps the solver stable. ``friction`` is the static coefficient
    # and ``dynamic_friction`` the sliding one (Isaac Sim 5.x models both as
    # resisting efforts). Tune magnitudes to taste.
    faucet_joint_static_friction: float = 0.4
    faucet_joint_dynamic_friction: float = 0.3
    faucet_joint_armature: float = 0.01

    # ---- Sink basin prop --------------------------------------------------
    sink_usd_path: str = SINK_USD_PATH
    sink_scale: tuple[float, float, float] = (0.83, 0.83, 0.83)  # TODO: tune to faucet size
    # 90 degrees about +z: (cos(45 deg), 0, 0, sin(45 deg)). Flip the sign of
    # the last component for the other 90deg direction.
    sink_init_rot: tuple[float, float, float, float] = tuple(R.from_euler("z", math.pi / 2).as_quat().tolist())
    # Placement of the sink relative to the faucet, applied every reset via
    # ``mdp.sync_object`` so the basin always tracks the faucet. (x, y) are in
    # the faucet's local frame (rotated by ``articulation_init_rot``); z is a
    # world-frame drop ("below"). Defaults put the basin a bit in front of and
    # ~0.2 m below the elevated faucet root — i.e. roughly at table height.
    # TODO: tune after previewing; which horizontal axis is "in front" depends
    # on the asset's authored orientation.
    sink_offset_from_faucet: tuple[float, float, float] = (0.0, -0.25, -0.1)

    # ---- Cosmetic faucet support pedestal --------------------------------
    # Purely visual rectangular cuboid that fills the gap between the tabletop
    # and the elevated faucet base (the faucet is lifted
    # ``articulation_half_height_est`` above the table, so otherwise it floats).
    # Collision is disabled, so it never affects the hand/faucet/sink. Its
    # height is derived from ``articulation_half_height_est`` in
    # ``__post_init__`` so the pedestal always spans table -> faucet base; only
    # the horizontal footprint and color are tuned here.
    # TODO: tune footprint after previewing so it reads as a base, not a post.
    support_footprint: tuple[float, float] = (0.12, 0.32)
    support_color: tuple[float, float, float] = (0.8, 0.8, 0.8)

    def __post_init__(self):
        super().__post_init__()
        # Success when any selected faucet joint moves by >= pi/4 from init.
        self.terminations.success.func = mdp.joint_relative_move
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "displacement",
            "op": ">=",
            "reduce": "any",
        }

        # Passive faucet joint. The base spawn already zeroes the drive
        # stiffness/damping/max_effort; this implicit actuator additionally
        # writes joint friction + armature (which JointDrivePropertiesCfg can't)
        # while keeping the drive at zero stiffness so there is no spring-back.
        self.scene.articulation.actuators = {
            "faucet_joint": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
                armature=self.faucet_joint_armature,
                friction=self.faucet_joint_static_friction,
                dynamic_friction=self.faucet_joint_dynamic_friction,
            )
        }

        # Spawn the sink near the faucet at table height; the reset event below
        # is the authoritative placement, so this just avoids an off-screen
        # spawn before the first reset.
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        fx, fy, _ = self.articulation_init_pos
        self.scene.sink = _build_sink_cfg(
            scale=self.sink_scale,
            init_rot=self.sink_init_rot,
            init_pos=(fx, fy, table_top_z),
        )

        # Lock the sink in front of / below the faucet every reset. Added after
        # super().__post_init__() so it runs *after* ``reset_articulation`` (the
        # event manager fires reset terms in insertion order), reading the
        # faucet's post-randomization pose. Horizontal offset is faucet-local;
        # the vertical drop is world-frame via ``z_offset``.
        ox, oy, oz = self.sink_offset_from_faucet
        self.events.reset_sink = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("sink"),
                "source_cfg": SceneEntityCfg("articulation"),
                "source_local_offset": (ox, oy, 0.0),
                "z_offset": float(oz),
                "quat": self.sink_init_rot,
            },
        )

        # Cosmetic pedestal under the elevated faucet. Its height equals the
        # faucet elevation, so the box spans table_top_z -> faucet base; its
        # center therefore sits half that height below the faucet root, while
        # (x, y) track the faucet exactly. Spawned here to avoid an off-screen
        # first frame; ``reset_support`` (added after super().__post_init__(),
        # so it fires after ``reset_articulation``) is the authoritative
        # placement and re-seats it under the randomized faucet each reset.
        support_height = float(self.articulation_half_height_est)
        sx, sy = self.support_footprint
        self.scene.faucet_support = _build_support_cfg(
            size=(sx, sy, support_height),
            color=self.support_color,
            init_pos=(fx, fy, table_top_z + support_height / 2.0),
        )
        self.events.reset_support = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("faucet_support"),
                "source_cfg": SceneEntityCfg("articulation"),
                "source_local_offset": (0.0, 0.0, 0.0),
                "z_offset": -support_height / 2.0,
                "quat": (1.0, 0.0, 0.0, 0.0),
            },
        )
