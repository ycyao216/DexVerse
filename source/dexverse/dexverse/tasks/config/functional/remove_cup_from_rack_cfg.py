# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Remove-cup-from-rack task.

A cup starts seated on a wooden cup holder (rack); the hand must lift it off
the rack and set it down *upright* at a goal spot on the tabletop.

Scene layout
~~~~~~~~~~~~
  - ``cup_holder``: a kinematic prop (the wooden stand). It is seated on the
    table and re-randomized each reset (``reset_cup_holder``). Its authored
    convex colliders are kept so the cup physically rests on it.
  - ``object`` (the cup): re-seated *relative to the holder* every reset via
    ``mdp.sync_object`` (``reset_cup_on_holder``), so wherever the holder lands
    *and however it is yawed* the cup tracks it in both position and
    orientation. The relative placement is fully tunable through
    :attr:`cup_offset_from_holder` (holder-local x, y, z) and
    :attr:`cup_rot_on_holder` (the cup's world-frame quat when the holder is at
    its init rotation; upright by default). The sync converts that to a
    holder-local quat and recomposes it with the holder's randomized rotation.
  - ``object_pose`` command: a position-only goal somewhere on the tabletop.

Success (``object_upright_at_goal``): the cup is within
:attr:`success_position_threshold` of the goal AND upright (its local +Z within
:attr:`success_max_tilt_rad` of world +Z, yaw-agnostic).

Reset ordering note: the cup-on-holder sync and holder randomization are added
in ``__post_init__`` *after* ``super().__post_init__()``, so they run after the
base ``reset_object`` (which zeroes the cup's velocity). The event manager fires
reset terms in insertion order, and we add the holder randomization before the
cup sync so the cup reads the holder's *post-randomization* pose.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from dexverse.assets import SYNTHESIS_DIR
from dexverse.tasks.config.bimanual.usd_helpers import ensure_single_rigid_body
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation as R

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .base_cfg import FunctionalGraspingEnvFloatingDexHandRightCfg

CUP_USD_PATH = str(SYNTHESIS_DIR / "Cup_B0CMD4LX4D_ForestGreen" / "model_Cup_B0CMD4LX4D_ForestGreen_69323.usd")
CUP_HOLDER_USD_PATH = str(SYNTHESIS_DIR / "cup_holder" / "model_woodenstand3.usd")

# Identity quaternion = the cup spawns upright (matches the cup used by the
# existing GraspCup task, whose "up" axis is local +Z at identity rotation).
CUP_UPRIGHT_ROT = (1.0, 0.0, 0.0, 0.0)
CUP_INIT_ROT = tuple(R.from_euler("x", -105, degrees=True).as_quat(scalar_first=True))


def _build_cup_holder_cfg(
    *,
    scale: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
    init_pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Static (kinematic) cup-holder prop.

    The holder USD authors its own (nested) ``RigidBodyAPI`` on a child link,
    which would collide with the top-level rigid body that
    ``spawn_usd_with_rigid_properties`` applies ("Failed to find a single rigid
    body ... Found multiple"). ``ensure_single_rigid_body`` strips the authored
    nested rigid/articulation/joint schemas (keeping the mesh colliders) so the
    spawn-applied kinematic body is the only one, and the ``reset_cup_holder``
    event can re-randomize it each reset. The asset's authored convex colliders
    are kept (``collision_props=None``) so the cup rests on it.
    """
    # The source USD authors its own (dynamic) ``RigidBodyAPI`` on a child link;
    # left in place it coexists with the spawn-applied top-level body, so the
    # holder is not purely kinematic (and the RigidObject resolver may abort on
    # multiple bodies). Collapse to a single body here so the only rigid body is
    # the kinematic one applied below.
    holder_usd_path = ensure_single_rigid_body(CUP_HOLDER_USD_PATH)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CupHolder",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=holder_usd_path,
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


@configclass
class RemoveCupFromRackEnvFloatingDexHandRightCfg(FunctionalGraspingEnvFloatingDexHandRightCfg):
    """Lift a cup off a wooden rack and place it upright on the tabletop."""

    # ---- Cup (the manipulated object) ------------------------------------
    usd_path: str = CUP_USD_PATH
    object_mass: float = 0.2
    object_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    object_half_height: float = 0.025
    object_static_friction: float | None = 2.0
    object_dynamic_friction: float | None = 2.0
    object_friction_combine_mode: str = "average"
    object_init_rot: tuple[float, float, float, float] = CUP_UPRIGHT_ROT

    # The cup does not randomize on its own — it is placed relative to the
    # holder by ``reset_cup_on_holder``. Zero every per-axis range so the base
    # ``reset_object`` only re-seats it at its init pose (and zeroes velocity).
    object_reset_x_range: tuple[float, float] = (0.0, 0.0)
    object_reset_y_range: tuple[float, float] = (0.0, 0.0)
    object_reset_yaw_range: tuple[float, float] = (0.0, 0.0)

    # ---- Cup holder (the rack) -------------------------------------------
    cup_holder_usd_path: str = CUP_HOLDER_USD_PATH
    cup_holder_scale: tuple[float, float, float] = (1.5, 1.5, 1.5)  # TODO: tune to cup size
    cup_holder_init_rot: tuple[float, float, float, float] = tuple(
        R.from_euler("z", -90, degrees=True).as_quat(scalar_first=True)
    )
    # Half-height used to seat the holder flush on the tabletop. TODO: tune so
    # the holder's base rests on the table (depends on the asset's origin).
    cup_holder_half_height: float = 0.0125
    # Where the holder is centred on the table (offsets from the table centre).
    cup_holder_init_x_offset: float = 0.0
    cup_holder_init_y_offset: float = 0.2
    # Holder reset randomization (offsets from its seated init pose). "Some
    # random initialization": modest x/y jitter, no yaw by default.
    cup_holder_reset_x_range: tuple[float, float] = (0.05, 0.15)
    cup_holder_reset_y_range: tuple[float, float] = (-0.1, 0.15)
    cup_holder_reset_yaw_range: tuple[float, float] = (-0.5, 0.5)

    # ---- Cup placement relative to the holder (tunable) ------------------
    # ``cup_offset_from_holder`` is in the holder's local frame; its z lifts the
    # cup up into the holder's seat. ``cup_rot_on_holder`` is the cup's absolute
    # (world-frame) orientation when seated — upright by default. Both are
    # applied every reset by ``mdp.sync_object``.
    # TODO: tune both after previewing the spawned holder + cup.
    cup_offset_from_holder: tuple[float, float, float] = (0.16, -0.01, 0.255)
    cup_rot_on_holder: tuple[float, float, float, float] = CUP_INIT_ROT

    # ---- Goal: place the cup upright on the tabletop ---------------------
    # Goal xy are offsets from the table centre; defaults bias toward the front,
    # away from the holder (seated at +y). Goal z is set in ``__post_init__`` so
    # the cup rests on the table (table_top + object_half_height + clearance).
    target_x_range: tuple[float, float] = (0.05, 0.15)
    target_y_range: tuple[float, float] = (-0.4, -0.25)
    place_clearance: float = 0.0

    # ---- Success: at goal position AND upright ---------------------------
    success_position_threshold: float = 0.04
    success_max_tilt_rad: float = 0.2617993878  # 15 degrees

    def __post_init__(self):
        super().__post_init__()

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        table_pos = self.scene.table.init_state.pos

        # --- Cup holder: seat on the table, then randomize each reset. -----
        holder_z = table_top_z + self.cup_holder_half_height
        holder_pos = (
            table_pos[0] + self.cup_holder_init_x_offset,
            table_pos[1] + self.cup_holder_init_y_offset,
            holder_z,
        )
        self.scene.cup_holder = _build_cup_holder_cfg(
            scale=self.cup_holder_scale,
            init_rot=self.cup_holder_init_rot,
            init_pos=holder_pos,
        )
        # Kinematic body -> use the kinematic-aware reset (skips velocity write).
        self.events.reset_cup_holder = EventTerm(
            func=mdp.reset_root_pose_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cup_holder"),
                "pose_range": {
                    "x": list(self.cup_holder_reset_x_range),
                    "y": list(self.cup_holder_reset_y_range),
                    "z": [0.0, 0.0],
                    "roll": [0.0, 0.0],
                    "pitch": [0.0, 0.0],
                    "yaw": list(self.cup_holder_reset_yaw_range),
                },
            },
        )

        # --- Cup seated on the holder (cosmetic init + authoritative reset).
        ox, oy, oz = self.cup_offset_from_holder
        self.scene.object.init_state.pos = (
            holder_pos[0] + ox,
            holder_pos[1] + oy,
            holder_pos[2] + oz,
        )
        self.scene.object.init_state.rot = self.cup_rot_on_holder
        # The cup is rigidly seated on the holder, so it must follow the
        # holder's re-randomized yaw — not just in position but in orientation.
        # ``cup_rot_on_holder`` is the cup's *world* orientation when the holder
        # sits at its init rotation; express it in the holder's local frame
        # (``inv(holder_init) * cup_world``) so ``sync_object`` recomposes it
        # with the holder's post-reset rotation (``R(holder) * quat_local``).
        holder_init_R = R.from_quat(self.cup_holder_init_rot, scalar_first=True)
        cup_world_R = R.from_quat(self.cup_rot_on_holder, scalar_first=True)
        cup_rot_local_on_holder = tuple(
            float(v) for v in (holder_init_R.inv() * cup_world_R).as_quat(scalar_first=True)
        )
        # Re-seat the cup relative to the (post-randomization) holder. Added
        # after reset_cup_holder so it reads the holder's new pose; the
        # local-frame offset and ``quat_local`` make both position and
        # orientation track the holder's randomized yaw.
        self.events.reset_cup_on_holder = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("object"),
                "source_cfg": SceneEntityCfg("cup_holder"),
                "source_local_offset": (ox, oy, oz),
                "z_offset": 0.0,
                "quat_local": cup_rot_local_on_holder,
            },
        )

        # --- Goal: place upright on the tabletop. -------------------------
        # Override the goal z (base derived it from target_height_range, which
        # is meant for lift goals) so the goal sits the cup *on* the table.
        place_z = table_top_z + self.object_half_height + self.place_clearance
        self.commands.object_pose.ranges.pos_z = (place_z, place_z)

        # Keep the goal command active but do not draw debug markers in renders.
        self.commands.object_pose.debug_vis = False

        # ``debug_vis`` also draws the command's *current-pose* visualizer at the
        # cup every step. Its default (``ALIGN_MARKER_CFG``) renders a
        # coordinate-frame triad (``frame_prim.usd``), which clutters the scene.
        # The debug callback always draws marker index 0, so swap that marker for
        # a fully transparent sphere: no triad is drawn, while the goal sphere
        # (a separate visualizer) is untouched.
        self.commands.object_pose.curr_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/body_pose",
            markers={
                "hidden": sim_utils.SphereCfg(
                    radius=1e-4,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.0),
                )
            },
        )

        # --- Success: at goal position AND upright (yaw-agnostic). --------
        # Replaces the base lift-to-goal success (position-only / zone-gated).
        self.terminations.success = DoneTerm(
            func=mdp.object_upright_at_goal,
            params={
                "command_name": "object_pose",
                "position_threshold": self.success_position_threshold,
                "max_tilt_rad": self.success_max_tilt_rad,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
