# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base for *fixate-then-manipulate* tabletop tasks.

Unlike :mod:`dexverse.tasks.config.articulation.articulation_base`, the
articulation here is *not* bolted to the world — its root link is a free
rigid body sitting on the tabletop. Actuating the target joint therefore
tends to tip or slide the whole object, so the robot has to pin the body
in place with one hand (or parts of one hand) while the other hand / free
fingers drive the joint past a threshold.

Concrete per-object configs (``open_laptop_cfg.py``, ``squeeze_scissors_cfg.py``,
etc.) subclass :class:`FixateArticulationEnvFloatingDexHandRightCfg` and set:

* ``articulation_usd_path`` — synthesis asset USD to load.
* ``articulation_scale`` / ``articulation_init_pos`` / ``articulation_init_rot``
  — placement on the table.
* ``articulation_half_height_est`` — height of the asset root above its
  bottom, used to seat the mesh flush on the tabletop.
* ``success_joint_names`` — regex or list of joint names used for reward /
  termination (defaults to ``".*"`` which matches every joint; override
  this when an asset has multiple joints of different semantics).
* ``success_threshold`` — scalar threshold in radians (revolute) or meters
  (prismatic); success fires when ``max(|joint_pos|)`` for
  ``success_joint_names`` crosses it.

The base adds a table-bounds ``out_of_bound`` termination so the episode
ends if the object falls / slides off the tabletop (the parent class
disables this term for fixed articulations).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import mdp
from ...dexverse_base_env_cfg import DEFAULT_TABLE_TOP_HEIGHT
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY

# All-zero reset pose range used by per-object configs that want the
# articulation to spawn at exactly the same pose every episode. Useful
# while we're focused on the manipulation itself rather than spatial
# robustness (e.g. objects resting on padding blocks would fall off if
# their xy is perturbed).
DETERMINISTIC_RESET_POSE_RANGE: dict[str, list[float]] = {
    "x": [0.0, 0.0],
    "y": [0.0, 0.0],
    "z": [0.0, 0.0],
    "roll": [0.0, 0.0],
    "pitch": [0.0, 0.0],
    "yaw": [0.0, 0.0],
}


@configclass
class FixateArticulationEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Base config for a free-root articulation that the robot must stabilise.

    Override ``articulation_*`` and ``success_*`` class attributes on a
    subclass (see the per-object configs in this package).

    The default ``robot_type`` (inherited from the parent) is
    ``floating_shadow_right``. Pass ``env.robot_type=floating_shadow_bimanual``
    (or ``bimanual_leap`` / ``xarm7_leap_bimanual``) on the CLI to switch to
    two hands — the scene, obs and rewards are all robot-agnostic and follow
    ``robot_config.fingertip_body_names`` automatically.
    """

    # Let the root link float freely. ``False`` explicitly disables any
    # authored fixed joint in the USD. If a particular synthesis asset fails
    # to spawn ("Failed to find a single articulation"), try ``None`` — that
    # keeps the USD's authored articulation root untouched, which some
    # synthesis USDs require because their ArticulationRootAPI is on a
    # non-rigid prim.
    articulation_fix_root_link: bool | None = False

    # Modest reset pose randomisation so the policy practises stabilising
    # the object from varying starts. Per-object configs may override.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.02, 0.02],
        "y": [-0.08, 0.08],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.2, 0.2],
    }

    # Recovery-from-topple + opening is slower than driving a fixed handle.
    episode_length_s_override: float = 8.0

    # ------------------------------------------------------------------
    # Damping / friction knobs
    # ------------------------------------------------------------------
    # ``articulation_base.make_articulation_cfg`` builds the asset with
    # ``linear_damping = angular_damping = None`` and ``joint_drive.damping = 0``,
    # and never attaches a physics material. That's fine while the root is
    # bolted to the world, but once we unfix it the body glides forever on
    # the table and the hinges/sliders whip around like they're greased.
    # The values below add just enough dissipation and contact friction to
    # feel physical without making the object feel sticky. Per-object
    # configs can override any of these (e.g. a waxy laptop lid may want
    # less table friction than a rubber-footed stapler).

    # Rigid body damping on the articulation root/links. Applied to
    # ``spawn.rigid_props`` so the physics engine dissipates translation
    # and rotation continuously. These are unitless PhysX damping
    # coefficients; 0.1 is "barely anything", 1.0 is "distinctly damped".
    articulation_linear_damping: float = 0.5
    articulation_angular_damping: float = 0.5

    # Viscous joint damping via the physics joint drive. With stiffness
    # still 0 this acts as pure velocity resistance (F = -D * joint_vel),
    # which keeps lids / blades from flopping open freely under gravity
    # while still leaving them fully passive — the robot is what actuates
    # the joint, not a servo.
    articulation_joint_damping: float = 2.0

    # Contact friction between the articulation's collision shapes and
    # anything else (table, robot hands). Ranges are sampled per env at
    # startup so trained policies see some variation.
    articulation_static_friction_range: tuple[float, float] = (0.8, 1.2)
    articulation_dynamic_friction_range: tuple[float, float] = (0.6, 1.0)

    # Coulomb-style joint friction (torque threshold that must be
    # overcome before the joint moves). Small, mostly to prevent the
    # joint from drifting when the robot just barely touches it.
    articulation_joint_friction_range: tuple[float, float] = (0.05, 0.15)

    # ------------------------------------------------------------------
    # Padding blocks
    # ------------------------------------------------------------------
    # For thin / small articulations (scissors, knife, phone) it helps to
    # prop the object up on two little support blocks so the robot's
    # fingers can slide underneath to pin or pry. When enabled, a pair of
    # static kinematic cuboids are added to the scene directly below the
    # articulation spawn; the articulation is seated on top of them
    # instead of flat on the table.
    articulation_use_padding_blocks: bool = False

    # (width, depth, height) of each block in metres. Height matters most
    # — it's how much finger clearance you get under the object.
    articulation_padding_block_size: tuple[float, float, float] = (0.04, 0.04, 0.015)

    # 2D (dx, dy) offsets of each block centre from the articulation's
    # ``articulation_init_pos`` xy in world frame. The default spacing
    # works for objects laid long along world-y (e.g. +90 yawed scissors,
    # knife); override per object when the long axis is different.
    articulation_padding_block_offsets: tuple[tuple[float, float], ...] = (
        (0.0, -0.03),
        (0.0, 0.03),
    )

    def __post_init__(self):
        super().__post_init__()

        # The parent strips ``object_out_of_bound`` because it assumes a
        # fixed articulation that cannot leave the table. Add it back,
        # clamped to the table footprint, so a toppled object ends the
        # episode instead of dragging the sim.
        table_size = self.scene.table.spawn.size
        self.terminations.articulation_out_of_bound = DoneTerm(
            func=mdp.out_of_bound,
            params={
                "in_bound_range": {
                    "x": (-table_size[0] * 0.5, table_size[0] * 0.5),
                    "y": (-table_size[1] * 0.5, table_size[1] * 0.5),
                    "z": (-0.05, 1.5),
                },
                "asset_cfg": SceneEntityCfg(ARTICULATION_KEY),
            },
        )

        self.episode_length_s = self.episode_length_s_override

        # --------------------------------------------------------------
        # Damping and friction.
        # --------------------------------------------------------------
        # These all piggy-back on the ``ArticulationCfg`` the parent
        # built in step (1) of ``ArticulationBaseEnvCfg.__post_init__``.
        spawn = self.scene.articulation.spawn

        # Rigid-body damping: dissipates translation / rotation so a
        # toppled or shoved object settles instead of gliding forever.
        spawn.rigid_props.linear_damping = self.articulation_linear_damping
        spawn.rigid_props.angular_damping = self.articulation_angular_damping

        # Viscous joint damping. Stiffness stays at 0 — we don't want
        # the joint to spring back to a target, we only want it to
        # resist velocity so free-swinging lids / blades don't whip.
        spawn.joint_drive_props.damping = self.articulation_joint_damping

        # Disable startup-time articulation friction/material randomization.
        self.events.articulation_physics_material = None
        self.events.articulation_joint_friction = None

        # --------------------------------------------------------------
        # Padding blocks
        # --------------------------------------------------------------
        # Spawned last because they need the final articulation xyz
        # (parent already seated it on the table in step 5), and because
        # we also need to bump the articulation up by the block height
        # so it rests *on* the blocks rather than clipping through.
        if self.articulation_use_padding_blocks:
            self._attach_padding_blocks()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _attach_padding_blocks(self) -> None:
        """Add static support cuboids under the articulation and re-seat it.

        Each entry in ``articulation_padding_block_offsets`` becomes one
        kinematic ``RigidObjectCfg`` whose top surface is flush with
        ``table_top + block_height``. The articulation is then shifted
        up by ``block_height`` so it lands on the blocks instead of the
        table.
        """
        block_w, block_d, block_h = self.articulation_padding_block_size
        block_z_center = DEFAULT_TABLE_TOP_HEIGHT + block_h * 0.5

        # Use the articulation's xy from what the parent already resolved
        # (which is the class-level ``articulation_init_pos`` xy).
        ax, ay, az = self.scene.articulation.init_state.pos

        for i, (dx, dy) in enumerate(self.articulation_padding_block_offsets):
            block_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/PaddingBlock_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(block_w, block_d, block_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.35, 0.35, 0.38),
                        roughness=0.8,
                    ),
                    visible=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + dx, ay + dy, block_z_center),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            setattr(self.scene, f"padding_block_{i}", block_cfg)

        # Seat the articulation on top of the blocks.
        self.scene.articulation.init_state.pos = (ax, ay, az + block_h)
