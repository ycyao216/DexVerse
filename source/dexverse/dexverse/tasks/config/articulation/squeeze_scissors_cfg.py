# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: open / close scissors (synthesis/scissors001)."""

from dexverse.assets import SYNTHESIS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

SCISSORS_USD_PATH = str(SYNTHESIS_DIR / "scissors010" / "scissors_backup.usd")


@configclass
class SqueezeScissorsEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Open / close scissor blades while the body is free on the table.

    The scissors have two blades hinged about a central pivot
    (``E_scissors_l_2`` and ``E_scissors_r_3``). Success fires when the
    larger of the two hinge angles crosses ``success_threshold``. A typical
    bimanual solution: one hand pins one handle, the other squeezes / pulls
    on the opposite handle.

    Scissors are small and thin, so we scale them up 1.5x and rest them
    on a pair of padding blocks for finger clearance.
    """

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = SCISSORS_USD_PATH
    # 1.5x to compensate for small authored scale + give finger-width.
    articulation_scale: tuple = (1.2, 1.2, 1.2)
    articulation_init_pos: tuple = (0.0, 0.05, 0.0)
    # Rotated +90 deg CCW about world z from the prior R_z(-90) so the
    # scissors' long axis (asset +y) lies along world y, aligned with the
    # padding stands. The +90 cancels the -90, netting out to identity.
    articulation_init_rot: tuple = (1.0, 0.0, 0.0, 0.0)
    articulation_init_joint_pos: dict[str, float] = {
        "RevoluteJoint_scissors_backup_left": 0.0,
        "RevoluteJoint_scissors_backup_right": 0.0,
    }
    # Implicit actuator (set in __post_init__) owns the hinge drive; keep the
    # base implicit joint-drive damping off so it doesn't double up.
    articulation_joint_damping: float = 0.0
    # Scissors are thin; small half-height seats them on the table.
    articulation_half_height_est: float = 0.012

    # Detent / "stays where you leave it" hold for the two scissor hinges --
    # the same friction-based model as the phone / knife / laptop. The implicit
    # PD actuator below runs at *zero stiffness* (no restoring spring, so a blade
    # never snaps back to closed) and PhysX joint friction supplies the Coulomb
    # "stays put" force. The hinge axis is vertical (the scissors lie flat), so
    # gravity does not load the joint -- friction only resists incidental
    # contact, so a gentle coefficient holds. Raise it if a blade drifts under
    # contact; lower it if the robot can't swing it open. ``dynamic_friction``
    # only takes effect on Isaac Sim >= 5.0; ``hinge_damping`` is the
    # version-agnostic viscous term.
    articulation_hinge_static_friction: float = 0.8
    articulation_hinge_dynamic_friction: float = 0.6
    articulation_hinge_damping: float = 1.0

    # Yaw the scissors about world +z, just like slide_utility_knife_cfg. The
    # reset uses ``reset_articulation_with_supports_uniform`` (wired in
    # __post_init__), which co-rotates the support stands about the scissors'
    # pivot, so the stands keep cradling them at any sampled yaw.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [0.0, 0.0],
        "y": [0.0, 0.0],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.3, 0.3],
    }

    # Padding blocks: scaled scissors are ~25–30 cm long after yaw,
    # oriented along world y. Two blocks spaced ~16 cm apart along y
    # support them near the handle and the blade tip.
    articulation_use_padding_blocks: bool = True
    articulation_padding_block_size: tuple[float, float, float] = (0.06, 0.02, 0.06)
    articulation_padding_block_offsets: tuple[tuple[float, float], ...] = (
        (0.0, -0.09),
        (0.0, 0.09),
    )

    success_joint_names: list[str] = [
        "RevoluteJoint_scissors_backup_left",
        "RevoluteJoint_scissors_backup_right",
    ]
    # ~40 deg of a 50 deg blade range (progress fraction). 1.0 would require
    # driving a blade all the way to its hard stop.
    success_threshold: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        # Friction-detent hinge drive (matches phone / knife / laptop): an
        # implicit actuator at zero stiffness + PhysX Coulomb joint friction.
        # Implicit (not IdealPDActuator) so the friction reaches the PhysX joint
        # and the holding torque is actually applied -- an explicit actuator's
        # effort is clipped to the USD joint's authored maxForce. Zero stiffness
        # => no restoring spring, so the blades stay wherever they are left.
        self.scene.articulation.actuators = {
            "scissors_hinge_stabilizer": ImplicitActuatorCfg(
                joint_names_expr=self.success_joint_names,
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=self.articulation_hinge_damping,
                friction=self.articulation_hinge_static_friction,
                dynamic_friction=self.articulation_hinge_dynamic_friction,
                armature=0.005,
            ),
        }
        # The blades start closed via the base ``reset_articulation_joints``
        # (``reset_joints_to_init`` seats both hinges at their 0 deg init). With
        # stiffness 0 the drive target is irrelevant, so no target-writing
        # reset/startup events are needed -- the joint friction holds the blades
        # wherever they are left.

        # Reset the scissors AND their padding stands as one rigid rig (a single
        # shared transform), like slide_utility_knife_cfg / open_huawei_phone_cfg,
        # so the stands stay cradling the scissors at any sampled yaw. The base
        # reset moves only the articulation, which would yaw the scissors off the
        # (static) stands.
        if self.articulation_use_padding_blocks:
            support_cfgs = [
                SceneEntityCfg(f"padding_block_{i}") for i in range(len(self.articulation_padding_block_offsets))
            ]
            self.events.reset_articulation.func = mdp.reset_articulation_with_supports_uniform
            self.events.reset_articulation.params = {
                "asset_cfg": SceneEntityCfg("articulation"),
                "support_cfgs": support_cfgs,
                "pose_range": self.articulation_reset_pose_range,
            }

        self.terminations.success.func = mdp.joint_relative_move
        # ~40 deg over a 50 deg blade range.
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
        }
