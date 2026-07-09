# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: open a laptop lid (synthesis/laptop001).

The laptop is a *free* two-link articulation: a base body (``E_body_1``)
sitting loose on the table and a lid (``E_displayer_5``) joined to it by a
single revolute hinge (``RevoluteJoint_computer_9_up``). Nothing is bolted to
the world, so pushing the laptop slides it and lifting the lid while the base
is unpinned tips the whole laptop -- the policy pins the base with one hand
while the other pries the lid open.

Hinge convention (from the USD): the joint range is ``[-110 deg, 0 deg]`` where
``0 deg`` is fully *closed* (lid flat on the base) and negative angles *open*
the lid (toward ``-110 deg``). The lid starts ajar at a negative angle.

Hinge model -- *friction detent* (the mechanism validated in the debug env).
The lid is held by an implicit actuator at **zero stiffness** (no position
servo, so it never springs back to a setpoint) plus PhysX Coulomb **joint
friction** -- a static, load-proportional holding torque that resists gravity
at whatever angle the lid sits. The lid therefore stays wherever it is left,
and the robot must overcome the static friction to pry it open; a little
``damping`` keeps it from whipping. (An earlier position-drive "hold at the
current angle" scheme could not statically balance gravity and let the lid
creep shut.)
"""

import math

import isaaclab.sim as sim_utils
from dexverse.assets import SYNTHESIS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

LAPTOP_USD_PATH = str(SYNTHESIS_DIR / "laptop001" / "model_computer_9.usd")


@configclass
class OpenLaptopEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Open a laptop lid while the laptop is free to slide / topple."""

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = LAPTOP_USD_PATH
    articulation_scale: tuple = (1.0, 1.0, 1.0)
    articulation_init_pos: tuple = (0.0, 0.0, 0.0)
    # Clockwise 90 deg yaw around +z (viewed from +z towards -z).
    articulation_init_rot: tuple = (math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5))
    # The implicit hinge actuator (set in __post_init__) owns the drive; keep
    # the base implicit joint damping off so it doesn't double up.
    articulation_joint_damping: float = 0.0
    # Lid starts ajar. Negative = open (range [-110 deg, 0 deg], 0 deg ==
    # closed). -45 deg is the angle the friction detent was tuned at; a smaller
    # crack (nearer 0) loads the hinge harder under gravity and may need more
    # static friction to hold.
    articulation_init_joint_pos: dict[str, float] = {
        "RevoluteJoint_computer_9_up": math.radians(-15.0),
    }
    # The asset's origin is at the *bottom* of the closed laptop, so only a
    # hair of clearance is needed to seat it flush on the table.
    articulation_half_height_est: float = 0.005
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.3, 0.0],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.2, 0.2],
    }

    # Moderate, deterministic contact friction: high enough for a hand to pin
    # the base, low enough that a push still slides the laptop. The base nulls
    # friction randomization, so __post_init__ re-attaches a physics material.
    articulation_static_friction_range: tuple[float, float] = (0.8, 0.8)
    articulation_dynamic_friction_range: tuple[float, float] = (0.6, 0.6)

    # Friction-detent hinge (validated in debug_laptop_cfg).
    #
    # stiffness = 0  -> no position servo, so the lid never springs back to a
    #                   setpoint; it stays wherever it currently is.
    # friction       -> PhysX Coulomb joint friction: a static, load-
    #                   proportional holding torque that resists gravity at any
    #                   angle. Raise it if the lid droops untouched (e.g. when
    #                   started nearer closed); lower it if the robot can't pry
    #                   the lid open.
    # damping        -> a little viscous resistance so the lid doesn't whip.
    # ``hinge_dynamic_friction`` only takes effect on Isaac Sim >= 5.0.
    articulation_hinge_stiffness: float = 0.0
    articulation_hinge_damping: float = 1.0
    articulation_hinge_static_friction: float = 1.0
    articulation_hinge_dynamic_friction: float = 1.0

    success_joint_names: list[str] = ["RevoluteJoint_computer_9_up"]
    success_threshold: float = 0.70

    def __post_init__(self):
        super().__post_init__()

        # Free, movable rigid laptop. ``fix_root_link`` is already False
        # (inherited) and the hinge connects two real bodies, so the base body
        # floats: pushing slides it, lifting the lid tips it. Gravity on.
        self.scene.articulation.spawn.rigid_props.disable_gravity = False
        self.scene.articulation.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.4)

        # Re-enable a (deterministic) contact physics material so the friction
        # ranges above take effect on table / hand contacts (the base nulls
        # articulation friction randomization). This is *contact* friction --
        # separate from the *joint* friction detent below.
        self.events.articulation_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("articulation"),
                "static_friction_range": self.articulation_static_friction_range,
                "dynamic_friction_range": self.articulation_dynamic_friction_range,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )

        # Friction-detent hinge: implicit actuator at zero stiffness + Coulomb
        # joint friction. Implicit (not IdealPDActuator) so the friction goes
        # straight into the PhysX joint and the holding torque is actually
        # applied -- an explicit actuator's effort is clipped to the USD joint's
        # authored maxForce (~0 here), which is why a position drive let the lid
        # fall. With stiffness 0 there is no setpoint, so the lid holds wherever
        # it is left and never springs back.
        self.scene.articulation.actuators = {
            "laptop_hinge": ImplicitActuatorCfg(
                joint_names_expr=self.success_joint_names,
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=self.articulation_hinge_stiffness,
                damping=self.articulation_hinge_damping,
                friction=self.articulation_hinge_static_friction,
                dynamic_friction=self.articulation_hinge_dynamic_friction,
            ),
        }
        # The lid starts ajar via the base ``reset_articulation_joints``
        # (``reset_joints_to_init`` seats the hinge at its init angle). With
        # stiffness 0 the drive target is irrelevant, so no target-writing
        # reset/startup events are needed -- the joint friction holds the lid
        # wherever it is left.

        self.terminations.success.func = mdp.joint_relative_move
        # Open the lid to ~70% of the remaining hinge travel from the init.
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
        }
