# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: unfold the temples of a pair of glasses (synthesis/glasses003).

The glasses USD is a three-body articulation: a front frame (``body_8``,
which carries the ArticulationRootAPI) and two temple arms
(``leg1_7`` / ``leg2_6``) each joined to the frame by a revolute hinge
(``RevoluteJoint_glasses3_left`` / ``..._right``). Nothing is bolted to the
world, so the frame is free on the tabletop -- the policy has to pin the frame
with one hand while the other swings a temple open (mirrors
:mod:`open_huawei_phone_cfg`, which also has two symmetric hinges).

Hinge convention (from the USD): the left hinge range is ``[-90 deg, 0 deg]``
and the right hinge range is ``[0 deg, 90 deg]``, both about the frame's local
+Z axis; ``0 deg`` is fully *folded* (temples flat against the frame). The
glasses spawn cracked just open and the task is to unfold both temples.

Placement: the asset is authored "worn-upright" (lens plane vertical, hinge
axis along local +Z). Standing that on the table balances on a thin frame edge
and topples instantly, so we lay the glasses flat (lenses up) with a +90 deg
roll about X. After that roll the temples rest folded on top of the frame and
the hinge axis is horizontal, so unfolding swings each temple up about a
gravity-loaded hinge -- the same regime as the laptop lid.

Hinge model: like the laptop / phone / knife, an *implicit* PD actuator at zero
stiffness plus PhysX joint friction gives a "stays where you leave it" detent.
With an implicit actuator the friction/damping go straight into the PhysX joint
drive, so the holding torque is actually applied -- unlike an explicit
``IdealPDActuator``, whose effort PhysX clips to the USD joint's authored
``maxForce`` (0 here). Zero stiffness means no restoring spring, so a temple
never snaps back to folded; the friction holds it against gravity.
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

GLASSES_USD_PATH = str(SYNTHESIS_DIR / "glasses003" / "glasses3.usd")


@configclass
class OpenGlassesEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Unfold both temples of a free pair of glasses on the tabletop."""

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = GLASSES_USD_PATH
    # 1.2x brings the ~13 cm frame to ~16 cm so the thin temples are graspable.
    articulation_scale: tuple = (1.2, 1.2, 1.2)
    articulation_init_pos: tuple = (0.0, 0.0, 0.0)
    # +90 deg roll about X: lay the worn-upright asset flat (lenses up) so it
    # rests stably and the temples sit folded on top of the frame.
    articulation_init_rot: tuple = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    # Implicit actuator (set in __post_init__) owns the hinge drive; keep the
    # base joint-drive damping off so it doesn't double up.
    articulation_joint_damping: float = 0.0
    # Both temples start cracked just open (a few degrees off the folded 0 deg
    # limit to avoid sitting on the hard stop). Left hinge opens negative,
    # right hinge opens positive.
    articulation_init_joint_pos: dict[str, float] = {
        "RevoluteJoint_glasses3_left": math.radians(-17.0),
        "RevoluteJoint_glasses3_right": math.radians(3.0),
    }
    # After the +90 deg roll and 1.2x scale the frame's geometric bottom sits
    # ~6 mm below the root; a hair more clearance seats it flush on the table.
    articulation_half_height_est: float = 0.008

    # Contact friction so the light frame grips the hands / table instead of
    # squirting away while one hand pins it and the other pries a temple. The
    # base disables friction randomization, so __post_init__ re-attaches a
    # deterministic (low == high) physics material with these ranges.
    articulation_static_friction_range: tuple[float, float] = (1.2, 1.2)
    articulation_dynamic_friction_range: tuple[float, float] = (0.9, 0.9)

    # Detent / "stays where you leave it" hold for the two temple hinges.
    #
    # Modeled with the implicit PD actuator below at *zero stiffness* (no
    # restoring force toward any target) plus PhysX joint friction, which
    # supplies the Coulomb "stays put" force that resists the (small) gravity
    # torque of an unfolded temple. The hinge axis is horizontal after the lay-
    # flat roll, so gravity does load the joint. The glasses are very light
    # (40 g), so a low coefficient still holds a temple while keeping the
    # *static* (breakaway) friction small enough for the robot to swing it.
    # Raise the friction if a temple droops untouched; lower it if the robot
    # still can't move it. ``dynamic_friction`` only takes effect on Isaac Sim
    # >= 5.0; ``hinge_damping`` is the version-agnostic viscous term.
    articulation_hinge_static_friction: float = 0.25
    articulation_hinge_dynamic_friction: float = 0.15
    articulation_hinge_damping: float = 1.0

    # Per-episode pose variation. The glasses lie flat with no support stands,
    # so yaw can vary freely; keep roll/pitch/z at 0 to stay flat on the table.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.03, 0.03],
        "y": [-0.08, 0.08],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.3, 0.3],
    }

    success_joint_names: list[str] = [
        "RevoluteJoint_glasses3_left",
        "RevoluteJoint_glasses3_right",
    ]
    # Success when BOTH hinges have unfolded past this fraction of their
    # reachable range from the cracked-open start (~0.7 * 90 deg ~= 63 deg).
    success_threshold: float = 0.8

    def __post_init__(self):
        super().__post_init__()

        # Light pair of glasses.
        self.scene.articulation.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.04)

        # The authored glasses materials are a thin dark metal frame plus a
        # transparent ``Clear_Glass`` lens, so the glasses barely render. Bind
        # one opaque, bright material at the articulation root -- ``spawn_from_usd``
        # binds it ``stronger_than_descendants``, so it overrides the per-mesh
        # bindings and the whole pair shows up clearly in the camera / viewport.
        # Tune ``diffuse_color`` to taste.
        self.scene.articulation.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.15, 0.35, 0.75),
            roughness=0.4,
            metallic=0.0,
        )

        # Enable self-collision so the temples collide with the frame (and each
        # other) instead of passing through them as they fold/unfold. The base
        # spawn disables self-collision by default; all three bodies use convex
        # colliders, so inter-link collision resolves cleanly.
        self.scene.articulation.spawn.articulation_props.enabled_self_collisions = True

        # Re-enable a PhysX material on the glasses' collision shapes (the base
        # nulls articulation friction randomization) so the contact-friction
        # ranges above actually take effect on table / hand contacts.
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

        # Implicit actuator owns the temple-hinge drive. With an implicit
        # actuator the stiffness/damping/friction go straight into the PhysX
        # joint drive, so the Coulomb friction is actually applied -- unlike the
        # explicit IdealPDActuator, whose effort is clipped to the USD joint's
        # authored maxForce (0 here). Zero stiffness => no restoring force toward
        # any target, so a temple never springs back to folded; the detent
        # "hold" is the joint friction below.
        self.scene.articulation.actuators = {
            "glasses_hinge_stabilizer": ImplicitActuatorCfg(
                joint_names_expr=self.success_joint_names,
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=self.articulation_hinge_damping,
                friction=self.articulation_hinge_static_friction,
                dynamic_friction=self.articulation_hinge_dynamic_friction,
                # The glasses USD authors no mass/inertia on the temple legs
                # (mass=None on every body); the spawn mass_props=0.04 above only
                # sets the root frame, so each hinge auto-computes a ~zero
                # rotational inertia. That ill-conditions the hinge: the temples
                # overshoot their [-90,0]/[0,90] deg limits, jitter, and make the
                # (self-collision-on) temple-vs-frame contacts unstable. Armature
                # adds inertia straight into the joint-space mass matrix, steadying
                # the hinge + the friction detent. Same fix as squeeze_scissors /
                # tong_lift_pastry; raise toward 0.01 if temples still overshoot,
                # lower if a temple is too hard to swing.
                armature=0.005,
            ),
        }

        # The temples start cracked-open via the base ``reset_articulation_joints``
        # (``reset_joints_to_init`` seats every joint at its init angle). With
        # stiffness=0 the drive target is irrelevant, so no per-step "hold at
        # current" target event and no reset/startup target-writing events are
        # needed -- the joint friction holds the temples wherever they are left.

        # Success requires BOTH hinges unfolded past the progress threshold.
        self.terminations.success.func = mdp.joint_relative_move
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "all",
        }
