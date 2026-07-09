# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: unfold a Huawei Mate X5 foldable phone.

The Mate X5 USD exposes two symmetric RevoluteJoint hinges (left/right)
that fold the two screen halves around the center body. The task is
symmetric with :mod:`open_laptop_cfg` — pin one half while levering the
other half open — but the phone is small enough that we scale it and rest
it on padding blocks for finger clearance.

This config mirrors the real-world treatment used in
:mod:`slide_utility_knife_cfg`:

* The phone and its support stands reset as one rigid rig, so the stands
  stay cradling the phone at any sampled pose instead of sliding out from
  under it.
* The hinges hold whatever angle they're left at (a detent), via a
  zero-stiffness *implicit* actuator plus PhysX joint friction. With an
  implicit actuator the friction/damping go straight into the PhysX joint
  drive, so the holding torque is actually applied -- unlike an explicit
  ``IdealPDActuator``, whose effort PhysX clips to the USD joint's authored
  ``maxForce`` (0 here), which is why the folded halves still fell. A folded
  half therefore stays folded under gravity until a hand actually pries it.
* Success requires the phone to be lifted off its supports *and* both
  hinges opened, so the policy can't "open" the phone while it's still
  pinned to the table / stands.
"""

import math

from dexverse.assets import SYNTHESIS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation as R

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

HUAWEI_PHONE_USD_PATH = str(SYNTHESIS_DIR / "huawei mate x5001" / "model_Huawei Mate01 X5.usd")


@configclass
class OpenHuaweiPhoneEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Unfold a Mate X5 phone while the body is free on the table."""

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = HUAWEI_PHONE_USD_PATH
    # 1.2x brings the folded phone to roughly palm-size.
    articulation_scale: tuple = (1.2, 1.2, 1.2)
    articulation_init_pos: tuple = (0.0, 0.05, 0.0)
    # Implicit actuator (set in __post_init__) owns the hinge drive; keep the
    # base joint-drive damping off so it doesn't double up.
    articulation_joint_damping: float = 0.0
    articulation_init_rot: tuple = tuple(
        (R.from_euler("x", math.radians(90)) * R.from_euler("z", math.radians(-90))).as_quat(scalar_first=True).tolist()
    )
    # Both hinges start folded (the two screen halves closed around the body).
    articulation_init_joint_pos: dict[str, float] = {
        "RevoluteJoint_Huawei_Mate01_X5_left": math.radians(75.0),
        "RevoluteJoint_Huawei_Mate01_X5_right": math.radians(-90.0),
    }
    articulation_half_height_est: float = 0.01

    # Elevated contact friction so the phone body grips the hands / stands
    # instead of squirting out while a hand pins one half and the other is
    # pried open. The base disables friction randomization, so __post_init__
    # re-attaches a physics material that applies these ranges. Collapsed
    # (low == high) bounds make it a deterministic coefficient on every
    # collision shape.
    articulation_static_friction_range: tuple[float, float] = (2.0, 2.0)
    articulation_dynamic_friction_range: tuple[float, float] = (2.0, 2.0)

    # Detent / "stays where you fold it" hold for the two hinges.
    #
    # A real foldable phone holds whatever angle you leave it at. We model
    # this with the implicit PD actuator below at *zero stiffness* (no
    # restoring force pulling a half back to a target) plus PhysX joint
    # friction, which supplies the Coulomb "stays put" force that resists the
    # gravity torque of an unsupported screen half.
    #
    # NOTE: PhysX joint *static* friction is load-proportional -- the holding
    # torque scales with the spatial force transmitted through the joint, so a
    # single coefficient holds roughly independent of the half's mass. Raise
    # these if a folded half droops/creeps when untouched; lower them if the
    # robot can't pry the half open. ``dynamic_friction`` only takes effect on
    # Isaac Sim >= 5.0; ``hinge_damping`` is the version-agnostic viscous term
    # (applied through the implicit actuator).
    articulation_hinge_static_friction: float = 0.4
    articulation_hinge_dynamic_friction: float = 0.3
    articulation_hinge_damping: float = 2.0

    # --- Per-task wrist drive override (scoped to this phone task) -----------
    # Same treatment as slide_utility_knife_cfg: the phone rests on kinematic
    # (infinite-mass) support stands and is gripped firmly with high contact
    # friction, so the floating hand's default wrist drive (effort_limit 15 N,
    # stiffness 2000, damping 400) stalls under the grasp+contact load -- the
    # hand "locks" once it grabs the phone and can't pin one half / pry the
    # other. Stiffen the wrist and raise its force ceiling so it can drag / lift
    # the loaded grip. Only the wrist DOFs (the ``*_translation_joint`` /
    # ``*_rotation_joint`` prismatic+revolute chain) are touched; the fingers
    # stay at their defaults. Applied in __post_init__ to ``self.scene.robot``,
    # a deep copy made by ``ArticulationCfg.replace()`` in the robot-setup
    # builder, so this does NOT mutate the shared FLOATING_SHADOW_*_CFG globals
    # used by other tasks. Set any value to None to leave that field at the
    # robot default.
    wrist_effort_limit: float | None = 100.0
    wrist_stiffness: float | None = 4000.0
    # Bumped alongside stiffness so the drive stays roughly critically damped
    # (critical damping ~ sqrt(stiffness)). Lower if the wrist feels sluggish;
    # raise if it oscillates / overshoots.
    wrist_damping: float | None = 600.0

    # Per-episode pose variation. Applied to the phone AND its support stands
    # as one shared rigid transform (see __post_init__) so the stands keep
    # cradling the phone. Keep roll/pitch/z at 0: the stands are kinematic
    # cuboids meant to sit flush on the table, and a shared roll/pitch would
    # tilt them off the surface. Widen x/y (metres) and yaw (radians) for more
    # spatial robustness.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.1, 0.0],
        "y": [-0.15, 0.15],
        # Yaw the phone about world +z, just like slide_utility_knife_cfg. The
        # reset uses ``reset_articulation_with_supports_uniform`` (wired in
        # __post_init__), which co-rotates the support stands about the phone's
        # pivot, so the stands keep cradling the phone at any sampled yaw.
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.3, 0.3],
    }
    articulation_use_padding_blocks: bool = True
    # (dx, dy) of each stand centre from the phone spawn xy. One stand under
    # each screen half so the folded phone lies flat across both.
    articulation_padding_block_offsets: tuple[tuple[float, float], ...] = (
        (-0.08, -0.05),
        (0.08, -0.050),
    )
    articulation_padding_block_size: tuple[float, float, float] = (0.03, 0.1, 0.08)
    success_joint_names: list[str] = [
        "RevoluteJoint_Huawei_Mate01_X5_left",
        "RevoluteJoint_Huawei_Mate01_X5_right",
    ]
    success_threshold: float = 0.5
    # The phone must also be lifted at least this far (m) above its spawn
    # height (i.e. off the support stands) for success -- this prevents the
    # policy from unfolding the hinges while the phone is still pinned to the
    # table / stands.
    success_lift_min_height: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        # Stiffen / raise the wrist force ceiling so the hand can move while
        # gripping the phone (see the wrist_* attributes above). Done first so
        # it runs regardless of the active robot_type's actuator layout.
        self._boost_wrist_drive()
        # Re-enable a PhysX material on the phone's collision shapes (the base
        # nulls articulation friction randomization) so the elevated contact
        # friction ranges above actually take effect on table / hand contacts.
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
        # Implicit actuator owns the hinge drive. With an implicit actuator the
        # stiffness/damping/friction go straight into the PhysX joint drive, so
        # the Coulomb friction is actually applied -- unlike the explicit
        # IdealPDActuator, whose effort is clipped to the USD joint's authored
        # maxForce (0 here), which is why the folded halves still fell. Zero
        # stiffness => no restoring force toward any target, so a half never
        # springs back to folded (or open); the detent "hold" is the joint
        # friction below.
        self.scene.articulation.actuators = {
            "phone_hinge_stabilizer": ImplicitActuatorCfg(
                joint_names_expr=self.success_joint_names,
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=self.articulation_hinge_damping,
                friction=self.articulation_hinge_static_friction,
                dynamic_friction=self.articulation_hinge_dynamic_friction,
                # The phone USD authors no mass/inertia on the screen halves
                # (mass=None on every body), so each hinge auto-computes a ~zero
                # rotational inertia. PhysX joint friction is load-proportional and
                # its solve is ill-conditioned at ~zero inertia, so the detent
                # can't hold a half that hangs mid-range under gravity (e.g. the
                # left hinge at its +75 deg init) -- it creeps. (The right hinge
                # only *looks* held because its -90 deg init sits exactly on its
                # hard lower limit, so the limit constraint holds it, not friction.)
                # Armature adds inertia straight into the joint-space mass matrix,
                # conditioning the friction solve so the detent actually bites.
                # Same fix as squeeze_scissors / tong_lift_pastry / open_glasses.
                # If a mid-range half still creeps after this, raise
                # articulation_hinge_static_friction / _dynamic_friction.
                armature=0.005,
            ),
        }
        # The hinges start folded via the base ``reset_articulation_joints``
        # (``reset_joints_to_init`` seats every joint at its init angle). With
        # stiffness=0 the drive target is irrelevant, so no per-step "hold at
        # current" target event and no reset/startup target-writing events are
        # needed -- the joint friction holds the folded halves from frame 0.
        # Randomize the phone and its stands together with a single shared
        # transform so the stands keep cradling the phone at any sampled
        # position / yaw. The base reset would move only the phone, dropping it
        # off the (static) stands.
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
        # Success requires BOTH: every hinge unfolded past the progress
        # threshold AND the phone lifted off its supports (so it can't be
        # unfolded while pinned to the table / stands).
        self.terminations.success.func = mdp.joint_moved_and_object_lifted
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "min_height": self.success_lift_min_height,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "all",
        }

    def _boost_wrist_drive(self) -> None:
        """Override the floating-hand wrist drive for this task only.

        Writes ``wrist_effort_limit`` / ``wrist_stiffness`` / ``wrist_damping``
        onto every wrist DOF (the ``*_translation_joint`` / ``*_rotation_joint``
        prismatic+revolute chain) of ``self.scene.robot``. Works for both the
        single-hand (``(x|y|z)_..._joint``) and bimanual
        (``(lh|rh)_(x|y|z)_..._joint``) key forms; finger joints are left alone.

        Only per-joint dict fields are edited, so a scalar field that would also
        cover the fingers is skipped. ``self.scene.robot`` is a deep copy from
        ``ArticulationCfg.replace()``, so this stays scoped to this task.
        """
        overrides = {
            "effort_limit_sim": self.wrist_effort_limit,
            "stiffness": self.wrist_stiffness,
            "damping": self.wrist_damping,
        }
        if all(value is None for value in overrides.values()):
            return

        actuators = getattr(getattr(self.scene, "robot", None), "actuators", None) or {}
        for actuator in actuators.values():
            for attr_name, new_value in overrides.items():
                if new_value is None:
                    continue
                limits = getattr(actuator, attr_name, None)
                if not isinstance(limits, dict):
                    # Scalar field would also cover the fingers; skip so we only
                    # ever touch the wrist DOFs via their explicit dict entries.
                    continue
                for joint_expr in limits:
                    if "translation_joint" in joint_expr or "rotation_joint" in joint_expr:
                        limits[joint_expr] = float(new_value)
