# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: extend a utility-knife blade (synthesis/utility knife002)."""

import isaaclab.sim as sim_utils
from dexverse.assets import SYNTHESIS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

UTILITY_KNIFE_USD_PATH = str(SYNTHESIS_DIR / "utility knife002" / "model_utility_knife_3.usd")


@configclass
class SlideUtilityKnifeEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Extend a retractable utility-knife blade along its slider.

    The current USD exposes one actuable joint: the prismatic blade slider
    (``PrismaticJoint_utility_knife_3_middle``). Keep success scoped to this
    joint explicitly for stable reward shaping.

    Small, thin asset: scaled 1.5x and rested on a pair of padding blocks
    so fingers can slip underneath.
    """

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = UTILITY_KNIFE_USD_PATH
    # 1.2x for easier grasping; the authored knife is ~16 cm long so
    # this lands around 19 cm.
    articulation_scale: tuple = (1.3, 1.3, 1.3)
    articulation_mass_kg: float = 0.05
    articulation_init_pos: tuple = (0.0, 0.0, 0.0)
    # Net 0 deg yaw: the previous -90 deg plus a +90 deg CCW rotation. The
    # knife's native long axis is world-y, so it now lies along y.
    articulation_init_rot: tuple = (1.0, 0.0, 0.0, 0.0)
    # Implicit actuator owns this prismatic joint's drive; keep the base
    # joint-drive damping off so it doesn't double up.
    articulation_joint_damping: float = 0.0
    # Scaled with the mesh (~0.022 * 1.2) so the knife still spawns flush on
    # the support blocks instead of clipping into them.
    articulation_half_height_est: float = 0.022

    # Elevated contact friction so the knife body grips the hands / padding
    # blocks instead of sliding around while the blade is driven out. The
    # base disables friction randomization, so __post_init__ re-attaches a
    # physics material that applies these ranges. Collapsed (low == high)
    # bounds make it a deterministic coefficient on every collision shape.
    articulation_static_friction_range: tuple[float, float] = (2.0, 2.0)
    articulation_dynamic_friction_range: tuple[float, float] = (2.0, 2.0)

    # Robot-hand contact friction. The robot here is a floating Shadow hand, so
    # this is the grip friction of the fingers / palm. The base nulls robot
    # physics-material randomization, so __post_init__ re-attaches a
    # deterministic (low == high) material with these ranges. Raise to grip the
    # knife more firmly; lower if the hand sticks. Set to None to leave the
    # robot at its default (PhysX ~0.5) material.
    robot_static_friction_range: tuple[float, float] | None = (1.5, 1.5)
    robot_dynamic_friction_range: tuple[float, float] | None = (1.5, 1.5)

    # --- Per-task wrist drive override (scoped to this knife task) -----------
    # The knife rests on kinematic (infinite-mass) support stands and is gripped
    # firmly with high contact friction, so the floating hand's default wrist
    # drive (effort_limit 15 N, stiffness 2000, damping 400) stalls under the
    # grasp+contact load -- the hand "locks" once it grabs the knife. Stiffen the
    # wrist and raise its force ceiling so it can drag / lift the loaded grip.
    #
    # Only the wrist DOFs (the ``*_translation_joint`` / ``*_rotation_joint``
    # prismatic+revolute chain) are touched; the fingers stay at their defaults.
    # Applied in __post_init__ to ``self.scene.robot``, which is a deep copy made
    # by ``ArticulationCfg.replace()`` in the robot-setup builder, so this does
    # NOT mutate the shared FLOATING_SHADOW_*_CFG globals used by other tasks.
    # Set any value to None to leave that field at the robot default.
    wrist_effort_limit: float | None = 60.0
    wrist_stiffness: float | None = 4000.0
    # Bumped alongside stiffness so the drive stays roughly critically damped
    # (critical damping ~ sqrt(stiffness)). Lower if the wrist feels sluggish;
    # raise if it oscillates / overshoots.
    wrist_damping: float | None = 600.0

    # Detent / "stays where you push it" hold for the blade slider.
    #
    # A real utility knife holds the blade at whatever notch you slide it to.
    # We model this with the implicit PD actuator below at *zero stiffness*
    # (so there is no restoring force pulling the blade back to a target) plus
    # PhysX joint friction, which supplies the Coulomb "stays put" force.
    #
    # NOTE: PhysX joint *static* friction is load-proportional -- the holding
    # force scales with the spatial force transmitted through the joint, not a
    # fixed value. On this light, horizontal blade that transmitted force is
    # small, so the hold is gentle. Raise these if the blade drifts under
    # contact; lower them if the robot can't slide it out. ``dynamic_friction``
    # only takes effect on Isaac Sim >= 5.0; ``blade_damping`` is the
    # version-agnostic viscous term (applied through the implicit actuator).
    articulation_blade_static_friction: float = 1.0
    articulation_blade_dynamic_friction: float = 0.8
    articulation_blade_damping: float = 1.2

    # Per-episode pose variation. Applied to the knife AND its support stands as
    # one shared rigid transform (see __post_init__) so the knife always lands
    # cradled on the stands. Keep roll/pitch/z at 0 to stay flat on the table;
    # widen x/y (metres) and yaw (radians) for more spatial robustness.
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [-0.05, 0.05],
        "y": [-0.15, 0.05],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.3, 0.3],
    }

    # The knife's long axis lies along world-y after the init rotation, so the
    # two stands are separated along y -- one under each end -- and centered on
    # the knife's spawn xy, so the knife lies flat across both.
    articulation_use_padding_blocks: bool = True
    # Taller stand (height 0.04 vs the 0.015 base default) so there is more
    # clearance under the knife for fingers to slip in and grasp. The block
    # height also sets how far the knife is lifted off the table. Footprint
    # (0.04 x 0.04) kept from the base.
    articulation_padding_block_size: tuple[float, float, float] = (0.06, 0.02, 0.06)
    # (dx, dy) from the knife spawn xy. Separated ~14 cm along y (the long
    # axis) so each stand sits near an end of the ~19 cm knife; same x so both
    # stay centered under it.
    articulation_padding_block_offsets: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.2),
    )

    success_joint_names: list[str] = ["PrismaticJoint_utility_knife_3_middle"]
    success_threshold: float = 0.4
    # The knife must also be lifted at least this far (m) above its spawn height
    # (i.e. off the support stands) for success -- this prevents the policy from
    # driving the blade while the knife is still pinned to the table / stands.
    success_lift_min_height: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        # Stiffen / raise the wrist force ceiling so the hand can move while
        # gripping the knife (see the wrist_* attributes above). Done first so
        # it runs regardless of the active robot_type's actuator layout.
        self._boost_wrist_drive()
        self.scene.articulation.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=self.articulation_mass_kg)
        # Re-enable a PhysX material on the knife's collision shapes (the base
        # nulls articulation friction randomization) so the elevated friction
        # ranges above actually take effect on the table / hand contacts.
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
        # Set the robot hand's contact friction (the base nulls
        # ``robot_physics_material``). ``SceneEntityCfg("robot")`` covers every
        # collision shape of the floating hand(s); pass ``body_names=[...]`` to
        # scope it to just the fingertips. Skipped when the ranges are None.
        if self.robot_static_friction_range is not None:
            self.events.robot_physics_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "static_friction_range": self.robot_static_friction_range,
                    "dynamic_friction_range": self.robot_dynamic_friction_range,
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            )
        # Implicit actuator owns the blade-slider drive. With an implicit
        # actuator the stiffness/damping/friction go straight into the PhysX
        # joint drive, so the Coulomb friction is actually applied -- unlike the
        # explicit IdealPDActuator, whose effort is clipped to the USD joint's
        # authored maxForce (0 here), which is why the blade wouldn't hold. Zero
        # stiffness => no restoring force toward any target, so the blade never
        # springs back to the retracted start; the detent "hold" is the joint
        # friction below.
        self.scene.articulation.actuators = {
            "utility_knife_slider_stabilizer": ImplicitActuatorCfg(
                joint_names_expr=self.success_joint_names,
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=self.articulation_blade_damping,
                friction=self.articulation_blade_static_friction,
                dynamic_friction=self.articulation_blade_dynamic_friction,
            ),
        }
        # The blade starts retracted via the base ``reset_articulation_joints``
        # (``reset_joints_to_init`` seats every joint at its init position). With
        # stiffness=0 the drive target is irrelevant, so no reset/startup
        # target-writing events are needed -- the joint friction holds the blade
        # wherever it is left.
        # Randomize the knife and its stands together with a single shared
        # transform so the stands keep cradling the knife at any sampled
        # position / yaw. The base reset would move only the knife, dropping it
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
        # Success requires BOTH: the blade slid past the stroke threshold AND
        # the knife lifted off its supports (so it can't be driven while pinned
        # to the table / stands).
        self.terminations.success.func = mdp.joint_moved_and_object_lifted
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "min_height": self.success_lift_min_height,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
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
