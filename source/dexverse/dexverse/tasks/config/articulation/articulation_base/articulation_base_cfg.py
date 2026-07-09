# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration classes for tabletop articulated-object tasks.

The success criterion is that the articulation's target joint(s) reach a
threshold position (angle for revolute joints, distance for prismatic joints).

A child task overrides the class attributes prefixed with ``articulation_`` and
``success_`` on a subclass of :class:`ArticulationBaseEnvFloatingDexHandRightCfg`,
then registers that class as a gym environment.
"""

from __future__ import annotations

import os
from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .... import dexverse_base_env_cfg as dexverse_base_env
from .... import mdp
from ...floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from .usd_helpers import (
    collect_articulation_usds_from_dir,
    ensure_single_articulation_root,
    is_usd_file_path,
    make_floating_articulation_root,
    normalize_joint_limits,
    simplify_collision_approximation,
)

# Standardized scene field name used by all observation/reward/termination/event
# entries below. Children should not change this — they swap the ArticulationCfg
# stored at this key via the class attributes.
ARTICULATION_KEY = "articulation"
ARTICULATION_PRIM_PATH = "{ENV_REGEX_NS}/Articulation"


def _make_single_usd_spawn(
    *,
    usd_path: str,
    scale: tuple[float, float, float],
    fix_root_link: bool | None,
) -> sim_utils.UsdFileCfg:
    """Per-asset UsdFileCfg with the standard tabletop articulation physics props."""
    return sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            max_effort=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=fix_root_link,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    )


def make_articulation_cfg(
    *,
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    init_pos: tuple[float, float, float] = (0.3, 0.0, 0.1),
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    init_joint_pos: dict[str, float] | None = None,
    fix_root_link: bool | None = True,
    actuators: dict | None = None,
    prim_path: str = ARTICULATION_PRIM_PATH,
    collision_approximation: str | None = "convexDecomposition",
) -> ArticulationCfg:
    """Build a single-asset :class:`ArticulationCfg` with tabletop physics props.

    The USD is run through :func:`ensure_single_articulation_root` so assets
    that author multiple ``ArticulationRootAPI`` prims (e.g. some synthesis
    USDs) load successfully, then through
    :func:`simplify_collision_approximation` so expensive ``sdf`` colliders
    don't stall / hang scene load (``collision_approximation=None`` disables).
    """
    usd_path = ensure_single_articulation_root(usd_path)
    usd_path = simplify_collision_approximation(usd_path, replace_with=collision_approximation)
    usd_path = normalize_joint_limits(usd_path)
    # For free-root tasks, relocate the articulation root onto the root rigid
    # body (fix_base assets author it on the world-fixed ``root_joint``, which
    # PhysX rejects for a floating articulation).
    if fix_root_link is False:
        usd_path = make_floating_articulation_root(usd_path)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=_make_single_usd_spawn(usd_path=usd_path, scale=scale, fix_root_link=fix_root_link),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=init_joint_pos or {},
        ),
        actuators=actuators or {},
    )


def make_multi_articulation_cfg(
    *,
    usd_paths: list[str],
    random_choice: bool = False,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    init_pos: tuple[float, float, float] = (0.3, 0.0, 0.1),
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    init_joint_pos: dict[str, float] | None = None,
    fix_root_link: bool | None = True,
    actuators: dict | None = None,
    prim_path: str = ARTICULATION_PRIM_PATH,
    collision_approximation: str | None = "convexDecomposition",
) -> ArticulationCfg:
    """Build an :class:`ArticulationCfg` that spawns a different USD per env.

    Uses :class:`sim_utils.MultiAssetSpawnerCfg` (not ``MultiUsdFileCfg``)
    because articulations need per-asset ``ArticulationRootPropertiesCfg``.

    With ``random_choice=False`` the assignment is deterministic round-robin
    (``usd_paths[i % N]`` for env i), which lets us control the (env_idx ->
    asset) mapping ourselves and randomise it manually if we want -- mirrors
    the convention used in ``pickup_object/base_cfg.py``.

    Each USD is repaired with :func:`ensure_single_articulation_root` and
    :func:`simplify_collision_approximation` (see :func:`make_articulation_cfg`).
    """
    if not usd_paths:
        raise ValueError("usd_paths must be a non-empty list")
    cleaned = [ensure_single_articulation_root(p) for p in usd_paths]
    cleaned = [simplify_collision_approximation(p, replace_with=collision_approximation) for p in cleaned]
    cleaned = [normalize_joint_limits(p) for p in cleaned]
    # See make_articulation_cfg: free-root tasks need the articulation root on
    # the root rigid body, not the authored world-fixed ``root_joint``.
    if fix_root_link is False:
        cleaned = [make_floating_articulation_root(p) for p in cleaned]
    assets_cfg = [_make_single_usd_spawn(usd_path=p, scale=scale, fix_root_link=fix_root_link) for p in cleaned]
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=assets_cfg,
            random_choice=random_choice,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=init_joint_pos or {},
        ),
        actuators=actuators or {},
    )


def _default_reset_pose_range() -> dict[str, list[float]]:
    return {
        "x": [0.0, 0.0],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [0.0, 0.0],
    }


@configclass
class ArticulationBaseObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for tabletop articulated-object tasks.

    Splits the articulation info across two groups:
      - ``state`` (observable, deployable; no velocities): articulation link
        poses, joint angles, and max-joint "open amount". The articulation is
        placed at a randomized pose per episode, so its pose is essential state
        for the policy.
      - ``privileged`` (sim-only): articulation link velocities and joint
        velocities (plus the inherited robot ``joint_vel`` / ``hand_tips``).

    No ``goal`` group: the "open it" target is a per-task constant (uninformative
    for imitation learning) and would otherwise have to be sourced from the
    reward — the demonstrated motion teaches the target. ``proprio`` stays as the
    base's joint-pos-only (robot) group.
    """

    @configclass
    class StateObsCfg(ObsGroup):
        articulation_pose_b = ObsTerm(
            func=mdp.body_pose_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg(ARTICULATION_KEY),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )
        articulation_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=".*")},
        )
        articulation_open_amount = ObsTerm(
            func=mdp.max_joint_pos,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=".*")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedObsCfg(dexverse_base_env.ObservationsCfg.PrivilegedObsCfg):
        articulation_vel_b = ObsTerm(
            func=mdp.body_vel_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={
                "body_asset_cfg": SceneEntityCfg(ARTICULATION_KEY),
                "base_asset_cfg": SceneEntityCfg("table"),
            },
        )
        articulation_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            params={"asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=".*")},
        )

    state: StateObsCfg = StateObsCfg()
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()


@configclass
class ArticulationBaseRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward for driving the target joint past a threshold."""

    open_amount = RewTerm(
        func=mdp.joint_open_reward,
        weight=5.0,
        params={
            "threshold_rad": MISSING,
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=".*"),
        },
    )


@configclass
class ArticulationBaseTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when the target joint crosses the threshold."""

    success = DoneTerm(
        func=mdp.joint_reach_threshold,
        params={
            "threshold": MISSING,
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=".*"),
        },
    )


@configclass
class ArticulationBaseEventCfg(dexverse_base_env.EventCfg):
    """Reset events for the articulated object."""

    reset_articulation = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "pose_range": _default_reset_pose_range(),
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY),
        },
    )

    reset_articulation_joints = EventTerm(
        func=mdp.reset_joints_to_init,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=".*"),
        },
    )


@configclass
class ArticulationBaseSceneCfg(dexverse_base_env.SceneCfg):
    """Scene with a single articulation slot. Child sets ``articulation`` via attrs."""

    articulation: ArticulationCfg = MISSING
    object = None


@configclass
class ArticulationBaseEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Robot-agnostic base config for tabletop articulated-object manipulation.

    Children override the ``articulation_*`` and ``success_*`` class attributes
    to specify the asset and success condition.
    """

    # ---- Articulation parameters (children override) ----
    # Path to a single USD file *or* a directory containing one
    # ``<asset_id>/mobility.usd`` per asset (the layout produced by
    # ``partnet_mobility_to_usd.py``). When a directory is given we spawn
    # a different USD per parallel environment via ``MultiAssetSpawnerCfg``,
    # mirroring the pattern used in ``pickup_object/base_cfg.py``.
    articulation_usd_path: str = MISSING
    articulation_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    articulation_init_pos: tuple[float, float, float] = (0.3, 0.0, 0.1)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_init_joint_pos: dict[str, float] | None = None
    # Half-height estimate used to seat the articulation on top of the table.
    articulation_half_height_est: float = 0.1
    # True  -> IsaacLab adds a fixed joint between world and the articulation root
    #          (root prim must have RigidBodyAPI; works for partnet-mobility assets).
    # False -> IsaacLab disables any authored fixed joint (root is free).
    # None  -> leave the USD's authored state alone (use this for assets whose
    #          ArticulationRootAPI is on a non-rigid-body prim, e.g. synthesis assets).
    articulation_fix_root_link: bool | None = True
    # Collider approximation override applied to the asset at load time. Many
    # synthesis USDs author every collider as ``sdf``; with no resolution set
    # PhysX cooks them at a 256^3 default that stalls / hard-hangs scene load
    # (before the first physics step) on a cold cooking cache. We rewrite them
    # to a cheap, robust approximation. Set to ``None`` to keep the USD's
    # authored colliders (e.g. ``sdf``) untouched.
    articulation_collision_approximation: str | None = "convexDecomposition"
    # Pose randomization applied at reset. None -> default y in [-0.2, 0.2].
    articulation_reset_pose_range: dict[str, list[float]] | None = None

    # ---- Multi-asset spawn (only used when articulation_usd_path is a dir) ----
    # Optional substring filter over the discovered ``<asset_id>/mobility.usd``
    # paths. None means "use everything found in the directory".
    articulation_ids: list[str] | None = None
    # Optional ``meta.json target_slot`` filter, e.g. ``"target_joint_prismatic"``.
    articulation_target_slot_filter: str | None = None
    # ``False`` (default) keeps the (env_idx -> asset) mapping deterministic
    # round-robin, so we control randomisation ourselves; ``True`` lets
    # IsaacLab pick randomly per spawn.
    articulation_multi_random_choice: bool = False
    # When ``True`` and the path is a directory, ``scene.num_envs`` is forced
    # to the number of discovered USDs so every env gets a unique asset.
    articulation_multi_auto_num_envs: bool = True

    # ---- Success criterion (children override) ----
    # Joint name(s) used for the success / reward / observation entries.
    # May be a regex string ("joint_0", ".*") or list of names.
    success_joint_names: Any = ".*"
    # Threshold value the (max) joint position must reach to count as success.
    # ``None`` means "not applicable" — typically because the leaf replaced
    # the default ``open_amount`` reward and ``success`` termination with
    # task-specific terms (e.g. ratio-mode ``joint_relative_move``). The
    # wiring in ``__post_init__`` skips threshold patching when this is None.
    success_threshold: float | None = None

    # ---- Standard config bundles ----
    observations: ArticulationBaseObservationsCfg = ArticulationBaseObservationsCfg()
    rewards: ArticulationBaseRewardsCfg = ArticulationBaseRewardsCfg()
    terminations: ArticulationBaseTerminationsCfg = ArticulationBaseTerminationsCfg()
    events: ArticulationBaseEventCfg = ArticulationBaseEventCfg()
    scene: ArticulationBaseSceneCfg = ArticulationBaseSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
    )

    def __post_init__(self):
        # 1. Build the articulation from the class attributes. Decide between
        #    single-USD spawn and multi-asset spawn by inspecting the path.
        usd_path = self.articulation_usd_path
        if usd_path is MISSING or usd_path is None:
            raise ValueError("articulation_usd_path must be set (file or directory).")
        if os.path.isdir(usd_path):
            usd_paths = collect_articulation_usds_from_dir(
                usd_path,
                articulation_ids=self.articulation_ids,
                target_slot=self.articulation_target_slot_filter,
            )
            if self.articulation_multi_auto_num_envs:
                self.scene.num_envs = len(usd_paths)
            self.scene.articulation = make_multi_articulation_cfg(
                usd_paths=usd_paths,
                random_choice=self.articulation_multi_random_choice,
                scale=self.articulation_scale,
                init_pos=self.articulation_init_pos,
                init_rot=self.articulation_init_rot,
                init_joint_pos=self.articulation_init_joint_pos,
                fix_root_link=self.articulation_fix_root_link,
                collision_approximation=self.articulation_collision_approximation,
            )
        elif os.path.isfile(usd_path) or is_usd_file_path(usd_path):
            self.scene.articulation = make_articulation_cfg(
                usd_path=usd_path,
                scale=self.articulation_scale,
                init_pos=self.articulation_init_pos,
                init_rot=self.articulation_init_rot,
                init_joint_pos=self.articulation_init_joint_pos,
                fix_root_link=self.articulation_fix_root_link,
                collision_approximation=self.articulation_collision_approximation,
            )
        else:
            raise FileNotFoundError(f"articulation_usd_path is neither a USD file nor a directory: {usd_path}")

        # 2. Wire the success joint names + threshold into obs/reward/termination.
        # Observation wiring is unconditional because every subclass inherits
        # ``ArticulationBaseObservationsCfg``. Reward / termination wiring is
        # defensive: leaves may override these with task-specific terms
        # (different reward function, different success criterion) that don't
        # have ``open_amount`` / ``success`` fields, in which case we skip.
        joint_names = self.success_joint_names
        self.observations.state.articulation_joint_pos.params["asset_cfg"] = SceneEntityCfg(
            ARTICULATION_KEY, joint_names=joint_names
        )
        self.observations.privileged.articulation_joint_vel.params["asset_cfg"] = SceneEntityCfg(
            ARTICULATION_KEY, joint_names=joint_names
        )
        self.observations.state.articulation_open_amount.params["asset_cfg"] = SceneEntityCfg(
            ARTICULATION_KEY, joint_names=joint_names
        )
        # Only patch the reward / termination if they are the defaults supplied
        # by articulation_base. Leaves are free to replace them with
        # task-specific terms (e.g. ``joint_open_fraction_reward``,
        # ``joint_relative_move`` with ``mode="ratio"``); those are left alone
        # and the leaf is responsible for wiring joint_names + thresholds.
        default_open_amount = getattr(self.rewards, "open_amount", None)
        if default_open_amount is not None and default_open_amount.func is mdp.joint_open_reward:
            default_open_amount.params["asset_cfg"] = SceneEntityCfg(ARTICULATION_KEY, joint_names=joint_names)
            if self.success_threshold is not None:
                default_open_amount.params["threshold_rad"] = self.success_threshold
        default_success = getattr(self.terminations, "success", None)
        if default_success is not None and default_success.func is mdp.joint_reach_threshold:
            default_success.params["asset_cfg"] = SceneEntityCfg(ARTICULATION_KEY, joint_names=joint_names)
            if self.success_threshold is not None:
                default_success.params["threshold"] = self.success_threshold

        # 3. Run the parent setup (configures robot, sim params, episode length).
        super().__post_init__()

        # 4. Disable object-related terms inherited from the tabletop base.
        # Observation groups (rgb / depth / pointcloud / proprio / privileged /
        # contact) are intentionally kept — every sub-env inherits the base
        # observation surface; only commands / events / terminations that are
        # specific to a rigid object are nulled here.
        if hasattr(self.commands, "object_pose"):
            self.commands.object_pose = None
        self.events.object_physics_material = None
        self.events.object_scale_mass = None
        self.events.reset_object = None
        self.terminations.object_out_of_bound = None

        # 5. Seat the articulation on top of the tabletop.
        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT
        articulation_z = table_top_z + self.articulation_half_height_est
        ax, ay, _ = self.articulation_init_pos
        self.scene.articulation.init_state.pos = (ax, ay, articulation_z)

        # 6. Apply the user's reset pose range (or keep the default).
        if self.articulation_reset_pose_range is not None and self.events.reset_articulation is not None:
            self.events.reset_articulation.params["pose_range"] = self.articulation_reset_pose_range

        # 7. Set up fingertip contact sensors against the articulation, if enabled.
        if self.robot_config.setup_contact_sensors:
            tip_prim_prefix = "{ENV_REGEX_NS}/Robot/"
            finger_tip_body_list = self.robot_config.fingertip_body_names

            for link_name in finger_tip_body_list:
                sensor_path = f"{tip_prim_prefix}{link_name}"
                setattr(
                    self.scene,
                    f"{link_name}_articulation_s",
                    ContactSensorCfg(
                        prim_path=sensor_path,
                        filter_prim_paths_expr=[ARTICULATION_PRIM_PATH],
                    ),
                )

            self.observations.contact.contact = ObsTerm(
                func=mdp.fingers_contact_force_b,
                params={"contact_sensor_names": [f"{link}_articulation_s" for link in finger_tip_body_list]},
                clip=(-20.0, 20.0),
            )
        else:
            self.observations.contact = None

        # 8. Override observation body names with robot-specific names.
        self.observations.privileged.hand_tips_state_b.params["body_asset_cfg"].body_names = (
            self.robot_config.hand_tips_body_names
        )


@configclass
class ArticulationBaseEnvFloatingDexHandRightCfg(ArticulationBaseEnvCfg):
    """Floating DexHand variant (Shadow / Leap, right-handed) of the base env.

    Children typically subclass *this* class and override only the
    ``articulation_*`` / ``success_*`` attributes.
    """

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)
