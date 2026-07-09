# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base classes for all pickup-object task variants.

Defines:
  - Default object (diverse YCB-style set via MultiAssetSpawner) and geometry constants
  - PickupObjectRewardsCfg  -- fingers_to_object, lift_when_grasping
  - PickupObjectTerminationsCfg -- object_out_of_bound
  - PickupObjectEnvCfg -- robot-agnostic base: table placement, contact
    sensors, observation/reward body-name wiring
"""

from __future__ import annotations

import glob
import os
import warnings

import isaaclab.sim as sim_utils

# ---------------------------------------------------------------------------
# Default object: diverse object pool from `make_default_object_cfg`
# ---------------------------------------------------------------------------
from dexverse.assets import MANI_TWIN_SELECTED_DIR as _MANI_TWIN_SELECTED_DIR
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp, object_annotations

# Discover USDs + sibling ``manipulation_annotations.json`` once and shuffle
# with a fixed seed so the (env_idx -> object) mapping is reproducible across
# runs but still deterministic within a run.
#
# This runs at import time, and ``dexverse.tasks`` imports *every* config module
# (incl. unrelated ones that merely share ``PickupObjectObservationsCfg`` from
# here). The mani_twin_selected pool is a separately downloaded bundle, so a hard
# failure here would abort the whole task registry. Degrade gracefully instead:
# leave the pool empty, warn once, and raise a clear, actionable error only when
# a task that actually needs the default pool is instantiated (see __post_init__).
_OBJECT_POOL_SEED = 0
_DEFAULT_POOL_UNAVAILABLE_MSG = (
    "The default object pool (mani_twin_selected) is not present. Tasks that use "
    "it (e.g. the default LiftObject / RelocateObject pool) need it downloaded: "
    "`python scripts/asset_tools/download_assets.py --mani-twin` (or rebuild it "
    "from upstream with `python scripts/asset_tools/download_selected_objects.py`). "
    "Tasks with an explicit `usd_path`, primitive tasks, and all other tasks are "
    "unaffected."
)
try:
    _OBJECT_USD_PATHS, _OBJECT_PLACEMENTS = object_annotations.collect_object_pool(
        str(_MANI_TWIN_SELECTED_DIR),
        shuffle=True,
        seed=_OBJECT_POOL_SEED,
    )
except (ValueError, OSError) as _pool_exc:
    warnings.warn(f"{_DEFAULT_POOL_UNAVAILABLE_MSG} (cause: {_pool_exc})", stacklevel=2)
    _OBJECT_USD_PATHS, _OBJECT_PLACEMENTS = [], []

DEFAULT_OBJECT_USD_PATHS: tuple[str, ...] = tuple(_OBJECT_USD_PATHS)
DEFAULT_OBJECT_PLACEMENTS: tuple[object_annotations.ObjectPlacement, ...] = tuple(_OBJECT_PLACEMENTS)

DEFAULT_OBJECT_CFG = dexverse_base_env.make_default_object_cfg(
    usd_paths=list(DEFAULT_OBJECT_USD_PATHS),
    random_choice=False,
    init_pos=(0.0, 0.0, 0.0),
    mass=0.3,
)

DEFAULT_OBJECT_HALF_HEIGHT: float = 0.05


# ---------------------------------------------------------------------------
# Object spawn helpers (single file + per-env MultiUsdFileCfg)
# ---------------------------------------------------------------------------


_USD_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")


def _is_usd_file_path(path: str) -> bool:
    """Return True if ``path`` looks like a single USD asset file."""
    lower = path.lower()
    return any(lower.endswith(ext) for ext in _USD_EXTENSIONS)


def build_object_cfg_from_usd(
    usd_path: str,
    mass: float = 0.3,
    scene_name: str = "object",
    prim_name: str = "Object",
) -> tuple[str, RigidObjectCfg]:
    """Build a (scene_name, RigidObjectCfg) pair for a single USD file.

    ``collision_props`` is required for IsaacLab to apply ``PhysxCollisionAPI``
    at spawn time -- without it PhysX doesn't register the per-mesh colliders
    even when the USD already has ``UsdPhysics.CollisionAPI`` authored. The
    convex-decomposition approximation baked into the USD by
    ``scripts/download_selected_objects.py`` is preserved either way, since
    ``modify_collision_properties`` does not touch ``MeshCollisionAPI``.
    """
    cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.UsdFileCfg(
            func=dexverse_base_env.spawn_usd_with_rigid_properties,
            usd_path=usd_path,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.707107, -0.707107, 0.0, 0.0),
        ),
    )
    return scene_name, cfg


def collect_usd_files_from_dir(
    usd_parent_dir: str,
    object_ids: list[str] | None = None,
) -> list[str]:
    """Recursively list USD files under ``usd_parent_dir`` (deterministic order).

    ``object_ids`` optionally filters the list by path substring, so a given
    USD always maps to the same env index when ``random_choice=False``.
    """
    usd_files_all = []
    for ext in _USD_EXTENSIONS:
        usd_files_all.extend(glob.glob(os.path.join(usd_parent_dir, f"**/*{ext}"), recursive=True))
    usd_files_all = sorted(set(usd_files_all))
    usd_files = [f for f in usd_files_all if "instanceable_meshes" not in f]
    usd_files = object_annotations.prefer_unpacked_usd(usd_files)

    if object_ids is not None:
        usd_files = [f for f in usd_files if any(oid in f for oid in object_ids)]

    if not usd_files:
        raise ValueError(f"No USD files found in {usd_parent_dir}")

    return usd_files


def build_multi_usd_object_cfg(
    usd_paths: list[str],
    *,
    mass: float = 0.3,
    random_choice: bool = False,
    prim_name: str = "Object",
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    init_rot: tuple[float, float, float, float] = (0.707107, -0.707107, 0.0, 0.0),
) -> RigidObjectCfg:
    """Build a RigidObjectCfg that spawns a different USD per environment.

    Uses IsaacLab's :class:`sim_utils.MultiUsdFileCfg` so each clone of
    ``{ENV_REGEX_NS}/<prim_name>`` can draw a distinct USD from ``usd_paths``.
    With ``random_choice=False`` the mapping is deterministic round-robin
    (``usd_paths[i % N]``) so setting ``scene.num_envs == len(usd_paths)``
    guarantees every environment gets a unique object.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=list(usd_paths),
            random_choice=random_choice,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
    )


# ---------------------------------------------------------------------------
# Shared observations
# ---------------------------------------------------------------------------


@configclass
class PickupObjectObservationsCfg(dexverse_base_env.ObservationsCfg):
    """Observation layout for tabletop pickup-object tasks.

    ``state`` (observable, no velocities): object pose. ``privileged``: object
    linear / angular velocities (+ inherited robot ``joint_vel`` / ``hand_tips``).
    ``proprio`` stays as the base's joint-pos-only group.
    """

    @configclass
    class StateObsCfg(ObsGroup):
        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))

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


# ---------------------------------------------------------------------------
# Shared rewards
# ---------------------------------------------------------------------------


@configclass
class PickupObjectRewardsCfg(dexverse_base_env.RewardsCfg):
    """Rewards shared by all pickup-object variants."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "distance_gain": 10.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
        },
    )


# ---------------------------------------------------------------------------
# Shared terminations
# ---------------------------------------------------------------------------


@configclass
class PickupObjectTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Terminations shared by all pickup-object variants."""

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            # tightened to table footprint in __post_init__
            "in_bound_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.2, 1.5)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


# ---------------------------------------------------------------------------
# Base env config
# ---------------------------------------------------------------------------


@configclass
class PickupObjectEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Robot-agnostic base for all pickup-object variants.

    Handles:
    - Placing the object flush on the table surface
    - Randomising the object reset pose within a centre square
    - Clamping the out-of-bound range to the table footprint
    - Wiring contact sensors and body names from robot_config

    Subclasses define their own scene / rewards / terminations / commands.
    Override ``object_half_height`` when using a non-can object.

    Custom objects
    ~~~~~~~~~~~~~~
    ``usd_path`` selects the object(s) used by the task:

    - ``None``  -- use :data:`DEFAULT_OBJECT_CFG` (diverse multi-object pool).
    - Path to a single ``.usd`` / ``.usda`` / ``.usdc`` file -- replace
      ``scene.object`` with this asset (single-object mode).
    - Path to a directory -- recursively collect every USD under it and spawn
      a different one per parallel environment via IsaacLab's
      :class:`sim_utils.MultiUsdFileCfg`.  ``scene.num_envs`` is auto-sized to
      the number of discovered USDs (unless ``multi_auto_num_envs=False``).
      ``scene.object`` stays a single entity, so all existing rewards /
      terminations / observations keep working unchanged.
    """

    object_half_height: float = DEFAULT_OBJECT_HALF_HEIGHT
    center_square_size: float = 0.45

    # Extra vertical clearance added to the support surface at reset time, on
    # top of the per-object bbox-derived lift in ``object_placements``. Keeps a
    # tiny margin against floating-point penetration of the table.
    support_clearance: float = 0.001

    # Static / dynamic friction applied to the object's collision shapes at
    # sim startup via PhysX material properties. Set both to the same value for
    # a deterministic coefficient; set to ``None`` to leave whatever the USD /
    # default material already specifies untouched.
    object_static_friction: float | None = 2.0
    object_dynamic_friction: float | None = 2.0

    usd_path: str | None = None
    object_ids: list[str] | None = None
    multi_random_choice: bool = False
    multi_auto_num_envs: bool = True

    # Per-env upright placements derived from ``manipulation_annotations.json``.
    # When ``None`` and ``usd_path`` is also ``None`` we auto-populate from the
    # default object pool. Setting to an empty tuple disables annotation-driven
    # resets (falls back to the generic uniform reset).
    object_placements: tuple[object_annotations.ObjectPlacement, ...] | None = None

    observations: PickupObjectObservationsCfg = PickupObjectObservationsCfg()
    rewards: PickupObjectRewardsCfg = PickupObjectRewardsCfg()
    terminations: PickupObjectTerminationsCfg = PickupObjectTerminationsCfg()

    def __post_init__(self):
        # --- Object setup: resolve single file vs. directory of USDs ---
        if self.usd_path is not None:
            if os.path.isfile(self.usd_path) or _is_usd_file_path(self.usd_path):
                # Single USD file -> swap out the default object, stay in single-object mode.
                _, single_cfg = build_object_cfg_from_usd(self.usd_path)
                self.scene.object = single_cfg
            elif os.path.isdir(self.usd_path):
                # Directory -> spawn a different USD per env via MultiUsdFileCfg.
                usd_files = collect_usd_files_from_dir(self.usd_path, self.object_ids)
                if self.multi_auto_num_envs:
                    self.scene.num_envs = len(usd_files)
                self.scene.object = build_multi_usd_object_cfg(
                    usd_files,
                    random_choice=self.multi_random_choice,
                )
            else:
                raise FileNotFoundError(f"usd_path does not exist or is not a USD file / directory: {self.usd_path}")
        elif self.object_placements is None:
            # Default path: use the shuffled mani_twin_selected pool and its
            # annotations. ``DEFAULT_OBJECT_CFG`` already spawns these USDs in
            # round-robin order so entry ``i`` of the placements matches env i.
            if not DEFAULT_OBJECT_USD_PATHS:
                raise FileNotFoundError(_DEFAULT_POOL_UNAVAILABLE_MSG)
            self.object_placements = DEFAULT_OBJECT_PLACEMENTS

        super().__post_init__()

        table_top_z = dexverse_base_env.DEFAULT_TABLE_TOP_HEIGHT

        # Place object flush on the table surface.
        obj_pos = self.scene.object.init_state.pos
        self.scene.object.init_state.pos = (obj_pos[0], obj_pos[1], table_top_z + self.object_half_height)

        half_side = self.center_square_size * 0.5

        use_annotation_reset = bool(self.object_placements)
        if use_annotation_reset and self.events.reset_object is not None:
            # Cycle the placements to match scene.num_envs so every env gets an
            # aligned ``(position_offset, quat)`` entry.
            per_env = object_annotations.placements_for_num_envs(self.object_placements, self.scene.num_envs)
            per_env_positions = [list(p.position_offset) for p in per_env]
            per_env_quats = [list(p.quat) for p in per_env]

            self.events.reset_object = EventTerm(
                func=mdp.reset_object_from_place_annotations,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "per_env_positions": per_env_positions,
                    "per_env_quats": per_env_quats,
                    "support_z": table_top_z + self.support_clearance,
                    "pose_range": {
                        "x": [0.0, 0.0],
                        "y": [-half_side, half_side],
                        "z": [0.0, 0.0],
                        "roll": [0.0, 0.0],
                        "pitch": [0.0, 0.0],
                        "yaw": [0.0, 0.0],
                    },
                },
            )
        elif self.events.reset_object is not None:
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [-half_side, half_side],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            }

        if self.terminations.object_out_of_bound is not None:
            table_size = self.scene.table.spawn.size
            self.terminations.object_out_of_bound.params["in_bound_range"] = {
                "x": (-table_size[0] * 0.5, table_size[0] * 0.5),
                "y": (-table_size[1] * 0.5, table_size[1] * 0.5),
                "z": (-0.2, 1.5),
            }

        # PhysX material on the object's collision shapes. Sets static / dynamic
        # friction at startup using IsaacLab's randomize_rigid_body_material:
        # equal (low, high) bounds + num_buckets=1 collapses the "randomization"
        # to a single deterministic coefficient applied to every shape.
        if self.object_static_friction is not None and self.object_dynamic_friction is not None:
            self.events.object_physics_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "static_friction_range": (float(self.object_static_friction),) * 2,
                    "dynamic_friction_range": (float(self.object_dynamic_friction),) * 2,
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            )

        # Contact sensors and contact observation.
        if self.robot_config.setup_contact_sensors:
            tip_prim_prefix = "{ENV_REGEX_NS}/Robot/"
            filter_paths = ["{ENV_REGEX_NS}/Object"]
            for link_name in self.robot_config.fingertip_body_names:
                setattr(
                    self.scene,
                    f"{link_name}_object_s",
                    ContactSensorCfg(
                        prim_path=f"{tip_prim_prefix}{link_name}",
                        filter_prim_paths_expr=filter_paths,
                    ),
                )
            sensor_names = [f"{link}_object_s" for link in self.robot_config.fingertip_body_names]
            self.observations.contact.contact = ObsTerm(
                func=mdp.fingers_contact_force_b,
                params={"contact_sensor_names": sensor_names},
                clip=(-20.0, 20.0),
            )
        else:
            self.observations.contact = None

        # Wire robot-specific body names into observations and rewards.
        self.observations.privileged.hand_tips_state_b.params["body_asset_cfg"].body_names = (
            self.robot_config.hand_tips_body_names
        )
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
