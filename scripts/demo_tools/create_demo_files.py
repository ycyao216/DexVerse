# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Headless converter: replay teleop demonstrations and stack them into HDF5
files ready for learning pipelines.

This is the production sibling of ``replay_demos.py``. It:

* Runs headless (no keyboard UI, no pause), so it is safe to call from
  cron / cluster jobs.
* Discovers source pickles under ``--demos-root`` (default:
  ``source/dexverse/demonstrations`` — the path ``download_demos.py``
  syncs into) and supports ``--task``, ``--file``, ``--all`` selection.
* Captures any combination of observation groups by name
  (``--obs-groups proprio perception vision``). The chosen group names
  are stored as a root attribute in the output HDF5 so downstream
  learners can pull only the groups they need.
* Encodes RGB images as ``uint8`` and depth images as ``float16`` for
  compact on-disk storage. Point clouds are produced already cropped by
  the env's ``camera_point_cloud_w`` term (table size + insets).
* Groups source pickles by task directory and merges every pickle in a
  directory into a single demonstration. Task directories often
  accumulate multiple ``.pkl`` files when teleop is run on different
  days; this script concatenates their episode lists (renumbering
  ``episode_index`` so each episode is unique) and writes one HDF5 per
  task directory, named after the directory and mirroring the source
  layout under ``--output-dir`` when given. ``--file <path>`` still
  produces a one-pickle output named after the pickle itself.

Only single-env ``dexverse_trajectory`` pickles are supported here. Batch
``dexverse_trajectory_batch`` bundles produced by ``merge_demos.py`` still
need ``replay_demos.py``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import NamedTuple

import torch
from _active_object_masks import apply_recorded_active_masks
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Convert dexverse teleop pickles into HDF5 demo files (headless).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

# --- Discovery / selection -------------------------------------------------
parser.add_argument(
    "--demos-root",
    type=Path,
    default=Path(__file__).resolve().parents[2] / "source" / "dexverse" / "demonstrations",
    help="Root directory containing demonstration pickles (default: source/dexverse/demonstrations).",
)
parser.add_argument(
    "--task",
    action="append",
    default=[],
    help="Task subdirectory to convert (relative to --demos-root). Repeatable.",
)
parser.add_argument(
    "--file",
    action="append",
    default=[],
    help="Specific pickle file to convert. Repeatable.",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Convert every pickle found under --demos-root.",
)

# --- Output ---------------------------------------------------------------
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help=(
        "Output directory; mirrors the relative layout of each source pickle. "
        "Defaults to writing the HDF5 next to its source pickle."
    ),
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing output HDF5 files (default: skip).",
)
parser.add_argument(
    "--compression",
    type=str,
    default="gzip",
    choices=["none", "gzip", "lzf"],
    help="HDF5 dataset compression (default: gzip).",
)
parser.add_argument(
    "--compression-opts",
    type=int,
    default=4,
    help="HDF5 gzip compression level (1..9); ignored for lzf/none.",
)

# --- Observation groups ---------------------------------------------------
parser.add_argument(
    "--obs-groups",
    nargs="+",
    default=None,
    help=(
        "Observation selection. Accepts either:\n"
        "  - a single observation-preset name "
        "(`rgb`, `rgb_depth` / `rgbd`, `pointcloud`, `state`) — applies the "
        "preset on the env cfg before construction so the obs space is "
        "narrowed accordingly (and `policy` / `proprio` history stacking is "
        "configured); or\n"
        "  - one or more observation-group names that exist on the env's "
        "observation manager (e.g. `--obs-groups proprio rgb`).\n"
        "When omitted (the default), every active group on the env is "
        "captured — downstream consumers can pick the subset they need."
    ),
)
parser.add_argument(
    "--rgb-dtype",
    type=str,
    default="uint8",
    choices=["uint8", "float32"],
    help="Storage dtype for RGB image observation terms (default: uint8).",
)
parser.add_argument(
    "--depth-dtype",
    type=str,
    default="float16",
    choices=["float16", "float32"],
    help="Storage dtype for depth/distance image observation terms (default: float16).",
)

# --- Parallel replay ------------------------------------------------------
parser.add_argument(
    "--num-parallel-envs",
    type=int,
    default=0,
    help=(
        "Number of parallel envs used to replay one pickle. ``0`` (default) "
        "auto-sets it to the number of episodes in the pickle so every "
        "trajectory is replayed in parallel. Override with a positive "
        "integer to cap (or expand) the parallelism; if smaller than the "
        "pickle's episode count, only the first N episodes are replayed "
        "(chunking across pickles is not yet supported — split the pickle "
        "if you need more episodes than VRAM allows)."
    ),
)

# --- Episode-level filters ------------------------------------------------
parser.add_argument(
    "--select-episodes",
    type=int,
    nargs="+",
    default=[],
    help="Subset of episode indices to convert. Empty = all.",
)
parser.add_argument(
    "--set-state",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Replay by restoring recorded scene states instead of stepping actions. "
        "Requires episode['states'] with length T+1. On by default; pass "
        "--no-set-state to step the recorded actions through the simulator instead."
    ),
)

# --- Environment-level options -------------------------------------------
parser.add_argument(
    "--task-override",
    type=str,
    default=None,
    help="Override the task stored in the pickle (applied to every pickle).",
)
parser.add_argument(
    "--robot-type-override",
    type=str,
    default=None,
    help=(
        "Override the robot_type stored in the pickle (applied to every pickle). "
        "If the override changes the action layout (e.g. floating_shadow_right -> "
        "ur10e_shadow_right), recorded actions are converted on the fly."
    ),
)
parser.add_argument(
    "--json-path",
    type=str,
    default=None,
    help="Template JSON spec path. Required for *Template environments.",
)
parser.add_argument(
    "--enable-pinocchio",
    action="store_true",
    help="Enable Pinocchio (required for dex-retargeting / some IK controllers).",
)

# --- Video recording ------------------------------------------------------
parser.add_argument(
    "--record-video",
    action="store_true",
    help="Render one MP4 per (episode, env_id) into --video-dir.",
)
parser.add_argument(
    "--video-camera",
    type=str,
    default="third_person_camera",
    help="Name of the scene Camera sensor to render from (default: third_person_camera).",
)
parser.add_argument(
    "--video-fps",
    type=int,
    default=30,
    help="Frames per second for written MP4s (default: 30).",
)
parser.add_argument(
    "--video-dir",
    type=Path,
    default=None,
    help=(
        "Output directory for videos; mirrors the relative source layout under it. "
        "Defaults to a `videos/` sibling of each HDF5 output."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless. AppLauncher's flag name is `--headless`.
args_cli.headless = True

# Resolve --obs-groups into either a "preset" mode or a "groups" mode:
#   - a single arg matching a known preset name (alias-aware) -> preset mode.
#   - otherwise (including None / empty) -> groups mode with the literal list,
#     or all groups when None / empty.
# Preset names are kept in sync with
# ``dexverse.tasks.dexverse_base_env_cfg.OBSERVATION_PRESET_NAMES``; we
# hard-code them here so we don't have to import the env cfg before
# ``AppLauncher`` initializes (Isaac Sim must boot first to import IsaacLab).
_OBS_PRESET_NAMES = (
    "rgb",
    "rgb_depth",
    "pointcloud",
    "state",
    "3view_rgb",
    "3view_rgb_depth",
    "3view_pointcloud",
)
_OBS_PRESET_ALIASES = {"rgbd": "rgb_depth", "3view_rgbd": "3view_rgb_depth"}
_obs_preset_arg = None
if args_cli.obs_groups is not None and len(args_cli.obs_groups) == 1:
    _maybe = args_cli.obs_groups[0]
    _canonical = _OBS_PRESET_ALIASES.get(_maybe, _maybe)
    if _canonical in _OBS_PRESET_NAMES:
        _obs_preset_arg = _canonical
        # Drop the preset name from the "individual groups" list; the env-cfg
        # method will null-out the disabled groups and we'll resolve the
        # capture-group list from the env's obs manager after construction.
        args_cli.obs_groups = None

# Isaac Lab refuses to spawn Camera sensors unless --enable_cameras is set.
# Auto-enable when:
#  - the user opts into video recording, OR
#  - the user explicitly requests a camera-driven obs group, OR
#  - the user picked a preset that includes a camera-driven group, OR
#  - the user did not pass ``--obs-groups`` (default = capture everything,
#    which may include rgb/depth/pointcloud depending on the env).
_CAMERA_OBS_GROUPS = {"rgb", "depth", "pointcloud", "perception", "vision"}
_obs_groups_lower = {g.lower() for g in (args_cli.obs_groups or [])}
_capture_all_groups = args_cli.obs_groups is None and _obs_preset_arg is None
_preset_has_camera = _obs_preset_arg in {
    "rgb",
    "rgb_depth",
    "pointcloud",
    "3view_rgb",
    "3view_rgb_depth",
    "3view_pointcloud",
}
if args_cli.record_video or _capture_all_groups or _preset_has_camera or (_obs_groups_lower & _CAMERA_OBS_GROUPS):
    args_cli.enable_cameras = True

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import inspect  # noqa: E402

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import numpy as np  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.managers.manager_base import ManagerTermBase  # noqa: E402

try:
    from tqdm.auto import tqdm  # noqa: E402

    _HAS_TQDM = True
except ImportError:  # pragma: no cover - tqdm is in the env, but fall back cleanly
    _HAS_TQDM = False

    def tqdm(iterable=None, **kwargs):  # type: ignore[no-redef]
        """No-op fallback: returns the iterable unchanged when tqdm is unavailable."""
        return iterable if iterable is not None else []


def _evaluate_termination_term(term_cfg, env, instance_cache: dict):
    """Evaluate a ``TerminationTermCfg`` and return a ``(num_envs,)`` bool tensor.

    Handles both shapes that IsaacLab terminations can take:

    * Plain function: ``cfg.func(env, **cfg.params)`` — used by simple terms
      like ``mdp.time_out`` or ``mdp.joint_relative_move``.
    * :class:`ManagerTermBase` subclass: instantiate ``cfg.func(cfg=cfg, env=env)``
      once (cached), then call ``instance(env, **cfg.params)`` per step.
      This matches what :class:`TerminationManager._prepare_terms` does
      internally, so per-step results line up with what the env's own
      manager would emit during training.
    """
    func = term_cfg.func
    params = term_cfg.params or {}
    if inspect.isclass(func) and issubclass(func, ManagerTermBase):
        key = id(term_cfg)
        instance = instance_cache.get(key)
        if instance is None:
            instance = func(cfg=term_cfg, env=env)
            instance_cache[key] = instance
        return instance(env, **params)
    return func(env, **params)


import contextlib

import dexverse.tasks  # noqa: F401, E402

# ============================================================================
# Helpers copied (verbatim where possible) from replay_demos.py so this script
# stays self-contained. Keep these in sync if upstream behavior changes.
# ============================================================================

_SINGLE_FORMAT = "dexverse_trajectory"
_BATCH_FORMAT = "dexverse_trajectory_batch"


def _maybe_convert_actions(
    actions_per_env: list[np.ndarray],
    source_robot_type: str | None,
    target_robot_type: str | None,
    env_cfg,
) -> list[np.ndarray]:
    """Convert recorded action arrays in-place when robot_type changes the action layout.

    Currently supports: floating_shadow_right -> ur10e_shadow_right (28-dim
    palm-Euler -> 29-dim palm-quat via FK on the floating wrist + frame swap;
    fingers pass through). Returns the input unchanged when no conversion rule
    applies, so callers always get a list of action arrays.
    """
    if source_robot_type == target_robot_type or target_robot_type is None:
        return actions_per_env

    if (source_robot_type, target_robot_type) == ("floating_shadow_right", "ur10e_shadow_right"):
        from dexverse.robot_agents.shadow._floating_to_ur10e_actions import (
            DEFAULT_FLOATING_BASE_POS,
            DEFAULT_FLOATING_BASE_ROT,
            convert_floating_shadow_right_to_ur10e_actions,
        )

        # Pull the target base pose from the env_cfg (so a user that customized
        # UR10E base placement still gets a correct conversion).
        ur10e_init = env_cfg.scene.robot.init_state
        ur10e_base_pos = tuple(ur10e_init.pos)
        ur10e_base_rot = tuple(ur10e_init.rot)
        print(
            f"  [convert] {source_robot_type} -> {target_robot_type}: "
            f"converting {len(actions_per_env)} action stream(s). "
            f"Floating base assumed at {DEFAULT_FLOATING_BASE_POS}; "
            f"UR10e base read from env at {ur10e_base_pos}."
        )
        return [
            (
                convert_floating_shadow_right_to_ur10e_actions(
                    a,
                    floating_base_pos=DEFAULT_FLOATING_BASE_POS,
                    floating_base_rot=DEFAULT_FLOATING_BASE_ROT,
                    ur10e_base_pos=ur10e_base_pos,
                    ur10e_base_rot=ur10e_base_rot,
                )
                if a.size
                else a
            )
            for a in actions_per_env
        ]

    raise ValueError(
        f"No action converter registered for {source_robot_type!r} -> {target_robot_type!r}. "
        "Add a rule in _maybe_convert_actions, or drop --robot-type-override and replay with "
        "the original robot."
    )


def _load_trajectory_pickle(path: str) -> dict:
    with open(path, "rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"File {path!r} is not a valid pickle dictionary.")
    fmt = payload.get("format")
    if fmt == _SINGLE_FORMAT:
        if "episodes" not in payload:
            raise ValueError(f"Trajectory pickle {path!r} has no 'episodes' entry.")
        return payload
    if fmt == _BATCH_FORMAT:
        raise ValueError(
            f"Pickle {path!r} is a batch trajectory ({_BATCH_FORMAT!r}); use replay_demos.py "
            "to replay merged bundles. create_demo_files.py only handles single-env pickles."
        )
    raise ValueError(
        f"File {path!r} does not contain a single-env dexverse trajectory pickle "
        f"(expected format={_SINGLE_FORMAT!r}; got {fmt!r})."
    )


def _tensorize_state(data, device):
    if isinstance(data, dict):
        return {key: _tensorize_state(value, device) for key, value in data.items()}
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data, device=device)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


# Entities owned by the env config and not by the demo recording. ``reset_to``
# will use the env's post-reset values for these (config-defined positions)
# instead of whatever the pickle says. This unblocks pickles recorded before
# new cosmetic props (e.g. table_leg_*) were added to the scene.
_ENV_MANAGED_ENTITY_PREFIXES: tuple[str, ...] = ("table",)


def _is_env_managed_entity(name: str) -> bool:
    return any(name == p or name.startswith(p + "_") for p in _ENV_MANAGED_ENTITY_PREFIXES)


def _build_replay_state(
    recorded,
    live: dict,
    *,
    extra_skip_entity_names: set[str] | None = None,
) -> dict:
    """Merge a recorded state onto a live post-reset state for ``scene.reset_to``.

    Env-managed entities (e.g. the table and its cosmetic legs) keep their
    live values so recordings made before those entities existed still
    replay. ``extra_skip_entity_names`` is used when the env's robot was
    swapped to a different articulation than the one that was recorded —
    forcing the live (env-default) joint values to be kept instead of the
    recorded ones, which would otherwise have the wrong joint count.
    """
    skip = set(extra_skip_entity_names or ())
    out: dict = {category: dict(entries) for category, entries in live.items()}
    if not isinstance(recorded, dict):
        return out
    for category, rec_entities in recorded.items():
        if not isinstance(rec_entities, dict):
            continue
        bucket = out.setdefault(category, {})
        for entity_name, rec_state in rec_entities.items():
            if _is_env_managed_entity(entity_name) or entity_name in skip:
                continue
            bucket[entity_name] = rec_state
    return out


# (source_robot_type, target_robot_type) pairs for which the articulation differs
# and the recorded robot state should be dropped at scene.reset_to time.
_ARTICULATION_MISMATCH_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("floating_shadow_right", "ur10e_shadow_right"),
})


def _entities_to_skip_for_robot_type_change(source_robot_type: str | None, target_robot_type: str | None) -> set[str]:
    """Return scene entity names whose recorded state should be ignored when the
    robot articulation differs from what was recorded. ``"robot"`` is the only
    entity affected today, but routed through a helper so callers stay generic."""
    if source_robot_type is None or target_robot_type is None:
        return set()
    if source_robot_type == target_robot_type:
        return set()
    if (source_robot_type, target_robot_type) in _ARTICULATION_MISMATCH_PAIRS:
        return {"robot"}
    return set()


def _force_clear_sim_context() -> None:
    """Best-effort teardown of Isaac Lab's SimulationContext singleton.

    Called after a partial env build so the next ``gym.make`` doesn't trip
    on "Simulation context already exists. Cannot create a new one." Silent
    no-op if the API surface isn't available — never raises from cleanup.
    """
    try:
        from isaaclab.sim import SimulationContext  # type: ignore
    except Exception:
        return
    try:
        instance = SimulationContext.instance()
    except Exception:
        instance = None
    if instance is None:
        return
    for method_name in ("clear_instance", "clear_all_callbacks", "stop", "close"):
        method = getattr(instance, method_name, None)
        if callable(method):
            with contextlib.suppress(Exception):
                method()


def _get_runtime_obs(env, *, update_history: bool = False):
    # update_history must be True on the per-step set-state path: history-enabled
    # obs terms (e.g. proprio/policy with history_length>0) only advance their
    # circular buffer when compute(update_history=True) is called. env.step does
    # this internally on the action path, but the set-state path recomputes obs
    # manually, so without this every history obs stays frozen at the initial fill.
    if hasattr(env, "observation_manager"):
        env.obs_buf = env.observation_manager.compute(update_history=update_history)
        return env.obs_buf
    if hasattr(env, "get_observations"):
        return env.get_observations()
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    obs, _ = env.reset()
    return obs


def _refresh_after_set_state(env):
    """Pull a just-written scene state into the asset/sensor ``.data`` buffers.

    ``scene.reset_to`` writes poses/velocities/joint state into PhysX, but the
    cached ``.data`` buffers the observation manager reads are only refreshed by
    ``scene.update(dt)`` (and RTX cameras need a few renders to latch the new
    frame). ``ManagerBasedRLEnv.step`` does exactly this every control step; the
    ``--set-state`` path must replicate it or *every* observation (proprio and
    images alike) stays frozen at the initial-reset values.
    """
    env.sim.forward()
    rerenders = int(getattr(env.cfg, "num_rerenders_on_reset", 0) or 0)
    if env.sim.has_rtx_sensors() and rerenders > 0:
        for _ in range(rerenders):
            env.sim.render()
    else:
        env.sim.render()
    env.scene.update(dt=env.physics_dt)


def _set_last_action(env, action) -> None:
    """Write a recorded action into the action manager's buffers.

    The ``--set-state`` path never calls ``env.step``, so the action manager's
    ``_action``/``_prev_action`` buffers stay at their zero reset value. Any obs
    term reading them (notably ``mdp.last_action``, i.e. the whole ``policy``
    group) would otherwise be recorded as all-zeros for the entire episode. We
    set the buffers directly rather than calling ``process_action`` so there are
    no action-term side effects and the prev/current history stays correct:
    ``mdp.last_action`` returns the raw ``action_manager.action`` regardless.
    """
    am = getattr(env, "action_manager", None)
    if am is None:
        return
    am._prev_action[:] = am._action
    am._action[:] = action.to(am.device)


def _collect_multi_spawn_entities(scene_cfg):
    multi_asset_entities, multi_usd_entities = {}, {}
    for name in dir(scene_cfg):
        if name.startswith("_"):
            continue
        entity_cfg = getattr(scene_cfg, name, None)
        if entity_cfg is None or not hasattr(entity_cfg, "spawn"):
            continue
        spawn_cfg = getattr(entity_cfg, "spawn", None)
        if spawn_cfg is None:
            continue
        catalog_id = f"scene.{name}"
        spawn_type = type(spawn_cfg).__name__
        if spawn_type == "MultiAssetSpawnerCfg":
            multi_asset_entities[catalog_id] = entity_cfg
        elif spawn_type == "MultiUsdFileCfg":
            multi_usd_entities[catalog_id] = entity_cfg
    return multi_asset_entities, multi_usd_entities


def _apply_episode_bindings(*, bindings, entities, kind, picker):
    applied = set()
    for item in bindings:
        if not isinstance(item, dict):
            continue
        catalog_id = item.get("catalog_id")
        if not isinstance(catalog_id, str) or not catalog_id or catalog_id in applied:
            continue
        if catalog_id not in entities:
            raise ValueError(f"[Converter] {kind} catalog_id not found in scene: {catalog_id!r}")
        picker(item, entities[catalog_id], catalog_id)
        applied.add(catalog_id)


def _apply_per_env_episode_bindings(
    *,
    per_ep_bindings: list,
    entities: dict,
    kind: str,
    picker_per_env,
) -> None:
    """Apply per-env asset/USD bindings so env_id ``i`` matches episode ``i``.

    ``per_ep_bindings`` is a list of length ``num_envs`` whose entries are
    each episode's recorded bindings list (``ep["multi_assets"]`` or
    ``ep["multi_usds"]``). For each ``catalog_id``, we collect the per-episode
    item and let ``picker_per_env`` rewrite ``entity_cfg.spawn`` so that, with
    ``random_choice=False``, env_id ``i`` picks index ``i``'s entry.
    """
    if not entities:
        return
    sentinel = None
    for catalog_id, entity_cfg in entities.items():
        per_ep_items: list = []
        for ep_bindings in per_ep_bindings:
            picked = sentinel
            if isinstance(ep_bindings, list):
                for item in ep_bindings:
                    if isinstance(item, dict) and item.get("catalog_id") == catalog_id:
                        picked = item
                        break
            per_ep_items.append(picked)
        # Skip catalogs with no recorded binding in any episode — keeps backward
        # compat with pickles that didn't record this entity.
        if all(item is sentinel for item in per_ep_items):
            continue
        picker_per_env(per_ep_items, entity_cfg, catalog_id, kind=kind)


def _pick_multi_asset_binding_per_env(per_ep_items, entity_cfg, catalog_id, *, kind="multi_assets"):
    """Set ``assets_cfg`` so ``assets_cfg[i]`` is what env_id ``i`` should spawn."""
    spawn_cfg = entity_cfg.spawn
    original = list(getattr(spawn_cfg, "assets_cfg", []))
    if not original:
        return
    picked: list = []
    for i, item in enumerate(per_ep_items):
        if not isinstance(item, dict):
            raise ValueError(f"[Converter] {kind} binding missing for env_id {i} ({catalog_id!r}).")
        asset_idx = int(item.get("asset_idx", -1))
        if asset_idx < 0 or asset_idx >= len(original):
            raise ValueError(
                f"[Converter] {kind} asset_idx out of range for {catalog_id!r} env_id {i}: "
                f"{asset_idx} not in [0, {len(original) - 1}]"
            )
        picked.append(original[asset_idx])
    spawn_cfg.assets_cfg = picked
    spawn_cfg.random_choice = False


def _pick_multi_usd_binding_per_env(per_ep_items, entity_cfg, catalog_id, *, kind="multi_usds"):
    """Set ``usd_path`` so ``usd_path[i]`` is what env_id ``i`` should spawn."""
    spawn_cfg = entity_cfg.spawn
    usd_pool = getattr(spawn_cfg, "usd_path", None)
    usd_paths = [usd_pool] if isinstance(usd_pool, str) else list(usd_pool or [])
    if not usd_paths:
        return
    picked: list[str] = []
    for i, item in enumerate(per_ep_items):
        if not isinstance(item, dict):
            raise ValueError(f"[Converter] {kind} binding missing for env_id {i} ({catalog_id!r}).")
        selected_path = str(item.get("usd_path", ""))
        chosen: str | None = None
        if selected_path:
            if selected_path in usd_paths:
                chosen = selected_path
            else:
                target_name = os.path.basename(selected_path)
                for p in usd_paths:
                    if os.path.basename(p) == target_name:
                        chosen = p
                        break
        if chosen is None:
            usd_idx = int(item.get("usd_idx", -1))
            if 0 <= usd_idx < len(usd_paths):
                chosen = usd_paths[usd_idx]
        if chosen is None:
            raise ValueError(
                f"[Converter] {kind} binding for env_id {i} ({catalog_id!r}) "
                "could not be resolved in the local USD pool."
            )
        picked.append(chosen)
    spawn_cfg.usd_path = picked
    spawn_cfg.random_choice = False


def _stack_at_leaves(items: list):
    """Recursively stack a list of nested-dict values into a single nested dict.

    - At tensor leaves: each input is expected to have shape ``(1, ...)``;
      we squeeze the env dim, stack across the input list, and return a tensor
      of shape ``(N, ...)``.
    - At ``dict`` nodes: recurse key-by-key (every input must have the same keys).
    - Everything else (None, scalars): returned as-is from the first input.
    """
    head = items[0]
    if isinstance(head, dict):
        out: dict = {}
        for key in head.keys():
            out[key] = _stack_at_leaves([entry[key] for entry in items])
        return out
    if isinstance(head, torch.Tensor):
        pieces = []
        for entry in items:
            pieces.append(entry.squeeze(0) if entry.shape[0] == 1 else entry[0])
        return torch.stack(pieces, dim=0)
    return head


def _stack_state_dicts(state_dicts: list[dict], device) -> dict:
    """Stack per-env recorded states (nested dict of arrays) along dim 0.

    Each input is the same shape as ``ep["initial_state"]`` (or one entry of
    ``ep["states"]``): nested dict whose leaves are arrays of shape
    ``(1, ...)``. Returns the same nesting with tensor leaves of shape
    ``(N, ...)`` on ``device``.
    """
    tensorised = [_tensorize_state(s, device) for s in state_dicts]
    return _stack_at_leaves(tensorised)


def _pick_multi_asset_binding(item, entity_cfg, catalog_id):
    spawn_cfg = entity_cfg.spawn
    assets = list(getattr(spawn_cfg, "assets_cfg", []))
    asset_idx = int(item.get("asset_idx", -1))
    if asset_idx < 0 or asset_idx >= len(assets):
        raise ValueError(
            f"[Converter] multi_assets asset_idx out of range for {catalog_id!r}: "
            f"{asset_idx} not in [0, {len(assets) - 1}]"
        )
    picked = assets[asset_idx]
    spawn_cfg.assets_cfg = [picked] + assets[:asset_idx] + assets[asset_idx + 1 :]
    spawn_cfg.random_choice = False


def _pick_multi_usd_binding(item, entity_cfg, catalog_id):
    """Resolve a recorded multi-USD binding to an existing path in the env's pool.

    Matching strategy, in order:
      1. Exact path match (same-machine reruns).
      2. Basename match (e.g. ``model_basket_22.usd`` — recordings made on
         another machine carry a foreign parent directory, but the leaf USD
         filename is stable; the env's local pool has the same leaves under
         the correct local parent path).
      3. ``usd_idx`` fallback (position in the pool).
    """
    spawn_cfg = entity_cfg.spawn
    usd_pool = getattr(spawn_cfg, "usd_path", None)
    usd_paths = [usd_pool] if isinstance(usd_pool, str) else list(usd_pool or [])
    if not usd_paths:
        return
    selected_path = str(item.get("usd_path", ""))
    picked_path: str | None = None
    if selected_path:
        if selected_path in usd_paths:
            picked_path = selected_path
        else:
            target_name = os.path.basename(selected_path)
            for p in usd_paths:
                if os.path.basename(p) == target_name:
                    picked_path = p
                    break
    if picked_path is None:
        usd_idx = int(item.get("usd_idx", -1))
        if usd_idx < 0 or usd_idx >= len(usd_paths):
            raise ValueError(
                f"[Converter] multi_usds binding for {catalog_id!r}: cannot resolve "
                f"recorded usd_path {selected_path!r} against this env's pool "
                f"(no exact or basename match), and usd_idx={usd_idx} is out of "
                f"range [0, {len(usd_paths) - 1}]."
            )
        picked_path = usd_paths[usd_idx]
    spawn_cfg.usd_path = [picked_path] + [p for p in usd_paths if p != picked_path]
    spawn_cfg.random_choice = False


# ============================================================================
# Discovery
# ============================================================================


class PickleGroup(NamedTuple):
    """A set of source pickles to merge into a single output H5.

    ``--file <path>`` selections produce a one-pickle group named after the
    pickle (legacy behavior). ``--task`` / ``--all`` selections collapse all
    ``*.pkl`` files in each task directory into a single group, named after
    the directory — so multiple recording sessions (e.g. demos collected on
    different days) merge into one demonstration H5.
    """

    label: str  # display name in logs (typically the task / pickle name)
    output_stem: str  # H5 filename stem, before the ``.[preset].demo.h5`` suffix
    anchor_dir: Path  # directory the group "belongs to"; used for output mirroring
    pickles: list[Path]  # source pickles in load order


def _discover_source_pickle_groups() -> list[PickleGroup]:
    """Resolve --all / --task / --file flags into pickle groups for conversion.

    ``--file`` is literal: each file becomes a one-pickle group whose H5 is
    named after the pickle. ``--task <name>`` and ``--all`` walk task
    directories under ``--demos-root`` (recursively) and group every
    ``*.pkl`` in each directory into a single merged demonstration; the
    H5 is named after the task directory.
    """
    groups: list[PickleGroup] = []
    demos_root = args_cli.demos_root.expanduser().resolve()

    for f in args_cli.file:
        p = Path(f).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--file path does not exist: {p}")
        groups.append(
            PickleGroup(
                label=p.name,
                output_stem=p.stem,
                anchor_dir=p.parent,
                pickles=[p],
            )
        )

    # Collect task directories — any directory under ``demos_root`` that
    # contains at least one pickle. ``--task <name>`` may point at the env
    # directory itself OR a parent (e.g. a category like ``grasping/``), so
    # we ``rglob`` and bucket by parent.
    task_dirs: set[Path] = set()
    for task in args_cli.task:
        sub = (demos_root / task).resolve()
        if not sub.is_dir():
            raise FileNotFoundError(f"--task directory not found under {demos_root}: {task}")
        for pkl in sub.rglob("*.pkl"):
            task_dirs.add(pkl.parent.resolve())

    if args_cli.all:
        if not demos_root.is_dir():
            raise FileNotFoundError(f"--demos-root not found: {demos_root}")
        for pkl in demos_root.rglob("*.pkl"):
            task_dirs.add(pkl.parent.resolve())

    for task_dir in sorted(task_dirs):
        pkls = sorted(p.resolve() for p in task_dir.glob("*.pkl"))
        if not pkls:
            continue
        groups.append(
            PickleGroup(
                label=task_dir.name,
                output_stem=task_dir.name,
                anchor_dir=task_dir,
                pickles=pkls,
            )
        )

    if not groups:
        raise SystemExit("Nothing selected. Pass --all, --task <name> (repeatable), or --file <path> (repeatable).")
    return groups


def _merge_trajectory_payloads(pickles: list[Path]) -> dict:
    """Concatenate the ``episodes`` lists from multiple single-env pickles.

    The merged payload inherits metadata (``env_name``, ``robot_type``,
    ``json_path``, ``format``, …) from the first pickle. All other pickles
    must agree on ``env_name`` / ``task``; mismatch raises. Episode indices
    are renumbered sequentially across the merged set so each ``demo_<i>``
    in the output H5 is unique.
    """
    if not pickles:
        raise ValueError("no pickles to merge")
    base = _load_trajectory_pickle(str(pickles[0]))
    base_env = base.get("env_name") or base.get("task")
    merged_episodes = list(base.get("episodes") or [])
    for extra in pickles[1:]:
        payload = _load_trajectory_pickle(str(extra))
        env = payload.get("env_name") or payload.get("task")
        if env != base_env:
            raise ValueError(
                f"env_name mismatch when merging {extra.name}: got {env!r}, "
                f"expected {base_env!r} (from {pickles[0].name})."
            )
        merged_episodes.extend(payload.get("episodes") or [])
    for i, ep in enumerate(merged_episodes):
        ep["episode_index"] = i
    merged = dict(base)
    merged["episodes"] = merged_episodes
    return merged


def _demo_h5_suffix() -> str:
    """Return the ``.h5`` suffix, prefixed with the active obs preset (if any).

    With a preset (e.g. ``rgb``), the suffix is ``.rgb.demo.h5`` so output
    files from different obs modes don't overwrite each other when the same
    pickle is converted multiple times.
    """
    return f".{_obs_preset_arg}.demo.h5" if _obs_preset_arg else ".demo.h5"


def _output_path_for_group(group: PickleGroup) -> Path:
    """H5 path for a pickle group. Mirrors ``--output-dir`` layout if given."""
    suffix = _demo_h5_suffix()
    out_name = group.output_stem + suffix
    if args_cli.output_dir is None:
        return group.anchor_dir / out_name
    demos_root = args_cli.demos_root.expanduser().resolve()
    out_root = args_cli.output_dir.expanduser().resolve()
    try:
        rel = group.anchor_dir.resolve().relative_to(demos_root)
    except ValueError:
        rel = Path(group.anchor_dir.name)
    return out_root / rel / out_name


# ============================================================================
# Observation capture + encoding
# ============================================================================


class ObsGroupCapture:
    """Compute one or more observation groups per call, with image dtype encoding.

    Each ``capture(env_id)`` returns a flat dict ``{<group>/<term>: ndarray}``
    for that env slice with images cast to the configured storage dtypes.
    """

    def __init__(self, env, group_names: list[str] | None, rgb_dtype: str, depth_dtype: str):
        self._manager = getattr(env, "observation_manager", None)
        if self._manager is None:
            raise RuntimeError("Environment has no observation_manager; cannot capture obs groups.")

        active = getattr(self._manager, "active_terms", {})
        if group_names is None:
            # Default: capture every active observation group on this env so
            # the resulting HDF5 contains a superset; downstream consumers
            # subscribe to whatever subset they need.
            group_names = sorted(active.keys())
        missing = [g for g in group_names if g not in active]
        if missing:
            available = sorted(active.keys())
            raise ValueError(f"Requested obs groups {missing} not present on env (available: {available}).")
        self._group_names = list(group_names)
        self._rgb_np_dtype = np.uint8 if rgb_dtype == "uint8" else np.float32
        self._depth_np_dtype = np.float16 if depth_dtype == "float16" else np.float32

        # Per-group term names (for splitting a concatenated group tensor back into terms).
        self._term_names = {g: list(active[g]) for g in self._group_names}
        self._term_dims = {g: list(self._manager.group_obs_term_dim[g]) for g in self._group_names}

    @property
    def group_names(self) -> list[str]:
        return list(self._group_names)

    def capture(self, env_id: int) -> dict[str, np.ndarray]:
        flat: dict[str, np.ndarray] = {}
        for group in self._group_names:
            group_obs = self._manager.compute_group(group)
            term_obs = self._decode_group(group, group_obs)
            for term_name, tensor in term_obs.items():
                value = tensor[env_id].detach().cpu().numpy()
                flat[f"{group}/{term_name}"] = self._encode(term_name, value)
        return flat

    def _decode_group(self, group: str, group_obs) -> dict[str, torch.Tensor]:
        if isinstance(group_obs, dict):
            return group_obs
        term_names = self._term_names[group]
        if len(term_names) == 1:
            return {term_names[0]: group_obs}
        term_dims = self._term_dims[group]
        split_sizes = [int(np.prod(dim)) for dim in term_dims]
        split_tensors = torch.split(group_obs, split_sizes, dim=-1)
        decoded: dict[str, torch.Tensor] = {}
        for name, dim, tensor in zip(term_names, term_dims, split_tensors):
            decoded[name] = tensor.reshape(tensor.shape[0], *dim)
        return decoded

    def _encode(self, term_name: str, value: np.ndarray) -> np.ndarray:
        name = term_name.lower()
        if "depth" in name or "distance" in name:
            return np.ascontiguousarray(value.astype(self._depth_np_dtype, copy=False))
        if "rgb" in name or "image" in name:
            if self._rgb_np_dtype == np.uint8:
                if value.dtype == np.uint8:
                    out = value
                elif np.issubdtype(value.dtype, np.floating):
                    # Treat <=1.0 floats as normalized; everything else as 0..255 already.
                    vmax = float(value.max()) if value.size else 0.0
                    scaled = value * 255.0 if vmax <= 1.0 + 1e-3 else value
                    out = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
                else:
                    out = np.clip(value, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(out)
            return np.ascontiguousarray(value.astype(np.float32, copy=False))
        return np.ascontiguousarray(value)


# ============================================================================
# Video recorder
# ============================================================================


class VideoRecorder:
    """Render one MP4 per (episode, env_id) by sampling a scene Camera sensor.

    Frames are captured by calling :py:meth:`capture` at each timestep after
    observations are available, then flushed as an MP4 on
    :py:meth:`finalize_episode`. The recorder gracefully no-ops if the named
    camera is not in the scene or if ``imageio`` is missing — it prints a
    warning and disables itself instead of aborting the whole conversion.
    """

    def __init__(self, env, *, camera_name: str, fps: int, output_dir: Path):
        self._enabled = False
        self._writer_factory = None
        self._camera = None
        self._fps = int(fps)
        self._output_dir = Path(output_dir)
        self._active: dict[int, dict] = {}

        camera = getattr(env, "scene", None)
        camera = camera[camera_name] if camera is not None and camera_name in camera.keys() else None
        if camera is None:
            print(
                f"[VideoRecorder] Camera {camera_name!r} not found in scene "
                f"(available: {sorted(env.scene.keys())}). Video recording disabled."
            )
            return

        try:
            import imageio  # noqa: F401

            try:
                import imageio_ffmpeg  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "imageio_ffmpeg is required for MP4 video output. Install with `pip install imageio-ffmpeg`."
                ) from exc
        except ImportError as exc:
            print(f"[VideoRecorder] {exc}. Video recording disabled.")
            return

        import imageio as _imageio

        self._writer_factory = _imageio.get_writer
        self._camera = camera
        self._camera_name = camera_name
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_episode(self, env_id: int, episode_index: int, episode_name: str) -> None:
        if not self._enabled:
            return
        path = self._output_dir / f"episode_{episode_index:05d}__env_{env_id}.mp4"
        # Lazy-init the writer on the first frame so we know the frame size.
        self._active[env_id] = {
            "writer": None,
            "path": path,
            "episode_index": episode_index,
            "episode_name": episode_name,
        }

    def capture(self, env_ids: list[int]) -> None:
        if not self._enabled or not env_ids:
            return
        rgb = self._read_rgb_tensor()
        if rgb is None:
            return
        for env_id in env_ids:
            buf = self._active.get(env_id)
            if buf is None:
                continue
            frame = rgb[env_id].detach().cpu().numpy()
            frame = self._normalize_frame(frame)
            if buf["writer"] is None:
                buf["writer"] = self._writer_factory(
                    str(buf["path"]),
                    fps=self._fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                )
            buf["writer"].append_data(frame)

    def finalize_episode(self, env_id: int) -> None:
        if not self._enabled:
            return
        buf = self._active.pop(env_id, None)
        if buf is None:
            return
        writer = buf["writer"]
        if writer is None:
            return
        try:
            writer.close()
            print(f"  -> wrote {buf['path']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[VideoRecorder] failed to close {buf['path']}: {exc}")

    def finalize_all(self) -> None:
        for env_id in list(self._active.keys()):
            self.finalize_episode(env_id)

    def _read_rgb_tensor(self):
        data = getattr(self._camera, "data", None)
        if data is None:
            return None
        output = getattr(data, "output", None)
        if output is None:
            return None
        rgb = output.get("rgb")
        if rgb is None:
            print(
                f"[VideoRecorder] Camera {self._camera_name!r} has no 'rgb' output; disabling further capture this run."
            )
            self._enabled = False
            return None
        return rgb

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame[..., :3]
        if np.issubdtype(frame.dtype, np.floating):
            vmax = float(frame.max()) if frame.size else 0.0
            scaled = frame * 255.0 if vmax <= 1.0 + 1e-3 else frame
            return np.clip(scaled, 0.0, 255.0).astype(np.uint8)[..., :3]
        return np.clip(frame, 0, 255).astype(np.uint8)[..., :3]


# ============================================================================
# HDF5 writer
# ============================================================================


class PerPickleH5Writer:
    """One HDF5 file per pickle group; one ``data/demo_<i>`` group per episode.

    Layout::

        / (attrs: task, source_pickles, schema_version, obs_groups,
                  rgb_dtype, depth_dtype, num_episodes)
          data/
            demo_<i>/ (attrs: episode_index, episode_name, success,
                              has_success_flag, num_samples)
              actions             (T, D) float32
              source_actions      (T, D) float32
              obs/<group>/<term>  (T, ...)
              next_obs/<group>/<term>
              initial_obs/<group>/<term>
              final_obs/<group>/<term>
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        output_file: Path,
        *,
        task_name: str,
        source_pickles: list[Path],
        obs_groups: list[str],
        rgb_dtype: str,
        depth_dtype: str,
        compression: str,
        compression_opts: int,
        observation_preset: str | None = None,
    ):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self._output_file = output_file

        if compression in ("none", ""):
            self._compression = None
            self._compression_opts = None
        elif compression == "lzf":
            self._compression = "lzf"
            self._compression_opts = None
        else:
            self._compression = "gzip"
            self._compression_opts = int(compression_opts)

        self._h5 = h5py.File(self._output_file, "w")
        self._data = self._h5.create_group("data")
        self._h5.attrs["task"] = str(task_name)
        self._h5.attrs["source_pickles"] = json.dumps([str(p.resolve()) for p in source_pickles])
        self._h5.attrs["schema_version"] = self.SCHEMA_VERSION
        self._h5.attrs["obs_groups"] = json.dumps(list(obs_groups))
        self._h5.attrs["rgb_dtype"] = rgb_dtype
        self._h5.attrs["depth_dtype"] = depth_dtype
        # ``observation_preset`` records the preset name used at recording
        # time (e.g. "rgb", "rgb_depth", "pointcloud", "state") so downstream
        # loaders can spawn the env with the same obs space. Empty string
        # means no preset was applied (env captured all active groups).
        self._h5.attrs["observation_preset"] = observation_preset or ""

        self._counter = 0

    def write_episode(
        self,
        *,
        episode_index: int,
        episode_name: str,
        success: bool | None,
        actions: list[np.ndarray],
        source_actions: list[np.ndarray],
        obs: dict[str, list[np.ndarray]],
        next_obs: dict[str, list[np.ndarray]],
        initial_obs: dict[str, np.ndarray],
        final_obs: dict[str, np.ndarray],
        terminations: dict[str, list[bool]] | None = None,
    ) -> None:
        group_name = f"demo_{self._counter}"
        self._counter += 1
        g = self._data.create_group(group_name)
        g.attrs["episode_index"] = int(episode_index)
        g.attrs["episode_name"] = str(episode_name)
        g.attrs["num_samples"] = len(actions)
        g.attrs["success"] = bool(success) if success is not None else False
        g.attrs["has_success_flag"] = success is not None

        if len(actions) > 0:
            self._dataset(g, "actions", np.stack(actions, axis=0).astype(np.float32, copy=False))
            self._dataset(
                g,
                "source_actions",
                np.stack(source_actions, axis=0).astype(np.float32, copy=False),
            )
            self._write_obs_dict(g.create_group("obs"), obs, stacked=True)
            self._write_obs_dict(g.create_group("next_obs"), next_obs, stacked=True)
            if terminations:
                term_group = g.create_group("terminations")
                for name, values in terminations.items():
                    if not values:
                        continue
                    self._dataset(
                        term_group,
                        name,
                        np.asarray(values, dtype=bool),
                    )
        self._write_obs_dict(g.create_group("initial_obs"), initial_obs, stacked=False)
        self._write_obs_dict(g.create_group("final_obs"), final_obs, stacked=False)

    def flush(self) -> None:
        self._h5.attrs["num_episodes"] = int(self._counter)
        path = self._output_file
        self._h5.flush()
        self._h5.close()
        print(f"  -> wrote {path} ({self._counter} episode{'s' if self._counter != 1 else ''})")

    def _dataset(self, group, name: str, array: np.ndarray) -> None:
        kwargs: dict = {}
        # Chunked / compressed datasets require every chunk dim > 0. Disable
        # chunking + compression when any non-leading axis is zero (e.g. an
        # obs term that returns shape (T, 0) — scene_vis side-effect terms).
        has_zero_inner_dim = array.ndim > 1 and any(d == 0 for d in array.shape[1:])
        if self._compression is not None and not has_zero_inner_dim:
            kwargs["compression"] = self._compression
            if self._compression_opts is not None:
                kwargs["compression_opts"] = self._compression_opts
            if array.ndim >= 1 and array.shape[0] > 0:
                kwargs["chunks"] = (min(64, array.shape[0]),) + array.shape[1:]
        group.create_dataset(name, data=array, **kwargs)

    def _write_obs_dict(self, parent, flat: dict, *, stacked: bool) -> None:
        for key, value in flat.items():
            if stacked:
                if not value:
                    continue
                array = np.stack(value, axis=0)
            else:
                array = np.asarray(value)
            # Skip zero-shape obs terms outright. ``scene_vis`` markers
            # (sphere goal, frame axes, zone visualizers, …) emit shape
            # ``(E, 0)`` because their job is the USD-render side effect,
            # not actually producing observations. Recording a 0-dim
            # dataset has no value and trips h5py's chunking guard.
            if array.size == 0:
                continue
            self._dataset(parent, key, array)


# ============================================================================
# Replay loop
# ============================================================================


def _resolve_num_envs(num_episodes: int) -> int:
    """Decide how many parallel envs to spawn for a pickle with ``num_episodes`` trajectories.

    ``--num-parallel-envs 0`` (default) auto-sets to ``num_episodes``. A
    positive integer is used verbatim and capped to ``num_episodes`` (since
    chunking across pickles is not yet supported — extra env slots would go
    unused and waste VRAM). A negative value falls back to 1.
    """
    requested = int(getattr(args_cli, "num_parallel_envs", 0) or 0)
    if requested <= 0:
        return max(1, num_episodes)
    return max(1, min(requested, num_episodes))


def _build_env_for_pickle(payload: dict, *, episodes: list):
    """Construct the runtime env that matches a trajectory pickle.

    ``episodes`` is the list of episodes to be replayed (post any
    ``--select-episodes`` filtering). The env is built with one slot per
    episode (or per ``--num-parallel-envs`` if set) so episode ``i`` plays
    in env_id ``i`` in parallel.
    """
    dataset_env_name = payload.get("env_name") or payload.get("task")
    env_name = dataset_env_name
    if args_cli.task_override is not None:
        env_name = args_cli.task_override.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not found in the pickle and was not overridden.")
    task_name = args_cli.task_override if args_cli.task_override is not None else env_name

    num_envs = _resolve_num_envs(len(episodes))
    # Truncate the chunk to the env count actually built. Caller passes back
    # the trimmed list so the main loop and writer agree on the active set.
    episodes_active = episodes[:num_envs]

    json_path = args_cli.json_path if args_cli.json_path is not None else payload.get("json_path")
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs, json_path=json_path)

    # We intentionally do NOT carry over `payload["usd_path"]` to the env
    # config. That field stores the path local to the recording machine and
    # often does not exist here. For single-asset envs the config's own
    # default is correct on this machine; for multi-asset/pool envs the
    # per-episode `multi_usds` binding (handled below) picks the right object
    # by basename out of the local pool.
    override_kwargs: dict = {}
    robot_type = payload.get("robot_type")
    if args_cli.robot_type_override is not None:
        robot_type = args_cli.robot_type_override
    # Apply robot_type if the user passed --robot-type-override (always) OR no
    # task override is in play (carry over the pickle's robot_type).
    if (
        robot_type is not None
        and hasattr(env_cfg, "robot_type")
        and (args_cli.task_override is None or args_cli.robot_type_override is not None)
    ):
        override_kwargs["robot_type"] = robot_type
    if override_kwargs:
        cfg_cls = type(env_cfg)
        env_cfg = cfg_cls(**override_kwargs)
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.scene.num_envs = num_envs
    env_cfg.env_name = env_name

    # Apply per-env multi-asset / multi-USD bindings so env_id ``i`` spawns
    # the variant recorded for episode ``i``. ``random_choice`` is set to
    # False so MultiAssetSpawnerCfg / MultiUsdFileCfg use ``[i]`` instead of
    # sampling.
    multi_asset_entities, multi_usd_entities = _collect_multi_spawn_entities(env_cfg.scene)
    if multi_asset_entities:
        per_ep_assets = [ep.get("multi_assets") for ep in episodes_active]
        _apply_per_env_episode_bindings(
            per_ep_bindings=per_ep_assets,
            entities=multi_asset_entities,
            kind="multi_assets",
            picker_per_env=_pick_multi_asset_binding_per_env,
        )
    if multi_usd_entities:
        per_ep_usds = [ep.get("multi_usds") for ep in episodes_active]
        _apply_per_env_episode_bindings(
            per_ep_bindings=per_ep_usds,
            entities=multi_usd_entities,
            kind="multi_usds",
            picker_per_env=_pick_multi_usd_binding_per_env,
        )

    env_cfg.recorders = {}
    # Capture original termination cfgs BEFORE wiping. We re-evaluate them in
    # the replay loop so the H5 carries a per-step boolean for each
    # termination (success, out-of-bound, etc.). The env's own termination
    # manager is left empty so the replay doesn't auto-reset envs that
    # would have ``done=True``.
    termination_cfgs: dict = {}
    for term_name in dir(env_cfg.terminations):
        if term_name.startswith("_"):
            continue
        term = getattr(env_cfg.terminations, term_name, None)
        if term is None or not isinstance(term, DoneTerm):
            continue
        # Skip time-out terms — they aren't a function of scene state, just
        # the episode clock, and replays already have a known T.
        if getattr(term, "time_out", False):
            continue
        termination_cfgs[term_name] = term
    env_cfg.terminations = {}

    # Apply observation preset (if any) BEFORE gym.make so the env is
    # constructed with the narrowed obs space + history stacking. The env-cfg
    # method also handles preset aliases (e.g. "rgbd" -> "rgb_depth").
    if _obs_preset_arg is not None and hasattr(env_cfg, "_apply_observation_preset"):
        env_cfg.observation_preset = _obs_preset_arg
        env_cfg._apply_observation_preset(_obs_preset_arg)

    env = gym.make(task_name, cfg=env_cfg).unwrapped
    return env, env_cfg, env_name, task_name, episodes_active, termination_cfgs


def _video_dir_for_group(group: PickleGroup, output_path: Path) -> Path:
    """Mirror the per-group video output under --video-dir (or alongside HDF5)."""
    stem = group.output_stem
    if args_cli.video_dir is None:
        return output_path.parent / "videos" / stem
    demos_root = args_cli.demos_root.expanduser().resolve()
    out_root = args_cli.video_dir.expanduser().resolve()
    try:
        rel = group.anchor_dir.resolve().relative_to(demos_root).parent
    except ValueError:
        rel = Path()
    return out_root / rel / stem


def _convert_one_group(group: PickleGroup) -> tuple[int, int, int]:  # noqa: C901
    """Convert one pickle group into a single H5. Returns ``(succ, fail, total)``."""
    output_path = _output_path_for_group(group)
    if output_path.is_file() and not args_cli.overwrite:
        print(f"[skip] {output_path} exists (pass --overwrite to replace).")
        return (0, 0, 0)

    if len(group.pickles) == 1:
        print(f"[load] {group.pickles[0]}")
    else:
        print(f"[load] {group.label}: merging {len(group.pickles)} pickle(s)")
        for p in group.pickles:
            print(f"  + {p.name}")

    payload = _merge_trajectory_payloads(group.pickles)

    episodes = payload["episodes"]
    if not episodes:
        print(f"  (no episodes in {group.label}; skipping)")
        return (0, 0, 0)
    if args_cli.select_episodes:
        wanted = set(args_cli.select_episodes)
        selected = [ep for ep in episodes if int(ep.get("episode_index", -1)) in wanted]
        if not selected:
            selected = [episodes[i] for i in args_cli.select_episodes if 0 <= i < len(episodes)]
        episodes = selected
        if not episodes:
            print("  (no episodes matched --select-episodes; skipping)")
            return (0, 0, 0)

    succeeded = 0
    failed = 0
    env = None
    video = None
    try:
        requested_count = len(episodes)
        env, env_cfg, env_name, task_name, episodes, termination_cfgs = _build_env_for_pickle(
            payload, episodes=episodes
        )
        num_envs = env.num_envs
        if num_envs < requested_count:
            print(f"  [info] replaying {num_envs}/{requested_count} episode(s) in parallel (--num-parallel-envs cap)")
        else:
            print(f"  [info] replaying {num_envs} episode(s) in parallel")

        capture = ObsGroupCapture(
            env,
            args_cli.obs_groups,
            rgb_dtype=args_cli.rgb_dtype,
            depth_dtype=args_cli.depth_dtype,
        )
        writer = PerPickleH5Writer(
            output_path,
            task_name=task_name,
            source_pickles=list(group.pickles),
            obs_groups=capture.group_names,
            rgb_dtype=args_cli.rgb_dtype,
            depth_dtype=args_cli.depth_dtype,
            compression=args_cli.compression,
            compression_opts=args_cli.compression_opts,
            observation_preset=_obs_preset_arg,
        )
        if args_cli.record_video:
            video = VideoRecorder(
                env,
                camera_name=args_cli.video_camera,
                fps=args_cli.video_fps,
                output_dir=_video_dir_for_group(group, output_path),
            )

        all_env_ids = torch.arange(num_envs, device=env.device)
        action_dim = int(env.action_space.shape[-1])
        idle_action = (
            env_cfg.idle_action.repeat(num_envs, 1)
            if hasattr(env_cfg, "idle_action")
            else torch.zeros((num_envs, action_dim), device=env.device)
        )

        # Per-env episode metadata + recorded data.
        ep_indices = [int(ep.get("episode_index", -1)) for ep in episodes]
        ep_names = [ep.get("episode_name", f"demo_{ep_indices[i]}") for i, ep in enumerate(episodes)]
        actions_np_per_env: list[np.ndarray] = [np.asarray(ep.get("actions", []), dtype=np.float32) for ep in episodes]
        # If the env's robot_type was overridden to one with a different action
        # layout (e.g. floating_shadow_right -> ur10e_shadow_right), translate
        # each per-env action stream to the new layout before the step loop.
        source_robot_type = payload.get("robot_type")
        target_robot_type = getattr(env_cfg, "robot_type", None)
        actions_np_per_env = _maybe_convert_actions(
            actions_np_per_env,
            source_robot_type=source_robot_type,
            target_robot_type=target_robot_type,
            env_cfg=env_cfg,
        )
        # When the articulation changes, the recorded robot state has a different
        # joint count and can't be written back to the new articulation. Skip it
        # so scene.reset_to uses the env's IK-derived default init joints.
        replay_state_skip_entities = _entities_to_skip_for_robot_type_change(source_robot_type, target_robot_type)
        if replay_state_skip_entities:
            print(
                f"  [convert] skipping entities {sorted(replay_state_skip_entities)} "
                "from recorded state (articulation changed)."
            )
        ep_lengths = [a.shape[0] for a in actions_np_per_env]
        if not ep_lengths:
            print("  (no episodes to replay; skipping)")
            return (0, 0, 0)
        T_max = max(ep_lengths)

        states_seq_per_env: list[list | None] = [None] * num_envs
        if args_cli.set_state:
            for i, ep in enumerate(episodes):
                seq = ep.get("states")
                if seq is None or len(seq) != ep_lengths[i] + 1:
                    raise ValueError(
                        "--set-state requires states len T+1 for episode "
                        f"{ep_indices[i]} (env_id {i}) in {group.label}."
                    )
                states_seq_per_env[i] = seq

        env.reset()

        with torch.inference_mode():
            # Reset every env to its per-episode initial state in a single shot.
            live_state = env.scene.get_state(is_relative=True)
            initial_state_stacked = _stack_state_dicts([ep["initial_state"] for ep in episodes], env.device)
            initial_state = _build_replay_state(
                initial_state_stacked,
                live_state,
                extra_skip_entity_names=replay_state_skip_entities,
            )
            env.reset_to(initial_state, all_env_ids, is_relative=True)
            for i, ep in enumerate(episodes):
                env_id_i = torch.tensor([i], device=env.device, dtype=torch.long)
                apply_recorded_active_masks(env, ep, env_id_i)
            if args_cli.set_state:
                _refresh_after_set_state(env)
            _get_runtime_obs(env)

            # Initial obs per env_id.
            initial_flat_per_env = [capture.capture(i) for i in range(num_envs)]
            term_keys = list(initial_flat_per_env[0].keys())

            obs_buf_per_env: list[dict[str, list[np.ndarray]]] = [{k: [] for k in term_keys} for _ in range(num_envs)]
            next_obs_buf_per_env: list[dict[str, list[np.ndarray]]] = [
                {k: [] for k in term_keys} for _ in range(num_envs)
            ]
            actions_buf_per_env: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
            source_actions_buf_per_env: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
            last_flat_per_env = list(initial_flat_per_env)
            # Per-step termination booleans: ``terminations_buf_per_env[i][name]``
            # gets one bool appended each step the env is active.
            terminations_buf_per_env: list[dict[str, list[bool]]] = [
                {name: [] for name in termination_cfgs} for _ in range(num_envs)
            ]
            # Class-based termination terms (``ManagerTermBase`` subclasses)
            # must be instantiated once with ``(cfg, env)`` and then called
            # per step; cache the instances by cfg id.
            termination_instance_cache: dict = {}

            if video is not None:
                for i in range(num_envs):
                    video.start_episode(i, ep_indices[i], str(ep_names[i]))
                video.capture(env_ids=list(range(num_envs)))

            # ``active[i]`` flips to False after env_id i has consumed all of
            # its recorded actions. The obs manager is still ticked for those
            # envs (we step the whole batch) but their post-step obs are not
            # captured into the per-env buffers, so the H5 episodes stay
            # truncated to T_i steps each.
            active = np.array([T_max > 0 and ep_lengths[i] > 0 for i in range(num_envs)], dtype=bool)

            step_bar = tqdm(
                range(T_max),
                desc=f"  replay {group.label}",
                unit="step",
                leave=False,
                total=T_max,
                disable=not _HAS_TQDM,
            )
            for step_idx in step_bar:
                if not simulation_app.is_running() or simulation_app.is_exiting():
                    break
                if not active.any():
                    break
                if _HAS_TQDM:
                    n_active = int(active.sum())
                    step_bar.set_postfix({"active": n_active, "of": num_envs})

                # Build batched actions. Inactive envs get the idle action so
                # their dynamics don't drift in a way that disrupts active envs.
                actions = idle_action.clone()
                for i in range(num_envs):
                    if not active[i]:
                        continue
                    if step_idx >= ep_lengths[i]:
                        continue
                    source_action_i = torch.as_tensor(
                        actions_np_per_env[i][step_idx],
                        device=env.device,
                        dtype=torch.float32,
                    )
                    if source_action_i.numel() != action_dim:
                        raise ValueError(
                            f"Recorded action dim {source_action_i.numel()} != env action "
                            f"dim {action_dim} in episode {ep_indices[i]} (env_id {i}) "
                            f"of {group.label}."
                        )
                    actions[i] = source_action_i

                if args_cli.set_state:
                    # Per-env state-set: stack from per-env step states.
                    per_env_step_states = []
                    for i in range(num_envs):
                        seq = states_seq_per_env[i]
                        if seq is not None and step_idx + 1 < len(seq):
                            per_env_step_states.append(seq[step_idx + 1])
                        else:
                            # Inactive / past-end env: reuse final state.
                            per_env_step_states.append(seq[-1] if seq else episodes[i]["initial_state"])
                    step_state = _build_replay_state(
                        _stack_state_dicts(per_env_step_states, env.device),
                        env.scene.get_state(is_relative=True),
                        extra_skip_entity_names=replay_state_skip_entities,
                    )
                    env.scene.reset_to(step_state, all_env_ids, is_relative=True)
                    _refresh_after_set_state(env)
                    _set_last_action(env, actions)
                    _get_runtime_obs(env, update_history=True)
                else:
                    env.step(actions)

                # Capture post-step obs only for envs still recording demos.
                active_ids_now = [i for i in range(num_envs) if active[i] and step_idx < ep_lengths[i]]
                for i in active_ids_now:
                    post_flat = capture.capture(i)
                    for k, v in last_flat_per_env[i].items():
                        obs_buf_per_env[i][k].append(v)
                    for k, v in post_flat.items():
                        next_obs_buf_per_env[i][k].append(v)
                    actions_buf_per_env[i].append(actions[i].detach().cpu().numpy().astype(np.float32))
                    source_actions_buf_per_env[i].append(actions[i].detach().cpu().numpy().astype(np.float32))
                    last_flat_per_env[i] = post_flat
                if video is not None and active_ids_now:
                    video.capture(env_ids=active_ids_now)

                # Per-step terminations: evaluate every recorded termination
                # function and append its bool for each active env_id. Inactive
                # envs don't get an entry, so the per-env arrays line up with
                # ``actions_buf_per_env`` (same T_i length). The helper
                # handles both plain-function and ``ManagerTermBase``-class
                # term shapes; class instances are cached across steps.
                if active_ids_now and termination_cfgs:
                    for term_name, term_cfg in termination_cfgs.items():
                        try:
                            term_value = _evaluate_termination_term(
                                term_cfg,
                                env,
                                termination_instance_cache,
                            )
                        except Exception as exc:  # noqa: BLE001
                            # A misconfigured termination shouldn't kill the
                            # whole replay; warn once per term and keep going
                            # with False as the recorded value.
                            if not getattr(term_cfg, "_warned", False):
                                print(f"  [warn] termination {term_name!r} failed: {exc}")
                                term_cfg._warned = True  # type: ignore[attr-defined]
                            term_value = None
                        for i in active_ids_now:
                            if term_value is None:
                                terminations_buf_per_env[i][term_name].append(False)
                            else:
                                terminations_buf_per_env[i][term_name].append(bool(term_value[i].item()))

                # Flip envs that just finished to inactive.
                for i in range(num_envs):
                    if step_idx + 1 >= ep_lengths[i]:
                        active[i] = False

            # Write each env's collected trajectory as one episode.
            for i in range(num_envs):
                ep_success = episodes[i].get("success")
                writer.write_episode(
                    episode_index=ep_indices[i],
                    episode_name=str(ep_names[i]),
                    success=ep_success,
                    actions=actions_buf_per_env[i],
                    source_actions=source_actions_buf_per_env[i],
                    obs=obs_buf_per_env[i],
                    next_obs=next_obs_buf_per_env[i],
                    initial_obs=initial_flat_per_env[i],
                    final_obs=last_flat_per_env[i],
                    terminations=terminations_buf_per_env[i],
                )
                if video is not None:
                    video.finalize_episode(i)
                if ep_success is True:
                    succeeded += 1
                elif ep_success is False:
                    failed += 1

        writer.flush()
    finally:
        if video is not None:
            try:
                video.finalize_all()
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] video.finalize_all() failed: {exc}")
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] env.close() failed: {exc}")
                _force_clear_sim_context()
        else:
            # Env build failed partway. Isaac Lab's SimulationContext is a
            # process-global singleton; if it was created before the failure
            # the next pickle's `gym.make` will raise "Simulation context
            # already exists. Cannot create a new one." Tear it down now.
            _force_clear_sim_context()

    total = len(episodes)
    unknown = total - succeeded - failed
    parts = [f"success {succeeded}/{total}"]
    if failed:
        parts.append(f"failed {failed}")
    if unknown:
        parts.append(f"no-flag {unknown}")
    print(f"  ({', '.join(parts)})")
    return (succeeded, failed, total)


def main() -> int:
    groups = _discover_source_pickle_groups()
    total_pickles = sum(len(g.pickles) for g in groups)
    if _obs_preset_arg is not None:
        print(
            f"Converting {len(groups)} task group(s) ({total_pickles} pickle(s)); "
            f"observation_preset={_obs_preset_arg!r}"
        )
    else:
        print(f"Converting {len(groups)} task group(s) ({total_pickles} pickle(s)); obs_groups={args_cli.obs_groups}")
    total_succeeded = 0
    total_failed = 0
    total_episodes = 0
    group_bar = tqdm(
        groups,
        desc="tasks",
        unit="task",
        total=len(groups),
        disable=not _HAS_TQDM,
    )
    for group in group_bar:
        if not simulation_app.is_running() or simulation_app.is_exiting():
            print("Simulation app exiting; stopping.")
            break
        try:
            succ, fail, tot = _convert_one_group(group)
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"[error] {group.label}: {exc}")
            traceback.print_exc()
            continue
        total_succeeded += succ
        total_failed += fail
        total_episodes += tot
        if _HAS_TQDM:
            group_bar.set_postfix({
                "success": total_succeeded,
                "failed": total_failed,
                "episodes": total_episodes,
            })
    unknown = total_episodes - total_succeeded - total_failed
    print(
        f"Done. Across {len(groups)} task group(s) ({total_pickles} pickle(s)): "
        f"success {total_succeeded}/{total_episodes}"
        + (f", failed {total_failed}" if total_failed else "")
        + (f", no-flag {unknown}" if unknown else "")
    )
    return 0


if __name__ == "__main__":
    main()
    simulation_app.close()
