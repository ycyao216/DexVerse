# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sequential, single-env sibling of :mod:`create_demo_files`.

Where :mod:`create_demo_files` replays N episodes in N parallel envs, this
script builds a single env (``num_envs=1``) and replays one episode at a
time in a Python loop. Before each episode it calls ``env.sim.reset()`` to
match the recorder's physics restart, then ``env.reset_to(...)``,
which routes through ``ManagerBasedEnv._reset_idx`` and therefore fires
the event manager's ``reset`` mode — so randomizers like
``reset_environment_background`` (HDRI) and ``reset_table_texture`` get
a fresh sample for every episode, instead of being baked once at the
start of a parallel run.

Output H5 layout is identical to :mod:`create_demo_files` so it works as
a drop-in replacement for downstream consumers (IL training pipelines,
``inspect_replay_h5.py``, ``render_demo_video.py``).

Trade-offs vs the parallel version:
  * Much slower (no parallelism over episodes); use this when the
    per-episode randomization matters more than wall-clock time.
  * Multi-asset / multi-USD envs are not supported here: the asset
    binding is baked at ``gym.make`` time, and a per-episode switch
    would require rebuilding the env (huge startup cost). A warning is
    printed and env_id 0 keeps whatever the first episode picked.

Example::

    python scripts/demo_tools/create_demo_files_sequential.py \
        --file source/dexverse/demonstrations/dexterous/Dexverse-FunctionalHammerStrike-v0/Dexverse-FunctionalHammerStrike-v0.pkl \
        --obs-groups rgb \
        --output-dir outputs/h5_sequential
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from demo_selection import discover_groups, normalize_demo_root, validate_selected_identity, worker_arguments, PickleGroup
from conversion_output import conversion_request, exit_conversion, request_json, validate_existing_output

import torch
from _active_object_masks import apply_recorded_active_masks
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Sequential single-env replayer for dexverse teleop pickles.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

# --- Discovery / selection -------------------------------------------------
parser.add_argument(
    "--demos-root",
    type=Path,
    default=Path(__file__).resolve().parents[2] / "source" / "dexverse" / "demonstrations",
    help="Versioned download directory or prepared release root (default: %(default)s).",
)
parser.add_argument(
    "--task",
    action="append",
    default=[],
    help="Versioned task ID, e.g. Dexverse-PushT-v0 or Dexverse-PushT-v1. Also accepts version/category/task paths. Repeatable.",
)
parser.add_argument(
    "--file",
    action="append",
    default=[],
    help="Explicit pickle path(s) to convert. Repeatable.",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Convert available curated baseline demos; report missing tasks. Never scans raw sessions by default.",
)
parser.add_argument("--version", choices=("all", "v0", "v1"), default="all",
                    help="Filter --all or check the version in --task (never substitute versions).")
parser.add_argument("--legacy-collections", action="store_true",
                    help="Explicit raw-directory discovery/merging instead of curated demos.pkl selection.")
parser.add_argument("--dry-run", action="store_true", help="List selected pickles without launching Isaac Sim or loading pickles.")
parser.add_argument("--worker-group", help=argparse.SUPPRESS)
parser.add_argument("--task-timeout", type=int, default=3600,
                    help="Timeout in seconds per isolated task when selecting multiple tasks (default: 3600).")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Output directory for the .h5 files. Defaults to the pickle's directory.",
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
        "Observation preset name (rgb / rgb_depth / pointcloud / state) "
        "or list of group names. Omitted = every active group."
    ),
)
parser.add_argument("--rgb-dtype", default="uint8", choices=["uint8", "float32"])
parser.add_argument("--depth-dtype", default="float16", choices=["float16", "float32"])

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
        "Replay by restoring recorded scene states each step (requires "
        "ep['states'] with length T+1) instead of stepping actions. On by "
        "default; pass --no-set-state to step the recorded actions through "
        "the simulator instead."
    ),
)
parser.add_argument(
    "--strip-cameras",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "For camera-free captures (e.g. --obs-groups state without --record-video) "
        "remove the scene's camera sensors before building the env, so no RTX frame "
        "is rendered per step. Pass --no-strip-cameras to keep them."
    ),
)

# --- Environment-level options -------------------------------------------
parser.add_argument("--task-override", default=None)
parser.add_argument("--robot-type-override", default=None)
parser.add_argument("--json-path", default=None)
parser.add_argument("--enable-pinocchio", action="store_true")
parser.add_argument("--seed", type=int, default=None, help="Replay reset/randomization seed; recorded state still takes precedence.")

# --- Video recording ------------------------------------------------------
parser.add_argument(
    "--record-video",
    action="store_true",
    help="Render one MP4 per episode into --video-dir from the named camera.",
)
parser.add_argument("--video-camera", default="third_person_camera")
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-dir", type=Path, default=None)
parser.add_argument("--video-only-camera", action="store_true", help="Keep only --video-camera (for low-cost video review with state observations).")
parser.add_argument("--video-size", type=int, default=None, help="Optional square resolution for the review camera.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()

args_cli.demos_root = normalize_demo_root(args_cli.demos_root)
if (args_cli.task or args_cli.all) and args_cli.task_override:
    parser.error("--task selects the exact replay task; use --file with --task-override for deliberate remapping.")
if args_cli.seed is not None and not 0 <= args_cli.seed < 2**32:
    parser.error("--seed must be in [0, 2**32)")
if any(i < 0 for i in args_cli.select_episodes) or len(set(args_cli.select_episodes)) != len(args_cli.select_episodes):
    parser.error("--select-episodes must contain distinct nonnegative indices")
try:
    if args_cli.worker_group:
        _worker = json.loads(args_cli.worker_group)
        _worker["anchor_dir"] = Path(_worker["anchor_dir"])
        _worker["pickles"] = [Path(p) for p in _worker["pickles"]]
        _selected_groups, _selection_warnings = [PickleGroup(**_worker)], []
    else:
        _selected_groups, _selection_warnings = discover_groups(
            args_cli.demos_root, tasks=args_cli.task, files=args_cli.file, all_tasks=args_cli.all,
            version=args_cli.version, legacy_collections=args_cli.legacy_collections,
        )
except (ValueError, FileNotFoundError) as exc:
    parser.error(str(exc))
for _warning in _selection_warnings:
    print(f"[coverage] {_warning}", flush=True)
if args_cli.dry_run:
    for _group in _selected_groups:
        print(f"{_group.label}: " + ", ".join(str(p) for p in _group.pickles))
    raise SystemExit(0)

if args_cli.task_timeout < 1:
    parser.error("--task-timeout must be positive")
if len(_selected_groups) > 1:
    # Isaac scene teardown/reinitialization can hang. Isolate each task process;
    # this parent never launches Kit, and still reports a nonzero status on failure.
    import subprocess

    _failures = []
    for _group in _selected_groups:
        print(f"[task worker] {_group.label}", flush=True)
        _command = [sys.executable, "-u", str(Path(__file__).resolve()),
                    *worker_arguments(_group, sys.argv[1:]), "--demos-root", str(args_cli.demos_root)]
        try:
            _completed = subprocess.run(_command, timeout=args_cli.task_timeout, check=False)
            if _completed.returncode:
                _failures.append(_group.label)
        except subprocess.TimeoutExpired:
            print(f"[error] {_group.label} exceeded --task-timeout", flush=True)
            _failures.append(_group.label)
    if _failures:
        print(f"Failed task groups: {_failures}", flush=True)
    raise SystemExit(1 if _failures else 0)

args_cli.headless = True

# --- Resolve --obs-groups -------------------------------------------------
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
        args_cli.obs_groups = None

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
_NEEDS_CAMERAS = bool(
    args_cli.record_video or _capture_all_groups or _preset_has_camera or (_obs_groups_lower & _CAMERA_OBS_GROUPS)
)
if _NEEDS_CAMERAS or not args_cli.strip_cameras:
    args_cli.enable_cameras = True
# Camera-free replay strips sensor configs before scene construction below.

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import inspect  # noqa: E402

import dexverse.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import numpy as np  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg, prune_stale_obs_refs, strip_camera_cfgs  # noqa: E402
from dexverse.teleop_utils.episode_task_state import restore_episode_task_state  # noqa: E402
from dexverse.benchmark import task_identity, validate_replay_identity  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.managers.manager_base import ManagerTermBase  # noqa: E402

try:
    from tqdm.auto import tqdm  # noqa: E402

    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


# ============================================================================
# Helpers — copied / trimmed from create_demo_files.py. Keep in sync if the
# pickle schema, env cfg, or H5 layout changes upstream.
# ============================================================================

_SINGLE_FORMAT = "dexverse_trajectory"
_BATCH_FORMAT = "dexverse_trajectory_batch"


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
            f"Pickle {path!r} is a batch trajectory ({_BATCH_FORMAT!r}); use replay_demos.py for merged bundles."
        )
    raise ValueError(
        f"File {path!r} does not contain a single-env dexverse trajectory pickle "
        f"(expected format={_SINGLE_FORMAT!r}; got {fmt!r})."
    )


def _tensorize_state(data, device):
    if isinstance(data, dict):
        return {k: _tensorize_state(v, device) for k, v in data.items()}
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data, device=device)
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


# Entities whose live values (config-default placement) should win over the
# recorded values — typically cosmetic scene props added after recording.
_ENV_MANAGED_ENTITY_PREFIXES: tuple[str, ...] = ("table",)


def _is_env_managed_entity(name: str) -> bool:
    return any(name == p or name.startswith(p + "_") for p in _ENV_MANAGED_ENTITY_PREFIXES)


def _build_replay_state(
    recorded,
    live: dict,
    *,
    extra_skip_entity_names: set[str] | None = None,
) -> dict:
    """Merge a recorded state onto a live post-reset state for ``scene.reset_to``."""
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


_ARTICULATION_MISMATCH_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("floating_shadow_right", "ur10e_shadow_right"),
})


def _entities_to_skip_for_robot_type_change(source_robot_type: str | None, target_robot_type: str | None) -> set[str]:
    if not source_robot_type or not target_robot_type:
        return set()
    if source_robot_type == target_robot_type:
        return set()
    if (source_robot_type, target_robot_type) in _ARTICULATION_MISMATCH_PAIRS:
        return {"robot"}
    return set()


def _maybe_convert_actions_one(
    actions: np.ndarray,
    source_robot_type: str | None,
    target_robot_type: str | None,
    env_cfg,
) -> np.ndarray:
    """Single-episode equivalent of create_demo_files._maybe_convert_actions."""
    if source_robot_type == target_robot_type or target_robot_type is None:
        return actions
    if (source_robot_type, target_robot_type) == ("floating_shadow_right", "ur10e_shadow_right"):
        from dexverse.robot_agents.shadow._floating_to_ur10e_actions import (
            DEFAULT_FLOATING_BASE_POS,
            DEFAULT_FLOATING_BASE_ROT,
            convert_floating_shadow_right_to_ur10e_actions,
        )

        ur10e_init = env_cfg.scene.robot.init_state
        return (
            convert_floating_shadow_right_to_ur10e_actions(
                actions,
                floating_base_pos=DEFAULT_FLOATING_BASE_POS,
                floating_base_rot=DEFAULT_FLOATING_BASE_ROT,
                ur10e_base_pos=tuple(ur10e_init.pos),
                ur10e_base_rot=tuple(ur10e_init.rot),
            )
            if actions.size
            else actions
        )
    raise ValueError(f"No action converter for {source_robot_type!r} -> {target_robot_type!r}.")


def _evaluate_termination_term(term_cfg, env, instance_cache: dict):
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


def _set_last_action(env, action: torch.Tensor) -> None:
    """Write a recorded action into the action manager's buffers.

    The ``--set-state`` path never calls ``env.step``, so the action manager's
    ``_action`` / ``_prev_action`` buffers stay at their zero reset value, and any
    observation term reading them -- notably ``mdp.last_action``, which is the
    whole ``policy`` group -- gets recorded as all-zeros for the entire episode.
    Policies trained on such a recording see zeros there but their own previous
    action at evaluation time, and fail. (DP3 ignores the ``policy`` group, which
    is why the point-cloud replays never surfaced this; the state-based diffusion
    policy consumes it.)

    The buffers are set directly rather than through ``process_action`` so no
    action term applies side effects to the scene we just restored.
    """
    manager = getattr(env, "action_manager", None)
    if manager is None:
        return
    manager._prev_action[:] = manager._action
    manager._action[:] = action.to(manager.device)


def _has_multi_asset_or_usd(scene_cfg) -> bool:
    """Spot the per-episode binding case so we can warn before silently breaking it."""
    for name in dir(scene_cfg):
        if name.startswith("_"):
            continue
        entity_cfg = getattr(scene_cfg, name, None)
        if entity_cfg is None or not hasattr(entity_cfg, "spawn"):
            continue
        spawn_cfg = getattr(entity_cfg, "spawn", None)
        if spawn_cfg is None:
            continue
        cls_name = type(spawn_cfg).__name__
        if "MultiAsset" in cls_name or "MultiUsd" in cls_name:
            return True
    return False


# ============================================================================
# Observation capture + HDF5 writer (same schema as create_demo_files.py)
# ============================================================================


class ObsGroupCapture:
    """Compute one or more observation groups per call, with image-dtype encoding."""

    def __init__(self, env, group_names, rgb_dtype: str, depth_dtype: str):
        self._manager = getattr(env, "observation_manager", None)
        if self._manager is None:
            raise RuntimeError("Environment has no observation_manager.")
        active = getattr(self._manager, "active_terms", {})
        if group_names is None:
            group_names = sorted(active.keys())
        missing = [g for g in group_names if g not in active]
        if missing:
            raise ValueError(f"Requested obs groups {missing} not on env (available: {sorted(active.keys())}).")
        self._group_names = list(group_names)
        self._rgb_np_dtype = np.uint8 if rgb_dtype == "uint8" else np.float32
        self._depth_np_dtype = np.float16 if depth_dtype == "float16" else np.float32
        self._term_names = {g: list(active[g]) for g in self._group_names}
        self._term_dims = {g: list(self._manager.group_obs_term_dim[g]) for g in self._group_names}

    @property
    def group_names(self) -> list[str]:
        return list(self._group_names)

    @property
    def term_names_by_group(self) -> dict[str, list[str]]:
        """Per-group term order as declared on the ObservationManager.

        This is the order a ``concatenate_terms=True`` group uses internally, so
        writers persist it and readers that re-concatenate the per-term datasets
        can reproduce the env's own layout instead of guessing.
        """
        return {g: list(self._term_names[g]) for g in self._group_names}

    def capture(self, env_id: int = 0) -> dict[str, np.ndarray]:
        flat: dict[str, np.ndarray] = {}
        for group in self._group_names:
            group_obs = self._manager.compute_group(group)
            term_obs = self._decode_group(group, group_obs)
            for term_name, tensor in term_obs.items():
                value = tensor[env_id].detach().cpu().numpy()
                flat[f"{group}/{term_name}"] = self._encode(term_name, value)
        return flat

    def _decode_group(self, group: str, group_obs):
        if isinstance(group_obs, dict):
            return group_obs
        term_names = self._term_names[group]
        if len(term_names) == 1:
            return {term_names[0]: group_obs}
        term_dims = self._term_dims[group]
        split_sizes = [int(np.prod(dim)) for dim in term_dims]
        split_tensors = torch.split(group_obs, split_sizes, dim=-1)
        return {name: t.reshape(t.shape[0], *dim) for name, dim, t in zip(term_names, term_dims, split_tensors)}

    def _encode(self, term_name: str, value: np.ndarray) -> np.ndarray:
        name = term_name.lower()
        if "depth" in name or "distance" in name:
            return np.ascontiguousarray(value.astype(self._depth_np_dtype, copy=False))
        if "rgb" in name or "image" in name:
            if self._rgb_np_dtype == np.uint8:
                if value.dtype == np.uint8:
                    out = value
                elif np.issubdtype(value.dtype, np.floating):
                    vmax = float(value.max()) if value.size else 0.0
                    scaled = value * 255.0 if vmax <= 1.0 + 1e-3 else value
                    out = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
                else:
                    out = np.clip(value, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(out)
            return np.ascontiguousarray(value.astype(np.float32, copy=False))
        return np.ascontiguousarray(value)


class PerPickleH5Writer:
    """Same on-disk schema as create_demo_files.PerPickleH5Writer."""

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
        term_order_by_group: dict[str, list[str]] | None = None,
        source_benchmark_revision: str | None = None,
        source_action_layout: dict | None = None,
        source_task: str | None = None,
        source_sim_device: str | None = None,
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
        self._h5.attrs["source_task"] = source_task or str(task_name)
        self._h5.attrs["source_sim_device"] = source_sim_device or "unspecified"
        self._h5.attrs["source_benchmark_revision"] = source_benchmark_revision or "unspecified"
        self._h5.attrs["replay_benchmark_revision"] = task_identity(task_name)["benchmark_revision"]
        self._h5.attrs["replay_task_version"] = task_identity(task_name)["task_version"]
        if source_action_layout is not None:
            self._h5.attrs["source_action_layout"] = json.dumps(source_action_layout)
        self._h5.attrs["source_pickles"] = json.dumps([str(p.resolve()) for p in source_pickles])
        self._h5.attrs["schema_version"] = self.SCHEMA_VERSION
        self._h5.attrs["obs_groups"] = json.dumps(list(obs_groups))
        self._h5.attrs["rgb_dtype"] = rgb_dtype
        self._h5.attrs["depth_dtype"] = depth_dtype
        self._h5.attrs["observation_preset"] = observation_preset or ""
        # Marker so readers can tell sequential outputs from parallel ones.
        self._h5.attrs["replay_mode"] = "sequential"
        # ObservationManager term order per group. Readers that rebuild a flat
        # observation vector from the per-term datasets (e.g.
        # scripts/diffusion/build_dataset.py) need this; alphabetical key order
        # silently produces a different layout than the live environment.
        if term_order_by_group:
            self._h5.attrs["term_order"] = json.dumps({g: list(terms) for g, terms in term_order_by_group.items()})
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
            self._dataset(g, "source_actions", np.stack(source_actions, axis=0).astype(np.float32, copy=False))
            self._write_obs_dict(g.create_group("obs"), obs, stacked=True)
            self._write_obs_dict(g.create_group("next_obs"), next_obs, stacked=True)
            if terminations:
                term_group = g.create_group("terminations")
                for name, values in terminations.items():
                    if not values:
                        continue
                    self._dataset(term_group, name, np.asarray(values, dtype=bool))
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
            if array.size == 0:
                continue
            self._dataset(parent, key, array)


# ============================================================================
# Optional MP4 video recorder
# ============================================================================


class VideoRecorder:
    """One MP4 per episode, sampled from a named scene Camera at each capture()."""

    def __init__(self, env, *, camera_name: str, fps: int, output_dir: Path):
        self._enabled = False
        self._writer = None
        self._fps = int(fps)
        self._output_dir = Path(output_dir)
        self._camera = None
        self._camera_name = camera_name
        self._current_path: Path | None = None

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
            import imageio_ffmpeg  # noqa: F401
        except ImportError as exc:
            print(f"[VideoRecorder] {exc}. Video recording disabled.")
            return
        import imageio as _imageio

        self._imageio = _imageio
        self._camera = camera
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_episode(self, episode_index: int) -> None:
        if not self._enabled:
            return
        self._current_path = self._output_dir / f"episode_{episode_index:05d}.mp4"
        self._writer = None  # lazy on first frame

    def capture(self) -> None:
        if not self._enabled or self._current_path is None:
            return
        rgb = self._read_rgb_tensor()
        if rgb is None:
            return
        frame = rgb[0].detach().cpu().numpy()
        frame = self._normalize_frame(frame)
        if self._writer is None:
            self._writer = self._imageio.get_writer(
                str(self._current_path),
                fps=self._fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
        self._writer.append_data(frame)

    def finalize_episode(self) -> None:
        if not self._enabled or self._writer is None:
            self._writer = None
            self._current_path = None
            return
        try:
            self._writer.close()
            print(f"  -> wrote {self._current_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[VideoRecorder] failed to close {self._current_path}: {exc}")
        self._writer = None
        self._current_path = None

    def _read_rgb_tensor(self):
        data = getattr(self._camera, "data", None)
        if data is None:
            return None
        output = getattr(data, "output", None)
        if output is None:
            return None
        rgb = output.get("rgb")
        if rgb is None:
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
# Discovery
# ============================================================================


def _discover_source_pickle_groups() -> list[PickleGroup]:
    return _selected_groups


def _merge_trajectory_payloads(pickles: list[Path]) -> dict:
    if not pickles:
        raise ValueError("no pickles to merge")
    base = _load_trajectory_pickle(str(pickles[0]))
    base_env = base.get("env_name") or base.get("task")
    merged_episodes = list(base.get("episodes") or [])
    for extra in pickles[1:]:
        payload = _load_trajectory_pickle(str(extra))
        for key in ("robot_type", "benchmark_revision", "action_layout", "sim_device"):
            if payload.get(key) != base.get(key):
                raise ValueError(f"{key} mismatch when merging {extra.name} with {pickles[0].name}")
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
    merged["num_episodes"] = len(merged_episodes)
    return merged


def _demo_h5_suffix() -> str:
    # Tag sequential outputs so they don't collide with parallel outputs.
    base = f".{_obs_preset_arg}" if _obs_preset_arg else ""
    return f"{base}.seq.demo.h5"


def _output_path_for_group(group: PickleGroup) -> Path:
    out_name = group.output_stem + _demo_h5_suffix()
    if args_cli.output_dir is None:
        return group.anchor_dir / out_name
    demos_root = args_cli.demos_root.expanduser().resolve()
    out_root = args_cli.output_dir.expanduser().resolve()
    try:
        rel = group.anchor_dir.resolve().relative_to(demos_root)
    except ValueError:
        rel = Path(group.anchor_dir.name)
    return out_root / rel / out_name


def _video_dir_for_group(group: PickleGroup, output_path: Path) -> Path:
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


# ============================================================================
# Env build (single env)
# ============================================================================


def _build_env_for_pickle(payload: dict):
    """Construct a single-env runtime that matches the pickle's task/robot.

    Multi-asset / multi-USD spawns are *not* rebound here. With one env there
    is one slot, and rebuilding the env per episode would dominate runtime.
    We warn and let the spawner pick its default (typically asset 0).
    """
    dataset_env_name = payload.get("env_name") or payload.get("task")
    env_name = dataset_env_name
    if args_cli.task_override is not None:
        env_name = args_cli.task_override.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not found in the pickle and was not overridden.")
    task_name = args_cli.task_override if args_cli.task_override is not None else env_name
    validate_replay_identity(payload, env_name, explicit_override=args_cli.task_override is not None)

    json_path = args_cli.json_path if args_cli.json_path is not None else payload.get("json_path")
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1, json_path=json_path)

    override_kwargs: dict = {}
    robot_type = payload.get("robot_type")
    if args_cli.robot_type_override is not None:
        robot_type = args_cli.robot_type_override
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
        env_cfg.scene.num_envs = 1
    env_cfg.env_name = env_name
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    if _has_multi_asset_or_usd(env_cfg.scene):
        print(
            "  [warn] this env uses multi-asset / multi-USD spawns. Sequential mode "
            "cannot rebind per episode without rebuilding the env each time, so "
            "every episode will reuse the spawner's default selection. Use the "
            "parallel create_demo_files.py if per-episode asset variety matters."
        )

    env_cfg.recorders = {}

    # Pull termination cfgs out so we can re-evaluate them per step without
    # the env's TerminationManager auto-resetting.
    termination_cfgs: dict = {}
    for term_name in dir(env_cfg.terminations):
        if term_name.startswith("_"):
            continue
        term = getattr(env_cfg.terminations, term_name, None)
        if term is None or not isinstance(term, DoneTerm):
            continue
        if getattr(term, "time_out", False):
            continue
        termination_cfgs[term_name] = term
    env_cfg.terminations = {}

    if _obs_preset_arg is not None and hasattr(env_cfg, "_apply_observation_preset"):
        env_cfg.observation_preset = _obs_preset_arg
        env_cfg._apply_observation_preset(_obs_preset_arg)

    if args_cli.record_video and (args_cli.video_only_camera or args_cli.video_size is not None):
        from isaaclab.sensors import CameraCfg

        selected_camera = getattr(env_cfg.scene, args_cli.video_camera, None)
        if not isinstance(selected_camera, CameraCfg):
            raise ValueError(f"Video camera {args_cli.video_camera!r} is not configured")
        if args_cli.video_size is not None:
            if args_cli.video_size < 16 or args_cli.video_size % 2:
                raise ValueError("--video-size must be an even integer >= 16")
            selected_camera.width = selected_camera.height = args_cli.video_size
        if args_cli.video_only_camera:
            for name in dir(env_cfg.scene):
                if name != args_cli.video_camera and isinstance(getattr(env_cfg.scene, name, None), CameraCfg):
                    setattr(env_cfg.scene, name, None)
            prune_stale_obs_refs(env_cfg)

    if _NEEDS_CAMERAS or not args_cli.strip_cameras:
        # IsaacLab default is 0 — without it, the RTX camera buffer is stale after
        # reset, so per-episode lighting/texture randomizers won't appear in the
        # first captured frame. In sequential mode every episode is a fresh reset,
        # so we always want a few extra renders to let RTX settle.
        env_cfg.num_rerenders_on_reset = 4
    else:
        # Camera-free capture (e.g. the ``state`` preset): drop the scene's camera
        # sensors and any obs term that referenced them. Nothing reads the frames,
        # and without RTX sensors the per-step refresh no longer renders (nor
        # re-renders 4x per set-state step). Physics, joint and object state are
        # unaffected. ``--no-strip-cameras`` keeps the sensors.
        strip_camera_cfgs(env_cfg)
        prune_stale_obs_refs(env_cfg)
        env_cfg.num_rerenders_on_reset = 0
        print("  [info] camera-free capture: scene camera sensors stripped (--no-strip-cameras keeps them).")

    from dexverse.replay_rigid_object import configure_replay_rigid_objects

    configure_replay_rigid_objects(env_cfg.scene)
    env = gym.make(task_name, cfg=env_cfg).unwrapped
    return env, env_cfg, env_name, task_name, termination_cfgs


# ============================================================================
# Per-episode replay
# ============================================================================


def _replay_one_episode(
    env,
    env_cfg,
    *,
    episode: dict,
    capture: ObsGroupCapture,
    writer: PerPickleH5Writer,
    video: VideoRecorder | None,
    termination_cfgs: dict,
    payload: dict,
    label: str,
) -> bool | None:
    """Replay one episode and append it to the writer. Returns the recording label."""
    ep_index = int(episode.get("episode_index", -1))
    ep_name = episode.get("episode_name", f"demo_{ep_index}")
    ep_success = episode.get("success")

    actions_np = np.asarray(episode.get("actions", []), dtype=np.float32)
    source_robot_type = payload.get("robot_type")
    target_robot_type = getattr(env_cfg, "robot_type", None)
    actions_np = _maybe_convert_actions_one(
        actions_np,
        source_robot_type,
        target_robot_type,
        env_cfg,
    )
    T = actions_np.shape[0]
    action_dim = int(env.action_space.shape[-1])
    if T == 0:
        print(f"  [skip] episode {ep_index} has no actions.")
        return None

    states_seq = episode.get("states") if args_cli.set_state else None
    if args_cli.set_state and (states_seq is None or len(states_seq) != T + 1):
        raise ValueError(
            f"--set-state requires episode['states'] of length T+1={T+1} for episode {ep_index} in {label}."
        )

    skip = _entities_to_skip_for_robot_type_change(source_robot_type, target_robot_type)
    env_ids_one = torch.tensor([0], device=env.device, dtype=torch.long)

    with torch.inference_mode():
        # Match record_demos.handle_reset, including before the first episode.
        # Restoring poses/velocities alone does not reproduce the recorder's
        # physics initialization and can make CPU action replay diverge.
        env.sim.reset()
        # reset_to fires _reset_idx -> event_manager.apply(mode="reset"), so the
        # HDRI / table-texture randomizers resample for this episode. Then the
        # recorded scene state is laid on top.
        live_state = env.scene.get_state(is_relative=True)
        initial_state = _build_replay_state(
            _tensorize_state(episode["initial_state"], env.device),
            live_state,
            extra_skip_entity_names=skip,
        )
        env.reset_to(initial_state, env_ids_one, is_relative=True)
        apply_recorded_active_masks(env, episode, env_ids_one)
        skipped_task_state = restore_episode_task_state(
            env,
            episode.get("task_state"),
            env_index=0,
            legacy_goal_pose=episode.get("goal_pose"),
        )
        if skipped_task_state:
            print(f"  [warn] could not restore task state: {', '.join(skipped_task_state)}")
        if args_cli.set_state:
            _refresh_after_set_state(env)
        _get_runtime_obs(env)

        initial_flat = capture.capture(0)
        term_keys = list(initial_flat.keys())
        obs_buf: dict[str, list[np.ndarray]] = {k: [] for k in term_keys}
        next_obs_buf: dict[str, list[np.ndarray]] = {k: [] for k in term_keys}
        actions_buf: list[np.ndarray] = []
        source_actions_buf: list[np.ndarray] = []
        last_flat = initial_flat
        terminations_buf: dict[str, list[bool]] = {n: [] for n in termination_cfgs}
        termination_instance_cache: dict = {}
        # Reset pass: evaluate every termination term once while
        # ``episode_length_buf == 0`` (just after ``reset_to``) so env-resident,
        # step-keyed state -- stage-machine / cut-sweep *persistent* success
        # latches, hold counters -- is cleared before the episode is scored.
        # Without this, a success latched in the previous episode is reported
        # at step 0 of the next one (seen on CutStripScissors, Aug 2026).
        for _term_cfg in termination_cfgs.values():
            with contextlib.suppress(Exception):
                _evaluate_termination_term(_term_cfg, env, termination_instance_cache)

        if video is not None:
            video.start_episode(ep_index)
            video.capture()

        step_bar = tqdm(
            range(T),
            desc=f"  ep {ep_index}",
            unit="step",
            leave=False,
            total=T,
            disable=not _HAS_TQDM,
        )
        for step_idx in step_bar:
            if not simulation_app.is_running() or simulation_app.is_exiting():
                break

            action_t = torch.as_tensor(actions_np[step_idx], device=env.device, dtype=torch.float32)
            if action_t.numel() != action_dim:
                raise ValueError(
                    f"Recorded action dim {action_t.numel()} != env action dim "
                    f"{action_dim} in episode {ep_index} of {label}."
                )
            action_batched = action_t.unsqueeze(0)

            if args_cli.set_state:
                step_state = _build_replay_state(
                    _tensorize_state(states_seq[step_idx + 1], env.device),
                    env.scene.get_state(is_relative=True),
                    extra_skip_entity_names=skip,
                )
                env.scene.reset_to(step_state, env_ids_one, is_relative=True)
                _refresh_after_set_state(env)
                # Mirror ManagerBasedRLEnv.step: advance the per-env step counter.
                # Step-keyed runtime logic (stage-machine / cut-sweep reset detection
                # via ``episode_length_buf == 0``, per-step memoization, hold
                # counters) otherwise sees step 0 forever and never accumulates
                # progress -- e.g. the cut tasks' success term fired in 0/50
                # replayed demos before this (Aug 2026).
                if isinstance(getattr(env, "episode_length_buf", None), torch.Tensor):
                    env.episode_length_buf[env_ids_one] += 1
                _set_last_action(env, action_batched)
                _get_runtime_obs(env, update_history=True)
            else:
                env.step(action_batched)

            post_flat = capture.capture(0)
            for k, v in last_flat.items():
                obs_buf[k].append(v)
            for k, v in post_flat.items():
                next_obs_buf[k].append(v)
            action_np = action_batched[0].detach().cpu().numpy().astype(np.float32)
            actions_buf.append(action_np)
            source_actions_buf.append(action_np)
            last_flat = post_flat

            if video is not None:
                video.capture()

            # Debug trace of the cut-sweep internals (DEXVERSE_CUTSWEEP_TRACE=<csv path>).
            _trace_path = os.environ.get("DEXVERSE_CUTSWEEP_TRACE")
            if _trace_path and "success" in termination_cfgs:
                _tk = (termination_cfgs["success"].params or {}).get("task_key")
                if _tk:
                    from dexverse.baseline_v1.mdp.cut_sweep import evaluate_cut_sweep as _ecs

                    _o = _ecs(env, _tk)
                    _buf = getattr(env, "episode_length_buf", None)
                    from dexverse.baseline_v1.mdp.stage_machine import evaluate_stage_graph as _esg, get_stage_graph as _gsg

                    _fl = _esg(env, task_key=_tk, persistent=True, ordering_mode="strict")
                    _st = getattr(env, "_stage_graph_runtime_cache", {}).get(_tk)
                    _latched = getattr(_st, "latched_flags", None) or {}
                    _names = [st.name for st in _gsg(_tk).stages]
                    with open(_trace_path, "a") as _fp:
                        _fp.write(
                            f"{ep_index},{step_idx},{int(_buf[0]) if _buf is not None else -1},"
                            f"{float(_o['frontier_frac'][0]):.4f},{int(_o['ext_ok'][0])},{int(_o['gates_ok'][0])},"
                            + ",".join(f"{int(_fl[n][0])}" for n in _names) + ","
                            + ",".join(f"{int(_latched[n][0]) if n in _latched else -1}" for n in _names)
                            + f",{int(getattr(_st, 'latched_step_buf', torch.tensor([-1]))[0])}\n"
                        )
            if termination_cfgs:
                for term_name, term_cfg in termination_cfgs.items():
                    try:
                        term_value = _evaluate_termination_term(
                            term_cfg,
                            env,
                            termination_instance_cache,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if term_name == "success":
                            raise RuntimeError(
                                f"Cannot measure replay success for episode {ep_index}, step {step_idx}."
                            ) from exc
                        if not getattr(term_cfg, "_warned", False):
                            print(f"  [warn] termination {term_name!r} failed: {exc}")
                            term_cfg._warned = True  # type: ignore[attr-defined]
                        term_value = None
                    if term_value is None:
                        terminations_buf[term_name].append(False)
                    else:
                        terminations_buf[term_name].append(bool(term_value[0].item()))

        if len(actions_buf) != T:
            raise RuntimeError(f"Episode {ep_index} interrupted: captured {len(actions_buf)}/{T} steps")
        writer.write_episode(
            episode_index=ep_index,
            episode_name=str(ep_name),
            success=ep_success,
            actions=actions_buf,
            source_actions=source_actions_buf,
            obs=obs_buf,
            next_obs=next_obs_buf,
            initial_obs=initial_flat,
            final_obs=last_flat,
            terminations=terminations_buf,
        )
        if video is not None:
            video.finalize_episode()

    return ep_success


def _convert_one_group(group: PickleGroup) -> tuple[int, int, int]:
    output_path = _output_path_for_group(group)

    if len(group.pickles) == 1:
        print(f"[load] {group.pickles[0]}")
    else:
        print(f"[load] {group.label}: merging {len(group.pickles)} pickle(s)")
        for p in group.pickles:
            print(f"  + {p.name}")

    payload = _merge_trajectory_payloads(group.pickles)
    validate_selected_identity(payload, group.expected_task)
    episodes = payload["episodes"]
    if not episodes:
        raise ValueError(f"No episodes in {group.label}")
    if args_cli.select_episodes:
        wanted = set(args_cli.select_episodes)
        selected = [ep for ep in episodes if int(ep.get("episode_index", -1)) in wanted]
        if {int(ep.get("episode_index", -1)) for ep in selected} != wanted:
            raise ValueError(f"Some --select-episodes indices are absent from {group.label}: {sorted(wanted)}")
        episodes = selected

    task = (args_cli.task_override or payload.get("env_name") or payload.get("task") or "").split(":")[-1]
    validate_replay_identity(payload, task, explicit_override=args_cli.task_override is not None)
    option_names = (
        "task_override", "robot_type_override", "device", "seed", "set_state", "strip_cameras",
        "obs_groups", "rgb_dtype", "depth_dtype", "compression", "compression_opts",
        "record_video", "video_camera", "video_fps", "video_only_camera", "video_size", "rendering_mode",
    )
    options = {name: getattr(args_cli, name, None) for name in option_names}
    options["observation_preset"] = _obs_preset_arg
    json_path = args_cli.json_path if args_cli.json_path is not None else payload.get("json_path")
    if json_path is not None:
        from demo_release import sha256

        options["json_sha256"] = sha256(json_path)
    request = conversion_request(
        task=task, identity=task_identity(task), pickles=group.pickles, episodes=episodes, options=options,
    )
    if output_path.exists() and not args_cli.overwrite:
        validate_existing_output(output_path, request)
        # An H5 completion marker does not establish that external MP4s still
        # exist or finalized correctly. Require a deliberate rebuild for videos.
        if args_cli.record_video:
            raise ValueError(f"Video output reuse is not verified: {output_path}; use a new directory or --overwrite")
        print(f"[skip] {output_path}: matching sources, episodes and conversion settings.")
        return (0, 0, 0)

    succeeded = 0
    failed = 0
    env = None
    video = None
    writer = None
    try:
        env, env_cfg, env_name, task_name, termination_cfgs = _build_env_for_pickle(payload)
        print(f"  [info] sequential replay of {len(episodes)} episode(s) in {env.num_envs} env")

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
            term_order_by_group=capture.term_names_by_group,
            source_benchmark_revision=payload.get("benchmark_revision"),
            source_action_layout=payload.get("action_layout"),
            source_task=payload.get("env_name") or payload.get("task"),
            source_sim_device=payload.get("sim_device"),
        )
        writer._h5.attrs["set_state"] = bool(args_cli.set_state)
        writer._h5.attrs["replay_sim_device"] = str(env.device)
        writer._h5.attrs["replay_seed"] = int(env.cfg.seed)
        writer._h5.attrs["step_dt"] = float(env.step_dt)
        writer._h5.attrs["conversion_complete"] = False
        writer._h5.attrs["requested_episodes"] = len(episodes)
        writer._h5.attrs["conversion_request"] = request_json(request)
        if args_cli.record_video:
            video = VideoRecorder(
                env,
                camera_name=args_cli.video_camera,
                fps=args_cli.video_fps,
                output_dir=_video_dir_for_group(group, output_path),
            )
            if not video.enabled:
                raise RuntimeError("--record-video was requested but the video recorder could not start.")

        # Initial reset so the env reaches its post-startup steady state.
        env.reset()

        episode_bar = tqdm(
            episodes,
            desc=f"[{group.label}]",
            unit="ep",
            disable=not _HAS_TQDM,
            leave=True,
        )
        for ep in episode_bar:
            if not simulation_app.is_running() or simulation_app.is_exiting():
                break
            ep_success = _replay_one_episode(
                env,
                env_cfg,
                episode=ep,
                capture=capture,
                writer=writer,
                video=video,
                termination_cfgs=termination_cfgs,
                payload=payload,
                label=group.label,
            )
            if ep_success is True:
                succeeded += 1
            elif ep_success is False:
                failed += 1

        if writer._counter != len(episodes):
            raise RuntimeError(f"Conversion interrupted: wrote {writer._counter}/{len(episodes)} episodes")
        writer._h5.attrs["conversion_complete"] = True
        writer.flush()
    finally:
        if video is not None:
            video.finalize_episode()
        if writer is not None and writer._h5.id.valid:
            writer._h5.close()
        # This worker owns one scene. Release it at process exit after closing
        # files: env/Kit teardown can hang or terminate with an incorrect code.

    total = len(episodes)
    unknown = total - succeeded - failed
    parts = [f"recorded success {succeeded}/{total}"]
    if failed:
        parts.append(f"failed {failed}")
    if unknown:
        parts.append(f"unevaluated {unknown}")
    print(f"  ({', '.join(parts)})")
    return (succeeded, failed, total)


def main() -> int:
    groups = _discover_source_pickle_groups()
    total_pickles = sum(len(g.pickles) for g in groups)
    if _obs_preset_arg is not None:
        print(
            f"Converting {len(groups)} task group(s) ({total_pickles} pickle(s)); "
            f"obs preset = {_obs_preset_arg!r}; sequential mode."
        )
    else:
        print(f"Converting {len(groups)} task group(s) ({total_pickles} pickle(s)); sequential mode.")

    grand_succ = grand_fail = grand_total = 0
    for group in groups:
        if not simulation_app.is_running() or simulation_app.is_exiting():
            raise RuntimeError("Simulation app exited before conversion completed")
        s, f, t = _convert_one_group(group)
        grand_succ += s
        grand_fail += f
        grand_total += t

    print(
        f"\nDone. total={grand_total} recorded_success={grand_succ} failed={grand_fail} "
        f"unevaluated={grand_total - grand_succ - grand_fail}"
    )
    return 0


if __name__ == "__main__":
    exit_conversion(main)
