# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record demonstrations with DexVerse environments using VR teleoperation.

This script allows users to record demonstrations operated by VR hand tracking for a
specified task. Each session is stored as a single pickle file containing the task
name, optional object USD path / robot variant, the initial scene state captured
via ``env.scene.get_state(is_relative=True)`` (robot + object + any other scene
entity) and the per-step teleop actions. ``replay_demos.py`` can load this pickle
to reconstruct the environment, restore the initial state and replay the actions
to regenerate the corresponding observations.

Required arguments:
    --task                    Name of the task.

Optional arguments:
    -h, --help                Show this help message and exit
    --dataset_file            File path to export the trajectory pickle.
    --dataset_dir             Root directory to export the trajectory pickle. Output path will be
                              "<dataset_dir>/<task_name>/<task_name>_<time>.pkl".
    --record_state            Enable recording per-step scene states (T+1 snapshots per episode).
    --num_demos               Number of demonstrations to record. (default: 0, infinite)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
    --enable_pinocchio        Enable Pinocchio. Auto-enabled for handtracking / motion controllers
                              because dex-retargeting requires it.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import inspect
import logging
import os
import pickle
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Record demonstrations for DexVerse environments using VR.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Teleop device name. Should match the device key in environment config. Default: handtracking",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default=None,
    help="Path to export the trajectory pickle (must be a file path).",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=None,
    help="Root directory to export trajectory pickle as '<dataset_dir>/<task_name>/<task_name>_<time>.pkl'.",
)
parser.add_argument(
    "--record_state",
    action="store_true",
    default=True,
    help="Enable recording per-step scene states (T+1 snapshots per episode). Always enabled.",
)
parser.add_argument(
    "--num_demos", type=int, default=50, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio. Auto-enabled for handtracking / motion controllers because dex-retargeting requires it.",
)
parser.add_argument(
    "--json_path",
    type=str,
    default=None,
    help="Path to template JSON spec. Required for *Template environments.",
)
parser.add_argument(
    "--robot_type",
    type=str,
    default=None,
    help=(
        "Optional robot variant override for environments that expose 'robot_type' "
        "(floating_shadow_right, floating_shadow_left, floating_shadow_bimanual)."
    ),
)
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help=(
        "Optional object asset override for environments that expose 'usd_path' "
        "(e.g. pickup_object-based tasks). Accepts either a single .usd/.usda/.usdc "
        "file (single-object mode) or a directory of USDs (object-pool mode)."
    ),
)
parser.add_argument(
    "--enable_debug_vis",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Override the task's debug-visualization toggle (zone / reference-point "
        "markers). Use --enable_debug_vis to force on, --no-enable_debug_vis to "
        "force off; omit to use the task's default."
    ),
)
parser.add_argument(
    "--teleop_retargeter",
    type=str,
    default="relative",
    choices=("relative", "absolute"),
    help=(
        "Retargeter mode for VR hand-tracking. 'relative' (default) tracks"
        " displacement from the calibration pose; 'absolute' drives the robot"
        " wrist to the VR hand's world position (red-dot location)."
    ),
)
parser.add_argument(
    "--retargeting_scheme",
    type=str,
    default="dexpilot",
    choices=("dexpilot", "vector"),
    help=(
        "Dex-retargeting scheme for the fingers. 'dexpilot' (default) uses"
        " DexPilot pinch/wrist vectors; 'vector' matches wrist->fingertip"
        " vectors. Orthogonal to --teleop_retargeter. Available for every"
        " floating hand (each ships both a dexpilot and a vector config)."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()

# Support lightweight Hydra-style env override parity with teleop_agent.py.
# Examples: env.robot_type=floating_shadow_right, env.usd_path=/path/to/obj.usd,
#          env.teleop_retargeter=absolute
for raw_arg in unknown_args:
    if raw_arg.startswith("env.robot_type="):
        args_cli.robot_type = raw_arg.split("=", 1)[1]
    elif raw_arg.startswith("env.usd_path="):
        args_cli.usd_path = raw_arg.split("=", 1)[1]
    elif raw_arg.startswith("env.teleop_retargeter="):
        args_cli.teleop_retargeter = raw_arg.split("=", 1)[1]
    elif raw_arg.startswith("env.retargeting_scheme="):
        args_cli.retargeting_scheme = raw_arg.split("=", 1)[1]
    elif raw_arg.startswith("env.enable_debug_vis="):
        args_cli.enable_debug_vis = raw_arg.split("=", 1)[1].strip().lower() in ("1", "true", "yes")
    else:
        parser.error(f"unrecognized arguments: {raw_arg}")

if args_cli.task is None:
    parser.error("--task is required")


def _resolve_dataset_file_path(task_name: str, dataset_file: str | None, dataset_dir: str | None) -> str:
    """Resolve output pickle path from CLI options.

    Precedence:
    1) ``--dataset_file`` (must be a file path)
    2) ``--dataset_dir`` (always directory semantics as ``<dataset_dir>/<task_name>/<task_name>_<time>.pkl``)
    3) default ``datasets/<task_name>_<time>/trajectory.pkl``
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    auto_name = f"{task_name}_{timestamp}.pkl"

    if dataset_file:
        if dataset_file.endswith(("/", "\\")) or Path(dataset_file).is_dir():
            raise ValueError(
                "--dataset_file must be a file path, not a directory. "
                "Use --dataset_dir for directory-based auto naming."
            )
        output_path = Path(dataset_file)
    elif dataset_dir:
        output_path = Path(dataset_dir) / task_name / auto_name
    else:
        output_path = Path("datasets") / f"{task_name}_{timestamp}" / "trajectory.pkl"

    if output_path.suffix.lower() not in {".pkl", ".pickle"}:
        output_path = output_path.with_suffix(".pkl")
    return str(output_path)


env_name = args_cli.task.split(":")[-1]
try:
    args_cli.dataset_file = _resolve_dataset_file_path(env_name, args_cli.dataset_file, args_cli.dataset_dir)
except ValueError as exc:
    parser.error(str(exc))

app_launcher_args = vars(args_cli)

uses_xr_teleop = (
    "handtracking" in args_cli.teleop_device.lower() or "motion_controllers" in args_cli.teleop_device.lower()
)

if uses_xr_teleop and not args_cli.enable_pinocchio:
    logger.info("Auto-enabling Pinocchio for handtracking / motion-controller teleoperation.")
    args_cli.enable_pinocchio = True

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one
    # installed by Isaac Sim. Pinocchio is required by the Pink IK controllers and dex-retargeting utilities.
    import pinocchio  # noqa: F401

if uses_xr_teleop:
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dexverse.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from dexverse.tasks.config.floating_teleop import (  # noqa: E402
    apply_teleop_retargeter_mode,
    apply_teleop_retargeting_scheme,
)
from dexverse.tasks.utils import (  # noqa: E402
    parse_env_cfg,
    prune_stale_obs_refs,
    strip_camera_cfgs,
)
from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402


class TrajectoryPickleRecorder:
    """Save teleop demos as replayable trajectory pickles.

    Each recorded session is written to a single pickle file that contains the
    environment metadata (task name, object USD path, robot variant), the initial
    scene state captured via ``env.scene.get_state(is_relative=True)`` and the
    per-step teleop actions. ``replay_demos.py`` consumes this pickle to rebuild
    the environment, restore the initial robot + object state and replay the
    actions to regenerate observations.
    """

    SCHEMA_VERSION = 3
    FORMAT_TAG = "dexverse_trajectory"

    def __init__(
        self,
        output_file: str,
        *,
        task_name: str,
        env_name: str,
        record_state: bool = True,
        usd_path: str | None = None,
        robot_type: str | None = None,
        json_path: str | None = None,
    ):
        self._output_file = output_file
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        self._episodes: list[dict] = []
        self._active_episode: dict | None = None
        self._record_state = bool(record_state)
        self._metadata = {
            "format": self.FORMAT_TAG,
            "schema_version": self.SCHEMA_VERSION,
            "task": task_name,
            "env_name": env_name,
            "record_state": self._record_state,
            "usd_path": usd_path,
            "robot_type": robot_type,
            "json_path": json_path,
        }

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def has_active_episode(self) -> bool:
        return self._active_episode is not None

    def start_episode(
        self,
        initial_state,
        goal_pose=None,
        multi_assets=None,
        multi_usds=None,
        active_object_metadata=None,
    ) -> None:
        initial_state_pkl = self._to_pickleable(initial_state)
        self._active_episode = {
            "episode_index": len(self._episodes),
            "episode_name": f"demo_{len(self._episodes)}",
            "initial_state": initial_state_pkl,
            "goal_pose": self._to_pickleable(goal_pose) if goal_pose is not None else None,
            "actions": [],
            "success": None,
        }
        if self._record_state:
            # ``states`` always stores T+1 snapshots: initial state + one state per action step.
            self._active_episode["states"] = [initial_state_pkl]
        if multi_assets is not None:
            self._active_episode["multi_assets"] = self._to_pickleable(multi_assets)
        if multi_usds is not None:
            self._active_episode["multi_usds"] = self._to_pickleable(multi_usds)
        if active_object_metadata is not None:
            self._active_episode["active_object_metadata"] = self._to_pickleable(active_object_metadata)

    def record_action(self, action) -> None:
        if self._active_episode is None:
            return
        self._active_episode["actions"].append(self._to_pickleable(action))

    def record_state(self, state) -> None:
        if not self._record_state or self._active_episode is None:
            return
        states = self._active_episode.get("states")
        if not isinstance(states, list):
            return
        states.append(self._to_pickleable(state))

    def finalize_episode(self, success: bool = True) -> None:
        if self._active_episode is None:
            return
        actions = self._active_episode["actions"]
        if len(actions) > 0:
            stacked = np.stack([np.asarray(a).reshape(-1) for a in actions], axis=0)
            self._active_episode["actions"] = stacked.astype(np.float32, copy=False)
        else:
            self._active_episode["actions"] = np.zeros((0, 0), dtype=np.float32)
        self._active_episode["num_steps"] = int(self._active_episode["actions"].shape[0])
        if self._record_state:
            states = self._active_episode.get("states")
            if not isinstance(states, list):
                raise ValueError("Episode states buffer is missing or malformed.")
            expected_len = int(self._active_episode["num_steps"]) + 1
            if len(states) != expected_len:
                raise ValueError(
                    f"Episode states length mismatch: got {len(states)}, expected {expected_len} "
                    "(must be T+1: initial + per-step post-state)."
                )
        self._active_episode["success"] = bool(success)
        self._episodes.append(self._active_episode)
        self._active_episode = None
        self.flush()

    def discard_episode(self) -> None:
        self._active_episode = None

    def flush(self) -> None:
        payload = dict(self._metadata)
        payload["num_episodes"] = len(self._episodes)
        payload["episodes"] = self._episodes
        with open(self._output_file, "wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _to_pickleable(cls, data):
        if isinstance(data, dict):
            return {key: cls._to_pickleable(value) for key, value in data.items()}
        if isinstance(data, list):
            return [cls._to_pickleable(value) for value in data]
        if isinstance(data, tuple):
            return tuple(cls._to_pickleable(value) for value in data)
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().copy()
        if isinstance(data, np.ndarray):
            return np.array(data, copy=True)
        return data


def _get_goal_pose_from_env(env) -> torch.Tensor | None:
    """Read the current goal pose command, if the task defines one."""
    if not hasattr(env, "command_manager"):
        return None
    try:
        goal_pose = env.command_manager.get_command("object_pose")
    except (AttributeError, KeyError):
        return None
    return goal_pose[0].detach().clone()


def _get_active_object_metadata_from_env(env, env_index: int = 0) -> dict | None:
    getter = getattr(env, "get_active_object_metadata", None)
    if not callable(getter):
        return None
    try:
        return getter(env_index)
    except Exception as exc:
        print(f"[WARN] Failed to read active object metadata: {exc}")
        return None


def _unwrap_obs(reset_output):
    """Normalize IsaacLab reset output to the observation object."""
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        return reset_output[0]
    return reset_output


def _canonical_env_prim_path(path: str) -> str:
    """Normalize env-scoped prim paths to use ``env_.*`` wildcard."""
    path = str(path)
    if "{ENV_REGEX_NS}" in path:
        path = path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    path = re.sub(r"/env_\d+/", "/env_.*/", path)
    return path


def _collect_multi_spawn_catalogs(scene_cfg) -> dict[str, dict]:
    """Return ``canonical_spawn_prim_path -> catalog metadata`` for multi-asset/usd entities."""
    catalogs: dict[str, dict] = {}
    for name in dir(scene_cfg):
        if name.startswith("_"):
            continue
        entity_cfg = getattr(scene_cfg, name, None)
        if entity_cfg is None or not hasattr(entity_cfg, "spawn"):
            continue
        spawn_cfg = getattr(entity_cfg, "spawn", None)
        if spawn_cfg is None:
            continue
        spawn_type = type(spawn_cfg).__name__
        if spawn_type not in ("MultiAssetSpawnerCfg", "MultiUsdFileCfg"):
            continue
        prim_path = getattr(entity_cfg, "prim_path", None)
        if prim_path is None:
            continue
        catalogs[_canonical_env_prim_path(str(prim_path))] = {
            "catalog_id": f"scene.{name}",
            "spawn_type": spawn_type,
            "spawn_cfg": spawn_cfg,
        }
    return catalogs


class MultiSpawnTrace:
    """Capture chosen assets for ``spawn_multi_asset`` during env construction."""

    def __init__(self, catalogs: dict[str, dict]):
        self._catalogs = catalogs
        self._multi_assets: list[dict] = []
        self._multi_usds: list[dict] = []
        self._enabled = bool(catalogs)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def multi_assets(self) -> list[dict]:
        return [dict(item) for item in self._multi_assets]

    @property
    def multi_usds(self) -> list[dict]:
        return [dict(item) for item in self._multi_usds]

    @staticmethod
    def _resolve_env_prim(spawn_prim_path: str, env_idx: int) -> str:
        return _canonical_env_prim_path(spawn_prim_path).replace("env_.*", f"env_{env_idx}")

    @staticmethod
    def _find_picked_index(seq: list, picked) -> int:
        for i, item in enumerate(seq):
            if item == picked:
                return i
        return -1

    @contextlib.contextmanager
    def _trace_multi_asset_cfg(self, *, canonical_prim: str, catalog_id: str, spawn_cfg, original_choice):
        original_func = spawn_cfg.func

        def _traced_spawn(prim_path, cfg, *args, **kwargs):
            if _canonical_env_prim_path(str(prim_path)) != canonical_prim:
                return original_func(prim_path, cfg, *args, **kwargs)
            if not getattr(cfg, "random_choice", False):
                return original_func(prim_path, cfg, *args, **kwargs)

            from isaaclab.sim.spawners.wrappers import wrappers as wrappers_mod

            picked_indices: list[int] = []

            def traced_choice(seq):
                picked = original_choice(seq)
                idx = self._find_picked_index(seq, picked)
                if idx >= 0:
                    picked_indices.append(idx)
                return picked

            wrappers_mod.random.choice = traced_choice
            try:
                prim = original_func(prim_path, cfg, *args, **kwargs)
            finally:
                wrappers_mod.random.choice = original_choice

            resolved_spawn_prim = _canonical_env_prim_path(str(prim_path))
            assets_cfg = list(getattr(cfg, "assets_cfg", []))
            for env_i, asset_idx in enumerate(picked_indices):
                if asset_idx < 0 or asset_idx >= len(assets_cfg):
                    continue
                self._multi_assets.append({
                    "catalog_id": catalog_id,
                    "spawn_prim_path": resolved_spawn_prim,
                    "resolved_prim_path": self._resolve_env_prim(str(prim_path), env_i),
                    "asset_idx": int(asset_idx),
                    "asset_id": type(assets_cfg[asset_idx]).__name__,
                })
            return prim

        spawn_cfg.func = _traced_spawn
        return original_func

    def _trace_multi_usd_cfg(self, *, canonical_prim: str, catalog_id: str, spawn_cfg, original_choice):
        original_func = spawn_cfg.func

        def _traced_spawn(prim_path, cfg, *args, **kwargs):
            if _canonical_env_prim_path(str(prim_path)) != canonical_prim:
                return original_func(prim_path, cfg, *args, **kwargs)
            if not getattr(cfg, "random_choice", False):
                return original_func(prim_path, cfg, *args, **kwargs)

            from isaaclab.sim.spawners.wrappers import wrappers as wrappers_mod

            picked_indices: list[int] = []

            def traced_choice(seq):
                picked = original_choice(seq)
                idx = self._find_picked_index(seq, picked)
                if idx >= 0:
                    picked_indices.append(idx)
                return picked

            wrappers_mod.random.choice = traced_choice
            try:
                prim = original_func(prim_path, cfg, *args, **kwargs)
            finally:
                wrappers_mod.random.choice = original_choice

            resolved_spawn_prim = _canonical_env_prim_path(str(prim_path))
            usd_pool = getattr(cfg, "usd_path", None)
            if isinstance(usd_pool, str):
                usd_paths = [usd_pool]
            else:
                usd_paths = list(usd_pool or [])
            for env_i, usd_idx in enumerate(picked_indices):
                if usd_idx < 0 or usd_idx >= len(usd_paths):
                    continue
                self._multi_usds.append({
                    "catalog_id": catalog_id,
                    "spawn_prim_path": resolved_spawn_prim,
                    "resolved_prim_path": self._resolve_env_prim(str(prim_path), env_i),
                    "usd_idx": int(usd_idx),
                    "usd_path": str(usd_paths[usd_idx]),
                })
            return prim

        spawn_cfg.func = _traced_spawn
        return original_func

    @contextlib.contextmanager
    def capture(self):
        if not self._enabled:
            yield
            return

        from isaaclab.sim.spawners.wrappers import wrappers as wrappers_mod

        original_choice = wrappers_mod.random.choice
        patched_spawn_funcs: list[tuple[object, object]] = []

        for canonical_prim, catalog in self._catalogs.items():
            spawn_cfg = catalog.get("spawn_cfg")
            if spawn_cfg is None or not hasattr(spawn_cfg, "func"):
                continue
            if str(catalog.get("spawn_type")) == "MultiUsdFileCfg":
                original_func = self._trace_multi_usd_cfg(
                    canonical_prim=canonical_prim,
                    catalog_id=str(catalog["catalog_id"]),
                    spawn_cfg=spawn_cfg,
                    original_choice=original_choice,
                )
            else:
                original_func = self._trace_multi_asset_cfg(
                    canonical_prim=canonical_prim,
                    catalog_id=str(catalog["catalog_id"]),
                    spawn_cfg=spawn_cfg,
                    original_choice=original_choice,
                )
            patched_spawn_funcs.append((spawn_cfg, original_func))

        try:
            yield
        finally:
            wrappers_mod.random.choice = original_choice
            for spawn_cfg, original_func in patched_spawn_funcs:
                spawn_cfg.func = original_func


def setup_output_file() -> str:
    """Ensure the trajectory pickle parent directory exists and return the path."""
    output_dir = os.path.dirname(args_cli.dataset_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    return args_cli.dataset_file


def create_environment_config() -> tuple["ManagerBasedRLEnvCfg | DirectRLEnvCfg", object | None]:
    """Parse + configure the env config for demo recording.

    Applies CLI overrides (``robot_type`` / ``usd_path``) via a clean rebuild so
    the config's ``__post_init__`` re-runs; extracts the success-termination term
    (disabling it in the env so we can check it manually); disables time-out
    termination; disables HDF5-style recorders; and strips camera configs when
    running in XR mode without camera rendering.
    """
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, json_path=args_cli.json_path)
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    override_kwargs: dict = {}
    if args_cli.robot_type is not None:
        if not hasattr(env_cfg, "robot_type"):
            raise ValueError(
                f"Task '{args_cli.task}' does not expose robot_type; cannot apply override '{args_cli.robot_type}'."
            )
        override_kwargs["robot_type"] = args_cli.robot_type
    if args_cli.usd_path is not None:
        if not hasattr(env_cfg, "usd_path"):
            raise ValueError(
                f"Task '{args_cli.task}' does not expose usd_path; cannot apply override '{args_cli.usd_path}'."
            )
        override_kwargs["usd_path"] = args_cli.usd_path
    if args_cli.enable_debug_vis is not None:
        if not hasattr(env_cfg, "enable_debug_vis"):
            raise ValueError(f"Task '{args_cli.task}' does not expose enable_debug_vis; cannot apply override.")
        override_kwargs["enable_debug_vis"] = args_cli.enable_debug_vis
    if override_kwargs:
        cfg_cls = type(env_cfg)
        env_cfg = cfg_cls(**override_kwargs)
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.scene.num_envs = 1
    env_cfg.env_name = args_cli.task.split(":")[-1]

    if "TopDownGrasp" in args_cli.task or "Lift" in args_cli.task:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
            env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    if args_cli.xr:
        if not args_cli.enable_cameras:
            # remove_camera_configs() uses delattr() and can expose class-level
            # camera defaults again; set camera cfgs to None instead.
            env_cfg = strip_camera_cfgs(env_cfg)
            env_cfg = prune_stale_obs_refs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # Replay regenerates observations, so no HDF5-style recorder manager is needed here.
    env_cfg.recorders = {}
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # Swap retargeter cfgs in place after the per-task __post_init__ has fully
    # built env_cfg.teleop_devices. Keeping this out of the env cfg itself
    # avoids plumbing the flag through every task config.
    if hasattr(env_cfg, "teleop_devices"):
        apply_teleop_retargeter_mode(
            env_cfg.teleop_devices,
            args_cli.teleop_retargeter,
            anchor_pos_offsets=getattr(env_cfg, "retargeter_anchor_pos_offsets", None),
        )
        apply_teleop_retargeting_scheme(env_cfg.teleop_devices, args_cli.retargeting_scheme)

    return env_cfg, success_term


def create_environment(
    env_cfg: "ManagerBasedRLEnvCfg | DirectRLEnvCfg", multi_spawn_trace: MultiSpawnTrace | None = None
) -> gym.Env:
    try:
        if multi_spawn_trace is not None and multi_spawn_trace.enabled:
            with multi_spawn_trace.capture():
                return gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)


def setup_teleop_device(env_cfg, callbacks: dict[str, Callable]) -> object:
    """Create a teleop device referenced by the environment config."""
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
            logger.info(f"Created teleop device '{args_cli.teleop_device}' from environment config.")
        else:
            available = list(env_cfg.teleop_devices.devices.keys()) if hasattr(env_cfg, "teleop_devices") else "None"
            logger.error(
                f"No teleop device '{args_cli.teleop_device}' found in environment config."
                f" Available devices: {available}"
            )
            exit(1)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        exit(1)

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        exit(1)

    return teleop_interface


def _resolve_term_callable(env: gym.Env, term: object) -> Callable:
    """Resolve a termination term's params-taking callable.

    Class-based terms (``ManagerTermBase`` subclasses, e.g. the functional
    ``lift_and_tilt_with_contact_zones`` success gate) must be *instantiated*
    with ``(cfg, env)`` before they are callable. Calling the class object
    directly routes the term's params into ``__init__`` instead, which raises
    ``TypeError: ... __init__() got an unexpected keyword argument 'min_height'``.
    Plain-function terms are returned unchanged. The resolved callable is
    cached so the instance (and its zone tensors) is built only once.
    """
    cached = _resolve_term_callable._cache.get(id(term))
    if cached is not None:
        return cached

    from isaaclab.managers import ManagerTermBase

    func = term.func
    if inspect.isclass(func) and issubclass(func, ManagerTermBase):
        func = func(term, env)
    _resolve_term_callable._cache[id(term)] = func
    return func


_resolve_term_callable._cache = {}


def check_success(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Return updated consecutive-success count and whether we've hit the threshold."""
    if success_term is None:
        return success_step_count, False

    success_func = _resolve_term_callable(env, success_term)
    success_flags = success_func(env, **success_term.params)
    if bool(success_flags[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0
    return success_step_count, False


def handle_reset(env: gym.Env) -> object:
    """Reset simulation + environment for a new demo attempt."""
    print("Resetting environment...")
    env.sim.reset()
    return _unwrap_obs(env.reset())


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object,
    success_term: object | None,
    trajectory_recorder: TrajectoryPickleRecorder,
    multi_assets: list[dict] | None = None,
    multi_usds: list[dict] | None = None,
) -> int:
    success_step_count = 0
    should_reset = False
    running = False  # Start inactive for VR (user activates with START gesture).

    def reset_recording_instance():
        nonlocal should_reset
        should_reset = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running
        running = True
        print("Recording started - Begin demonstration")
        if hasattr(teleop_interface, "_retargeters") and teleop_interface._retargeters:
            if hasattr(teleop_interface._retargeters[0], "calibrate_wrist_pose"):
                teleop_interface._retargeters[0].calibrate_wrist_pose()

    def stop_recording_instance():
        nonlocal running
        running = False
        print("Recording paused")

    teleop_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }
    for key, cb in teleop_callbacks.items():
        teleop_interface.add_callback(key, cb)

    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    print("=" * 60)
    print("VR Demo Recording Started")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Teleop Device: {args_cli.teleop_device}")
    print(f"Trajectory pickle: {args_cli.dataset_file}")
    print(f"Record per-step states: {args_cli.record_state}")
    print(f"Target Demos: {'Infinite' if args_cli.num_demos == 0 else args_cli.num_demos}")
    print("=" * 60)
    print("\nVR Controls:")
    print("  - Use START gesture/button to begin recording")
    print("  - Use STOP gesture/button to pause recording")
    print("  - Use RESET gesture/button to reset environment")
    print("  - Success condition will automatically complete and save the demo")
    print(f"  - Need {args_cli.num_success_steps} consecutive successful steps to mark as successful")
    print("=" * 60)

    _printed_bodies = False
    _debug_counter = 0

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            # Update robot wrist pose for the retargeter (mirrors teleop_agent.py).
            if hasattr(teleop_interface, "_retargeters") and teleop_interface._retargeters:
                try:
                    robot = env.scene["robot"]
                    body_names = robot.body_names
                    if not _printed_bodies:
                        logger.info(f"Available robot body names: {body_names}")
                        _printed_bodies = True
                    wrist_body_idx = None
                    wrist_body_name = None
                    for candidate in ("z_rotation_link", "palm_wrist", "palm"):
                        if candidate in body_names:
                            wrist_body_name = candidate
                            wrist_body_idx = body_names.index(candidate)
                            break
                    if wrist_body_idx is not None:
                        body_pose_w = robot.data.body_link_pose_w[0, wrist_body_idx, :7]
                        _debug_counter += 1
                        if _debug_counter % 60 == 0:
                            quat_wxyz = body_pose_w[3:].cpu().numpy()
                            logger.info(
                                f"Robot wrist ({wrist_body_name}) pose - "
                                f"Position: [{body_pose_w[0]:.3f}, {body_pose_w[1]:.3f}, {body_pose_w[2]:.3f}], "
                                f"Quaternion (w,x,y,z): [{quat_wxyz[0]:.3f}, {quat_wxyz[1]:.3f}, "
                                f"{quat_wxyz[2]:.3f}, {quat_wxyz[3]:.3f}]"
                            )
                        for retargeter in teleop_interface._retargeters:
                            if hasattr(retargeter, "set_robot_wrist_pose"):
                                retargeter.set_robot_wrist_pose(body_pose_w.cpu().numpy())
                except (AttributeError, KeyError, ValueError):
                    pass

            action = teleop_interface.advance()
            if running:
                # On the first step of a new demo, capture the initial scene state.
                if not trajectory_recorder.has_active_episode():
                    initial_scene_state = env.scene.get_state(is_relative=True)
                    trajectory_recorder.start_episode(
                        initial_state=initial_scene_state,
                        goal_pose=_get_goal_pose_from_env(env),
                        multi_assets=multi_assets,
                        multi_usds=multi_usds,
                        active_object_metadata=_get_active_object_metadata_from_env(env),
                    )

                trajectory_recorder.record_action(action.detach().clone())
                actions = action.repeat(env.num_envs, 1)
                env.step(actions)
                if args_cli.record_state:
                    post_step_state = env.scene.get_state(is_relative=True)
                    trajectory_recorder.record_state(post_step_state)

                success_step_count, success_reached = check_success(env, success_term, success_step_count)
                if success_reached:
                    trajectory_recorder.finalize_episode(success=True)
                    print(f"✓ Recorded {trajectory_recorder.num_episodes} successful demonstration(s).")
                    should_reset = True
                    running = False

                if args_cli.num_demos > 0 and trajectory_recorder.num_episodes >= args_cli.num_demos:
                    print(f"\nAll {trajectory_recorder.num_episodes} demonstrations recorded. Exiting...")
                    target_time = time.time() + 1.0
                    while time.time() < target_time:
                        env.sim.render()
                    break
            else:
                env.sim.render()

            if should_reset:
                if trajectory_recorder.has_active_episode():
                    trajectory_recorder.discard_episode()
                handle_reset(env)
                teleop_interface.reset()
                success_step_count = 0
                should_reset = False
                running = False

            if env.sim.is_stopped():
                break

    return trajectory_recorder.num_episodes


def main() -> None:
    output_file = setup_output_file()

    global env_cfg  # Exposed for setup_teleop_device parity with prior implementation.
    env_cfg, success_term = create_environment_config()

    multi_spawn_catalogs = _collect_multi_spawn_catalogs(env_cfg.scene)
    multi_spawn_trace = MultiSpawnTrace(multi_spawn_catalogs)

    env = create_environment(env_cfg, multi_spawn_trace=multi_spawn_trace)
    teleop_interface = setup_teleop_device(env_cfg, {})

    trajectory_recorder = TrajectoryPickleRecorder(
        output_file,
        task_name=args_cli.task,
        env_name=args_cli.task.split(":")[-1],
        record_state=args_cli.record_state,
        usd_path=getattr(env_cfg, "usd_path", None),
        robot_type=getattr(env_cfg, "robot_type", None),
        json_path=args_cli.json_path,
    )

    episode_multi_assets = multi_spawn_trace.multi_assets if multi_spawn_trace.multi_assets else None
    episode_multi_usds = multi_spawn_trace.multi_usds if multi_spawn_trace.multi_usds else None
    num_recorded = run_simulation_loop(
        env,
        teleop_interface,
        success_term,
        trajectory_recorder,
        multi_assets=episode_multi_assets,
        multi_usds=episode_multi_usds,
    )

    env.close()
    trajectory_recorder.flush()
    print(f"\nRecording session completed with {num_recorded} successful demonstration(s)")
    print(f"Trajectory pickle saved to: {output_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
