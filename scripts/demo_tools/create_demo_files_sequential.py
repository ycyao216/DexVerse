# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sequential, single-env sibling of :mod:`create_demo_files`.

Where :mod:`create_demo_files` replays N episodes in N parallel envs, this
script builds a single env (``num_envs=1``) and replays one episode at a
time in a Python loop. Between episodes it calls ``env.reset_to(...)``,
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
import pickle
from pathlib import Path
from typing import NamedTuple

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
    help="Root that holds task/<env>/<pickle>.pkl trees (default: %(default)s).",
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
    help="Explicit pickle path(s) to convert. Repeatable.",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Convert every pickle found under --demos-root.",
)
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

# --- Environment-level options -------------------------------------------
parser.add_argument("--task-override", default=None)
parser.add_argument("--robot-type-override", default=None)
parser.add_argument("--json-path", default=None)
parser.add_argument("--enable-pinocchio", action="store_true")

# --- Video recording ------------------------------------------------------
parser.add_argument(
    "--record-video",
    action="store_true",
    help="Render one MP4 per episode into --video-dir from the named camera.",
)
parser.add_argument("--video-camera", default="third_person_camera")
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-dir", type=Path, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
if args_cli.record_video or _capture_all_groups or _preset_has_camera or (_obs_groups_lower & _CAMERA_OBS_GROUPS):
    args_cli.enable_cameras = True

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
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
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


def _force_clear_sim_context() -> None:
    try:
        from isaaclab.sim import SimulationContext
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
        self._h5.attrs["observation_preset"] = observation_preset or ""
        # Marker so readers can tell sequential outputs from parallel ones.
        self._h5.attrs["replay_mode"] = "sequential"
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


class PickleGroup(NamedTuple):
    label: str
    output_stem: str
    anchor_dir: Path
    pickles: list[Path]


def _discover_source_pickle_groups() -> list[PickleGroup]:
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

    task_dirs: set[Path] = set()
    for task in args_cli.task:
        sub = (demos_root / task).resolve()
        if not sub.is_dir():
            raise FileNotFoundError(f"--task dir not found under {demos_root}: {task}")
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
        raise SystemExit("Nothing selected. Pass --all, --task <name>, or --file <path>.")
    return groups


def _merge_trajectory_payloads(pickles: list[Path]) -> dict:
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

    # IsaacLab default is 0 — without it, the RTX camera buffer is stale after
    # reset, so per-episode lighting/texture randomizers won't appear in the
    # first captured frame. In sequential mode every episode is a fresh reset,
    # so we always want a few extra renders to let RTX settle.
    env_cfg.num_rerenders_on_reset = 4

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
    """Replay one episode and append it to the writer. Returns the success flag."""
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
        return ep_success

    states_seq = episode.get("states") if args_cli.set_state else None
    if args_cli.set_state and (states_seq is None or len(states_seq) != T + 1):
        raise ValueError(
            f"--set-state requires episode['states'] of length T+1={T+1} for episode {ep_index} in {label}."
        )

    skip = _entities_to_skip_for_robot_type_change(source_robot_type, target_robot_type)
    env_ids_one = torch.tensor([0], device=env.device, dtype=torch.long)

    with torch.inference_mode():
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

            if termination_cfgs:
                for term_name, term_cfg in termination_cfgs.items():
                    try:
                        term_value = _evaluate_termination_term(
                            term_cfg,
                            env,
                            termination_instance_cache,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if not getattr(term_cfg, "_warned", False):
                            print(f"  [warn] termination {term_name!r} failed: {exc}")
                            term_cfg._warned = True  # type: ignore[attr-defined]
                        term_value = None
                    if term_value is None:
                        terminations_buf[term_name].append(False)
                    else:
                        terminations_buf[term_name].append(bool(term_value[0].item()))

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
        )
        if args_cli.record_video:
            video = VideoRecorder(
                env,
                camera_name=args_cli.video_camera,
                fps=args_cli.video_fps,
                output_dir=_video_dir_for_group(group, output_path),
            )

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

        writer.flush()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] env.close() failed: {exc}")
                _force_clear_sim_context()
        else:
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
            f"obs preset = {_obs_preset_arg!r}; sequential mode."
        )
    else:
        print(f"Converting {len(groups)} task group(s) ({total_pickles} pickle(s)); sequential mode.")

    grand_succ = grand_fail = grand_total = 0
    for group in groups:
        if not simulation_app.is_running() or simulation_app.is_exiting():
            print("  [info] simulation app exiting; stopping early.")
            break
        s, f, t = _convert_one_group(group)
        grand_succ += s
        grand_fail += f
        grand_total += t

    print(
        f"\nDone. total={grand_total} success={grand_succ} failed={grand_fail} "
        f"no-flag={grand_total - grand_succ - grand_fail}"
    )
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        simulation_app.close()
    raise SystemExit(rc)
