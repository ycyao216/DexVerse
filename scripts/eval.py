#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a policy in a single DexVerse environment.

Use ``--policy module`` to load a policy implementation from a Python module or
source file. The module must define the following factory:

    create_policy(env, checkpoint, device, args) -> policy

The returned object must implement ``act(observations)`` and may implement
``reset(observations)`` and ``close()``. Actions may have shape
``(action_dim,)`` or ``(1, action_dim)``. Policy modules own model loading and
observation/action transforms; this script manages the environment, episodes,
success metrics, and video output.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


from isaaclab.app import AppLauncher
from omegaconf import OmegaConf


# =============================================================================
# Command-line arguments and application launch
# =============================================================================

parser = argparse.ArgumentParser(description="Evaluate a DexVerse policy and record RGB observations.")
parser.add_argument("--task", type=str, required=True, help="Registered DexVerse task name.")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of evaluation episodes.")
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=None,
    help="Maximum steps per episode. By default, the task timeout is used.",
)
parser.add_argument(
    "--policy",
    choices=("checkpoint", "zero", "random", "module"),
    default="checkpoint",
    help=(
        "Policy source. 'checkpoint' validates a checkpoint with fallback actions; "
        "'module' loads a policy adapter."
    ),
)
parser.add_argument("--policy_module", type=str, default=None, help="Python module name or path to a policy file.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path passed to the policy factory.")
parser.add_argument(
    "--checkpoint_fallback_action",
    choices=("random", "zero"),
    default="random",
    help="Fallback action used by checkpoint validation mode. This mode does not run model inference.",
)
parser.add_argument("--output_dir", type=str, default="outputs/eval", help="Evaluation output directory.")
parser.add_argument("--seed", type=int, default=42, help="Environment and random-policy seed.")
parser.add_argument("--video_fps", type=int, default=30, help="Output video frame rate.")
parser.add_argument(
    "--video_stride", type=int, default=2, help="Record one frame every N environment steps."
)
parser.add_argument(
    "--camera_names",
    type=str,
    nargs="+",
    default=None,
    help="Scene camera names. All RGB cameras are discovered when omitted.",
)
parser.add_argument(
    "--observation_preset",
    type=str,
    default=None,
    help="Observation preset. Defaults to 'rgb' or 'state' based on the render mode.",
)
parser.add_argument(
    "--domain_randomization_config",
    type=str,
    default=None,
    help="YAML file with environment overrides and event randomization settings.",
)
parser.add_argument(
    "--eval_render_mode",
    choices=("gui", "rgb", "headless"),
    default="rgb",
    help="gui: window and cameras; rgb: headless with cameras; headless: no window, cameras, or video.",
)

# Task and asset configuration.
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O.")
parser.add_argument("--robot_type", type=str, default=None, help="Override the task robot type.")
parser.add_argument("--json_path", type=str, default=None, help="JSON specification for template environments.")
parser.add_argument("--object_usd", type=str, nargs="+", default=None, metavar="PATH", help="Custom object USD path.")
parser.add_argument("--object_half_height", type=float, default=None, help="Object half-height in meters.")
parser.add_argument("--object_mass", type=float, default=None, help="Object mass in kilograms.")
parser.add_argument(
    "--wrist_xyz",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Initial wrist position in world coordinates.",
)
parser.add_argument(
    "--wrist_rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("QW", "QX", "QY", "QZ"),
    help="Initial wrist orientation as a world-frame quaternion.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Configure the window and camera extensions before launching the application.
args_cli.headless = args_cli.eval_render_mode != "gui"
args_cli.enable_cameras = args_cli.eval_render_mode in ("gui", "rgb")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# Imports that require a running Isaac Sim application
# =============================================================================

import numpy as np
import torch

import dexverse.tasks  # noqa: F401  # Registers the DexVerse Gym environments.
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from dexverse.tasks.utils import parse_env_cfg, prune_stale_obs_refs, strip_camera_cfgs


# =============================================================================
# Environment configuration
# =============================================================================

def _set_cfg_value(cfg_obj: object, key_path: list[str], value: Any) -> None:
    """Set a nested value on a configclass, dictionary, or list."""
    target = cfg_obj
    for key in key_path[:-1]:
        if hasattr(target, key):
            target = getattr(target, key)
        elif isinstance(target, dict):
            target = target[key]
        elif isinstance(target, list):
            target = target[int(key)]
        else:
            raise AttributeError(f"Cannot traverse configuration path: {'.'.join(key_path)}")

    final_key = key_path[-1]
    if hasattr(target, final_key):
        setattr(target, final_key, value)
    elif isinstance(target, dict):
        target[final_key] = value
    elif isinstance(target, list):
        target[int(final_key)] = value
    else:
        raise AttributeError(f"Cannot set configuration path: {'.'.join(key_path)}")


def _apply_hydra_env_overrides(env_cfg: object, unknown_args: list[str]) -> None:
    """Apply command-line overrides in ``env.<path>=<value>`` form."""
    if not unknown_args:
        return

    dotlist: list[str] = []
    invalid: list[str] = []
    for raw in unknown_args:
        if "=" not in raw:
            invalid.append(raw)
            continue
        key, value = raw.split("=", maxsplit=1)
        key = key.lstrip("+")
        if key.startswith("env."):
            dotlist.append(f"{key[4:]}={value}")
        else:
            invalid.append(raw)
    if invalid:
        raise ValueError(f"Unsupported arguments: {invalid}. Expected env.<path>=<value> overrides.")

    if dotlist:
        overrides = OmegaConf.to_container(OmegaConf.from_dotlist(dotlist), resolve=True)

        def apply_recursive(prefix: list[str], item: Any) -> None:
            if isinstance(item, dict):
                for child_key, child_value in item.items():
                    apply_recursive([*prefix, child_key], child_value)
            else:
                _set_cfg_value(env_cfg, prefix, item)

        apply_recursive([], overrides)


def _apply_mapping_overrides(target: object, values: dict[str, Any], prefix: list[str] | None = None) -> None:
    """Recursively apply a YAML mapping to a configuration object."""
    prefix = prefix or []
    for key, value in values.items():
        path = [*prefix, key]
        if isinstance(value, dict):
            _apply_mapping_overrides(target, value, path)
        else:
            _set_cfg_value(target, path, value)


def apply_domain_randomization_config(env_cfg: object) -> None:
    """Apply environment and event overrides from a YAML file.

    The ``overrides`` mapping updates environment configuration fields. The
    ``events`` mapping updates, disables, or enables reset event terms.
    """
    if args_cli.domain_randomization_config is None:
        return
    config_path = Path(args_cli.domain_randomization_config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Domain-randomization configuration not found: {config_path}")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("The domain-randomization YAML root must be a mapping.")
    if not bool(raw.get("enabled", True)):
        print(f"[INFO] Domain randomization disabled by configuration: {config_path}")
        return

    overrides = raw.get("overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError("Domain-randomization 'overrides' must be a mapping.")
    _apply_mapping_overrides(env_cfg, overrides)

    events = raw.get("events", {})
    if not isinstance(events, dict):
        raise TypeError("Domain-randomization 'events' must be a mapping.")
    for event_name, event_spec in events.items():
        if not isinstance(event_spec, dict):
            raise TypeError(f"Event configuration for {event_name!r} must be a mapping.")
        enabled = bool(event_spec.get("enabled", True))
        current = getattr(env_cfg.events, event_name, None)
        if not enabled:
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
            continue

        # Create the background reset term when it is enabled but not registered.
        if current is None and event_name == "reset_environment_background":
            from dexverse.tasks import mdp
            from dexverse.tasks.dexverse_base_env_cfg import DEFAULT_BACKGROUND_HDRI_POOL
            from isaaclab.managers import EventTermCfg as EventTerm

            params = dict(event_spec.get("params", {}))
            params.setdefault("hdr_paths", DEFAULT_BACKGROUND_HDRI_POOL)
            current = EventTerm(func=mdp.reset_environment_background, mode="reset", params=params)
            setattr(env_cfg.events, event_name, current)
            continue

        if current is None:
            raise ValueError(
                f"Task {args_cli.task!r} does not define event {event_name!r}. "
                "New physics randomization events must be declared in the task configuration."
            )
        params = event_spec.get("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"Event parameters for {event_name!r} must be a mapping.")
        _apply_mapping_overrides(current.params, params)

    print(f"[INFO] Loaded domain-randomization configuration: {config_path}")


def build_env_cfg() -> object:
    """Build the environment configuration and apply command-line overrides."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        json_path=args_cli.json_path,
    )

    # Reconstruct the config so that __post_init__ recomputes derived fields.
    reinit_kwargs: dict[str, Any] = {}
    if args_cli.robot_type is not None:
        if not hasattr(env_cfg, "robot_type"):
            raise ValueError(f"Task {args_cli.task!r} does not support a robot_type override.")
        reinit_kwargs["robot_type"] = args_cli.robot_type

    if args_cli.object_usd is not None:
        if not hasattr(env_cfg, "object_usd_path"):
            raise ValueError(f"Task {args_cli.task!r} does not support an object USD override.")
        if len(args_cli.object_usd) == 1:
            reinit_kwargs["object_usd_path"] = args_cli.object_usd[0]
        else:
            reinit_kwargs["object_usd_paths"] = args_cli.object_usd

    for name, value in (
        ("object_half_height", args_cli.object_half_height),
        ("object_mass", args_cli.object_mass),
    ):
        if value is not None:
            if not hasattr(env_cfg, name):
                raise ValueError(f"Task {args_cli.task!r} does not support a {name} override.")
            reinit_kwargs[name] = value

    if reinit_kwargs:
        env_cfg = type(env_cfg)(**reinit_kwargs)
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
        env_cfg.scene.num_envs = 1

    _apply_hydra_env_overrides(env_cfg, hydra_args)
    if hasattr(env_cfg, "rebuild_object_from_params"):
        env_cfg.rebuild_object_from_params()

    # Camera-free evaluation uses state observations and omits RTX sensors.
    default_preset = "state" if args_cli.eval_render_mode == "headless" else "rgb"
    preset = args_cli.observation_preset or default_preset
    if preset and hasattr(env_cfg, "_apply_observation_preset"):
        env_cfg.observation_preset = preset
        env_cfg._apply_observation_preset(preset)

    if args_cli.wrist_xyz is not None or args_cli.wrist_rot is not None:
        from dexverse.tasks.config.robot_init import set_robot_wrist_init_world_pos

        wrist_kwargs: dict[str, Any] = {}
        if args_cli.wrist_xyz is not None:
            wrist_kwargs.update(zip(("x", "y", "z"), args_cli.wrist_xyz))
        if args_cli.wrist_rot is not None:
            wrist_kwargs["rot"] = tuple(args_cli.wrist_rot)
        set_robot_wrist_init_world_pos(env_cfg, **wrist_kwargs)

    apply_domain_randomization_config(env_cfg)
    if args_cli.eval_render_mode == "headless":
        env_cfg = strip_camera_cfgs(env_cfg)
        env_cfg = prune_stale_obs_refs(env_cfg)
    env_cfg.scene.num_envs = 1
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    return env_cfg


# =============================================================================
# Policy interface
# =============================================================================

class ZeroPolicy:
    """Return zero actions for environment and recording checks."""

    def __init__(self, env: gym.Env):
        self.shape = env.action_space.shape
        self.device = env.unwrapped.device

    def act(self, observations: Any) -> torch.Tensor:
        del observations
        return torch.zeros(self.shape, device=self.device)


class RandomPolicy:
    """Return actions sampled uniformly from ``[-1, 1]``."""

    def __init__(self, env: gym.Env):
        self.shape = env.action_space.shape
        self.device = env.unwrapped.device

    def act(self, observations: Any) -> torch.Tensor:
        del observations
        return 2.0 * torch.rand(self.shape, device=self.device) - 1.0


class CheckpointFallbackPolicy:
    """Inspect a structured checkpoint and return explicit fallback actions.

    This mode validates checkpoint loading and evaluation infrastructure. It
    does not instantiate a model or perform policy inference.
    """

    def __init__(self, env: gym.Env, checkpoint: str, action_mode: str):
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[CHECKPOINT] Loading: {checkpoint_path}", flush=True)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            raise TypeError(f"Checkpoint root must be a dictionary, got {type(payload).__name__}.")
        if "state_dicts" not in payload:
            raise KeyError("Checkpoint does not contain a 'state_dicts' entry.")
        cfg = payload.get("cfg")
        task_name = OmegaConf.select(cfg, "task.name", default="unknown") if cfg is not None else "unknown"
        policy_target = OmegaConf.select(cfg, "policy._target_", default="unknown") if cfg is not None else "unknown"
        state_dict_names = ", ".join(sorted(payload["state_dicts"]))
        print(
            f"[CHECKPOINT] Parsed: task={task_name}, policy={policy_target}, "
            f"state_dicts={state_dict_names}",
            flush=True,
        )
        print(
            f"[CHECKPOINT] Inference disabled; using {action_mode} fallback actions.",
            flush=True,
        )
        del payload
        self.shape = env.action_space.shape
        self.device = env.unwrapped.device
        self.action_mode = action_mode

    def act(self, observations: Any) -> torch.Tensor:
        del observations
        if self.action_mode == "zero":
            return torch.zeros(self.shape, device=self.device)
        return 2.0 * torch.rand(self.shape, device=self.device) - 1.0


def _load_policy_module(spec: str) -> ModuleType:
    """Load a policy from an import path or a Python source file."""
    path = Path(spec).expanduser()
    if path.is_file():
        module_spec = importlib.util.spec_from_file_location("dexverse_external_eval_policy", path.resolve())
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not load policy file: {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def create_policy(env: gym.Env) -> object:
    """Create the selected policy and validate its required interface."""
    if args_cli.policy == "checkpoint":
        if not args_cli.checkpoint:
            raise ValueError("--checkpoint is required when --policy=checkpoint.")
        return CheckpointFallbackPolicy(env, args_cli.checkpoint, args_cli.checkpoint_fallback_action)
    if args_cli.policy == "zero":
        return ZeroPolicy(env)
    if args_cli.policy == "random":
        return RandomPolicy(env)
    if not args_cli.policy_module:
        raise ValueError("--policy_module is required when --policy=module.")

    module = _load_policy_module(args_cli.policy_module)
    factory = getattr(module, "create_policy", None)
    if not callable(factory):
        raise TypeError("The policy module must define create_policy(env, checkpoint, device, args).")
    policy = factory(
        env=env,
        checkpoint=args_cli.checkpoint,
        device=env.unwrapped.device,
        args=args_cli,
    )
    if not callable(getattr(policy, "act", None)):
        raise TypeError("The object returned by create_policy must implement act(observations).")
    return policy


def _prepare_action(action: Any, env: gym.Env) -> torch.Tensor:
    """Convert and validate a policy action for the environment."""
    tensor = torch.as_tensor(action, device=env.unwrapped.device, dtype=torch.float32)
    expected = tuple(env.action_space.shape)
    if tensor.ndim == 1 and expected[0] == 1 and tensor.shape[0] == expected[1]:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Policy action shape {tuple(tensor.shape)} does not match {expected}.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Policy action contains NaN or Inf values.")
    return tensor


# =============================================================================
# Camera discovery and video recording
# =============================================================================

def _safe_filename(name: str) -> str:
    """Convert an arbitrary label into a safe filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "camera"


def _camera_output_name(scene_name: str) -> str:
    """Map scene camera names to concise output filenames."""
    aliases = {
        "third_person_camera": "head",
        "left_wrist_camera": "left_hand",
        "right_wrist_camera": "right_hand",
        "wrist_camera": "right_hand",
        "third_person_camera_left": "left_view",
        "third_person_camera_right": "right_view",
    }
    return aliases.get(scene_name, _safe_filename(scene_name))


def discover_rgb_cameras(env: gym.Env) -> dict[str, object]:
    """Find RGB-enabled cameras in the environment scene."""
    scene = env.unwrapped.scene
    requested = set(args_cli.camera_names or [])
    cameras: dict[str, object] = {}
    for name in scene.keys():
        if requested and name not in requested:
            continue
        entity = scene[name]
        output = getattr(getattr(entity, "data", None), "output", None)
        if output is not None and "rgb" in output:
            cameras[name] = entity

    if requested:
        missing = sorted(requested - set(cameras))
        if missing:
            raise ValueError(
                f"Cameras are missing or do not provide RGB output: {missing}. "
                f"Available: {sorted(scene.keys())}"
            )
    if not cameras:
        raise RuntimeError("No RGB cameras found. Check the observation preset and camera configuration.")
    return cameras


def rgb_to_uint8(rgb: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert an HWC RGB or RGBA batch to contiguous uint8 RGB."""
    frame = rgb[0].detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)[0]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(f"Unsupported camera frame shape: {frame.shape}")
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.nan_to_num(frame)
        # Scale normalized floating-point images while preserving [0, 255] inputs.
        if frame.size and float(frame.max()) <= 1.0 and float(frame.min()) >= 0.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(frame)


class MultiCameraVideoRecorder:
    """Encode one MP4 stream per RGB camera and episode."""

    def __init__(self, cameras: dict[str, object], output_dir: Path, fps: int):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video recording requires opencv-python (cv2).") from exc
        self.cv2 = cv2
        self.cameras = cameras
        self.output_dir = output_dir
        self.fps = fps
        self.writers: dict[str, Any] = {}
        self.latest_frames: dict[str, np.ndarray] = {}

    def start_episode(self, episode_index: int) -> None:
        """Create an episode directory and reset the video writers."""
        self.close_episode()
        self.episode_dir = self.output_dir / f"episode_{episode_index:03d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

    def capture(self) -> None:
        """Capture and encode the latest RGB frame from environment zero."""
        for name, camera in self.cameras.items():
            rgb = camera.data.output.get("rgb")
            if rgb is None:
                continue
            frame = rgb_to_uint8(rgb)
            self.latest_frames[name] = frame
            if name not in self.writers:
                height, width = frame.shape[:2]
                output_path = self.episode_dir / f"{_camera_output_name(name)}.mp4"
                writer = self.cv2.VideoWriter(
                    str(output_path),
                    self.cv2.VideoWriter_fourcc(*"mp4v"),
                    float(self.fps),
                    (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"Could not create video: {output_path}")
                self.writers[name] = writer
            # OpenCV VideoWriter expects BGR input.
            self.writers[name].write(np.ascontiguousarray(frame[..., ::-1]))

    def get_latest_frames(self) -> dict[str, np.ndarray]:
        """Return the latest CPU RGB frame for each camera."""
        return self.latest_frames

    def close_episode(self) -> None:
        """Release all writers and finalize their MP4 indexes."""
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()


class NullVideoRecorder:
    """No-op recorder used when cameras are disabled."""

    def start_episode(self, episode_index: int) -> None:
        del episode_index

    def capture(self) -> None:
        return

    def get_latest_frames(self) -> dict[str, np.ndarray]:
        return {}

    def close_episode(self) -> None:
        return


# =============================================================================
# Episode execution and result reporting
# =============================================================================

def current_task_success(env: gym.Env) -> bool:
    """Return the current value of the task's success termination term."""
    manager = env.unwrapped.termination_manager
    if "success" not in manager.active_terms:
        raise RuntimeError(f"Task {args_cli.task!r} does not define a 'success' termination term.")
    return bool(manager.get_term("success")[0].item())


def write_results(path: Path, completed: list[tuple[int, bool, int]]) -> None:
    """Atomically update the evaluation summary after each episode."""
    successes = [episode for episode, success, _ in completed if success]
    rate = len(successes) / len(completed) if completed else 0.0
    lines = [
        f"task: {args_cli.task}",
        f"policy: {args_cli.policy}",
        f"checkpoint: {args_cli.checkpoint or ''}",
        f"checkpoint_fallback_action: {args_cli.checkpoint_fallback_action if args_cli.policy == 'checkpoint' else ''}",
        f"target_episodes: {args_cli.num_episodes}",
        f"completed_episodes: {len(completed)}",
        f"successful_episodes: {', '.join(map(str, successes)) if successes else 'none'}",
        f"success_count: {len(successes)}",
        f"success_rate: {rate:.6f} ({rate * 100.0:.2f}%)",
        "",
        "episode_results:",
    ]
    lines.extend(
        f"episode={episode:03d}\tsuccess={str(success).lower()}\tsteps={steps}"
        for episode, success, steps in completed
    )
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _call_optional(policy: object, method_name: str, *args: Any) -> None:
    """Call an optional policy lifecycle method."""
    method = getattr(policy, method_name, None)
    if callable(method):
        method(*args)


def _current_robot_state_cpu(env: gym.Env) -> np.ndarray:
    """Read the 28-dimensional robot joint state from environment zero."""
    state = env.unwrapped.scene["robot"].data.joint_pos[0]
    if state.device.type != "cpu":
        raise RuntimeError(f"Robot state is on {state.device}; evaluation requires CPU simulation for this adapter.")
    return np.ascontiguousarray(state.detach().numpy(), dtype=np.float32)


def run_evaluation(env: gym.Env, policy: object, run_dir: Path) -> list[tuple[int, bool, int]]:
    """Run episodes sequentially and write videos and metrics."""
    observations, _ = env.reset(seed=args_cli.seed)
    video_enabled = args_cli.eval_render_mode in ("gui", "rgb")
    cameras = discover_rgb_cameras(env) if video_enabled else {}
    recorder = (
        MultiCameraVideoRecorder(cameras, run_dir, args_cli.video_fps)
        if video_enabled
        else NullVideoRecorder()
    )
    result_path = run_dir / "results.txt"
    completed: list[tuple[int, bool, int]] = []
    needs_manual_reset = False

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Policy: {args_cli.policy}; episodes: {args_cli.num_episodes}")
    print(f"[INFO] Render mode: {args_cli.eval_render_mode}")
    print(f"[INFO] RGB cameras: {', '.join(cameras) if cameras else 'disabled'}")
    print(f"[INFO] Output directory: {run_dir}")

    try:
        for episode in range(1, args_cli.num_episodes + 1):
            # A manual step limit does not trigger the environment's automatic reset.
            if needs_manual_reset:
                observations, _ = env.reset()
                needs_manual_reset = False
            print(f"\n[Episode {episode:03d}/{args_cli.num_episodes:03d}] Starting", flush=True)
            recorder.start_episode(episode)
            recorder.capture()
            _call_optional(policy, "set_rgb_frames", recorder.get_latest_frames())
            _call_optional(policy, "set_state", _current_robot_state_cpu(env))
            _call_optional(policy, "reset", observations)
            # Do not report an empty or single-frame recording as a completed episode.
            if not simulation_app.is_running():
                raise RuntimeError("Isaac Sim closed before the episode executed an environment step.")
            episode_success = False
            steps = 0

            while simulation_app.is_running():
                with torch.inference_mode():
                    _call_optional(policy, "set_state", _current_robot_state_cpu(env))
                    action = _prepare_action(policy.act(observations), env)
                    observations, _, terminated, truncated, _ = env.step(action)
                steps += 1

                # Read the term before the next step overwrites the manager buffer.
                success_now = current_task_success(env)
                done = bool(terminated[0].item() or truncated[0].item())
                forced_limit = args_cli.max_episode_steps is not None and steps >= args_cli.max_episode_steps
                if success_now:
                    episode_success = True
                    done = True
                # A terminated environment has already reset, so its camera frame belongs
                # to the next episode. Manual limits do not reset and may record the frame.
                if steps % args_cli.video_stride == 0 and not done:
                    recorder.capture()
                    _call_optional(policy, "set_rgb_frames", recorder.get_latest_frames())
                if done or forced_limit:
                    if forced_limit and not done:
                        needs_manual_reset = True
                    break

            recorder.close_episode()
            completed.append((episode, episode_success, steps))
            write_results(result_path, completed)
            label = "SUCCESS" if episode_success else "FALSE"
            rate = sum(item[1] for item in completed) / len(completed)
            print(
                f"[Episode {episode:03d}/{args_cli.num_episodes:03d}] {label} | "
                f"steps={steps} | success rate={rate * 100.0:.2f}%",
                flush=True,
            )
            if not simulation_app.is_running():
                print("[INFO] Isaac Sim closed; stopping evaluation early.")
                break
    finally:
        recorder.close_episode()

    return completed


def main() -> Path:
    """Create the environment and policy, then run evaluation."""
    if args_cli.num_episodes <= 0:
        raise ValueError("--num_episodes must be greater than zero.")
    if args_cli.video_fps <= 0 or args_cli.video_stride <= 0:
        raise ValueError("--video_fps and --video_stride must be greater than zero.")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    if args_cli.checkpoint:
        checkpoint_path = Path(args_cli.checkpoint).expanduser()
        checkpoint_label = (
            checkpoint_path.parent.parent.name
            if checkpoint_path.parent.name == "checkpoints"
            else checkpoint_path.stem
        )
    else:
        checkpoint_label = args_cli.policy
    run_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        / _safe_filename(checkpoint_label)
        / _safe_filename(args_cli.task)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=build_env_cfg())
    policy = None
    try:
        print(f"[INFO] Gym observation space: {env.observation_space}")
        print(f"[INFO] Gym action space: {env.action_space}")
        policy = create_policy(env)
        # Allow asynchronous policy adapters to keep rendering while inference is pending.
        _call_optional(policy, "set_keep_alive", env.unwrapped.sim.render)
        completed = run_evaluation(env, policy, run_dir)
        successes = sum(item[1] for item in completed)
        rate = successes / len(completed) if completed else 0.0
        print(f"\n[DONE] {successes}/{len(completed)} successful; success rate: {rate * 100.0:.2f}%")
        print(f"[DONE] Results: {run_dir / 'results.txt'}")
    finally:
        if policy is not None:
            _call_optional(policy, "close")
        env.close()
    return run_dir


if __name__ == "__main__":
    completed_run_dir = None
    try:
        completed_run_dir = main()
    finally:
        simulation_app.close()
