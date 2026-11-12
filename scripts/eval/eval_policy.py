# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online evaluation of a VLA baseline on the DexVerse baseline suites.

Runs the DexVerse (Isaac Lab) environment in this process and queries a policy
server running in another process — neither OpenVLA-OFT nor openpi can share an
interpreter with Isaac Sim. Start the server first:

    bash scripts/eval/serve_openvla.sh     # HTTP,      default port 8777
    bash scripts/eval/serve_pi0.sh         # WebSocket, default port 8000

Then evaluate:

    # one task, quick smoke test
    python scripts/eval/eval_policy.py --policy pi0 --enable_cameras --headless \\
        --task Dexverse-GraspCup-v0 --num_episodes 2 --max_steps 200 --save_videos

    # a whole suite in this process (see run_all_tasks.sh for one process per task)
    python scripts/eval/eval_policy.py --policy openvla --enable_cameras --headless \\
        --suite single --unnorm_key <dataset_statistics key>

Embodiment (single-hand vs bimanual) is read off ``env_cfg.robot_type`` rather
than assumed from the task name, and cross-checked against the suite the task
belongs to. Config class names are not reliable here: several bimanual tasks
are still named ``...EnvFloatingDexHandRightCfg``.

``--observation_preset`` controls the width of the proprio state vector and
therefore whether the policy receives the input distribution it was trained on.
It defaults per backend (openvla -> no preset, pi0 -> ``rgb``); the resulting
state dimension is always printed before the first request.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import NamedTuple

# scripts/eval/ is a plain directory of entry points, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_tasks  # noqa: E402
import policy_clients  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate a VLA baseline on DexVerse via a policy server.")

# NOTE: no argument here may use `required=True`. AppLauncher.add_app_launcher_args()
# calls parser.parse_known_args() internally before adding the simulator flags, so a
# required argument makes that internal parse exit(2) — which both swallows --help and
# silently drops --headless/--device/--enable_cameras. Required arguments are validated
# by hand after parse_args() instead.

# --- Backend ---
parser.add_argument(
    "--policy",
    type=str,
    default=None,
    choices=policy_clients.POLICY_NAMES,
    help="Which policy server to query. 'openvla' speaks HTTP, 'pi0' speaks the openpi websocket protocol.",
)
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument(
    "--port",
    type=int,
    default=None,
    help="Policy server port. Default: 8777 for --policy openvla, 8000 for --policy pi0.",
)
parser.add_argument(
    "--unnorm_key",
    type=str,
    default=None,
    help=(
        "Required for --policy openvla: the key into the checkpoint's dataset_statistics.json "
        "used to denormalize actions (e.g. the RLDS dataset name the model was finetuned on)."
    ),
)

# --- Task selection ---
parser.add_argument(
    "--suite",
    type=str,
    default=None,
    choices=baseline_tasks.SUITE_NAMES,
    help="Evaluate a whole baseline suite. Mutually exclusive with --task.",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Evaluate a single baseline task, e.g. Dexverse-GraspCup-v0.",
)

# --- Rollout ---
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--replan_interval",
    type=int,
    default=None,
    help=(
        "Execute N actions from each predicted chunk, then re-query the policy. "
        "Default: 8 for openvla (its full NUM_ACTIONS_CHUNK open-loop horizon), 5 for pi0."
    ),
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Override env_cfg.episode_length_s (env step size is 1/60 s). Default: each task's own value.",
)

# --- Observation space ---
parser.add_argument(
    "--observation_preset",
    type=str,
    default=None,
    help=(
        "Observation preset to apply before building the env (rgb, rgb_depth/rgbd, pointcloud, "
        "state, 3view_*), or 'none' to leave the task's own observation groups untouched. "
        "This sets the proprio history length and therefore the state width the server receives. "
        "Default: 'none' for openvla, 'rgb' for pi0."
    ),
)
parser.add_argument(
    "--state_dim",
    type=int,
    default=None,
    help=(
        "Truncate the proprio vector to this many dimensions before sending it. "
        "Default: 28/56 (single/bimanual) for openvla, whose RLDS transform clips to a single "
        "frame; unset for pi0, whose norm stats cover the full stacked state."
    ),
)

# --- Environment tweaks ---
parser.add_argument(
    "--disable_bg_randomize",
    action="store_true",
    default=True,
    help=(
        "Null out the reset_environment_background and reset_table_texture event terms, matching "
        "create_demo_files_sequential.py --disable-bg-randomize which produced the training set. "
        "On by default; pass --no_disable_bg_randomize to keep per-episode randomization."
    ),
)
parser.add_argument("--no_disable_bg_randomize", dest="disable_bg_randomize", action="store_false")
parser.add_argument(
    "--disable_object_out_of_bound",
    action="store_true",
    default=False,
    help="Detach the object_out_of_bound termination. Diagnostic: lets a dropped object play out.",
)

# --- Output ---
parser.add_argument("--output_dir", type=str, default=None, help="Default: runs/<policy>_eval_online.")
parser.add_argument("--save_videos", action="store_true", default=False)
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument(
    "--dump_first_frame",
    action="store_true",
    default=False,
    help="Write each task's first observation frame to <output_dir>/<task>/first_frame_*.png.",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---- Validate the arguments argparse could not mark required (see NOTE above) ----
if args.policy is None:
    raise SystemExit(f"--policy is required. Choose one of: {list(policy_clients.POLICY_NAMES)}")
if args.suite and args.task:
    raise SystemExit("--suite and --task are mutually exclusive.")
if not args.suite and not args.task:
    raise SystemExit("Pass --suite {single,bimanual} or --task <env id>.")
if args.policy == "openvla" and not args.unnorm_key:
    raise SystemExit("--policy openvla requires --unnorm_key (the key into the checkpoint's dataset_statistics.json).")

# Resolve the task list here rather than inside main(): baseline_tasks is pure
# Python, so a typo'd task name should fail now — not after the websocket client
# has connected and Isaac Sim has spent a minute booting.
try:
    TASK_PLAN = [baseline_tasks.find_task(args.task)] if args.task else list(baseline_tasks.get_suite(args.suite))
except ValueError as exc:
    raise SystemExit(str(exc)) from None

# ---- Resolve backend-dependent defaults ----

DEFAULT_PORTS = {"openvla": 8777, "pi0": 8000}
DEFAULT_REPLAN = {"openvla": 8, "pi0": 5}

if args.port is None:
    args.port = DEFAULT_PORTS[args.policy]
if args.replan_interval is None:
    args.replan_interval = DEFAULT_REPLAN[args.policy]
if args.observation_preset is None:
    args.observation_preset = policy_clients.DEFAULT_OBS_PRESET[args.policy]
if args.output_dir is None:
    args.output_dir = f"runs/{args.policy}_eval_online"

if args.replan_interval < 1:
    # 0 would consume the whole chunk into an empty queue and then pop from it.
    raise SystemExit("--replan_interval must be >= 1.")

LOG = f"[{args.policy}-eval]"


def log(message: str = "") -> None:
    """Print and flush immediately.

    Isaac Sim takes over stdout once the app is running and buffered writes are
    discarded when ``simulation_app.close()`` tears the process down — which
    silently swallows the per-episode lines and the final summary, i.e. exactly
    the output this script exists to produce. Every user-facing print goes
    through here.
    """
    print(message, flush=True)


# Build the client BEFORE AppLauncher. A missing backend dependency or an
# unreachable server then fails immediately, instead of ~1 min into Isaac Sim
# startup — and before simulation_app takes over stderr, which otherwise eats
# the traceback (see the __main__ guard at the bottom).
client = policy_clients.make_client(
    policy=args.policy,
    host=args.host,
    port=args.port,
    unnorm_key=args.unnorm_key,
)
log(f"{LOG} policy server: {client.describe()}")
if isinstance(client, policy_clients.OpenPiWebsocketClient):
    log(f"{LOG} server metadata: {client.server_metadata}")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Observation plumbing
# ---------------------------------------------------------------------------

# robot_type values that mean "two hands, 56 DoF, two wrist cameras".
BIMANUAL_ROBOT_TYPES = frozenset({"floating_shadow_bimanual"})

# Proprio width per embodiment for a single (non-stacked) frame.
SINGLE_FRAME_STATE_DIM = {"single": 28, "bimanual": 56}


def to_uint8_hwc(image):
    """Convert a possibly-batched, possibly-floating image tensor to (H, W, 3) uint8."""
    import numpy as np
    import torch

    if image is None:
        return None
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            image = image[0]
        array = image.detach().cpu().contiguous().numpy()
    else:
        array = np.asarray(image)
        if array.ndim == 4:
            array = array[0]
    if array.dtype != np.uint8:
        # Camera terms may hand back either normalized floats or raw 0-255.
        if array.max() <= 1.0 + 1e-6:
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return array


def extract_observation(obs, embodiment: str, state_dim: int | None):
    """Build an :class:`EvalObservation` from a DexVerse observation dict.

    The ``rgb`` group has ``concatenate_terms=False`` so it arrives as a dict of
    term name -> tensor. ``proprio`` is concatenated, and with the default
    history length its only term is ``joint_pos``.
    """
    import numpy as np
    import torch

    rgb_group = obs.get("rgb") if isinstance(obs, dict) else None
    if not isinstance(rgb_group, dict):
        rgb_group = {}

    third_person = to_uint8_hwc(rgb_group.get("rgb_image"))

    proprio = obs.get("proprio") if isinstance(obs, dict) else None
    if isinstance(proprio, dict):
        proprio = proprio.get("joint_pos")
    state = None
    raw_state_dim = None
    if proprio is not None:
        if isinstance(proprio, torch.Tensor):
            array = proprio.detach().cpu().contiguous().numpy()
        else:
            array = np.asarray(proprio)
        if array.ndim == 2:
            array = array[0]
        raw_state_dim = int(array.shape[-1])
        if state_dim is not None:
            array = array[:state_dim]
        state = array.astype(np.float32)

    if third_person is None or state is None:
        raise RuntimeError(
            "Observation is missing rgb.rgb_image or proprio.joint_pos. "
            "Pass --enable_cameras, and check that the task's observation preset keeps both groups."
        )

    if embodiment == "bimanual":
        left = to_uint8_hwc(rgb_group.get("left_wrist_rgb_image"))
        right = to_uint8_hwc(rgb_group.get("right_wrist_rgb_image"))
        # Zero-fill matches the padding used when building the training set.
        left = np.zeros_like(third_person) if left is None else left
        right = np.zeros_like(third_person) if right is None else right
        eval_obs = policy_clients.EvalObservation(
            third_person=third_person, state=state, left_wrist=left, right_wrist=right
        )
    else:
        wrist = to_uint8_hwc(rgb_group.get("wrist_rgb_image"))
        wrist = np.zeros_like(third_person) if wrist is None else wrist
        eval_obs = policy_clients.EvalObservation(third_person=third_person, state=state, wrist=wrist)
    return eval_obs, raw_state_dim


def save_video(frames, path, fps, imageio_mod, cv2_mod) -> None:
    """Write ``frames`` to ``path``, preferring imageio and falling back to cv2."""
    if not frames:
        return
    if imageio_mod is not None:
        imageio_mod.mimsave(str(path), frames, fps=fps)
        return
    if cv2_mod is not None:
        import numpy as np

        height, width = frames[0].shape[:2]
        writer = cv2_mod.VideoWriter(str(path), cv2_mod.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames:
            writer.write(cv2_mod.cvtColor(np.asarray(frame), cv2_mod.COLOR_RGB2BGR))
        writer.release()


# ---------------------------------------------------------------------------
# Success scoring
# ---------------------------------------------------------------------------


def first_env_bool(value) -> bool:
    """Collapse a per-env termination / success result to the env-0 boolean."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


class SuccessChecker:
    """Evaluate a detached success termination term on demand.

    The success term is removed from the env cfg so the episode does not end the
    instant it fires; we score it ourselves each step instead. Params are
    filtered against the callable's signature so class-based ``ManagerTermBase``
    terms — whose params are split between ``__init__`` and ``__call__`` — do not
    raise ``TypeError``.
    """

    def __init__(self, term_cfg, env) -> None:
        params = dict(getattr(term_cfg, "params", {}) or {})
        func = term_cfg.func
        if isinstance(func, type):
            self.instance = func(cfg=term_cfg, env=env)
            target = self.instance.__call__
        else:
            self.instance = None
            target = func
        try:
            signature = inspect.signature(target)
            accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
            self.params = params if accepts_var_kw else {k: v for k, v in params.items() if k in signature.parameters}
        except (TypeError, ValueError):
            # Builtin / C-extension callable with no introspectable signature.
            self.params = params
        if self.instance is not None:
            self._call = lambda env_: self.instance(env_, **self.params)
        else:
            self._call = lambda env_: func(env_, **self.params)

    def __call__(self, env) -> bool:
        return first_env_bool(self._call(env))


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def resolve_embodiment(env_cfg, task_name: str) -> tuple[str, str | None]:
    """Return ``(embodiment, robot_type)`` for a parsed env cfg.

    The embodiment comes from ``robot_type`` and nothing else — config class
    names lie here (see the note in :mod:`baseline_tasks`). It is cross-checked
    against the suite the task is declared in, because a silent disagreement
    means sending the wrong number of images and a wrong-width state.
    """
    robot_type = getattr(env_cfg, "robot_type", None)
    embodiment = "bimanual" if robot_type in BIMANUAL_ROBOT_TYPES else "single"
    declared_suite = baseline_tasks.suite_of(task_name)
    if embodiment != declared_suite:
        raise SystemExit(
            f"{LOG} embodiment mismatch for {task_name}: robot_type={robot_type!r} implies "
            f"{embodiment!r}, but the task is listed in the {declared_suite!r} suite. "
            "Fix baseline_tasks.py or BIMANUAL_ROBOT_TYPES before trusting these numbers."
        )
    log(f"{LOG} robot_type={robot_type!r} -> {embodiment} embodiment")
    return embodiment, robot_type


def apply_observation_preset(env_cfg) -> None:
    """Narrow the observation groups to the preset the policy was trained on.

    Must run after ``parse_env_cfg`` and before ``gym.make``: the base cfg
    deliberately does not apply presets in ``__post_init__``, because sub-bases
    and leaf configs are still populating observation terms at that point.
    """
    if args.observation_preset != "none" and hasattr(env_cfg, "_apply_observation_preset"):
        env_cfg.observation_preset = args.observation_preset
        env_cfg._apply_observation_preset(args.observation_preset)
        log(f"{LOG} applied observation preset: {args.observation_preset}")
    else:
        log(f"{LOG} no observation preset applied (task defaults kept)")


def check_rgb_group(env_cfg, embodiment: str, task_name: str) -> None:
    """Verify the camera terms the policy is fed are actually configured.

    The group is deliberately never rebuilt here: a fresh ``RgbObsCfg()`` carries
    every camera term (both side views, and both the single-wrist and
    left/right-wrist variants) while the per-robot nulling that
    ``__post_init__`` applied would be gone — env construction then dies on a
    missing scene entity.
    """
    rgb_group = getattr(getattr(env_cfg, "observations", None), "rgb", None)
    if rgb_group is None:
        raise SystemExit(
            f"{LOG} {task_name} has no 'rgb' observation group, but the policy is fed rendered RGB. "
            "Pass --observation_preset rgb (or another preset that enables it)."
        )
    # A wrist view the task never produces would be zero-filled at inference
    # time — a black image the policy was never trained on, and nothing
    # downstream would flag it. Check now, before the simulator boots.
    expected_terms = (
        ("rgb_image", "left_wrist_rgb_image", "right_wrist_rgb_image")
        if embodiment == "bimanual"
        else ("rgb_image", "wrist_rgb_image")
    )
    missing = [name for name in expected_terms if getattr(rgb_group, name, None) is None]
    if missing:
        log(f"{LOG} WARNING: rgb group is missing {missing}; those views will be sent as zeros.")


def detach_success_term(env_cfg):
    """Remove and return the task's ``success`` termination term.

    Success is scored manually each step instead, so an episode is not cut short
    the instant it first fires.
    """
    if not hasattr(env_cfg, "terminations") or not hasattr(env_cfg.terminations, "success"):
        return None
    success_cfg = env_cfg.terminations.success
    if success_cfg is not None:
        env_cfg.terminations.success = None
    return success_cfg


def apply_env_overrides(env_cfg) -> None:
    """Apply the CLI's environment tweaks: recorders, seed, terminations, events."""
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = {}
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed

    if (
        args.disable_object_out_of_bound
        and hasattr(env_cfg, "terminations")
        and getattr(env_cfg.terminations, "object_out_of_bound", None) is not None
    ):
        env_cfg.terminations.object_out_of_bound = None
        log(f"{LOG} detached object_out_of_bound termination")

    if args.episode_length_s is not None and hasattr(env_cfg, "episode_length_s"):
        log(f"{LOG} episode_length_s: {env_cfg.episode_length_s} -> {args.episode_length_s}")
        env_cfg.episode_length_s = float(args.episode_length_s)

    if args.disable_bg_randomize and hasattr(env_cfg, "events"):
        disabled = []
        for term_name in ("reset_environment_background", "reset_table_texture"):
            if getattr(env_cfg.events, term_name, None) is not None:
                setattr(env_cfg.events, term_name, None)
                disabled.append(term_name)
        if disabled:
            log(f"{LOG} disabled per-episode randomizers: {disabled}")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


class TaskContext(NamedTuple):
    """Everything a rollout needs that stays fixed for the whole task."""

    task: baseline_tasks.BaselineTask
    embodiment: str
    robot_type: str | None
    state_dim: int | None
    success_evaluator: SuccessChecker | None
    out_dir: str
    video_dir: str | None


class EpisodeResult(NamedTuple):
    """Outcome of a single episode."""

    episode_return: float
    length: int
    success: bool
    latencies_ms: list[float]
    frames: list


def dump_first_frame(eval_obs, ctx: TaskContext) -> None:
    """Write one PNG per camera, for eyeballing the visual gap against training."""
    try:
        import imageio

        named = {"third_person": eval_obs.third_person}
        if ctx.embodiment == "bimanual":
            named["left_wrist"] = eval_obs.left_wrist
            named["right_wrist"] = eval_obs.right_wrist
        else:
            named["wrist"] = eval_obs.wrist
        for label, image in named.items():
            imageio.imwrite(os.path.join(ctx.out_dir, f"first_frame_{label}.png"), image)
        log(f"{LOG} dumped first-frame PNGs to {ctx.out_dir}")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not break the rollout
        log(f"{LOG} could not dump first frame: {exc}")


def log_fired_terms(env, episode: int, step: int, success: bool) -> None:
    """Report which non-success termination terms ended the episode."""
    try:
        manager = env.unwrapped.termination_manager
        fired = [name for name in manager.active_terms if name != "success" and first_env_bool(manager.get_term(name))]
    except Exception:  # noqa: BLE001 - diagnostics only
        return
    if fired:
        log(f"{LOG}   ep {episode + 1} ended at step {step + 1}: fired={fired} success={success}")


def run_episode(env, episode: int, ctx: TaskContext) -> EpisodeResult:
    """Roll one episode out under the policy and score it."""
    import numpy as np
    import torch

    obs, _ = env.reset(seed=args.seed + episode)
    episode_return = 0.0
    success = False
    frames: list = []
    latencies_ms: list[float] = []
    action_queue: deque = deque()
    length = args.max_steps

    for step in range(args.max_steps):
        eval_obs, raw_state_dim = extract_observation(obs, ctx.embodiment, ctx.state_dim)
        first_request = episode == 0 and step == 0

        if ctx.video_dir is not None:
            frames.append(eval_obs.third_person)
        if args.dump_first_frame and first_request:
            dump_first_frame(eval_obs, ctx)

        if not action_queue:
            if first_request:
                # The single most useful line for diagnosing a silently wrong
                # eval: the state width the server actually receives.
                log(
                    f"{LOG} first request | {eval_obs.shape_summary()} "
                    f"(raw proprio dim={raw_state_dim}, sent={eval_obs.state.shape[0]}, "
                    f"preset={args.observation_preset})"
                )
            chunk, latency_ms = client.infer(eval_obs, ctx.task.prompt)
            latencies_ms.append(latency_ms)
            if first_request:
                log(f"{LOG} first response: chunk={chunk.shape} latency={latency_ms:.1f}ms")
            action_queue.extend(chunk[: args.replan_interval])
            if not action_queue:
                raise RuntimeError(
                    f"Policy server returned an empty action chunk (shape {chunk.shape}). "
                    "Check that the server is serving the checkpoint you think it is."
                )

        action = torch.from_numpy(np.asarray(action_queue.popleft(), dtype=np.float32))
        obs, reward, terminated, truncated, _info = env.step(action.unsqueeze(0).to(env.unwrapped.device))

        if isinstance(reward, torch.Tensor):
            episode_return += float(reward.reshape(-1)[0].item())
        else:
            episode_return += float(np.asarray(reward).reshape(-1)[0])

        if ctx.success_evaluator is not None:
            success = success or ctx.success_evaluator(env.unwrapped)

        if first_env_bool(terminated) or first_env_bool(truncated) or success:
            length = step + 1
            log_fired_terms(env, episode, step, success)
            break

    return EpisodeResult(episode_return, length, success, latencies_ms, frames)


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------


def build_task_context(task, out_dir: str, video_dir: str | None):
    """Parse and adapt the env cfg for ``task``, then build the env.

    Returns ``(env, ctx)``, or ``(None, error_message)`` when the env could not
    be constructed — one unbuildable task must not abort the whole sweep.
    """
    import gymnasium as gym
    from dexverse.tasks.utils import parse_env_cfg

    task_name = task.env_id
    env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=1)

    embodiment, robot_type = resolve_embodiment(env_cfg, task_name)
    apply_observation_preset(env_cfg)
    check_rgb_group(env_cfg, embodiment, task_name)
    success_cfg = detach_success_term(env_cfg)
    apply_env_overrides(env_cfg)

    state_dim = args.state_dim
    if state_dim is None and args.policy == "openvla":
        state_dim = SINGLE_FRAME_STATE_DIM[embodiment]

    try:
        env = gym.make(task_name, cfg=env_cfg)
    except Exception as exc:  # noqa: BLE001 - one bad task must not kill the sweep
        log(f"{LOG} FAILED to create env for {task_name}: {exc}. Skipping.")
        return None, str(exc)

    success_evaluator = None
    if success_cfg is not None:
        try:
            success_evaluator = SuccessChecker(success_cfg, env.unwrapped)
        except Exception as exc:  # noqa: BLE001
            log(f"{LOG} WARN: could not build success checker for {task_name}: {exc}. Success rate will be N/A.")

    ctx = TaskContext(
        task=task,
        embodiment=embodiment,
        robot_type=robot_type,
        state_dim=state_dim,
        success_evaluator=success_evaluator,
        out_dir=out_dir,
        video_dir=video_dir,
    )
    return env, ctx


def summarize_episodes(results: list[EpisodeResult], scored: bool) -> dict:
    """Aggregate per-episode results into the metrics recorded for the task."""
    import numpy as np

    latencies = [ms for result in results for ms in result.latencies_ms]
    successes = sum(1 for result in results if result.success)
    return {
        "num_episodes": args.num_episodes,
        "avg_return": float(np.mean([r.episode_return for r in results])) if results else 0.0,
        "avg_ep_len": float(np.mean([r.length for r in results])) if results else float(args.max_steps),
        "inference_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "success_rate": successes / max(1, args.num_episodes) if scored else None,
    }


def evaluate_task(task, imageio_mod, cv2_mod) -> dict:
    """Evaluate one task end to end and write its ``metrics.json``.

    Returns the metrics dict, or ``{"error": ...}`` if the env could not be built.
    """
    task_name = task.env_id
    log(f"\n{LOG} === Task {task.task_index}: {task_name} ===")
    log(f"{LOG} prompt: {task.prompt!r}")

    out_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(out_dir, exist_ok=True)
    video_dir = None
    if args.save_videos and (imageio_mod or cv2_mod):
        video_dir = os.path.join(out_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    env, ctx = build_task_context(task, out_dir, video_dir)
    if env is None:
        return {"error": ctx}

    results: list[EpisodeResult] = []
    try:
        for episode in range(args.num_episodes):
            result = run_episode(env, episode, ctx)
            results.append(result)
            if ctx.video_dir is not None and result.frames:
                suffix = "success" if result.success else "failure"
                save_video(
                    result.frames,
                    os.path.join(ctx.video_dir, f"episode_{episode:03d}_{suffix}.mp4"),
                    fps=args.video_fps,
                    imageio_mod=imageio_mod,
                    cv2_mod=cv2_mod,
                )
            log(
                f"{LOG} {task_name}  ep {episode + 1}/{args.num_episodes}  "
                f"return={result.episode_return:.3f}  success={result.success}  steps={result.length}"
            )
    finally:
        env.close()

    metrics = summarize_episodes(results, scored=ctx.success_evaluator is not None)
    eval_config = {
        "task": task_name,
        "task_index": task.task_index,
        "prompt": task.prompt,
        "policy": args.policy,
        "embodiment": ctx.embodiment,
        "robot_type": ctx.robot_type,
        "observation_preset": args.observation_preset,
        "state_dim_sent": ctx.state_dim,
        "host": args.host,
        "port": args.port,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "replan_interval": args.replan_interval,
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"config": eval_config, "metrics": metrics}, handle, ensure_ascii=False, indent=2)

    rate = metrics["success_rate"]
    printable = {**metrics, "success_rate": "N/A" if rate is None else round(rate, 4)}
    log(f"{LOG} {task_name} done: {json.dumps(printable)}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_preset_argument() -> None:
    """Canonicalize ``--observation_preset``, failing before the first env build.

    Also folds aliases (``rgbd`` -> ``rgb_depth``) so the preset recorded in
    ``metrics.json`` is unambiguous.
    """
    if args.observation_preset == "none":
        return
    from dexverse.tasks.dexverse_base_env_cfg import resolve_observation_preset

    try:
        args.observation_preset = resolve_observation_preset(args.observation_preset)
    except ValueError as exc:
        raise SystemExit(f"{LOG} {exc}") from None


def load_video_backend():
    """Return the ``(imageio, cv2)`` modules available for writing episode videos."""
    if not args.save_videos:
        return None, None
    try:
        import imageio

        return imageio, None
    except ImportError:
        pass
    try:
        import cv2

        return None, cv2
    except ImportError:
        log(f"{LOG} WARNING: --save_videos needs imageio or cv2; videos disabled.")
        return None, None


def write_summary(summary: dict[str, dict]) -> None:
    """Write the cross-task ``summary.json`` and print the per-task success rates."""
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    log(f"\n{LOG} summary written to {summary_path}")
    log(f"{LOG} per-task success rates:")
    for name, metrics in summary.items():
        if "error" in metrics:
            log(f"  {name:52s}  ERROR: {metrics['error']}")
        else:
            rate = metrics.get("success_rate")
            log(f"  {name:52s}  success_rate={'N/A' if rate is None else f'{rate:.3f}'}")


def main() -> None:
    # Importing dexverse.tasks is what registers the Dexverse-* env ids with
    # gymnasium. It has to happen after AppLauncher, hence the local import.
    import dexverse.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401

    log(f"{LOG} evaluating {len(TASK_PLAN)} task(s), {args.num_episodes} episode(s) each")
    resolve_preset_argument()
    imageio_mod, cv2_mod = load_video_backend()

    os.makedirs(args.output_dir, exist_ok=True)
    summary: dict[str, dict] = {}
    for task in TASK_PLAN:
        summary[task.env_id] = evaluate_task(task, imageio_mod, cv2_mod)
    write_summary(summary)


if __name__ == "__main__":
    import contextlib
    import traceback

    exit_code = 0
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        # simulation_app.close() below can SIGTERM the process before Python
        # flushes a traceback, which turns any failure into a silent exit.
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        exit_code = 1
    finally:
        with contextlib.suppress(BaseException):
            simulation_app.close()
    sys.exit(exit_code)
