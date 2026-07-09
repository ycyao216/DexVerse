# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
from omegaconf import OmegaConf

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
    "--json_path",
    type=str,
    default=None,
    help="Path to template JSON spec. Required for *Template environments.",
)
parser.add_argument(
    "--object_usd",
    type=str,
    nargs="+",
    default=None,
    metavar="PATH",
    help=(
        "Object USD input for pickup-object task variants. Accepts one of:\n"
        "  single file : --object_usd /path/to/mug.usd\n"
        "  directory   : --object_usd /path/to/objects_dir/   "
        "(all immediate *.usd files used; random per env)\n"
        "  explicit list: --object_usd /a.usd /b.usd /c.usd   "
        "(random per env)\n"
        "Triggers a full config re-instantiation so table placement and "
        "other derived values are computed with the new object."
    ),
)
parser.add_argument(
    "--object_half_height",
    type=float,
    default=None,
    help="Half-height of the object in metres (used to place it flush on the table).",
)
parser.add_argument(
    "--object_mass",
    type=float,
    default=None,
    help="Object mass in kg.",
)
parser.add_argument(
    "--observation_preset",
    type=str,
    default=None,
    help=(
        "Optional observation preset (e.g. rgb, rgb_depth/rgbd, pointcloud, state, "
        "3view_rgb, 3view_rgb_depth/3view_rgbd, 3view_pointcloud). The 3view_* presets "
        "also enable the two side-view third-person cameras."
    ),
)
parser.add_argument(
    "--wrist_xyz",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "World-frame target for the robot's palm (init pose). Calls "
        "dexverse.tasks.config.robot_init.set_robot_wrist_init_world_pos. "
        "Floating hands write the translation joints directly; arm+hand robots "
        "(fr3_/ur10e_/xarm7_*) run damped-LS IK to solve for arm joint angles."
    ),
)
parser.add_argument(
    "--wrist_rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("QW", "QX", "QY", "QZ"),
    help=(
        "World-frame quaternion (w, x, y, z) for the palm init orientation. "
        "Combined with --wrist_xyz to set a full pose; either can be used alone."
    ),
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help=(
        "If set, stop after N env steps and exit cleanly (good for smoke tests). Default: run until the sim is closed."
    ),
)
parser.add_argument(
    "--no_step",
    action="store_true",
    default=False,
    help=(
        "Debug mode: reset the env to spawn the scene, then just render without "
        "stepping physics. Useful for inspecting the post-spawn pose when joints "
        "go unstable on the first physics step."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse known args so we can accept Hydra-style env overrides
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dexverse.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from dexverse.tasks.utils import parse_env_cfg


def _set_cfg_value(cfg_obj, key_path: list[str], value):
    """Set a nested value on a config object using dot-list keys."""
    target = cfg_obj
    for key in key_path[:-1]:
        if hasattr(target, key):
            target = getattr(target, key)
        elif isinstance(target, dict):
            target = target[key]
        elif isinstance(target, list):
            target = target[int(key)]
        else:
            raise AttributeError(f"Cannot traverse key '{key}' in path '{'.'.join(key_path)}'.")

    final_key = key_path[-1]
    if hasattr(target, final_key):
        setattr(target, final_key, value)
    elif isinstance(target, dict):
        target[final_key] = value
    elif isinstance(target, list):
        target[int(final_key)] = value
    else:
        raise AttributeError(f"Cannot set key '{final_key}' in path '{'.'.join(key_path)}'.")


def _apply_hydra_env_overrides(env_cfg, unknown_args: list[str]):
    """Apply Hydra-style env overrides (e.g. env.robot_type=shadow) onto env_cfg."""
    if not unknown_args:
        return

    env_dotlist = []
    unsupported_args = []
    for arg in unknown_args:
        if "=" not in arg:
            unsupported_args.append(arg)
            continue

        key, value = arg.split("=", maxsplit=1)
        key = key.lstrip("+")
        if key.startswith("env."):
            env_dotlist.append(f"{key[len('env.'):]}={value}")
        else:
            unsupported_args.append(arg)

    if unsupported_args:
        raise ValueError(
            "Unsupported extra arguments: "
            f"{unsupported_args}. Only Hydra-style 'env.<path>=<value>' overrides are supported."
        )

    if not env_dotlist:
        return

    env_overrides = OmegaConf.to_container(OmegaConf.from_dotlist(env_dotlist), resolve=True)

    def _apply_recursive(prefix: list[str], item):
        if isinstance(item, dict):
            for child_key, child_value in item.items():
                _apply_recursive([*prefix, child_key], child_value)
        else:
            _set_cfg_value(env_cfg, prefix, item)

    _apply_recursive([], env_overrides)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        json_path=args_cli.json_path,
    )
    # Collect kwargs that require a config re-instantiation to take effect.
    # These are fields that __post_init__ derives other values from, so they
    # must be set before __post_init__ runs rather than patched in afterwards.
    reinit_kwargs = {}
    if args_cli.robot_type is not None:
        if not hasattr(env_cfg, "robot_type"):
            raise ValueError(
                f"Task '{args_cli.task}' does not expose robot_type; cannot apply override '{args_cli.robot_type}'."
            )
        reinit_kwargs["robot_type"] = args_cli.robot_type
    # --object_usd accepts 1..N paths: single file, directory, or explicit list.
    if args_cli.object_usd is not None:
        if not hasattr(env_cfg, "object_usd_path"):
            raise ValueError(
                f"Task '{args_cli.task}' does not expose 'object_usd_path'; cannot apply --object_usd override."
            )
        paths = args_cli.object_usd  # list[str] from nargs='+'
        if len(paths) == 1:
            # Single entry: could be a file or a directory; store in object_usd_path.
            reinit_kwargs["object_usd_path"] = paths[0]
        else:
            # Explicit list of files: store in object_usd_paths.
            reinit_kwargs["object_usd_paths"] = paths

    for attr, val in [
        ("object_half_height", args_cli.object_half_height),
        ("object_mass", args_cli.object_mass),
    ]:
        if val is not None:
            if not hasattr(env_cfg, attr):
                raise ValueError(f"Task '{args_cli.task}' does not expose '{attr}'; cannot apply override.")
            reinit_kwargs[attr] = val

    if reinit_kwargs:
        # Re-create config with all overridden params at once so that __post_init__
        # runs with the correct values (avoids stale derived state).
        cfg_cls = type(env_cfg)
        env_cfg = cfg_cls(**reinit_kwargs)
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
    _apply_hydra_env_overrides(env_cfg, hydra_args)
    # Hydra overrides patch fields after __post_init__, so if any object flat params
    # (object_usd_path, object_mass, etc.) were changed via "env.object_usd_path=..."
    # we must rebuild scene.object now to propagate those changes.
    if hasattr(env_cfg, "rebuild_object_from_params"):
        env_cfg.rebuild_object_from_params()

    # Apply observation preset (CLI arg wins; otherwise honour whatever
    # __post_init__/Hydra left on env_cfg.observation_preset). Presets must run
    # after Hydra overrides so the multiview-camera wiring sees final state.
    preset = args_cli.observation_preset or getattr(env_cfg, "observation_preset", None)
    if preset and hasattr(env_cfg, "_apply_observation_preset"):
        env_cfg.observation_preset = preset
        env_cfg._apply_observation_preset(preset)

    # Optional CLI overrides for the palm init pose (translation and/or rotation
    # in world frame). Routes to the floating-translation or arm IK path based
    # on env_cfg.robot_type.
    if args_cli.wrist_xyz is not None or args_cli.wrist_rot is not None:
        from dexverse.tasks.config.robot_init import set_robot_wrist_init_world_pos

        kwargs = {}
        if args_cli.wrist_xyz is not None:
            kwargs["x"], kwargs["y"], kwargs["z"] = args_cli.wrist_xyz
        if args_cli.wrist_rot is not None:
            kwargs["rot"] = tuple(args_cli.wrist_rot)
        print(f"[zero_agent] setting wrist init world pose: {kwargs}")
        set_robot_wrist_init_world_pos(env_cfg, **kwargs)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # Debug mode: render only, never step physics. Useful when joints blow up
    # on the first physics step — you can see the post-spawn pose without
    # the sim destabilizing further.
    if args_cli.no_step:
        sim = env.unwrapped.sim
        print("[zero_agent] --no_step: rendering only; ctrl+c or close the GUI to exit")
        while simulation_app.is_running():
            sim.render()
        env.close()
        return

    # simulate environment
    step_idx = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            actions *= 0.0
            # actions[0,22] = 0.1
            # apply actions
            next_obs, reward, terminated, truncated, info = env.step(actions)
        step_idx += 1
        if args_cli.max_steps is not None and step_idx >= args_cli.max_steps:
            print(f"[zero_agent] reached --max_steps={args_cli.max_steps}; exiting cleanly")
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
