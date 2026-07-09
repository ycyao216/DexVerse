# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with DexVerse environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers)."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for DexVerse environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
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
    "--json_path",
    type=str,
    default=None,
    help="Path to template JSON spec. Required for *Template environments.",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
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
        " DexPilot pinch/wrist vectors; 'vector' matches palm->fingertip"
        " vectors. Orthogonal to --teleop_retargeter. Currently wired for the"
        " Shadow hand variants."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for dex-retargeting and some IK controllers).",
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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, unknown_args = parser.parse_known_args()

# Support lightweight Hydra-style env override for common teleop use-cases.
# Examples: env.robot_type=floating_shadow_right, env.usd_path=/path/to/obj.usd,
#          env.teleop_retargeter=absolute
hydra_env_overrides: list[str] = []
for raw_arg in unknown_args:
    if "=" not in raw_arg:
        parser.error(f"unrecognized arguments: {raw_arg}")
    key, value = raw_arg.split("=", 1)
    key_without_add = key.lstrip("+")
    if key_without_add == "env.robot_type":
        args_cli.robot_type = raw_arg.split("=", 1)[1]
    elif key_without_add == "env.usd_path":
        args_cli.usd_path = value
    elif key_without_add == "env.teleop_retargeter":
        args_cli.teleop_retargeter = value
    elif key_without_add == "env.retargeting_scheme":
        args_cli.retargeting_scheme = value
    elif key_without_add == "env.enable_debug_vis":
        args_cli.enable_debug_vis = value.strip().lower() in ("1", "true", "yes")
    elif key_without_add.startswith("env."):
        hydra_env_overrides.append(raw_arg)
    else:
        parser.error(f"unrecognized arguments: {raw_arg}")

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim. Pinocchio is required by the Pink IK controllers and the
    # dex-retargeting utilities (e.g., LeapHandDexRetargeting)
    import pinocchio  # noqa: F401

if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging

import dexverse.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from dexverse.tasks.config.floating_teleop import (
    apply_teleop_retargeter_mode,
    apply_teleop_retargeting_scheme,
)
from dexverse.tasks.utils import parse_env_cfg, prune_stale_obs_refs, strip_camera_cfgs
from isaaclab.devices import (
    Se3Gamepad,
    Se3GamepadCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
)
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _set_cfg_value(cfg_obj, key_path: list[str], value) -> None:
    """Set a nested value on a config object using Hydra-style dot-list keys."""
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


def _apply_hydra_env_overrides(env_cfg, raw_overrides: list[str]) -> None:
    if not raw_overrides:
        return

    env_dotlist = []
    for raw_arg in raw_overrides:
        key, value = raw_arg.split("=", maxsplit=1)
        key = key.lstrip("+")
        env_dotlist.append(f"{key[len('env.'):]}={value}")

    env_overrides = OmegaConf.to_container(OmegaConf.from_dotlist(env_dotlist), resolve=True)

    def _apply_recursive(prefix: list[str], item) -> None:
        if isinstance(item, dict):
            for child_key, child_value in item.items():
                _apply_recursive([*prefix, child_key], child_value)
        else:
            _set_cfg_value(env_cfg, prefix, item)

    _apply_recursive([], env_overrides)


def main() -> None:  # noqa: C901
    """
    Run teleoperation with a DexVerse environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    json_path = getattr(args_cli, "json_path", None)
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, json_path=json_path, device=args_cli.device, num_envs=args_cli.num_envs)
    # Collect field-level overrides that require a clean rebuild (avoid stale
    # nested config state left by the previous __post_init__).
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
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
    _apply_hydra_env_overrides(env_cfg, hydra_env_overrides)
    env_cfg.env_name = args_cli.task

    # Swap retargeter cfgs in place after the per-task __post_init__ has
    # fully built env_cfg.teleop_devices. Keeping this out of the env cfg
    # itself avoids plumbing the flag through every task config.
    if hasattr(env_cfg, "teleop_devices"):
        apply_teleop_retargeter_mode(
            env_cfg.teleop_devices,
            args_cli.teleop_retargeter,
            anchor_pos_offsets=getattr(env_cfg, "retargeter_anchor_pos_offsets", None),
        )
        apply_teleop_retargeting_scheme(env_cfg.teleop_devices, args_cli.retargeting_scheme)

    # Template environments consume json_path internally via parse_env_cfg.

    # modify configuration
    env_cfg.terminations.time_out = None

    # Task-specific modifications
    if getattr(env_cfg, "supports_object_pose_command", False):
        # Set the resampling time range to large number to avoid resampling during teleoperation
        obj_cmd = getattr(getattr(env_cfg, "commands", None), "object_pose", None)
        if obj_cmd is not None:
            obj_cmd.resampling_time_range = (1.0e9, 1.0e9)
        # Add termination condition for reaching the goal otherwise the environment won't reset
        # Note: This assumes the task has an object_reached_goal function in its mdp module
        # You may need to adjust this based on your specific task structure
        if hasattr(env_cfg, "terminations"):
            # Only add if not already present
            if not hasattr(env_cfg.terminations, "object_reached_goal"):
                try:
                    from dexverse.tasks import mdp

                    env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
                except (ImportError, AttributeError):
                    logger.warning(
                        "Could not import object_reached_goal termination. Some tasks may not reset properly."
                    )

    if args_cli.xr:
        # Set camera cfgs to None (instead of delattr) so configclass fields stay valid.
        env_cfg = strip_camera_cfgs(env_cfg)
        # Null out any remaining obs terms (e.g. vision / perception groups)
        # that still point at removed cameras.
        env_cfg = prune_stale_obs_refs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach, we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Configure the teleop device in the environment config.")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")
    if args_cli.xr:
        print("For XR devices: Use START/STOP gestures or buttons to control teleoperation.")

    # Module-level variables for debug tracking
    _printed_bodies = False
    _debug_counter = 0

    def teleop_start_callback():
        start_teleoperation()
        print("teleop start callback")
        if hasattr(teleop_interface, "_retargeters") and teleop_interface._retargeters:
            if hasattr(teleop_interface._retargeters[0], "calibrate_wrist_pose"):
                teleop_interface._retargeters[0].calibrate_wrist_pose()

    def teleop_stop_callback():
        stop_teleoperation()

    teleop_interface.add_callback("START", teleop_start_callback)
    teleop_interface.add_callback("STOP", teleop_stop_callback)

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # Update robot wrist pose for visualization (if retargeter supports it)
                if hasattr(teleop_interface, "_retargeters") and teleop_interface._retargeters:
                    try:
                        # Get robot wrist body pose
                        # Try z_rotation_link first (actual wrist frame after all rotations)
                        # Fall back to palm_wrist if z_rotation_link doesn't exist
                        robot = env.scene["robot"]
                        body_names = robot.body_names

                        # Debug: Print available bodies (only once)
                        if not _printed_bodies:
                            logger.info(f"Available robot body names: {body_names}")
                            _printed_bodies = True

                        wrist_body_name = None
                        wrist_body_idx = None

                        # Try z_rotation_link first (this is the actual wrist frame)
                        if "z_rotation_link" in body_names:
                            wrist_body_name = "z_rotation_link"
                            wrist_body_idx = body_names.index("z_rotation_link")
                        elif "palm_wrist" in body_names:
                            wrist_body_name = "palm_wrist"
                            wrist_body_idx = body_names.index("palm_wrist")
                        elif "palm" in body_names:
                            wrist_body_name = "palm"
                            wrist_body_idx = body_names.index("palm")

                        if wrist_body_idx is not None:
                            # Get body pose in world frame [pos (3), quat (4)]
                            body_pose_w = robot.data.body_link_pose_w[0, wrist_body_idx, :7]  # Shape (7,)

                            # Debug: Print wrist orientation (only occasionally to avoid spam)
                            _debug_counter += 1
                            if _debug_counter % 60 == 0:  # Print every 60 frames (~1 second at 60fps)
                                quat_wxyz = body_pose_w[3:].cpu().numpy()  # w, x, y, z
                                logger.info(
                                    f"Robot wrist ({wrist_body_name}) pose - Position: [{body_pose_w[0]:.3f},"
                                    f" {body_pose_w[1]:.3f}, {body_pose_w[2]:.3f}], Quaternion (w,x,y,z):"
                                    f" [{quat_wxyz[0]:.3f}, {quat_wxyz[1]:.3f}, {quat_wxyz[2]:.3f}, {quat_wxyz[3]:.3f}]"
                                )

                            # Update retargeter with robot wrist pose
                            for retargeter in teleop_interface._retargeters:
                                if hasattr(retargeter, "set_robot_wrist_pose"):
                                    retargeter.set_robot_wrist_pose(body_pose_w.cpu().numpy())
                    except (AttributeError, KeyError, ValueError):
                        # Silently fail if robot or body not found (not all environments have wrist body)
                        pass

                # get device command
                action = teleop_interface.advance()

                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1)
                    # apply actions
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")

        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
