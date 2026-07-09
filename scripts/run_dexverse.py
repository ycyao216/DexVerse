# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run DexVerse environments with joint slider UI.")
parser.add_argument("--task", type=str, default="Dexverse-PourWineGlass-v0", help="Name of the task.")
parser.add_argument(
    "--json_path",
    type=str,
    default=None,
    help="Path to template JSON spec. Required for *Template environments.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed used for the environment (None=unset, -1=random).",
)
parser.add_argument(
    "--disable_background_randomization",
    action="store_true",
    help="Disable reset_environment_background event to keep skyLight/HDR unchanged.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
app = app_launcher.app

import dexverse.tasks  # noqa: F401  # triggers environment registration
import gymnasium as gym
from dexverse.tasks.utils import parse_env_cfg

AVAILABLE_ENVS = [
    "Dexverse-BimanualHandover-v0",
    "Dexverse-BimanualLiftBasket-v0",
    "Dexverse-BimanualLiftTray-v0",
    "Dexverse-FixateThenManipulate-CabinetDoubleDoor-v0",
    "Dexverse-FixateThenManipulate-LiftBasketHandle-v0",
    "Dexverse-FixateThenManipulate-OpenFlatFolder-v0",
    "Dexverse-FixateThenManipulate-OpenHuaweiPhone-v0",
    "Dexverse-FixateThenManipulate-OpenLaptop-v0",
    "Dexverse-FixateThenManipulate-SlideUtilityKnife-v0",
    "Dexverse-FixateThenManipulate-SqueezeScissors-v0",
    "Dexverse-FixateThenManipulate-SqueezeStapler-v0",
    "Dexverse-FixateThenManipulate-UnscrewCap-v0",
    "Dexverse-GarmentSoftbody-v0",
    "Dexverse-GearMesh-v0",
    "Dexverse-GraspBucket-v0",
    "Dexverse-GraspPot-v0",
    "Dexverse-GraspTwoItems-v0",
    "Dexverse-InHandApple-v0",
    "Dexverse-InsertPeg-v0",
    "Dexverse-LiftLid-v0",
    "Dexverse-LiftObject-v0",
    "Dexverse-LiftingPot-v0",
    "Dexverse-LongHorizon-ClutteredPhonePickup-v0",
    "Dexverse-LongHorizon-MicrowaveRetrievePlace-v0",
    "Dexverse-NutThread-v0",
    "Dexverse-OpenDoor-v0",
    "Dexverse-OpenDrawer-v0",
    "Dexverse-OpenFaucet-v0",
    "Dexverse-OpenMicrowave-v0",
    "Dexverse-PickFromClutter-v0",
    "Dexverse-PickThinObjectFromContainer-v0",
    "Dexverse-PickUpCan-v0",
    "Dexverse-PickUpCards-v0",
    "Dexverse-PickUpStick-v0",
    "Dexverse-PivotLargeCuboidAgainstWall-v0",
    "Dexverse-PlugCharger-v0",
    "Dexverse-PourWineGlass-v0",
    "Dexverse-PourObject-v0",
    "Dexverse-PushButton-v0",
    "Dexverse-PushSmallSphereObstacleSlope-v0",
    "Dexverse-PushSphereUpSlope-v0",
    "Dexverse-PushT-v0",
    "Dexverse-Relocate-v0",
    "Dexverse-RelocateObject-v0",
    "Dexverse-RotateKnob-v0",
    "Dexverse-SingleJointPool-v0",
    "Dexverse-TakeBookOffShelf-v0",
    "Dexverse-TexasPoker-v0",
    "Dexverse-TurnOnSwitch-v0",
]
print("[available envs]")
for env_id in AVAILABLE_ENVS:
    print(f"  - {env_id}")

task = args_cli.task
env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs, json_path=args_cli.json_path)
if args_cli.seed is not None:
    env_cfg.seed = args_cli.seed
if args_cli.disable_background_randomization and hasattr(env_cfg, "events"):
    if hasattr(env_cfg.events, "reset_environment_background"):
        env_cfg.events.reset_environment_background = None
        print("[INFO] Disabled reset_environment_background event via CLI.")
env_cfg.env_name = task
env = gym.make(task, cfg=env_cfg)
obs, _ = env.reset()

desired_pos = {}
for art in env.unwrapped.scene.articulations.values():
    desired_pos[art.cfg.prim_path] = art.data.joint_pos[0].detach().cpu().numpy().tolist()
    init_pos = torch.tensor([desired_pos[art.cfg.prim_path]], device=art.device, dtype=art.data.joint_pos.dtype)
    init_vel = torch.zeros_like(init_pos)
    env_ids = torch.tensor([0], device=art.device, dtype=torch.int32)
    art.set_joint_position_target(init_pos, env_ids=env_ids)
    art.set_joint_velocity_target(init_vel, env_ids=env_ids)

if "robot" in env.unwrapped.scene.articulations:
    robot = env.unwrapped.scene.articulations["robot"]
    cur = robot.data.joint_pos[0].detach().cpu()
    ref = robot.data.default_joint_pos[0].detach().cpu()
    max_abs_err = torch.max(torch.abs(cur - ref)).item()
    print("[verify][reset] robot joint names:")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i}] {name}")
    print(f"[verify][reset] max_abs_err(current_qpos - default_joint_pos) = {max_abs_err:.6e}")


while app.is_running():
    # Keep all articulations at slider/desired positions and advance simulation.
    for art_name, art in env.unwrapped.scene.articulations.items():
        pos = torch.tensor([desired_pos[art.cfg.prim_path]], device=art.device, dtype=art.data.joint_pos.dtype)
        vel = torch.zeros_like(pos)
        art.write_joint_state_to_sim(pos, vel, env_ids=torch.tensor([0], device=art.device, dtype=torch.int32))
    env.unwrapped.sim.step()
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
env.close()
app.close()
