# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Combined multi-object grasp environment templates."""

import gymnasium as gym
from isaaclab.devices.openxr import XrCfg
from isaaclab.utils import configclass

from .. import agents
from ..floating_teleop import DEFAULT_FLOATING_SHADOW_XR_CFG, setup_floating_teleop
from . import (
    graspbucket_rigid_variants_cfg,
    grasppot_rigid_variants_cfg,
    liftlid_rigid_variants_cfg,
    opendoor_rigid_variants_cfg,
    opendrawer_rigid_variants_cfg,
    openmicrowave_rigid_variants_cfg,
    pourmug_rigid_variants_cfg,
    pushbutton_rigid_variants_cfg,
    rotateknob_rigid_variants_cfg,
    turnonswitch_rigid_variants_cfg,
)


@configclass
class _ArticulationRigidTeleopEnvCfg:
    """Shared teleoperation wiring for articulation+rigid variants."""

    robot_type: str = "floating_shadow_right"
    xr: XrCfg = DEFAULT_FLOATING_SHADOW_XR_CFG

    def __post_init__(self):
        super().__post_init__()
        setup_floating_teleop(self)


@configclass
class OpenMicrowavePickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    openmicrowave_rigid_variants_cfg.OpenMicrowavePickUpCanEnvCfg,
):
    """Open-microwave + can teleop variant."""


@configclass
class OpenMicrowavePickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    openmicrowave_rigid_variants_cfg.OpenMicrowavePickUpStickEnvCfg,
):
    """Open-microwave + stick teleop variant."""


@configclass
class OpenMicrowavePourMugTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    openmicrowave_rigid_variants_cfg.OpenMicrowavePourMugEnvCfg,
):
    """Open-microwave + mug teleop variant."""


@configclass
class OpenMicrowaveRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    openmicrowave_rigid_variants_cfg.OpenMicrowaveRelocateSphereEnvCfg,
):
    """Open-microwave + relocate sphere teleop variant."""


@configclass
class OpenDoorPickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    opendoor_rigid_variants_cfg.OpenDoorPickUpCanEnvCfg,
):
    """Open-door + can teleop variant."""


@configclass
class OpenDoorPickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    opendoor_rigid_variants_cfg.OpenDoorPickUpStickEnvCfg,
):
    """Open-door + stick teleop variant."""


@configclass
class OpenDoorPourMugTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    opendoor_rigid_variants_cfg.OpenDoorPourMugEnvCfg,
):
    """Open-door + mug teleop variant."""


@configclass
class OpenDoorRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    opendoor_rigid_variants_cfg.OpenDoorRelocateSphereEnvCfg,
):
    """Open-door + relocate sphere teleop variant."""


@configclass
class OpenDrawerPickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    opendrawer_rigid_variants_cfg.OpenDrawerPickUpCanEnvCfg,
):
    """Open-drawer + can teleop variant."""


@configclass
class OpenDrawerPickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    opendrawer_rigid_variants_cfg.OpenDrawerPickUpStickEnvCfg,
):
    """Open-drawer + stick teleop variant."""


@configclass
class OpenDrawerPourMugTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    opendrawer_rigid_variants_cfg.OpenDrawerPourMugEnvCfg,
):
    """Open-drawer + mug teleop variant."""


@configclass
class OpenDrawerRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    opendrawer_rigid_variants_cfg.OpenDrawerRelocateSphereEnvCfg,
):
    """Open-drawer + relocate sphere teleop variant."""


@configclass
class LiftLidPickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    liftlid_rigid_variants_cfg.LiftLidPickUpCanEnvCfg,
):
    """Lift-lid + can teleop variant."""


@configclass
class LiftLidPickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    liftlid_rigid_variants_cfg.LiftLidPickUpStickEnvCfg,
):
    """Lift-lid + stick teleop variant."""


@configclass
class LiftLidPourMugTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    liftlid_rigid_variants_cfg.LiftLidPourMugEnvCfg,
):
    """Lift-lid + mug teleop variant."""


@configclass
class LiftLidRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    liftlid_rigid_variants_cfg.LiftLidRelocateSphereEnvCfg,
):
    """Lift-lid + relocate sphere teleop variant."""


@configclass
class GraspPotPickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    grasppot_rigid_variants_cfg.GraspPotPickUpCanEnvCfg,
):
    """Grasp-pot + can teleop variant."""


@configclass
class GraspPotPickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    grasppot_rigid_variants_cfg.GraspPotPickUpStickEnvCfg,
):
    """Grasp-pot + stick teleop variant."""


@configclass
class GraspPotPourMugTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    grasppot_rigid_variants_cfg.GraspPotPourMugEnvCfg,
):
    """Grasp-pot + mug teleop variant."""


@configclass
class GraspPotRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    grasppot_rigid_variants_cfg.GraspPotRelocateSphereEnvCfg,
):
    """Grasp-pot + relocate sphere teleop variant."""


@configclass
class PushButtonPickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    pushbutton_rigid_variants_cfg.PushButtonPickUpCanEnvCfg,
):
    """Push-button + can teleop variant."""


@configclass
class PushButtonPickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    pushbutton_rigid_variants_cfg.PushButtonPickUpStickEnvCfg,
):
    """Push-button + stick teleop variant."""


@configclass
class PushButtonPourMugTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    pushbutton_rigid_variants_cfg.PushButtonPourMugEnvCfg,
):
    """Push-button + mug teleop variant."""


@configclass
class PushButtonRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    pushbutton_rigid_variants_cfg.PushButtonRelocateSphereEnvCfg,
):
    """Push-button + relocate sphere teleop variant."""


@configclass
class RotateKnobPickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    rotateknob_rigid_variants_cfg.RotateKnobPickUpCanEnvCfg,
):
    """Rotate-knob + can teleop variant."""


@configclass
class RotateKnobPickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    rotateknob_rigid_variants_cfg.RotateKnobPickUpStickEnvCfg,
):
    """Rotate-knob + stick teleop variant."""


@configclass
class RotateKnobPourMugTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    rotateknob_rigid_variants_cfg.RotateKnobPourMugEnvCfg,
):
    """Rotate-knob + mug teleop variant."""


@configclass
class RotateKnobRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    rotateknob_rigid_variants_cfg.RotateKnobRelocateSphereEnvCfg,
):
    """Rotate-knob + relocate sphere teleop variant."""


@configclass
class TurnOnSwitchPickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    turnonswitch_rigid_variants_cfg.TurnOnSwitchPickUpCanEnvCfg,
):
    """Turn-on-switch + can teleop variant."""


@configclass
class TurnOnSwitchPickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    turnonswitch_rigid_variants_cfg.TurnOnSwitchPickUpStickEnvCfg,
):
    """Turn-on-switch + stick teleop variant."""


@configclass
class TurnOnSwitchPourMugTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    turnonswitch_rigid_variants_cfg.TurnOnSwitchPourMugEnvCfg,
):
    """Turn-on-switch + mug teleop variant."""


@configclass
class TurnOnSwitchRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    turnonswitch_rigid_variants_cfg.TurnOnSwitchRelocateSphereEnvCfg,
):
    """Turn-on-switch + relocate sphere teleop variant."""


@configclass
class GraspBucketPickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    graspbucket_rigid_variants_cfg.GraspBucketPickUpCanEnvCfg,
):
    """Grasp-bucket + can teleop variant."""


@configclass
class GraspBucketPickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    graspbucket_rigid_variants_cfg.GraspBucketPickUpStickEnvCfg,
):
    """Grasp-bucket + stick teleop variant."""


@configclass
class GraspBucketPourMugTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    graspbucket_rigid_variants_cfg.GraspBucketPourMugEnvCfg,
):
    """Grasp-bucket + mug teleop variant."""


@configclass
class GraspBucketRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    graspbucket_rigid_variants_cfg.GraspBucketRelocateSphereEnvCfg,
):
    """Grasp-bucket + relocate sphere teleop variant."""


@configclass
class PourMugPickUpCanTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    pourmug_rigid_variants_cfg.PourMugPickUpCanEnvCfg,
):
    """Pour-mug + can teleop variant."""


@configclass
class PourMugPickUpStickTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    pourmug_rigid_variants_cfg.PourMugPickUpStickEnvCfg,
):
    """Pour-mug + stick teleop variant."""


@configclass
class PourMugRelocateSphereTeleopEnvCfg(
    _ArticulationRigidTeleopEnvCfg,
    pourmug_rigid_variants_cfg.PourMugRelocateSphereEnvCfg,
):
    """Pour-mug + sphere teleop variant."""


_COMMON_KWARGS = {
    "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
}


def _register_env(base_id: str, class_name: str) -> None:
    kwargs = {
        **_COMMON_KWARGS,
        "env_cfg_entry_point": f"{__name__}:{class_name}",
    }
    gym.register(
        id=base_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs=kwargs,
    )


for _base_id, _class_name in (
    ("Dexverse-OpenMicrowavePickUpCan-v0", "OpenMicrowavePickUpCanTeleopEnvCfg"),
    ("Dexverse-OpenMicrowavePickUpStick-v0", "OpenMicrowavePickUpStickTeleopEnvCfg"),
    ("Dexverse-OpenMicrowavePourMug-v0", "OpenMicrowavePourMugTeleopEnvCfg"),
    ("Dexverse-OpenMicrowaveRelocateSphere-v0", "OpenMicrowaveRelocateSphereTeleopEnvCfg"),
    ("Dexverse-OpenDoorPickUpCan-v0", "OpenDoorPickUpCanTeleopEnvCfg"),
    ("Dexverse-OpenDoorPickUpStick-v0", "OpenDoorPickUpStickTeleopEnvCfg"),
    ("Dexverse-OpenDoorPourMug-v0", "OpenDoorPourMugTeleopEnvCfg"),
    ("Dexverse-OpenDoorRelocateSphere-v0", "OpenDoorRelocateSphereTeleopEnvCfg"),
    ("Dexverse-OpenDrawerPickUpCan-v0", "OpenDrawerPickUpCanTeleopEnvCfg"),
    ("Dexverse-OpenDrawerPickUpStick-v0", "OpenDrawerPickUpStickTeleopEnvCfg"),
    ("Dexverse-OpenDrawerPourMug-v0", "OpenDrawerPourMugTeleopEnvCfg"),
    ("Dexverse-OpenDrawerRelocateSphere-v0", "OpenDrawerRelocateSphereTeleopEnvCfg"),
    ("Dexverse-LiftLidPickUpCan-v0", "LiftLidPickUpCanTeleopEnvCfg"),
    ("Dexverse-LiftLidPickUpStick-v0", "LiftLidPickUpStickTeleopEnvCfg"),
    ("Dexverse-LiftLidPourMug-v0", "LiftLidPourMugTeleopEnvCfg"),
    ("Dexverse-LiftLidRelocateSphere-v0", "LiftLidRelocateSphereTeleopEnvCfg"),
    ("Dexverse-GraspPotPickUpCan-v0", "GraspPotPickUpCanTeleopEnvCfg"),
    ("Dexverse-GraspPotPickUpStick-v0", "GraspPotPickUpStickTeleopEnvCfg"),
    ("Dexverse-GraspPotPourMug-v0", "GraspPotPourMugTeleopEnvCfg"),
    ("Dexverse-GraspPotRelocateSphere-v0", "GraspPotRelocateSphereTeleopEnvCfg"),
    ("Dexverse-PushButtonPickUpCan-v0", "PushButtonPickUpCanTeleopEnvCfg"),
    ("Dexverse-PushButtonPickUpStick-v0", "PushButtonPickUpStickTeleopEnvCfg"),
    ("Dexverse-PushButtonPourMug-v0", "PushButtonPourMugTeleopEnvCfg"),
    ("Dexverse-PushButtonRelocateSphere-v0", "PushButtonRelocateSphereTeleopEnvCfg"),
    ("Dexverse-RotateKnobPickUpCan-v0", "RotateKnobPickUpCanTeleopEnvCfg"),
    ("Dexverse-RotateKnobPickUpStick-v0", "RotateKnobPickUpStickTeleopEnvCfg"),
    ("Dexverse-RotateKnobPourMug-v0", "RotateKnobPourMugTeleopEnvCfg"),
    ("Dexverse-RotateKnobRelocateSphere-v0", "RotateKnobRelocateSphereTeleopEnvCfg"),
    ("Dexverse-TurnOnSwitchPickUpCan-v0", "TurnOnSwitchPickUpCanTeleopEnvCfg"),
    ("Dexverse-TurnOnSwitchPickUpStick-v0", "TurnOnSwitchPickUpStickTeleopEnvCfg"),
    ("Dexverse-TurnOnSwitchPourMug-v0", "TurnOnSwitchPourMugTeleopEnvCfg"),
    ("Dexverse-TurnOnSwitchRelocateSphere-v0", "TurnOnSwitchRelocateSphereTeleopEnvCfg"),
    ("Dexverse-GraspBucketPickUpCan-v0", "GraspBucketPickUpCanTeleopEnvCfg"),
    ("Dexverse-GraspBucketPickUpStick-v0", "GraspBucketPickUpStickTeleopEnvCfg"),
    ("Dexverse-GraspBucketPourMug-v0", "GraspBucketPourMugTeleopEnvCfg"),
    ("Dexverse-GraspBucketRelocateSphere-v0", "GraspBucketRelocateSphereTeleopEnvCfg"),
    ("Dexverse-PourMugPickUpCan-v0", "PourMugPickUpCanTeleopEnvCfg"),
    ("Dexverse-PourMugPickUpStick-v0", "PourMugPickUpStickTeleopEnvCfg"),
    ("Dexverse-PourMugRelocateSphere-v0", "PourMugRelocateSphereTeleopEnvCfg"),
):
    _register_env(_base_id, _class_name)
