# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation task configurations (actuating articulated joints; single- and two-hand)."""

from ...utils.registration import register_env

register_env(__name__, "Dexverse-OpenCabinet-v0", "opencabinet_cfg", "OpenCabinetEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-LiftLid-v0", "liftlid_cfg", "LiftLidEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-OpenDoor-v0", "opendoor_cfg", "OpenDoorEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-RotateKnob-v0", "rotateknob_cfg", "RotateKnobEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-OpenFaucet-v0", "openfaucet_cfg", "OpenFaucetEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-OpenMicrowave-v0", "openmicrowave_cfg", "OpenMicrowaveEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-OpenDrawer-v0", "opendrawer_cfg", "OpenDrawerEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-TurnOnSwitch-v0", "turnonswitch_cfg", "TurnOnSwitchEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-GraspBucket-v0", "graspbucket_cfg", "GraspBucketEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-LiftBucket-v0", "lift_bucket_cfg", "LiftBucketEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-GraspPot-v0", "grasppot_cfg", "GraspPotEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-PushButton-v0", "pushbutton_cfg", "PushButtonEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-OpenLaptop-v0", "open_laptop_cfg", "OpenLaptopEnvFloatingDexHandRightCfg")
register_env(
    __name__, "Dexverse-SqueezeScissors-v0", "squeeze_scissors_cfg", "SqueezeScissorsEnvFloatingDexHandRightCfg"
)
register_env(
    __name__, "Dexverse-SlideUtilityKnife-v0", "slide_utility_knife_cfg", "SlideUtilityKnifeEnvFloatingDexHandRightCfg"
)
register_env(
    __name__, "Dexverse-LiftBasketHandle-v0", "lift_basket_handle_cfg", "LiftBasketHandleEnvFloatingDexHandRightCfg"
)
register_env(__name__, "Dexverse-OpenStapler-v0", "open_stapler_cfg", "OpenStaplerEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-OpenFlatFolder-v0", "open_flat_folder_cfg", "OpenFlatFolderEnvFloatingDexHandRightCfg")
register_env(
    __name__, "Dexverse-OpenHuaweiPhone-v0", "open_huawei_phone_cfg", "OpenHuaweiPhoneEnvFloatingDexHandRightCfg"
)
register_env(
    __name__, "Dexverse-CabinetDoubleDoor-v0", "cabinet_double_door", "CabinetDoubleDoorEnvFloatingDexHandRightCfg"
)
register_env(__name__, "Dexverse-UnscrewCap-v0", "unscrew_cap_cfg", "UnscrewCapEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-OpenGlasses-v0", "open_glasses_cfg", "OpenGlassesEnvFloatingDexHandRightCfg")
