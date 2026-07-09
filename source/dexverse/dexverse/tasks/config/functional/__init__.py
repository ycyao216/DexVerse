# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functional-grasping task configurations.

Each task fixes a single USD asset (one object per task) since each object's
functional grasp differs. Some tasks use a lift-to-goal success condition,
while pouring tasks require lifting and tilting the object into a pour pose.
"""

from ...utils.registration import register_env

register_env(__name__, "Dexverse-GraspBleach-v0", "grasp_bleach_cfg", "GraspBleachEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-GraspPan-v0", "grasp_pan_cfg", "GraspPanEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-GraspKettle-v0", "grasp_kettle_cfg", "GraspKettleEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-GraspCup-v0", "grasp_cup_cfg", "GraspCupEnvFloatingDexHandRightCfg")
register_env(
    __name__, "Dexverse-RemoveCupFromRack-v0", "remove_cup_from_rack_cfg", "RemoveCupFromRackEnvFloatingDexHandRightCfg"
)
register_env(__name__, "Dexverse-FunctionalPourCan-v0", "pour_can_cfg", "PourCanEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-FunctionalPourMug-v0", "pour_mug_cfg", "PourMugEnvFloatingDexHandRightCfg")
register_env(
    __name__, "Dexverse-FunctionalHammerStrike-v0", "hammer_strike_cfg", "HammerStrikeEnvFloatingDexHandRightCfg"
)
register_env(__name__, "Dexverse-FunctionalDrillApply-v0", "drill_apply_cfg", "DrillApplyEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-PourWineGlass-v0", "pour_wine_glass_cfg", "PourWineGlassEnvFloatingDexHandRightCfg")
