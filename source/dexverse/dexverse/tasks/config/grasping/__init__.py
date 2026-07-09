# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasping task configurations (basic prehensile pick / place / stack / relocate)."""

from ...utils.registration import register_env

register_env(__name__, "Dexverse-PickCube-v0", "pick_cube_cfg", "PickCubeEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-StackCube-v0", "stack_cube_cfg", "StackCubeEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-RelocateSphere-v0", "relocate_sphere_cfg", "RelocateEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-PickUpStick-v0", "pick_up_stick_cfg", "PickUpStickEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-RelocateObject-v0", "relocate_object_cfg", "RelocateObjectEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-GraspTwoItems-v0", "grasp_two_items_cfg", "GraspTwoItemsEnvFloatingDexHandRightCfg")
