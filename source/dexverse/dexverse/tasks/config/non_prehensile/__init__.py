# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Non-prehensile task configurations (push / pivot — move objects without grasping)."""

from ...utils.registration import register_env

register_env(
    __name__, "Dexverse-PushSphereUpSlope-v0", "push_sphere_up_slope_cfg", "PushSphereUpSlopeEnvFloatingDexHandRightCfg"
)
register_env(
    __name__,
    "Dexverse-PushSmallSphereObstacleSlope-v0",
    "push_small_sphere_obstacle_slope_cfg",
    "PushSmallSphereObstacleSlopeEnvFloatingDexHandRightCfg",
)
register_env(__name__, "Dexverse-PushT-v0", "pusht_cfg", "PushTEnvFloatingDexHandRightCfg")
register_env(
    __name__,
    "Dexverse-PivotLargeCuboidAgainstWall-v0",
    "pivot_large_cuboid_against_wall_cfg",
    "PivotLargeCuboidAgainstWallEnvFloatingDexHandRightCfg",
)
register_env(
    __name__, "Dexverse-TakeBookOffShelf-v0", "take_book_off_shelf_cfg", "TakeBookOffShelfEnvFloatingDexHandRightCfg"
)
