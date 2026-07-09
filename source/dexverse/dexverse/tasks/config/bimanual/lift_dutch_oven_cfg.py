# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual lift of a Dutch oven (functional/Dutch_Oven_B08F2TMKZB_White)."""


from dexverse.assets import SYNTHESIS_DIR
from isaaclab.utils import configclass

from .base_cfg import BimanualLiftObjectEnvFloatingShadowBimanualCfg

DUTCH_OVEN_USD_PATH = str(SYNTHESIS_DIR / "Dutch_Oven_B08F2TMKZB_White" / "model_Dutch_Oven_B08F2TMKZB_White_69323.usd")


@configclass
class LiftDutchOvenEnvFloatingShadowBimanualCfg(BimanualLiftObjectEnvFloatingShadowBimanualCfg):
    """Lift a Dutch oven off the tabletop with two Shadow hands."""

    usd_path: str = DUTCH_OVEN_USD_PATH
    scale: tuple[float, float, float] = (3.0, 3.0, 3.0)
    mass: float = 2.0
    object_half_height: float = 0.07
    table_clearance: float = 0.02
    object_init_x_offset: float = 0.0
    object_init_y_offset: float = 0.0
    # positive 90 degrees around z-axis, w,x,y,z
    object_init_rot: tuple[float, float, float, float] = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
    lift_height: float = 0.3
    obj_x_range: tuple[float, float] = (-0.02, 0.02)
    obj_y_range: tuple[float, float] = (-0.1, 0.1)
    obj_z_range: tuple[float, float] = (0.0, 0.0)
    obj_roll_range: tuple[float, float] = (0.0, 0.0)
    obj_pitch_range: tuple[float, float] = (0.0, 0.0)
    obj_yaw_range: tuple[float, float] = (-0.2, 0.2)
