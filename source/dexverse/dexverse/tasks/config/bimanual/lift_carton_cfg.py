# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual lift of a carton (functional/carton001)."""


from dexverse.assets import SYNTHESIS_DIR
from isaaclab.utils import configclass

from .base_cfg import BimanualLiftObjectEnvFloatingShadowBimanualCfg

CARTON_USD_PATH = str(SYNTHESIS_DIR / "carton001" / "model_carton.usd")


@configclass
class LiftCartonEnvFloatingShadowBimanualCfg(BimanualLiftObjectEnvFloatingShadowBimanualCfg):
    """Lift a carton off the tabletop with two Shadow hands."""

    usd_path: str = CARTON_USD_PATH
    scale: tuple[float, float, float] = (0.75, 0.75, 0.75)
    mass: float = 0.6
    object_half_height: float = 0.02
    table_clearance: float = 0.02
    object_init_x_offset: float = 0.1
    object_init_y_offset: float = 0.0
    object_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    lift_height: float = 0.2
    obj_x_range: tuple[float, float] = (0.05, 0.1)
    obj_y_range: tuple[float, float] = (-0.1, 0.1)
    obj_z_range: tuple[float, float] = (0.0, 0.0)
    obj_roll_range: tuple[float, float] = (0.0, 0.0)
    obj_pitch_range: tuple[float, float] = (0.0, 0.0)
    obj_yaw_range: tuple[float, float] = (0.0, 0.0)
