# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual lift of a shopping basket (synthesis/shopping basket002)."""

from dexverse.assets import SYNTHESIS_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import BimanualLiftObjectEnvFloatingShadowBimanualCfg

BASKET_USD_PATH = str(SYNTHESIS_DIR / "shopping basket002" / "model_basket_22.usd")
BASKET_SUCCESS_MAX_TILT_RAD = 0.174532925


@configclass
class LiftBasketEnvFloatingShadowBimanualCfg(BimanualLiftObjectEnvFloatingShadowBimanualCfg):
    """Lift a shopping basket off the tabletop with two Shadow hands."""

    usd_path: str = BASKET_USD_PATH
    # Tune to the authored mesh; baskets are taller than trays so expect a
    # larger half-height estimate.
    scale: tuple[float, float, float] = (0.8, 0.8, 0.8)
    mass: float = 0.6
    # Move basket slightly forward so both palms have more free space at reset.
    object_init_x_offset: float = 0.06
    object_init_y_offset: float = 0.0
    object_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    lift_height: float = 0.2
    obj_x_range: tuple[float, float] = (-0.04, 0.02)
    obj_y_range: tuple[float, float] = (-0.1, 0.1)
    obj_z_range: tuple[float, float] = (0.0, 0.0)
    obj_roll_range: tuple[float, float] = (0.0, 0.0)
    obj_pitch_range: tuple[float, float] = (0.0, 0.0)
    obj_yaw_range: tuple[float, float] = (-0.2, 0.2)

    success_max_tilt_rad: float = BASKET_SUCCESS_MAX_TILT_RAD

    def __post_init__(self):
        super().__post_init__()
        # Raise robot root a bit to reduce initial hand-object/table collisions.
        rx, ry, rz = self.scene.robot.init_state.pos
        self.scene.robot.init_state.pos = (rx, ry, rz + 0.1)
        self.terminations.success.func = mdp.object_upright_and_lifted
        self.terminations.success.params = {
            "object_cfg": SceneEntityCfg("object"),
            "min_height": self.lift_height,
            "max_tilt_rad": self.success_max_tilt_rad,
        }
