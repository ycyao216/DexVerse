# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: raise a shopping-basket handle (synthesis/shopping basket002)."""


from dexverse.assets import SYNTHESIS_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

BASKET_USD_PATH = str(SYNTHESIS_DIR / "shopping basket002" / "model_basket_22.usd")


@configclass
class LiftBasketHandleEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Swing the basket handle up towards vertical while the basket is free.

    The shopping-basket USD exposes hinged handle(s) connected to the body.
    Success fires when the max handle angle crosses ``success_threshold``.
    If the USD turns out not to be articulated (the mesh is sometimes used
    as a pure rigid object, e.g. ``bimanual_lift/lift_basket_cfg.py``), the
    spawn step will raise; in that case either use a different basket USD
    or remove this task from ``__init__.py``.
    """

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = BASKET_USD_PATH
    # Baskets are authored close to life size; 0.6x fits the tabletop.
    articulation_scale: tuple = (0.6, 0.6, 0.6)
    articulation_init_pos: tuple = (0.0, 0.0, 0.0)
    articulation_init_rot: tuple = (1.0, 0.0, 0.0, 0.0)
    # Baskets sit noticeably above the tabletop.
    articulation_half_height_est: float = 0.10
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [0.0, 0.0],
        "y": [-0.1, 0.1],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.2, 0.2],
    }

    success_joint_names: list[str] = [
        "RevoluteJoint_basket_22_right",
        "RevoluteJoint_basket_22_left",
    ]
    success_threshold: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.func = mdp.joint_relative_move
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
        }
