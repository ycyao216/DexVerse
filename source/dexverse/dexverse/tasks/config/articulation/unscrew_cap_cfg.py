# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: unscrew a centrifuge-tube cap (autobio/centrifuge_15ml_screw)."""

import math

from dexverse.assets import CORE_ASSETS_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

CENTRIFUGE_15ML_SCREW_USD_PATH = str(
    CORE_ASSETS_DIR / "autobio" / "centrifuge_15ml_screw" / "model_centrifuge_15ml_screw.usd"
)


@configclass
class UnscrewCapEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Loosen a centrifuge tube cap while the tube body is free on the table.

    Task semantics: one hand stabilizes the tube body, the other applies
    rotational motion on the cap. Success is triggered once the cap has
    rotated enough to be considered opened (no need to fully remove it).
    """

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = CENTRIFUGE_15ML_SCREW_USD_PATH
    # Authored tube is ~2.3 cm diameter and ~12.3 cm tall; scale up to
    # improve finger contact robustness for in-hand bimanual manipulation.
    articulation_scale: tuple = (1.8, 1.8, 1.8)
    articulation_init_pos: tuple = (0.0, 0.0, 0.0)
    # Rotate +90 deg about Y so the tube lies along world-x, making the
    # cap twist more accessible for one-hand hold + one-hand rotate.
    articulation_init_rot: tuple = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
    # When laid sideways, table clearance is roughly the scaled radius.
    articulation_half_height_est: float = 0.02

    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [0.0, 0.0],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.2, 0.2],
    }

    success_joint_names: list[str] = ["cap_revolute"]
    # Cap joint range is ~720 deg in the authored USD.
    success_threshold: float = math.radians(700.0)

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.func = mdp.joint_relative_move
        # Require near-complete unscrewing. Keep a tiny tolerance for
        # physics/contact jitter at the mechanical limit.
        self.terminations.success.params = {
            "threshold": 0.98,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
        }
