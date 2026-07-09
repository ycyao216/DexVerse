# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: press a stapler head (synthesis/green stapler)."""

import math

from dexverse.assets import SYNTHESIS_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

STAPLER_USD_PATH = str(SYNTHESIS_DIR / "green stapler" / "model_stapler_1.usd")


@configclass
class OpenStaplerEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Open the stapler head up onto the body while the base is free.

    Elements: ``E_shell_1`` (top shell), ``E_body_2`` (body / anvil),
    ``E_stapler_needle_3`` (inner plunger). The main rotary hinge between
    shell and body is what we ask the policy to actuate. A typical solution
    pins the body on the table while the palm / fingers press the shell
    downwards.
    """

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = STAPLER_USD_PATH
    articulation_scale: tuple = (1.0, 1.0, 1.0)
    articulation_init_pos: tuple = (0.0, 0.0, 0.0)
    # Rotate +90 deg about z (counterclockwise when viewed from +z to -z).
    articulation_init_rot: tuple = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    # Staplers sit on a flat base; a small half-height keeps the anvil on
    # the table.
    articulation_half_height_est: float = 0.03
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [0.0, 0.0],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.2, 0.2],
    }

    success_joint_names: list[str] = ["RevoluteJoint_stapler_1_up"]
    success_threshold: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.func = mdp.joint_relative_move
        # Deep press: 80% of the shell hinge travel.
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
        }
