# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push-button task: depress the button past 80% of its travel.

Inherits :class:`ArticulationBaseEnvFloatingDexHandRightCfg`. The button is
sprung (an ImplicitActuatorCfg gives it a return-to-rest stiffness), so
``__post_init__`` patches the articulation's ``actuators`` dict after the
base finishes building the ArticulationCfg.
"""

from __future__ import annotations

from dexverse.assets import DEXVERSE_AUTHORED_ARTICULATIONS_DIR
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY

ASSET_DIR = DEXVERSE_AUTHORED_ARTICULATIONS_DIR / "button"
BUTTON_USD_PATH = str(ASSET_DIR / "button.usd")
BUTTON_SCALE = (3.0, 3.0, 3.0)
BUTTON_HALF_HEIGHT_EST = 0.03
BUTTON_JOINT_NAME = "button_prismatic"
BUTTON_PRESS_DISTANCE_M = 0.01
SUCCESS_PRESS_RATIO = 0.8


@configclass
class PushButtonRewardsCfg(dexverse_base_env.RewardsCfg):
    """Reward proportional to button compression past the press threshold."""

    button_press = RewTerm(
        func=mdp.joint_displacement_reward,
        weight=5.0,
        params={
            "threshold_m": BUTTON_PRESS_DISTANCE_M,
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[BUTTON_JOINT_NAME]),
        },
    )


@configclass
class PushButtonTerminationsCfg(dexverse_base_env.TerminationsCfg):
    """Success when the button is pressed past ``SUCCESS_PRESS_RATIO`` of its range."""

    success = DoneTerm(
        func=mdp.joint_relative_move,
        params={
            "threshold": SUCCESS_PRESS_RATIO,
            # "progress" = abs(q - q_init) / reachable_from_init: 0 at rest, 1
            # fully pressed. NOT "ratio" ((q - lower)/(upper - lower)) -- the
            # button rests at its upper limit (q_init=0, limits [-0.015, 0]), so
            # "ratio" reads 1.0 at rest and drops toward 0 when pressed, i.e. it
            # would fire when released and never when pressed.
            "mode": "progress",
            "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=[BUTTON_JOINT_NAME]),
        },
    )


@configclass
class PushButtonEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Push-button env (floating dex-hand variant)."""

    articulation_usd_path: str = BUTTON_USD_PATH
    articulation_scale: tuple[float, float, float] = BUTTON_SCALE
    articulation_init_pos: tuple[float, float, float] = (0.3, 0.2, 0.0)
    articulation_init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = BUTTON_HALF_HEIGHT_EST
    articulation_fix_root_link: bool | None = True

    success_joint_names: list[str] = [BUTTON_JOINT_NAME]
    # success_threshold left at None: ratio-mode ``joint_relative_move``
    # doesn't read the base threshold.

    rewards: PushButtonRewardsCfg = PushButtonRewardsCfg()
    terminations: PushButtonTerminationsCfg = PushButtonTerminationsCfg()

    def __post_init__(self):
        # Start at rest (joint at 0). Set BEFORE super so the base wires it in.
        self.articulation_init_joint_pos = {BUTTON_JOINT_NAME: 0.0}
        super().__post_init__()
        # The button needs a return-to-rest spring; install it after the base
        # has built the ArticulationCfg.
        self.scene.articulation.actuators = {
            "button_spring": ImplicitActuatorCfg(
                joint_names_expr=[BUTTON_JOINT_NAME],
                stiffness=500.0,
                damping=10.0,
                effort_limit_sim=50.0,
            ),
        }


# Backward-compat alias.
PushButtonEnvCfg = PushButtonEnvFloatingDexHandRightCfg
