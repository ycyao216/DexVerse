# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Double-articulation co-completion task: open two joints close in time.

Template for tasks where an articulation (e.g. a double-door cabinet) exposes
two joints that must both be driven past a threshold, and success additionally
requires that each joint's *first* crossing into the satisfied region falls
within ``success_window_s`` of the other. This rules out finishing one door
well before the other.

Subclassing recipe: point ``articulation_usd_path`` at an asset whose
articulation has at least two target joints, list their exact names in
``success_joint_names``, and pick ``success_threshold`` plus
``success_window_s``. Register the concrete subclass as a gym env alongside
the other entries in ``__init__.py``.
"""

from __future__ import annotations

import math

from dexverse.assets import STORAGE_FURNITURE_REVOLUTE_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import mdp
from .articulation_base import ArticulationBaseEnvFloatingDexHandRightCfg
from .articulation_base.articulation_base_cfg import ARTICULATION_KEY

CABINET_DOUBLE_DOOR_USD_PATH = str(STORAGE_FURNITURE_REVOLUTE_DIR / "44781" / "mobility.usd")
CABINET_DOOR_JOINT_NAMES = ("target_joint_revolute", "joint_11")
CABINET_DOOR_INIT_OPEN_ANGLE_RAD = math.radians(10.0)
CABINET_DOOR_SUCCESS_RATIO = 0.8


@configclass
class CabinetDoubleDoorEnvFloatingDexHandRightCfg(ArticulationBaseEnvFloatingDexHandRightCfg):
    """Open two joints on a fixed articulation, with co-completion in time.

    Success = all ``success_joint_names`` currently satisfied AND the
    spread of their first-entry times <= ``success_window_s``.
    """

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = CABINET_DOUBLE_DOOR_USD_PATH
    articulation_scale: tuple = (0.4, 0.4, 0.4)
    articulation_init_pos: tuple = (0.45, 0.0, 0.0)
    articulation_init_rot: tuple = (1.0, 0.0, 0.0, 0.0)
    articulation_half_height_est: float = 0.2165

    articulation_fix_root_link: bool | None = True

    # Name both joints explicitly — a catch-all regex would also match any
    # internal hinges and break the co-completion check.
    success_joint_names: tuple[str, ...] = CABINET_DOOR_JOINT_NAMES
    success_threshold: float = CABINET_DOOR_SUCCESS_RATIO
    # Allowed gap between each joint's first-reach time, in seconds.
    success_window_s: float = 1.0

    def __post_init__(self):
        # Start both doors slightly open so the hands have a usable gap at t=0.
        self.articulation_init_joint_pos = {
            joint_name: CABINET_DOOR_INIT_OPEN_ANGLE_RAD for joint_name in self.success_joint_names
        }
        super().__post_init__()

        env_step_dt = self.sim.dt * self.decimation
        window_steps = max(1, int(round(self.success_window_s / env_step_dt)))

        self.terminations.success = DoneTerm(
            func=mdp.joint_co_completion,
            params={
                "threshold": self.success_threshold,
                "mode": "ratio",
                "window_steps": window_steps,
                "asset_cfg": SceneEntityCfg(ARTICULATION_KEY, joint_names=list(self.success_joint_names)),
            },
        )

        self.episode_length_s = 20.0
