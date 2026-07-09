# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixate-then-manipulate: open a flat folder / file (synthesis/flat folder002)."""


from dexverse.assets import SYNTHESIS_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import FixateArticulationEnvFloatingDexHandRightCfg

FLAT_FOLDER_USD_PATH = str(SYNTHESIS_DIR / "flat folder002" / "model_file_9.usd")


@configclass
class OpenFlatFolderEnvFloatingDexHandRightCfg(FixateArticulationEnvFloatingDexHandRightCfg):
    """Lift the folder envelope off the body along its single hinge.

    Elements: ``E_body_1`` (body / bottom flap), ``E_envelope_2`` (top
    flap). The envelope is typically very thin and light, so the manipulator
    hand tends to drag the whole folder across the table unless the other
    hand pins ``E_body_1`` down — a clear bimanual fixate-then-manipulate
    win.
    """

    robot_type: str = "floating_shadow_bimanual"
    articulation_usd_path: str = FLAT_FOLDER_USD_PATH
    articulation_scale: tuple = (1.2, 1.2, 1.2)
    articulation_init_pos: tuple = (0.0, 0.0, 0.0)
    articulation_init_rot: tuple = (1.0, 0.0, 0.0, 0.0)
    # Folders lie almost flat on the table.
    articulation_half_height_est: float = 0.005
    articulation_reset_pose_range: dict[str, list[float]] = {
        "x": [0.0, 0.0],
        "y": [-0.2, 0.2],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [-0.2, 0.2],
    }

    success_joint_names: list[str] = ["RevoluteJoint_file_9_up"]
    success_threshold: float = 0.6

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.func = mdp.joint_relative_move
        # ~180 deg of opening over a 190 deg hinge range.
        self.terminations.success.params = {
            "threshold": self.success_threshold,
            "asset_cfg": SceneEntityCfg("articulation", joint_names=self.success_joint_names),
            "mode": "progress",
            "op": ">=",
            "reduce": "any",
        }
