# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regressions for task wrist placement and teleop-origin alignment."""

from dataclasses import dataclass, replace
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "source/dexverse/dexverse"
ROBOTS = [f"floating_{hand}_{side}" for hand in ("allegro", "inspire", "shadow") for side in ("right", "left")]


@dataclass
class Config:
    """Use the same copy-on-replace interface as IsaacLab config objects."""

    pos: tuple = (1.0, 2.0, 3.0)
    rot: tuple = (2**-0.5, 0.0, 0.0, 2**-0.5)  # 90 degrees about world Z
    joint_pos: dict | None = None
    init_state: object = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@pytest.fixture(params=("tasks", "baseline_v1"))
def helpers(request):
    return runpy.run_path(str(SOURCE / request.param / "config/robot_init.py"))


@pytest.fixture(params=ROBOTS)
def env_cfg(request):
    initial = Config(joint_pos={
        "x_translation_joint": 0.2,
        "y_translation_joint": 0.3,
        "z_translation_joint": 0.4,
        "x_rotation_joint": 0.0,
        "y_rotation_joint": 0.0,
        "z_rotation_joint": 0.0,
        "finger_joint": 0.7,
    })
    retargeter = SimpleNamespace(wrist_joint_origin=None)
    return SimpleNamespace(
        robot_type=request.param,
        scene=SimpleNamespace(robot=Config(init_state=initial)),
        teleop_devices=SimpleNamespace(devices={
            "handtracking": SimpleNamespace(retargeters=[retargeter]),
        }),
    )


def test_x_only_placement_preserves_other_world_coordinates_and_shared_config(helpers, env_cfg):
    original = env_cfg.scene.robot
    helpers["set_robot_wrist_init_world_pos"](env_cfg, x=-0.4)
    joints = env_cfg.scene.robot.init_state.joint_pos
    # With the base rotated 90 degrees, local Y controls world X.
    assert joints == pytest.approx({**original.init_state.joint_pos, "y_translation_joint": 1.4})
    assert original.init_state.joint_pos["y_translation_joint"] == 0.3
    assert env_cfg.scene.robot is not original

    helpers["align_retargeter_wrist_origin_to_init"](env_cfg)
    origins = env_cfg.teleop_devices.devices["handtracking"].retargeters[0].wrist_joint_origin
    side = env_cfg.robot_type.rsplit("_", 1)[1]
    assert set(origins) == {side}
    np.testing.assert_allclose(origins[side]["pos"], [-0.4, 2.2, 3.4], atol=1e-12)
    np.testing.assert_allclose(origins[side]["rot"], original.init_state.rot, atol=1e-12)


def test_full_pose_and_explicit_tracking_side(helpers, env_cfg):
    helpers["set_robot_wrist_init_world_pos"](
        env_cfg, x=-0.4, y=0.5, z=1.0, rot=(0.0, 0.0, 0.0, 1.0),
    )
    joints = env_cfg.scene.robot.init_state.joint_pos
    np.testing.assert_allclose(
        [joints[f"{axis}_translation_joint"] for axis in "xyz"], [-1.5, 1.4, -2.0], atol=1e-12,
    )
    np.testing.assert_allclose(
        [joints[f"{axis}_rotation_joint"] for axis in "xyz"], [0.0, 0.0, np.pi / 2], atol=1e-12,
    )
    assert joints["finger_joint"] == 0.7
    # An explicitly requested tracking side still takes precedence over inference.
    helpers["align_retargeter_wrist_origin_to_init"](env_cfg, hand_key="left")
    origins = env_cfg.teleop_devices.devices["handtracking"].retargeters[0].wrist_joint_origin
    assert set(origins) == {"left"}
    np.testing.assert_allclose(origins["left"]["pos"], [-0.4, 0.5, 1.0], atol=1e-12)
    assert abs(origins["left"]["rot"][3]) == pytest.approx(1.0)
