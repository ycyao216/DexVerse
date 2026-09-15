# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PushT observation regressions using Isaac Sim and the current task assets.

Run with the Isaac Lab Python environment from the repository root::

    PYTHONPATH=source/dexverse python scripts/tests/test_pusht_observations.py
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import math
import unittest

import gymnasium as gym
import torch

import dexverse.tasks  # noqa: F401 -- registers the Gym task
from dexverse.tasks.config.non_prehensile.pusht_cfg import PushTEnvFloatingDexHandRightCfg


class TestPushTObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = PushTEnvFloatingDexHandRightCfg()
        cfg.sim.device = "cpu"
        cfg.scene.num_envs = 1
        cfg.seed = 42
        cfg.terminations = {}
        cfg.observation_preset = "state"
        cfg._apply_observation_preset("state")
        cls.env = gym.make("Dexverse-PushT-v0", cfg=cfg).unwrapped
        cls.env.sim.reset()
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _write_pose(self, name, values):
        self.env.scene[name].write_root_pose_to_sim(torch.tensor([values], dtype=torch.float32))
        self.env.sim.forward()
        self.env.scene.update(0.0)

    def _observations(self):
        return self.env.observation_manager.compute()

    def test_state_policy_observes_object_and_goal_in_the_correct_frames(self):
        self._write_pose("object", [0.10, -0.10, 0.70, 1, 0, 0, 0])
        half = math.sqrt(0.5)
        self._write_pose("goal_tee", [-0.10, 0.05, 0.63, half, 0, 0, half])
        obs = self._observations()
        self.assertNotIn("privileged", obs)
        self.assertEqual(obs["state"].shape, (1, 14))
        self.assertEqual(obs["goal"].shape, (1, 7))
        table = self.env.scene["table"].data.root_pos_w
        torch.testing.assert_close(self.env.scene["table"].data.root_quat_w,
                                   torch.tensor([[1.0, 0, 0, 0]]))
        torch.testing.assert_close(obs["state"][:, :3], torch.tensor([[0.10, -0.10, 0.70]]) - table)
        torch.testing.assert_close(obs["state"][:, 3:7], torch.tensor([[1.0, 0, 0, 0]]))
        # A 90-degree goal yaw maps the world displacement (0.20, -0.15, 0.07)
        # to (-0.15, -0.20, 0.07) in the goal frame.
        torch.testing.assert_close(obs["state"][:, 7:], torch.tensor([[-0.15, -0.20, 0.07, half, 0, 0, -half]]),
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(obs["goal"][:, :3], torch.tensor([[-0.10, 0.05, 0.63]]) - table)
        torch.testing.assert_close(obs["goal"][:, 3:], torch.tensor([[half, 0, 0, half]]), atol=1e-6, rtol=1e-6)

    def test_goal_changes_are_observable_and_velocity_does_not_enter_pose_features(self):
        self._write_pose("object", [0.10, -0.10, 0.70, 1, 0, 0, 0])
        self._write_pose("goal_tee", [-0.10, 0.05, 0.63, 1, 0, 0, 0])
        before = self._observations()
        self._write_pose("goal_tee", [0.20, -0.20, 0.63, 1, 0, 0, 0])
        moved = self._observations()
        torch.testing.assert_close(before["state"][:, :7], moved["state"][:, :7])
        self.assertFalse(torch.equal(before["goal"], moved["goal"]))
        self.assertFalse(torch.equal(before["state"][:, 7:], moved["state"][:, 7:]))
        self.env.scene["object"].write_root_velocity_to_sim(torch.tensor([[1.0, 2, 3, 4, 5, 6]]))
        after_velocity = self._observations()
        torch.testing.assert_close(moved["state"], after_velocity["state"])
        torch.testing.assert_close(moved["goal"], after_velocity["goal"])

    def test_visual_presets_keep_the_goal_and_exclude_object_state(self):
        for preset in ("rgb", "rgb_depth", "pointcloud", "3view_rgb", "3view_rgb_depth", "3view_pointcloud"):
            with self.subTest(preset=preset):
                cfg = PushTEnvFloatingDexHandRightCfg()
                cfg._apply_observation_preset(preset)
                self.assertIsNotNone(cfg.observations.goal)
                self.assertIsNone(cfg.observations.state)
                self.assertIsNone(cfg.observations.privileged)


if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        simulation_app.close()
