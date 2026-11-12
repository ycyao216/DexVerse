# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Obstacle-reset geometry regressions. Run with the Isaac Lab Python environment::

    PYTHONPATH=source/dexverse python scripts/tests/test_obstacle_spacing.py
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import math
from types import SimpleNamespace
import unittest

import torch
from isaaclab.managers import SceneEntityCfg

from dexverse.tasks.config.non_prehensile import push_small_sphere_obstacle_slope_cfg as sphere_cfg
from dexverse.tasks.mdp.resets import reset_random_inverted_v_obstacles


class CapturedRigidObject:
    """Record the poses/velocities the reset would write to the simulator."""

    def __init__(self, count):
        default = torch.zeros(count, 13)
        default[:, 3] = 1
        self.data = SimpleNamespace(default_root_state=default)
        self.pose = torch.full((count, 7), 77.0)
        self.velocity = torch.full((count, 6), 77.0)

    def write_root_pose_to_sim(self, pose, env_ids):
        self.pose[env_ids] = pose

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.velocity[env_ids] = velocity


class Scene(dict):
    def __init__(self, count, pool):
        super().__init__({f"obstacle_{i}_{side}": CapturedRigidObject(count)
                          for i in range(pool) for side in ("left", "right")})
        self.env_origins = torch.zeros(count, 3)
        self.env_origins[:, 0] = torch.arange(count) * 3.0


class TestObstacleSpacing(unittest.TestCase):
    def test_task_requests_twenty_one_centimeter_clearance(self):
        event = sphere_cfg.PushSmallSphereObstacleSlopeEventCfg().reset_random_inverted_v_obstacles
        self.assertEqual(event.params["min_apex_tangent_spacing"], 0.21)
        self.assertEqual(event.params["active_count_range"], (2, 2))

    def _reset(self, *, count=128, pool=5, active=(2, 2), ids=None, spacing=0.21):
        scene = Scene(count, pool)
        env = SimpleNamespace(num_envs=count, device="cpu", scene=scene)
        if ids is None:
            ids = torch.arange(count)
        angle = math.radians(15)
        reset_random_inverted_v_obstacles(
            env, ids,
            obstacle_asset_cfgs=tuple((SceneEntityCfg(f"obstacle_{i}_left"), SceneEntityCfg(f"obstacle_{i}_right"))
                                     for i in range(pool)),
            slope_center=(0.18, 0.0, 0.70),
            slope_quat=(math.cos(angle / 2), 0.0, -math.sin(angle / 2), 0.0),
            active_count_range=active,
            apex_tangent_range=(-0.30, 0.4 - 0.12 - 0.12 * math.cos(math.radians(38)) - 0.10),
            apex_tangent_jitter=0.005,
            min_apex_tangent_spacing=spacing,
            apex_lateral_range=(-0.25, 0.25),
            apex_normal_offset=0.055,
            leg_length=0.12,
            leg_yaw_angle_rad=math.radians(38),
        )
        return scene

    def test_active_obstacles_have_clearance_after_jitter(self):
        torch.manual_seed(42)
        scene = self._reset(count=2048)
        tangent = torch.tensor([math.cos(math.radians(15)), 0, math.sin(math.radians(15))])
        delta = scene["obstacle_0_left"].pose[:, :3] - scene["obstacle_1_left"].pose[:, :3]
        # Both left legs have the same orientation and apex offset, so their
        # separation along the slope equals the separation between the apices.
        clearance = torch.abs(delta @ tangent)
        self.assertGreaterEqual(float(clearance.min()), 0.21 - 1e-6)
        self.assertLessEqual(float(clearance.max()), 0.40)
        self.assertTrue(torch.all(scene["obstacle_0_left"].pose[:, 2] > 0))
        self.assertTrue(torch.all(scene["obstacle_1_left"].pose[:, 2] > 0))

    def test_subset_reset_preserves_other_envs_and_parks_inactive_obstacles(self):
        ids = torch.tensor([1, 3, 7])
        scene = self._reset(count=8, ids=ids)
        normal = torch.tensor([-math.sin(math.radians(15)), 0, math.cos(math.radians(15))])
        parked = scene.env_origins[ids] + torch.tensor([0.18, 0, 0.70]) - 2 * normal
        for i in range(5):
            for side in ("left", "right"):
                asset = scene[f"obstacle_{i}_{side}"]
                self.assertTrue(torch.all(asset.pose[[0, 2, 4, 5, 6]] == 77))
                self.assertTrue(torch.all(asset.velocity[ids] == 0))
                if i >= 2:
                    torch.testing.assert_close(asset.pose[ids, :3], parked)

    def test_zero_and_single_active_obstacle_need_no_pairwise_clearance(self):
        for pool, active in ((5, (0, 0)), (1, (1, 1)), (5, (1, 1))):
            with self.subTest(pool=pool, active=active):
                scene = self._reset(pool=pool, active=active)
                for i in range(pool):
                    self.assertEqual(bool(torch.all(scene[f"obstacle_{i}_left"].pose[:, 2] > 0)), i < active[0])

    def test_impossible_clearance_raises(self):
        with self.assertRaises(ValueError):
            self._reset(active=(3, 3))


if __name__ == "__main__":
    try:
        print(f"Reset sampler source: {reset_random_inverted_v_obstacles.__code__.co_filename}", flush=True)
        print(f"Task configuration source: {sphere_cfg.__file__}", flush=True)
        unittest.main()
    finally:
        simulation_app.close()
