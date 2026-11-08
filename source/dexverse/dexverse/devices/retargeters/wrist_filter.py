# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clean up tracked wrist poses before they drive the robot.

Hand tracking occasionally loses the wrist: in recorded sessions the pose jumped
over 100 degrees in one frame, froze on the wrong pose for most of a second, then
snapped back, flinging whatever the hand was holding. Smoothing alone would still
follow such a pose, so this filter does three things in order:

1. **Gate.** A pose further from the last good one than a hand could have moved is
   rejected and the last good pose is held. A hand that really did move during a
   dropout is picked up again once tracking looks alive: poses that keep changing,
   each believable from the one before. A jump or a frozen pose does not count.
2. **Smooth.** An adaptive low-pass (the One Euro filter) smooths strongly when the
   hand is nearly still, where tracking noise would shake a grasp, and barely at
   all when it moves, where lag would hurt.
3. **Ramp.** Output speed is capped, and capped harder while catching up after a
   rejection, so a reacquired pose is approached instead of jumped to.

The filter has no simulator dependencies, so it can be tested on recorded commands.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

_MIN_DT = 1.0 / 240.0
_MAX_DT = 0.1


@dataclass
class WristFilterCfg:
    """Settings for :class:`WristPoseFilter`. Speeds are in real time."""

    max_speed: float = 2.0
    """Fastest believable wrist translation [m/s]."""
    max_angular_speed: float = 10.0
    """Fastest believable wrist rotation [rad/s]."""
    position_slack: float = 0.01
    """Extra gate allowance per frame for position noise [m]."""
    angle_slack: float = math.radians(5.0)
    """Extra gate allowance per frame for rotation noise [rad]."""
    reacquire_time: float = 0.3
    """How long tracking must look alive before a new far-away pose is accepted [s]."""
    catch_up_speed: float = 0.5
    """Output speed limit while catching up to a reacquired pose [m/s]."""
    catch_up_angular_speed: float = math.radians(120.0)
    """Output rotation speed limit while catching up to a reacquired pose [rad/s]."""
    position_min_cutoff: float = 2.0
    """Low-pass cutoff with the wrist still [Hz]."""
    position_beta: float = 20.0
    """Cutoff increase with wrist speed [Hz per m/s]."""
    rotation_min_cutoff: float = 2.0
    """Low-pass cutoff with the wrist still [Hz]."""
    rotation_beta: float = 6.0
    """Cutoff increase with wrist angular speed [Hz per rad/s]."""
    speed_cutoff: float = 5.0
    """Low-pass cutoff for the speed estimates that drive the adaptive cutoff [Hz]."""


def _blend(cutoff_hz: float, dt: float) -> float:
    """First-order low-pass blend factor for a cutoff frequency and time step."""
    time_constant = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + time_constant / dt)


class WristPoseFilter:
    """Filters one wrist's (x, y, z, qw, qx, qy, qz) pose stream."""

    def __init__(self, cfg: WristFilterCfg, clock: Callable[[], float] = time.perf_counter):
        self.cfg = cfg
        self._clock = clock
        self._last_time = 0.0
        self._good_pos: np.ndarray | None = None
        self._good_rot: R | None = None
        self._out_pos: np.ndarray | None = None
        self._out_rot: R | None = None
        self._linear_rate = 0.0
        self._angular_rate = 0.0
        self._raw_pos: np.ndarray | None = None
        self._raw_rot: R | None = None
        self._rejecting = False
        self._alive_time = 0.0
        self._catching_up = False
        self.num_updates = 0
        self.num_rejected = 0

    def update(self, pose: np.ndarray) -> np.ndarray:
        """Take the latest tracked pose and return the pose to command."""
        pose = np.asarray(pose, dtype=np.float64)
        pos, rot = pose[:3], R.from_quat(pose[[4, 5, 6, 3]])
        now = self._clock()
        self.num_updates += 1

        if self._good_pos is None:
            self._good_pos, self._good_rot = pos, rot
            self._raw_pos, self._raw_rot = pos, rot
            self._out_pos, self._out_rot = pos.copy(), rot
            self._last_time = now
            return self._output()

        dt = float(np.clip(now - self._last_time, _MIN_DT, _MAX_DT))
        self._last_time = now
        self._gate(pos, rot, dt)
        self._raw_pos, self._raw_rot = pos, rot
        self._smooth(dt)
        return self._output()

    def last_output(self) -> np.ndarray | None:
        """The most recent output pose, or None before the first valid pose."""
        return None if self._out_pos is None else self._output()

    def _believable(self, pos: np.ndarray, rot: R, from_pos: np.ndarray, from_rot: R, dt: float) -> bool:
        """Whether a hand could have moved from one pose to the other in one frame."""
        cfg = self.cfg
        return bool(
            np.linalg.norm(pos - from_pos) <= cfg.max_speed * dt + cfg.position_slack
            and (rot * from_rot.inv()).magnitude() <= cfg.max_angular_speed * dt + cfg.angle_slack
        )

    def _gate(self, pos: np.ndarray, rot: R, dt: float) -> None:
        if self._believable(pos, rot, self._good_pos, self._good_rot, dt):
            self._accept(pos, rot)
            return

        self.num_rejected += 1
        self._rejecting = True
        frozen = np.array_equal(pos, self._raw_pos)
        alive = not frozen and self._believable(pos, rot, self._raw_pos, self._raw_rot, dt)
        self._alive_time = self._alive_time + dt if alive else 0.0
        if self._alive_time >= self.cfg.reacquire_time:
            self._accept(pos, rot)

    def _accept(self, pos: np.ndarray, rot: R) -> None:
        self._good_pos, self._good_rot = pos, rot
        self._catching_up = self._catching_up or self._rejecting
        self._rejecting = False
        self._alive_time = 0.0

    def _smooth(self, dt: float) -> None:
        cfg = self.cfg
        position_error = self._good_pos - self._out_pos
        rotation_error = (self._good_rot * self._out_rot.inv()).as_rotvec()
        linear_distance = float(np.linalg.norm(position_error))
        angular_distance = float(np.linalg.norm(rotation_error))

        speed_blend = _blend(cfg.speed_cutoff, dt)
        self._linear_rate += speed_blend * (linear_distance / dt - self._linear_rate)
        self._angular_rate += speed_blend * (angular_distance / dt - self._angular_rate)

        if self._catching_up:
            linear_cap, angular_cap = cfg.catch_up_speed * dt, cfg.catch_up_angular_speed * dt
        else:
            linear_cap, angular_cap = cfg.max_speed * dt, cfg.max_angular_speed * dt

        linear_step = linear_distance * _blend(cfg.position_min_cutoff + cfg.position_beta * self._linear_rate, dt)
        angular_step = angular_distance * _blend(cfg.rotation_min_cutoff + cfg.rotation_beta * self._angular_rate, dt)
        linear_step, angular_step = min(linear_step, linear_cap), min(angular_step, angular_cap)

        if linear_distance > 0.0:
            self._out_pos = self._out_pos + position_error * (linear_step / linear_distance)
        if angular_distance > 0.0:
            self._out_rot = R.from_rotvec(rotation_error * (angular_step / angular_distance)) * self._out_rot

        if self._catching_up and linear_distance <= cfg.position_slack and angular_distance <= cfg.angle_slack:
            self._catching_up = False

    def _output(self) -> np.ndarray:
        x, y, z, w = self._out_rot.as_quat()
        return np.array([*self._out_pos, w, x, y, z], dtype=np.float32)
