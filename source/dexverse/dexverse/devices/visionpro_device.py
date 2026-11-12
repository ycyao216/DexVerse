# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apple Vision Pro hand-tracking device backed by ``avp_stream``."""

from __future__ import annotations

import os
import signal
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.retargeter_base import RetargeterBase


# avp_stream joint index -> Isaac Lab/OpenXR-compatible joint name. The Vision Pro skeleton has 25 joints including the wrist; OpenXR additionally has palm.
VISIONPRO_JOINT_NAMES = (
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)


def _identity_pose() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    """Convert a homogeneous matrix to ``[x, y, z, qw, qx, qy, qz]``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Expected a finite 4x4 transform, got shape {matrix.shape}.")

    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.array(
        [
            matrix[0, 3],
            matrix[1, 3],
            matrix[2, 3],
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        ],
        dtype=np.float32,
    )


def _retargeting_rotation_from_hand_geometry(poses: dict[str, np.ndarray]) -> np.ndarray:
    """Build the Shadow retargeting root frame from Vision Pro joint geometry.

    The Shadow URDF rotates its palm under the floating-hand root. In that root
    frame, finger extension is +X, the index-finger side is +Y, and the palm
    normal is +Z. Vision Pro's wrist local axes do not match OpenXR's, so derive
    these axes from stable knuckle positions instead of wrist orientation.
    """
    wrist = poses["wrist"][:3]

    x_axis = poses["middle_proximal"][:3] - wrist
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1.0e-6:
        raise ValueError("Cannot construct hand frame: wrist-to-middle direction is degenerate.")
    x_axis /= x_norm

    y_axis = poses["index_proximal"][:3] - poses["little_proximal"][:3]
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1.0e-6:
        raise ValueError("Cannot construct hand frame: index-to-little direction is degenerate.")
    y_axis /= y_norm

    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Retargeting uses row vectors: world_vector @ retargeting_rotation.
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def visionpro_hand_to_joint_poses(hand_matrices: np.ndarray) -> dict[str, np.ndarray]:
    """Convert avp_stream world-frame hand matrices to DexVerse joint poses."""
    matrices = np.asarray(hand_matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or matrices.shape[0] < len(VISIONPRO_JOINT_NAMES):
        raise ValueError(f"Expected at least 25 hand transforms with shape (N, 4, 4), got {matrices.shape}.")

    poses = {name: matrix_to_pose(matrices[index]) for index, name in enumerate(VISIONPRO_JOINT_NAMES)}

    # The retargeters currently use wrist and finger joints, not palm. Supplying
    # a palm pose nevertheless keeps the raw layout compatible with OpenXR.
    palm_position = np.mean(
        [
            poses["index_metacarpal"][:3],
            poses["middle_metacarpal"][:3],
            poses["ring_metacarpal"][:3],
            poses["little_metacarpal"][:3],
        ],
        axis=0,
    )
    poses["palm"] = poses["wrist"].copy()
    poses["palm"][:3] = palm_position
    poses["_retargeting_rotation"] = _retargeting_rotation_from_hand_geometry(poses)
    return poses


def rotate_world_matrices(matrices: np.ndarray, yaw_offset_deg: float) -> np.ndarray:
    """Rotate world-frame transforms around the Isaac scene's Z axis."""
    matrices = np.asarray(matrices)
    single_matrix = matrices.ndim == 2
    if single_matrix:
        matrices = matrices[None, ...]
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"Expected transforms with shape (4, 4) or (N, 4, 4), got {matrices.shape}.")

    world_rotation = np.eye(4, dtype=np.float64)
    world_rotation[:3, :3] = Rotation.from_euler("z", yaw_offset_deg, degrees=True).as_matrix()
    rotated = world_rotation @ matrices
    return rotated[0] if single_matrix else rotated


class VisionProDevice(DeviceBase):
    """Stream Vision Pro hand tracking and expose it as a DexVerse device."""

    _KEY_COMMANDS = {"S": "START", "P": "STOP", "R": "RESET", "Q": "QUIT"}

    def __init__(self, cfg: VisionProDeviceCfg, retargeters: list[RetargeterBase] | None = None):
        super().__init__(retargeters)
        import carb
        import omni

        self._carb = carb
        self._ip = cfg.ip
        self._yaw_offset_deg = cfg.yaw_offset_deg
        self._additional_callbacks: dict[str, Callable] = {}
        '''
        avp_stream installs process-wide SIGINT/SIGTERM handlers when it is
        imported. Isaac Sim already installed handlers suited to Kit, so keep
        those instead of letting avp_stream raise KeyboardInterrupt from an
        arbitrary render/physics callback.
        '''
        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        from avp_stream import VisionProStreamer

        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        self._streamer = VisionProStreamer(ip=self._ip, record=False)

        default_pose = _identity_pose()
        self._previous_joint_poses_left = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_joint_poses_right = {name: default_pose.copy() for name in HAND_JOINT_NAMES}
        self._previous_head_pose = default_pose.copy()

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = self._carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def __del__(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        if getattr(self, "_streamer", None) is not None:
            self._streamer.cleanup()

    def __str__(self) -> str:
        return (
            f"Vision Pro Hand Tracking Device: {self.__class__.__name__}\n"
            f"\tVision Pro address: {self._ip}\n"
            f"\tScene yaw offset: {self._yaw_offset_deg:.1f} degrees\n"
            "\tControls: S=start/calibrate, P=pause, R=reset, Q=quit"
        )

    def reset(self):
        # Keep the last valid tracking pose across environment resets. This
        # avoids a one-frame jump to the origin while the stream is healthy.
        pass

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key.upper()] = func

    def _get_raw_data(self) -> dict[DeviceBase.TrackingTarget, Any]:
        latest = self._streamer.get_latest(use_cache=True)
        if latest is not None:
            try:
                left = rotate_world_matrices(latest.left, self._yaw_offset_deg)
                right = rotate_world_matrices(latest.right, self._yaw_offset_deg)
                self._previous_joint_poses_left = visionpro_hand_to_joint_poses(left)
                self._previous_joint_poses_right = visionpro_hand_to_joint_poses(right)
                if latest.head is not None:
                    head = rotate_world_matrices(latest.head, self._yaw_offset_deg)
                    self._previous_head_pose = matrix_to_pose(head)
            except (TypeError, ValueError):
                # A tracking dropout can produce an incomplete frame. Reusing
                # the last complete pose is safer than sending zeros.
                pass

        data: dict[DeviceBase.TrackingTarget, Any] = {}
        if RetargeterBase.Requirement.HAND_TRACKING in self._required_features:
            data[DeviceBase.TrackingTarget.HAND_LEFT] = self._previous_joint_poses_left
            data[DeviceBase.TrackingTarget.HAND_RIGHT] = self._previous_joint_poses_right
        if RetargeterBase.Requirement.HEAD_TRACKING in self._required_features:
            data[DeviceBase.TrackingTarget.HEAD] = self._previous_head_pose
        return data

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != self._carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key = event.input.name.upper()
        command = self._KEY_COMMANDS.get(key, key)
        callback = self._additional_callbacks.get(command)
        if callback is not None:
            callback()
        return True


@dataclass
class VisionProDeviceCfg(DeviceCfg):
    """Configuration for :class:`VisionProDevice`."""

    ip: str = field(default_factory=lambda: os.environ.get("VISIONPRO_IP", "192.168.3.37"))
    yaw_offset_deg: float = field(
        default_factory=lambda: float(os.environ.get("VISIONPRO_YAW_OFFSET_DEG", "-90.0"))
    )
    class_type: type[DeviceBase] = VisionProDevice
