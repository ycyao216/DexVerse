# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Absolute hand-retargeter that drives the robot wrist to the tracked hand pose.

This retargeter reuses :class:`SimpleRelativeRetargeter` for finger retargeting
and hand-keypoint visualization, but changes the wrist command so the robot's
wrist is driven to the *absolute* world position of the VR hand (the red-dot
location in the XR visualization), not an offset from a calibration pose.

Concretely, for floating-hand robots whose wrist pose is commanded via
individual prismatic / revolute joints (``x/y/z_translation_joint`` +
``x/y/z_rotation_joint``), the joint commands are computed as::

    joint_trans = R_origin^T @ (hand_pos_world - joint_origin_pos_world)
    joint_rot   = normalize_hand_rotation(hand_quat_world)  (in the same frame)

where ``joint_origin_pos_world`` / ``joint_origin_rot_world`` is the world-frame
placement of the translation-joint origin for a given hand (typically the
robot's USD root pose for a single-arm floating hand). The rotation is
expressed in the target order (``xyz`` or ``yaw_pitch_roll``) to match the
action layout used by :class:`SimpleRelativeRetargeter`.

The ``quat_absolute`` wrist representation (used by arm-based IK controllers)
is also supported: the target EE pose is the hand pose transformed into the
robot base frame, instead of ``ee_default_pose_b + (hand_pos - base_pose[:3])``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase
from scipy.spatial.transform import Rotation as R

from .simple_relative_retargeting import (
    SIMPLE_RETARGETER_LAYOUT_SOURCES,
    SimpleRelativeRetargeter,
    SimpleRelativeRetargeterCfg,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

# Per-robot_type override for the attribute that holds the per-hand wrist joint
# origin on the robot layout module. Modules should expose a dict of the form::
#
#     {"right": {"pos": (x, y, z), "rot": (w, x, y, z)}, ...}
#
# where ``pos`` / ``rot`` is the world placement of the wrist translation-joint
# origin (typically the robot USD root pose for a single-arm floating hand).
SIMPLE_ABSOLUTE_WRIST_ORIGIN_ATTR_NAME = "SIMPLE_ABSOLUTE_WRIST_ORIGIN"
SIMPLE_ABSOLUTE_WRIST_ORIGIN_ATTR_OVERRIDES: dict[str, str] = {
    "floating_shadow_right": "FLOATING_SHADOW_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_shadow_left": "FLOATING_SHADOW_LEFT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_shadow_bimanual": "FLOATING_SHADOW_BIMANUAL_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_leap_right": "FLOATING_LEAP_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_leap_bimanual": "FLOATING_LEAP_BIMANUAL_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "bimanual_leap": "BIMANUAL_LEAP_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_sharpa_right": "FLOATING_SHARPA_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_sharpa_left": "FLOATING_SHARPA_LEFT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_sharpa_bimanual": "FLOATING_SHARPA_BIMANUAL_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_wuji_right": "FLOATING_WUJI_RIGHT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_wuji_left": "FLOATING_WUJI_LEFT_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
    "floating_wuji_bimanual": "FLOATING_WUJI_BIMANUAL_SIMPLE_ABSOLUTE_WRIST_ORIGIN",
}


def _default_origin() -> dict[str, tuple]:
    """Identity origin: zero position, unit quaternion (w,x,y,z)."""
    return {"pos": (0.0, 0.0, 0.0), "rot": (1.0, 0.0, 0.0, 0.0)}


class SimpleAbsoluteRetargeter(SimpleRelativeRetargeter):
    """Wrist retargeter that drives the robot wrist to the absolute hand pose.

    The finger retargeting logic is inherited unchanged from
    :class:`SimpleRelativeRetargeter`; only the wrist command assignment is
    overridden to use the world-frame hand pose directly (offset by the
    per-hand ``wrist_joint_origin``) instead of a delta from a calibration
    pose.
    """

    def __init__(self, cfg: SimpleAbsoluteRetargeterCfg):
        super().__init__(cfg)
        self.cfg: SimpleAbsoluteRetargeterCfg = cfg  # type: ignore[assignment]
        self._wrist_joint_origins = self._resolve_wrist_joint_origins()

    def _resolve_wrist_joint_origins(self) -> dict[DeviceBase.TrackingTarget, dict[str, np.ndarray]]:
        """Build the per-hand wrist joint-origin map, in world frame.

        Priority (highest first):
          1. Entries explicitly provided in ``cfg.wrist_joint_origin``.
          2. Module-level constant named by
             ``SIMPLE_ABSOLUTE_WRIST_ORIGIN_ATTR_OVERRIDES[robot_type]``, or
             ``SIMPLE_ABSOLUTE_WRIST_ORIGIN`` as a default fallback.
          3. Identity (zero position + unit quaternion), with a warning.
        """
        user_override = dict(self.cfg.wrist_joint_origin or {})

        module_spec: dict[str, Any] | None = None
        module_name, _ = SIMPLE_RETARGETER_LAYOUT_SOURCES[self.cfg.robot_type]
        module = import_module(module_name)
        attr_name = SIMPLE_ABSOLUTE_WRIST_ORIGIN_ATTR_OVERRIDES.get(
            self.cfg.robot_type, SIMPLE_ABSOLUTE_WRIST_ORIGIN_ATTR_NAME
        )
        module_spec = getattr(module, attr_name, None)

        resolved: dict[DeviceBase.TrackingTarget, dict[str, np.ndarray]] = {}
        for hand_target in self._tracked_hands:
            hand_key = hand_target.name.replace("HAND_", "").lower()

            entry: dict[str, Any] | None = None
            for candidate in (user_override.get(hand_target), user_override.get(hand_key)):
                if candidate is not None:
                    entry = candidate
                    break

            if entry is None and module_spec is not None:
                entry = module_spec.get(hand_key)

            if entry is None:
                entry = _default_origin()
                print(
                    "[SimpleAbsoluteRetargeter] No wrist_joint_origin configured for hand "
                    f"'{hand_key}' (robot_type='{self.cfg.robot_type}'). Falling back to "
                    "identity; the robot wrist may not track the VR hand correctly."
                )

            resolved[hand_target] = {
                "pos": np.asarray(entry.get("pos", (0.0, 0.0, 0.0)), dtype=np.float32),
                "rot": np.asarray(entry.get("rot", (1.0, 0.0, 0.0, 0.0)), dtype=np.float32),
            }

        return resolved

    def _assign_hand_wrist_command(self, action: np.ndarray, hand: DeviceBase.TrackingTarget) -> None:
        """Write one hand's absolute wrist command into the action vector."""
        wrist_pose = self.latest_wrist_poses.get(hand)
        if wrist_pose is None:
            return

        wrist_pos = np.asarray(wrist_pose[:3], dtype=np.float32).copy()
        wrist_quat = np.asarray(wrist_pose[3:], dtype=np.float32).copy()

        hand_layout = self._layout["hands"][hand]
        rot_repr = hand_layout.get("wrist_rot_repr", "euler")

        origin = self._wrist_joint_origins[hand]
        origin_pos = origin["pos"]
        origin_rot = R.from_quat(wxyz_to_xyzw(origin["rot"]))
        origin_rot_inv = origin_rot.inv()

        if rot_repr == "quat_absolute":
            self._assign_absolute_wrist_command_world(
                action, hand_layout, wrist_pos, wrist_quat, origin_pos, origin_rot_inv
            )
            return

        # Translate the hand into the joint-origin frame. Joint action values
        # are interpreted as offsets from this frame.
        wrist_pos_local = origin_rot_inv.apply(wrist_pos - origin_pos).astype(np.float32)

        normalized_world = self._get_normalized_wrist_rotation(wrist_quat)
        normalized_local = origin_rot_inv * normalized_world

        if rot_repr == "euler":
            euler_xyz = normalized_local.as_euler("XYZ", degrees=False)
            order = hand_layout["wrist_rot_order"]
            if order == "xyz":
                rot_cmd = np.array(euler_xyz, dtype=np.float32)
            elif order == "yaw_pitch_roll":
                rot_cmd = np.array([euler_xyz[2], euler_xyz[1], euler_xyz[0]], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported wrist rotation order '{order}'.")
        elif rot_repr == "rotvec":
            rot_cmd = np.asarray(normalized_local.as_rotvec(), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported wrist_rot_repr '{rot_repr}'. Use 'euler', 'rotvec', or 'quat_absolute'.")

        rot_signs = np.asarray(hand_layout.get("wrist_rot_signs", (1.0, 1.0, 1.0)), dtype=np.float32)
        rot_cmd = rot_cmd * rot_signs

        self._assign(action, hand_layout["wrist_trans_indices"], wrist_pos_local)
        self._assign(action, hand_layout["wrist_rot_indices"], rot_cmd)

    def _assign_absolute_wrist_command_world(
        self,
        action: np.ndarray,
        hand_layout: dict[str, Any],
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
        origin_pos: np.ndarray,
        origin_rot_inv: R,
    ) -> None:
        """Write an absolute EE target pose expressed in the robot base frame.

        The hand pose in the world frame is transformed directly into the robot
        base frame without any reference to a calibration pose or a default
        home EE pose — the robot EE is commanded to coincide with the VR hand.
        """
        target_pos_b = origin_rot_inv.apply(wrist_pos - origin_pos).astype(np.float32)

        hand_rot_world = self._get_normalized_wrist_rotation(wrist_quat_wxyz)
        target_rot_b = origin_rot_inv * hand_rot_world
        target_quat_wxyz = xyzw_to_wxyz(target_rot_b.as_quat().astype(np.float32))

        self._assign(action, hand_layout["wrist_trans_indices"], target_pos_b)
        self._assign(action, hand_layout["wrist_rot_indices"], target_quat_wxyz)


@dataclass
class SimpleAbsoluteRetargeterCfg(SimpleRelativeRetargeterCfg):
    """Configuration for :class:`SimpleAbsoluteRetargeter`.

    In addition to all fields of :class:`SimpleRelativeRetargeterCfg`, this cfg
    accepts an explicit ``wrist_joint_origin`` mapping, used to express the
    tracked hand pose in the robot's wrist-joint coordinate frame.

    Attributes:
        wrist_joint_origin: Per-hand override for the world placement of the
            wrist translation-joint origin. Keys may be either
            :class:`DeviceBase.TrackingTarget` values or the lowercase hand
            names (``"right"`` / ``"left"``). Values are dicts of the form
            ``{"pos": (x, y, z), "rot": (w, x, y, z)}`` in world coordinates.
            If omitted, the retargeter looks up
            ``SIMPLE_ABSOLUTE_WRIST_ORIGIN`` (or the per-robot override) on
            the robot's layout module.
    """

    wrist_joint_origin: dict[Any, dict[str, tuple]] = field(default_factory=dict)
    retargeter_type: type[RetargeterBase] = SimpleAbsoluteRetargeter
