# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for deriving the absolute retargeter's wrist joint origin from
a robot's :class:`ArticulationCfg`.

SimpleAbsoluteRetargeter needs, per hand, the world pose the robot's wrist
occupies when every wrist-joint action is zero. Because Isaac Lab's
``JointPositionActionCfg`` defaults to ``use_default_offset=True``, the sim
applies ``joint_target = action * scale + default_joint_pos``, so the correct
origin is the *home-wrist world pose*::

    origin_pos = base_pos + R(base_rot) @ home_translation_joint_values
    origin_rot = base_rot * R_joint_chain(home_rotation_joint_values)

For floating hands with identity ``base_rot`` and zero home rotation joints
(the common case) this reduces to ``origin_pos = base_pos + home_translation``
and ``origin_rot = base_rot``.

This module has *no* dependency on :mod:`dexverse.devices.retargeters` (which
already imports from :mod:`dexverse.robot_agents`), so it is safe to import
from robot-agent modules at import time without creating a cycle.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from isaaclab.assets.articulation import ArticulationCfg
from scipy.spatial.transform import Rotation as R


def _wxyz_to_xyzw(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _resolve_joint_default(joint_pos: dict[str, float], joint_name: str) -> float:
    """Return the default value for ``joint_name`` in ``joint_pos``.

    Keys in ``init_state.joint_pos`` may be either explicit joint names or
    Isaac Lab regex patterns (e.g. ``"FFJ(1|2|3|4)"``). Exact-match wins;
    otherwise fall back to the first regex key that matches the whole joint
    name. Raises :class:`KeyError` if neither is found.
    """
    if joint_name in joint_pos:
        return float(joint_pos[joint_name])
    for key, value in joint_pos.items():
        try:
            if re.fullmatch(key, joint_name) is not None:
                return float(value)
        except re.error:
            continue
    raise KeyError(
        f"Joint '{joint_name}' not found in init_state.joint_pos. Available keys: {sorted(joint_pos.keys())}."
    )


def compute_wrist_joint_origin(
    articulation_cfg: ArticulationCfg,
    translation_joint_names: Sequence[str],
    rotation_joint_names: Sequence[str] | None = None,
    rotation_axes: str = "XYZ",
    base_pos: Sequence[float] | None = None,
    base_rot: Sequence[float] | None = None,
    mount_offset: Sequence[float] | None = None,
) -> dict[str, tuple]:
    """Derive a SimpleAbsoluteRetargeter wrist joint origin from a robot cfg.

    The result is the world pose of the robot's wrist when every wrist-joint
    action is zero — i.e. the pose :class:`SimpleAbsoluteRetargeter` must
    subtract from the tracked hand pose to produce a valid joint-offset
    command.

    Args:
        articulation_cfg: The robot's :class:`ArticulationCfg`. ``init_state``
            is read for base pose and home joint values; it is **not**
            modified.
        translation_joint_names: The wrist translation joints in the action
            order used by the action cfg (typically
            ``("x_translation_joint", "y_translation_joint", "z_translation_joint")``).
            Their defaults from ``init_state.joint_pos`` form a 3-vector
            interpreted as an ``xyz`` offset in the robot base frame.
        rotation_joint_names: Optional wrist rotation joints, **in URDF chain
            order from parent (outer) to child (inner)**. Used together with
            ``rotation_axes`` to compose the home-wrist rotation. When
            omitted, the origin rotation is ``base_rot`` unchanged.
        rotation_axes: scipy intrinsic axes string matching
            ``rotation_joint_names`` order. For the standard floating-hand
            chain ``parent → x_rot → y_rot → z_rot → wrist`` this is
            ``"XYZ"``.
        base_pos: Override for the robot's world position, matching the value
            the setup builder will pass to ``ArticulationCfg.replace(pos=...)``.
            Defaults to ``articulation_cfg.init_state.pos``.
        base_rot: Override for the robot's world rotation (``(w, x, y, z)``),
            matching ``ArticulationCfg.replace(rot=...)``. Defaults to
            ``articulation_cfg.init_state.rot``.
        mount_offset: Fixed translation, in the robot base frame, from the
            articulation root to the link the wrist translation-joint chain
            actually emanates from. For single-hand floating robots the chain
            starts at the root, so this is ``(0, 0, 0)`` (the default). For
            multi-hand robots each hand's chain starts at a per-hand mount link
            that is offset from the root inside the USD (e.g. the bimanual
            Shadow hands sit at ``(0, -0.3, 0)`` / ``(0, +0.3, 0)``); pass that
            offset here so the origin reflects the hand's true home pose.
            Assumes the mount link is unrotated relative to the base.

    Returns:
        ``{"pos": (x, y, z), "rot": (w, x, y, z)}`` suitable for dropping into
        a ``SIMPLE_ABSOLUTE_WRIST_ORIGIN`` per-hand entry.
    """
    init_state = articulation_cfg.init_state

    effective_base_pos = np.asarray(base_pos if base_pos is not None else init_state.pos, dtype=np.float64)
    effective_base_rot_wxyz = np.asarray(base_rot if base_rot is not None else init_state.rot, dtype=np.float64)
    base_rotation = R.from_quat(_wxyz_to_xyzw(effective_base_rot_wxyz))

    if len(translation_joint_names) != 3:
        raise ValueError(
            f"translation_joint_names must have exactly 3 entries (xyz order); got {len(translation_joint_names)}."
        )
    home_trans = np.array(
        [_resolve_joint_default(init_state.joint_pos, name) for name in translation_joint_names],
        dtype=np.float64,
    )

    mount = np.zeros(3, dtype=np.float64) if mount_offset is None else np.asarray(mount_offset, dtype=np.float64)

    origin_pos = effective_base_pos + base_rotation.apply(mount + home_trans)

    if rotation_joint_names:
        if len(rotation_joint_names) != len(rotation_axes):
            raise ValueError(
                f"rotation_joint_names length ({len(rotation_joint_names)}) must match "
                f"rotation_axes length ({len(rotation_axes)})."
            )
        home_rot_angles = [_resolve_joint_default(init_state.joint_pos, name) for name in rotation_joint_names]
        home_rotation = R.from_euler(rotation_axes, home_rot_angles, degrees=False)
        origin_rot_scipy = base_rotation * home_rotation
        origin_rot_wxyz = _xyzw_to_wxyz(origin_rot_scipy.as_quat())
    else:
        origin_rot_wxyz = effective_base_rot_wxyz

    return {
        "pos": tuple(float(v) for v in origin_pos),
        "rot": tuple(float(v) for v in origin_rot_wxyz),
    }
