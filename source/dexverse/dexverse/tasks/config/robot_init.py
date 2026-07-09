# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-agnostic wrist (palm) init-pose helpers for task configs.

Tasks that want the robot's wrist to spawn at a specific world location pre-
dexverse-cross-embodiment-refactor expressed that intent by writing directly to
``init_state.joint_pos["x_translation_joint"]`` etc. — which only works for
single-arm floating Shadow / Leap robots. On the bimanual variants those keys
don't exist (they're prefixed ``rh_``/``lh_``).

:func:`set_robot_wrist_init_world_pos` takes the desired palm pose in
*world* coordinates and dispatches per ``env_cfg.robot_type`` to the correct
mechanism so the same task cfg works across embodiments.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

_SINGLE_ARM_FLOATING_TRANSLATION_ROBOTS = (
    "floating_shadow_right",
    "floating_shadow_left",
    "floating_leap_right",
    "floating_sharpa_right",
    "floating_sharpa_left",
    "floating_wuji_right",
    "floating_wuji_left",
)
_FLOATING_BIMANUAL_TRANSLATION_ROBOTS = (
    "floating_shadow_bimanual",
    "floating_sharpa_bimanual",
    "floating_wuji_bimanual",
)

_SINGLE_FLOATING_JOINTS: dict[str, tuple[tuple[str, str, str], tuple[str, str, str]]] = {
    "floating_shadow_right": (
        ("x_translation_joint", "y_translation_joint", "z_translation_joint"),
        ("x_rotation_joint", "y_rotation_joint", "z_rotation_joint"),
    ),
    "floating_shadow_left": (
        ("x_translation_joint", "y_translation_joint", "z_translation_joint"),
        ("x_rotation_joint", "y_rotation_joint", "z_rotation_joint"),
    ),
    "floating_leap_right": (
        ("x_translation_joint", "y_translation_joint", "z_translation_joint"),
        ("x_rotation_joint", "y_rotation_joint", "z_rotation_joint"),
    ),
    "floating_sharpa_right": (
        ("right_x_joint", "right_y_joint", "right_z_joint"),
        ("right_roll_joint", "right_pitch_joint", "right_yaw_joint"),
    ),
    "floating_sharpa_left": (
        ("left_x_joint", "left_y_joint", "left_z_joint"),
        ("left_roll_joint", "left_pitch_joint", "left_yaw_joint"),
    ),
    "floating_wuji_right": (
        ("right_x_joint", "right_y_joint", "right_z_joint"),
        ("right_roll_joint", "right_pitch_joint", "right_yaw_joint"),
    ),
    "floating_wuji_left": (
        ("left_x_joint", "left_y_joint", "left_z_joint"),
        ("left_roll_joint", "left_pitch_joint", "left_yaw_joint"),
    ),
}

_FLOATING_BIMANUAL_JOINTS: dict[str, dict[str, tuple[tuple[str, str, str], tuple[str, str, str]]]] = {
    "floating_shadow_bimanual": {
        "right": (
            ("rh_x_translation_joint", "rh_y_translation_joint", "rh_z_translation_joint"),
            ("rh_x_rotation_joint", "rh_y_rotation_joint", "rh_z_rotation_joint"),
        ),
        "left": (
            ("lh_x_translation_joint", "lh_y_translation_joint", "lh_z_translation_joint"),
            ("lh_x_rotation_joint", "lh_y_rotation_joint", "lh_z_rotation_joint"),
        ),
    },
    "floating_sharpa_bimanual": {
        "right": (
            ("right_x_joint", "right_y_joint", "right_z_joint"),
            ("right_roll_joint", "right_pitch_joint", "right_yaw_joint"),
        ),
        "left": (
            ("left_x_joint", "left_y_joint", "left_z_joint"),
            ("left_roll_joint", "left_pitch_joint", "left_yaw_joint"),
        ),
    },
    "floating_wuji_bimanual": {
        "right": (
            ("right_x_joint", "right_y_joint", "right_z_joint"),
            ("right_roll_joint", "right_pitch_joint", "right_yaw_joint"),
        ),
        "left": (
            ("left_x_joint", "left_y_joint", "left_z_joint"),
            ("left_roll_joint", "left_pitch_joint", "left_yaw_joint"),
        ),
    },
}


def _supported_robot_types_for_wrist_init() -> tuple[str, ...]:
    return (
        *_SINGLE_ARM_FLOATING_TRANSLATION_ROBOTS,
        *_FLOATING_BIMANUAL_TRANSLATION_ROBOTS,
    )


def _single_floating_joint_names(robot_type: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    try:
        return _SINGLE_FLOATING_JOINTS[robot_type]
    except KeyError as exc:
        raise NotImplementedError(f"No single-floating joint map for robot_type={robot_type!r}.") from exc


def _floating_bimanual_joint_names(
    robot_type: str,
) -> dict[str, tuple[tuple[str, str, str], tuple[str, str, str]]]:
    try:
        return _FLOATING_BIMANUAL_JOINTS[robot_type]
    except KeyError as exc:
        raise NotImplementedError(f"No bimanual-floating joint map for robot_type={robot_type!r}.") from exc


def _bimanual_hand_mount_offsets(robot_type: str) -> dict[str, tuple]:
    """Per-hand mount offsets (base frame) for a bimanual floating hand.

    Imported lazily so this module stays free of a hard ``robot_agents``
    dependency at import time.
    """
    if robot_type == "floating_shadow_bimanual":
        from dexverse.robot_agents.shadow.floating import (
            FLOATING_SHADOW_BIMANUAL_HAND_MOUNT_OFFSET,
        )

        return FLOATING_SHADOW_BIMANUAL_HAND_MOUNT_OFFSET

    if robot_type == "floating_sharpa_bimanual":
        from dexverse.robot_agents.sharpa.floating import (
            FLOATING_SHARPA_BIMANUAL_HAND_MOUNT_OFFSET,
        )

        return FLOATING_SHARPA_BIMANUAL_HAND_MOUNT_OFFSET

    if robot_type == "floating_wuji_bimanual":
        from dexverse.robot_agents.wuji.floating import (
            FLOATING_WUJI_BIMANUAL_HAND_MOUNT_OFFSET,
        )

        return FLOATING_WUJI_BIMANUAL_HAND_MOUNT_OFFSET

    raise NotImplementedError(f"No bimanual mount-offset map for robot_type={robot_type!r}.")


def _wxyz_to_xyzw(q) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _xyzw_to_wxyz_tuple(q) -> tuple[float, float, float, float]:
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def set_robot_wrist_init_world_pos(
    env_cfg,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    rot: tuple[float, float, float, float] | None = None,
) -> None:
    """Set the robot's wrist (palm) initial world pose.

    ``x``/``y``/``z`` are absolute world-frame coords in metres; ``None`` leaves
    that component at the robot's default. ``rot`` is a world-frame quaternion
    ``(w, x, y, z)`` for the palm; ``None`` preserves the default orientation.
    The conversion to joint values is robot-specific:

    * Floating Shadow/Leap/Sharpa single-arm robots: writes the robot-specific
      three translation joints so that ``base_pos + R(base_rot) @ joint`` equals
      the requested palm world position, and writes the three rotation joints
      (XYZ intrinsic euler) for the orientation.
    * Bimanual floating Shadow/Sharpa/Wuji: writes the robot-specific translation
      and rotation joints for *both* hands, accounting for each hand's fixed
      mount offset.
    """
    if x is None and y is None and z is None and rot is None:
        return
    robot_type = getattr(env_cfg, "robot_type", None)
    if robot_type in _SINGLE_ARM_FLOATING_TRANSLATION_ROBOTS:
        _apply_floating_pose(env_cfg, x=x, y=y, z=z, rot=rot)
    elif robot_type in _FLOATING_BIMANUAL_TRANSLATION_ROBOTS:
        _apply_floating_bimanual_pose(env_cfg, x=x, y=y, z=z, rot=rot)
    else:
        raise NotImplementedError(
            f"set_robot_wrist_init_world_pos: robot_type={robot_type!r} is not "
            "supported yet. Supported: "
            f"{_supported_robot_types_for_wrist_init()}."
        )


def _apply_floating_pose(env_cfg, *, x, y, z, rot) -> None:
    """Set translation (and optionally rotation) joints on a floating hand."""
    robot_type = getattr(env_cfg, "robot_type", None)
    translation_joint_names, rotation_joint_names = _single_floating_joint_names(robot_type)
    init_state = env_cfg.scene.robot.init_state
    base_pos = np.asarray(init_state.pos, dtype=np.float64)
    base_rot_sp = R.from_quat(_wxyz_to_xyzw(init_state.rot))
    joint_pos = dict(init_state.joint_pos)

    cur_joint_trans = np.array(
        [float(joint_pos.get(joint_name, 0.0)) for joint_name in translation_joint_names],
        dtype=np.float64,
    )
    cur_palm_w = base_pos + base_rot_sp.apply(cur_joint_trans)
    new_palm_w = np.array(
        [
            float(x) if x is not None else cur_palm_w[0],
            float(y) if y is not None else cur_palm_w[1],
            float(z) if z is not None else cur_palm_w[2],
        ],
        dtype=np.float64,
    )
    new_joint_trans = base_rot_sp.inv().apply(new_palm_w - base_pos)
    for joint_name, joint_value in zip(translation_joint_names, new_joint_trans):
        joint_pos[joint_name] = float(joint_value)

    if rot is not None:
        # palm_rot_w = base_rot @ joint_rot, so joint_rot = base_rot^-1 @ palm_rot_w.
        # The wrist chain is XYZ intrinsic; mirrors compute_wrist_joint_origin.
        target_palm_rot_w = R.from_quat(_wxyz_to_xyzw(rot))
        joint_rot_sp = base_rot_sp.inv() * target_palm_rot_w
        for joint_name, joint_value in zip(rotation_joint_names, joint_rot_sp.as_euler("XYZ", degrees=False)):
            joint_pos[joint_name] = float(joint_value)

    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        init_state=init_state.replace(joint_pos=joint_pos),
    )


def _apply_floating_bimanual_pose(env_cfg, *, x, y, z, rot) -> None:
    """Set wrist translation and rotation joints on both hands of a floating bimanual robot."""
    robot_type = getattr(env_cfg, "robot_type", None)
    init_state = env_cfg.scene.robot.init_state
    base_pos = np.asarray(init_state.pos, dtype=np.float64)
    base_rot_sp = R.from_quat(_wxyz_to_xyzw(init_state.rot))
    joint_pos = dict(init_state.joint_pos)
    mount_offsets = _bimanual_hand_mount_offsets(robot_type)
    joint_names_by_hand = _floating_bimanual_joint_names(robot_type)

    for hand_key in ("right", "left"):
        translation_joint_names, rotation_joint_names = joint_names_by_hand[hand_key]
        mount = np.asarray(mount_offsets[hand_key], dtype=np.float64)
        cur_joint_trans = np.array(
            [float(joint_pos.get(joint_name, 0.0)) for joint_name in translation_joint_names],
            dtype=np.float64,
        )
        cur_palm_w = base_pos + base_rot_sp.apply(mount + cur_joint_trans)
        new_palm_w = np.array(
            [
                float(x) if x is not None else cur_palm_w[0],
                float(y) if y is not None else cur_palm_w[1],
                float(z) if z is not None else cur_palm_w[2],
            ],
            dtype=np.float64,
        )
        new_joint_trans = base_rot_sp.inv().apply(new_palm_w - base_pos) - mount
        for joint_name, joint_value in zip(translation_joint_names, new_joint_trans):
            joint_pos[joint_name] = float(joint_value)

        if rot is not None:
            target_palm_rot_w = R.from_quat(_wxyz_to_xyzw(rot))
            joint_rot_sp = base_rot_sp.inv() * target_palm_rot_w
            for joint_name, joint_value in zip(rotation_joint_names, joint_rot_sp.as_euler("XYZ", degrees=False)):
                joint_pos[joint_name] = float(joint_value)

    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        init_state=init_state.replace(joint_pos=joint_pos),
    )


def align_retargeter_wrist_origin_to_init(
    env_cfg,
    hand_key: str = "right",
) -> None:
    """Stamp every retargeter's ``wrist_joint_origin`` to the current init frame.

    Call this after :func:`set_robot_wrist_init_world_pos` (so the joint state
    already reflects the desired init) AND after ``teleop_devices`` has been
    built. Floating-hand absolute retargeting needs the world pose the wrist
    sits at when actions are zero.

    Robot dispatch matches :func:`set_robot_wrist_init_world_pos`. For bimanual
    robots both ``"right"`` and ``"left"`` origins are stamped (``hand_key`` is
    ignored).
    """
    origin_entry = _compute_retargeter_origin_entry(env_cfg, hand_key)
    teleop_devices = getattr(env_cfg, "teleop_devices", None)
    if teleop_devices is None:
        return
    devices = getattr(teleop_devices, "devices", None) or {}
    if not devices:
        return
    for device_cfg in devices.values():
        for rtg in getattr(device_cfg, "retargeters", None) or []:
            if hasattr(rtg, "wrist_joint_origin"):
                rtg.wrist_joint_origin = origin_entry


def _compute_retargeter_origin_entry(env_cfg, hand_key: str) -> dict[str, dict]:
    """Return the ``{hand_key: {"pos", "rot"}}`` map for the retargeter origin."""
    robot_type = getattr(env_cfg, "robot_type", None)
    if robot_type in _SINGLE_ARM_FLOATING_TRANSLATION_ROBOTS:
        return {hand_key: _palm_world_pose_floating(env_cfg)}
    if robot_type in _FLOATING_BIMANUAL_TRANSLATION_ROBOTS:
        return _palm_world_pose_floating_bimanual(env_cfg)
    raise NotImplementedError(
        f"align_retargeter_wrist_origin_to_init: robot_type={robot_type!r} "
        "is not supported yet. Supported: "
        f"{_supported_robot_types_for_wrist_init()}."
    )


def _palm_world_pose_floating(env_cfg) -> dict[str, tuple]:
    robot_type = getattr(env_cfg, "robot_type", None)
    translation_joint_names, rotation_joint_names = _single_floating_joint_names(robot_type)
    init_state = env_cfg.scene.robot.init_state
    base_pos = np.asarray(init_state.pos, dtype=np.float64)
    base_rot_sp = R.from_quat(_wxyz_to_xyzw(init_state.rot))
    joint_pos = init_state.joint_pos

    trans = np.array(
        [float(joint_pos.get(joint_name, 0.0)) for joint_name in translation_joint_names],
        dtype=np.float64,
    )
    palm_pos_w = base_pos + base_rot_sp.apply(trans)

    # The wrist chain is base → x_rot → y_rot → z_rot → palm with XYZ intrinsic
    # composition; matches dexverse.devices.wrist_origin.compute_wrist_joint_origin.
    joint_rot_values = [float(joint_pos.get(joint_name, 0.0)) for joint_name in rotation_joint_names]
    joint_rot_sp = R.from_euler("XYZ", joint_rot_values, degrees=False)
    palm_rot_w_sp = base_rot_sp * joint_rot_sp
    palm_quat_w_wxyz = _xyzw_to_wxyz_tuple(palm_rot_w_sp.as_quat())

    return {
        "pos": (float(palm_pos_w[0]), float(palm_pos_w[1]), float(palm_pos_w[2])),
        "rot": palm_quat_w_wxyz,
    }


def _palm_world_pose_floating_bimanual(env_cfg) -> dict[str, dict]:
    """Return per-hand palm world pose for a floating bimanual setup."""
    robot_type = getattr(env_cfg, "robot_type", None)
    init_state = env_cfg.scene.robot.init_state
    base_pos = np.asarray(init_state.pos, dtype=np.float64)
    base_rot_sp = R.from_quat(_wxyz_to_xyzw(init_state.rot))
    joint_pos = init_state.joint_pos
    mount_offsets = _bimanual_hand_mount_offsets(robot_type)
    joint_names_by_hand = _floating_bimanual_joint_names(robot_type)

    poses: dict[str, dict] = {}
    for hand_key in ("right", "left"):
        translation_joint_names, rotation_joint_names = joint_names_by_hand[hand_key]
        mount = np.asarray(mount_offsets[hand_key], dtype=np.float64)
        trans = np.array(
            [float(joint_pos.get(joint_name, 0.0)) for joint_name in translation_joint_names],
            dtype=np.float64,
        )
        palm_pos_w = base_pos + base_rot_sp.apply(mount + trans)

        joint_rot_values = [float(joint_pos.get(joint_name, 0.0)) for joint_name in rotation_joint_names]
        joint_rot_sp = R.from_euler("XYZ", joint_rot_values, degrees=False)
        palm_rot_w_sp = base_rot_sp * joint_rot_sp
        palm_quat_w_wxyz = _xyzw_to_wxyz_tuple(palm_rot_w_sp.as_quat())

        poses[hand_key] = {
            "pos": (float(palm_pos_w[0]), float(palm_pos_w[1]), float(palm_pos_w[2])),
            "rot": palm_quat_w_wxyz,
        }
    return poses


__all__ = [
    "set_robot_wrist_init_world_pos",
    "align_retargeter_wrist_origin_to_init",
]
