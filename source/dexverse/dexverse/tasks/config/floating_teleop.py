# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared teleoperation helpers for floating tabletop configs."""

from dataclasses import fields, is_dataclass

from dexverse.devices.retargeters import (
    SimpleAbsoluteRetargeterCfg,
    SimpleRelativeRetargeterCfg,
)
from dexverse.devices.retargeters.simple_relative_retargeting import (
    SIMPLE_RETARGETER_LAYOUT_SOURCES,
)
from isaaclab.devices import DevicesCfg, OpenXRDeviceCfg
from isaaclab.devices.openxr import XrCfg

# Supported values for the ``--teleop_retargeter`` CLI flag / equivalent
# ``env.teleop_retargeter=`` Hydra-style override.
TELEOP_RETARGETER_MODES = ("relative", "absolute")

# Supported values for the ``--retargeting_scheme`` CLI flag / equivalent
# ``env.retargeting_scheme=`` Hydra-style override. This is orthogonal to the
# relative/absolute mode above: it selects which dex-retargeting YAML the finger
# optimizer loads (DexPilot pinch vectors vs. palm->fingertip vectors).
TELEOP_RETARGETING_SCHEMES = ("dexpilot", "vector")

# Default XR configuration for floating hands
DEFAULT_FLOATING_SHADOW_XR_CFG = XrCfg(
    anchor_pos=[-0.5, 0.0, 0.1],
    anchor_rot=[0.7071068, 0.0, 0.0, -0.7071068],
)

DEFAULT_FLOATING_LEAP_XR_CFG = XrCfg(
    anchor_pos=[-0.25, 0.0, 0.1],
    anchor_rot=[0.7071068, 0.0, 0.0, -0.7071068],
)

# Additive world-frame offset applied to each device's ``XrCfg.anchor_pos``
# depending on the selected retargeter mode. This is the global default used by
# every environment; individual env cfgs can override it by setting a
# ``retargeter_anchor_pos_offsets`` attribute to a dict of the same shape (or
# a partial dict — unspecified modes fall back to the default).
#
# Rationale: the absolute retargeter drives the robot wrist to the VR hand's
# world pose directly, so the VR hand's natural resting position should sit a
# bit lower than in the relative retargeter (which offsets from the robot's
# calibration pose).
DEFAULT_RETARGETER_ANCHOR_POS_OFFSET: dict[str, tuple[float, float, float]] = {
    "relative": (0.0, 0.0, 0.0),
    "absolute": (0.1, 0.0, -0.3),
}


def maybe_set_leap_xr_cfg(env_cfg) -> None:
    """Use Leap-style XR anchor when `robot_type` is a Leap variant."""
    rt = getattr(env_cfg, "robot_type", None)
    if rt in (
        "floating_leap_right",
        "floating_leap_bimanual",
        "bimanual_leap",
    ):
        env_cfg.xr = DEFAULT_FLOATING_LEAP_XR_CFG


def _devices_simple_relative(sim_device, xr_cfg: XrCfg, simple_robot_type: str) -> DevicesCfg:
    """Single code path: `SimpleRelativeRetargeter` with layout from `simple_relative_retargeting.py`."""
    return DevicesCfg(
        devices={
            "handtracking": OpenXRDeviceCfg(
                retargeters=[
                    SimpleRelativeRetargeterCfg(
                        sim_device=sim_device,
                        robot_type=simple_robot_type,
                    )
                ],
                sim_device=sim_device,
                xr_cfg=xr_cfg,
            ),
        }
    )


def _devices_simple_absolute(sim_device, xr_cfg: XrCfg, simple_robot_type: str) -> DevicesCfg:
    """Single code path for :class:`SimpleAbsoluteRetargeter`.

    The absolute retargeter drives the robot wrist to the VR hand's world pose
    (the red-dot location in the XR visualization) rather than to a delta from
    a calibration pose.
    """
    return DevicesCfg(
        devices={
            "handtracking": OpenXRDeviceCfg(
                retargeters=[
                    SimpleAbsoluteRetargeterCfg(
                        sim_device=sim_device,
                        robot_type=simple_robot_type,
                    )
                ],
                sim_device=sim_device,
                xr_cfg=xr_cfg,
            ),
        }
    )


def build_floating_teleop_devices(
    sim_device,
    xr_cfg: XrCfg,
    hand_joint_names: list[str],
    wrist_position_offset: tuple[float, float, float],
    retargeter_config_filename: str | None,
    retargeter_urdf_path: str | None = None,
    apply_shadow_specific_postprocess: bool = False,
    enable_visualization: bool = True,
    num_open_xr_hand_joints: int = 26,
    env_robot_type: str | None = None,
    use_absolute_retargeter: bool = False,
) -> DevicesCfg:
    """Build teleoperation devices for floating tabletop robots.

    All supported robots use :class:`~dexverse.devices.retargeters.simple_relative_retargeting.SimpleRelativeRetargeter`
    so finger joint ordering matches ``SIMPLE_RELATIVE_RETARGETER_LAYOUT`` / dex specs in each robot module.

    Args:
        sim_device: Simulation device (e.g., ``cuda:0``).
        xr_cfg: XR configuration.
        hand_joint_names: Legacy; unused for SimpleRelative (kept for call-site compatibility).
        wrist_position_offset: Legacy; unused for SimpleRelative.
        retargeter_config_filename: Legacy filename hint; only used if ``env_robot_type`` is missing.
        retargeter_urdf_path: Legacy; unused.
        apply_shadow_specific_postprocess: Legacy; unused.
        enable_visualization: Legacy; unused.
        num_open_xr_hand_joints: Legacy; unused.
        env_robot_type: Must match :attr:`DexVerseBaseEnvCfg.robot_type` (e.g. ``floating_shadow_right``).
            This selects the layout key in ``SIMPLE_RETARGETER_LAYOUT_SOURCES``. Prefer passing this always.
        use_absolute_retargeter: If True, use :class:`SimpleAbsoluteRetargeter`
            so the robot wrist tracks the VR hand's world pose directly. If
            False (default), use :class:`SimpleRelativeRetargeter` which tracks
            displacement from the calibration pose.

    Returns:
        DevicesCfg with OpenXR + retargeter (relative or absolute based on flag).
    """
    # Legacy parameters above are kept for call-site compatibility; layout is driven by env_robot_type.
    _ = (
        hand_joint_names,
        wrist_position_offset,
        retargeter_urdf_path,
        apply_shadow_specific_postprocess,
        enable_visualization,
        num_open_xr_hand_joints,
    )

    devices_builder = _devices_simple_absolute if use_absolute_retargeter else _devices_simple_relative

    # Primary: explicit env robot type (authoritative; avoids brittle filename substring rules).
    if env_robot_type is not None and env_robot_type in SIMPLE_RETARGETER_LAYOUT_SOURCES:
        return devices_builder(sim_device, xr_cfg, env_robot_type)

    # Legacy fallback: infer from retargeter_config_filename (e.g. custom tasks / old scripts).
    if retargeter_config_filename:
        name_lower = retargeter_config_filename.lower()
        if name_lower in (
            "floating_leap_bimanual",
            "bimanual_leap",
            "floating_shadow_left",
            "floating_shadow_bimanual",
            "floating_sharpa_right",
            "floating_sharpa_left",
            "floating_sharpa_bimanual",
            "floating_wuji_right",
            "floating_wuji_left",
            "floating_wuji_bimanual",
        ):
            return devices_builder(sim_device, xr_cfg, retargeter_config_filename)
        if "shadow" in name_lower:
            return devices_builder(sim_device, xr_cfg, "floating_shadow_right")
        if "leap" in name_lower:
            return devices_builder(sim_device, xr_cfg, "floating_leap_right")

    raise ValueError(
        "Could not select a SimpleRelative retargeter layout. Pass env_robot_type= to "
        "build_floating_teleop_devices (e.g. 'floating_shadow_right', 'floating_leap_right'). "
        f"Supported keys: {sorted(SIMPLE_RETARGETER_LAYOUT_SOURCES.keys())}."
    )


def _clone_retargeter_cfg(source, target_cls):
    """Return a new ``target_cls`` instance copying common dataclass fields from ``source``.

    Fields declared on ``source`` are copied. Fields that are *only* on
    ``target_cls`` are also copied if the source carries a matching runtime
    attribute (e.g. an env-cfg patch attached ``wrist_joint_origin`` to a
    relative cfg so it survives the relative→absolute swap). ``retargeter_type``
    is left at the ``target_cls`` default so the new cfg builds the intended
    retargeter.
    """
    if not (is_dataclass(source) and is_dataclass(target_cls)):
        return source
    src_field_names = {f.name for f in fields(source)}
    kwargs = {}
    for field in fields(target_cls):
        if field.name == "retargeter_type":
            continue
        if field.name in src_field_names or hasattr(source, field.name):
            kwargs[field.name] = getattr(source, field.name)
    return target_cls(**kwargs)


def _resolve_anchor_pos_offset(
    mode: str,
    anchor_pos_offsets: dict[str, tuple[float, float, float]] | None,
) -> tuple[float, float, float]:
    """Pick the anchor-pos offset for ``mode`` from an override dict or the default.

    ``anchor_pos_offsets`` may be a partial dict; unspecified modes fall back
    to :data:`DEFAULT_RETARGETER_ANCHOR_POS_OFFSET`.
    """
    if anchor_pos_offsets is not None and mode in anchor_pos_offsets:
        offset = anchor_pos_offsets[mode]
    else:
        offset = DEFAULT_RETARGETER_ANCHOR_POS_OFFSET.get(mode, (0.0, 0.0, 0.0))
    return (float(offset[0]), float(offset[1]), float(offset[2]))


def _apply_anchor_pos_offset_to_devices(
    devices_cfg,
    offset: tuple[float, float, float],
) -> None:
    """Add ``offset`` to ``xr_cfg.anchor_pos`` on every device in ``devices_cfg``."""
    if offset == (0.0, 0.0, 0.0):
        return
    devices = getattr(devices_cfg, "devices", None) or {}
    for device_cfg in devices.values():
        xr_cfg = getattr(device_cfg, "xr_cfg", None)
        if xr_cfg is None:
            continue
        ax, ay, az = xr_cfg.anchor_pos
        ox, oy, oz = offset
        device_cfg.xr_cfg = xr_cfg.replace(anchor_pos=(ax + ox, ay + oy, az + oz))


def apply_teleop_retargeter_mode(
    devices_cfg,
    mode: str,
    anchor_pos_offsets: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Swap retargeter cfgs in an already-built :class:`DevicesCfg` in place.

    This is the entry point used by CLI scripts (``teleop_agent.py`` /
    ``record_demos.py``) to switch between the relative and absolute retargeter
    after a task's env cfg has been fully built by its ``__post_init__`` — it
    avoids having every task cfg forward an extra flag.

    Each device's ``xr_cfg.anchor_pos`` is also nudged by a per-mode offset so
    the VR viewpoint matches what feels natural for that retargeter (in
    particular, the absolute retargeter wants a slightly lower anchor than the
    relative one). The offset comes from ``anchor_pos_offsets`` if provided,
    otherwise from :data:`DEFAULT_RETARGETER_ANCHOR_POS_OFFSET`.

    Args:
        devices_cfg: ``DevicesCfg`` (typically ``env_cfg.teleop_devices``) that
            may carry one or more retargeter cfgs in its device entries.
        mode: One of :data:`TELEOP_RETARGETER_MODES`.
        anchor_pos_offsets: Optional per-env override for the per-mode anchor
            offsets. Partial dicts are allowed — unspecified modes fall back to
            the module-level default.
    """
    if mode not in TELEOP_RETARGETER_MODES:
        raise ValueError(f"Unknown teleop retargeter mode '{mode}'. Supported: {TELEOP_RETARGETER_MODES}.")
    if devices_cfg is None:
        return

    offset = _resolve_anchor_pos_offset(mode, anchor_pos_offsets)
    _apply_anchor_pos_offset_to_devices(devices_cfg, offset)

    if mode == "relative":
        return

    mode_to_target = {
        "absolute": SimpleAbsoluteRetargeterCfg,
    }
    target_cls = mode_to_target[mode]

    devices = getattr(devices_cfg, "devices", None) or {}
    for device_cfg in devices.values():
        retargeters = getattr(device_cfg, "retargeters", None)
        if not retargeters:
            continue
        swapped = []
        for rtg in retargeters:
            # Only convert the *relative* cfg family; leave unrelated retargeters
            # (user-provided or arm-specific) untouched so this helper is safe
            # to run on arbitrary env cfgs.
            if isinstance(rtg, SimpleAbsoluteRetargeterCfg):
                swapped.append(rtg)
            elif isinstance(rtg, SimpleRelativeRetargeterCfg):
                swapped.append(_clone_retargeter_cfg(rtg, target_cls))
            else:
                swapped.append(rtg)
        device_cfg.retargeters = swapped


def apply_teleop_retargeting_scheme(devices_cfg, scheme: str) -> None:
    """Set the dex-retargeting ``scheme`` on every Simple retargeter cfg in place.

    Companion to :func:`apply_teleop_retargeter_mode`. Where that helper swaps
    the relative/absolute wrist behavior, this one selects which finger
    retargeting YAML each retargeter loads (``"dexpilot"`` vs ``"vector"``).
    Both :class:`SimpleRelativeRetargeterCfg` and
    :class:`SimpleAbsoluteRetargeterCfg` carry the ``retargeting_scheme`` field,
    so this is safe to call before or after the relative/absolute swap.

    Args:
        devices_cfg: ``DevicesCfg`` (typically ``env_cfg.teleop_devices``).
        scheme: One of :data:`TELEOP_RETARGETING_SCHEMES`.
    """
    if scheme not in TELEOP_RETARGETING_SCHEMES:
        raise ValueError(f"Unknown retargeting scheme '{scheme}'. Supported: {TELEOP_RETARGETING_SCHEMES}.")
    if devices_cfg is None:
        return

    devices = getattr(devices_cfg, "devices", None) or {}
    for device_cfg in devices.values():
        retargeters = getattr(device_cfg, "retargeters", None)
        if not retargeters:
            continue
        for rtg in retargeters:
            # Only the Simple retargeter family understands schemes; leave any
            # unrelated (user-provided / arm-specific) retargeters untouched.
            if isinstance(rtg, SimpleRelativeRetargeterCfg):
                rtg.retargeting_scheme = scheme


def setup_floating_teleop(env_cfg) -> None:
    """Wire up floating-hand teleoperation devices on ``env_cfg``.

    Encapsulates the boilerplate shared by every floating-hand robot variant
    (single-hand right/left and bimanual; the device layout is selected from
    ``env_cfg.robot_type``): apply the optional Leap XR override, then build the
    teleop devices from the env's ``_teleop_config`` (if present). Call from
    ``__post_init__`` *after* ``super().__post_init__()`` with ``robot_type`` and
    ``xr`` already set.
    """
    maybe_set_leap_xr_cfg(env_cfg)
    if hasattr(env_cfg, "_teleop_config"):
        teleop_config = env_cfg._teleop_config
        env_cfg.teleop_devices = build_floating_teleop_devices(
            sim_device=env_cfg.sim.device,
            xr_cfg=env_cfg.xr,
            hand_joint_names=teleop_config["hand_joint_names"],
            wrist_position_offset=teleop_config["wrist_position_offset"],
            retargeter_config_filename=teleop_config["retargeter_config_filename"],
            retargeter_urdf_path=teleop_config["retargeter_urdf_path"],
            apply_shadow_specific_postprocess=teleop_config["apply_shadow_specific_postprocess"],
            env_robot_type=env_cfg.robot_type,
        )
