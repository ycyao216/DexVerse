# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve bimanual handover contact links from the active robot type.

The handover task needs contact sensors split by hand. Keep the task surface
small by maintaining one explicit ``robot_type`` table here, then validate the
selected names against the active robot config / USD when that information is
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BimanualContactLinks:
    """Contact body names split by side."""

    all: tuple[str, ...]
    left: tuple[str, ...]
    right: tuple[str, ...]


def _make_links(
    *,
    right_palm: str,
    left_palm: str,
    right_fingertips: tuple[str, ...],
    left_fingertips: tuple[str, ...],
) -> BimanualContactLinks:
    right = (*right_fingertips, right_palm)
    left = (*left_fingertips, left_palm)
    return BimanualContactLinks(all=(*right, *left), left=left, right=right)


_SHADOW_LINKS = _make_links(
    right_palm="rh_palm",
    left_palm="lh_palm",
    right_fingertips=("rh_thtip", "rh_fftip", "rh_mftip", "rh_rftip", "rh_lftip"),
    left_fingertips=("lh_thtip", "lh_fftip", "lh_mftip", "lh_rftip", "lh_lftip"),
)

_LEAP_LINKS = _make_links(
    right_palm="right_palm_lower",
    left_palm="left_palm_lower_left",
    right_fingertips=("right_thumb_fingertip", "right_fingertip", "right_fingertip_2", "right_fingertip_3"),
    left_fingertips=("left_thumb_fingertip", "left_fingertip", "left_fingertip_2", "left_fingertip_3"),
)

_FLOATING_SHARPA_LINKS = _make_links(
    right_palm="right_hand_C_MC",
    left_palm="left_hand_C_MC",
    right_fingertips=(
        "right_thumb_fingertip",
        "right_index_fingertip",
        "right_middle_fingertip",
        "right_ring_fingertip",
        "right_pinky_fingertip",
    ),
    left_fingertips=(
        "left_thumb_fingertip",
        "left_index_fingertip",
        "left_middle_fingertip",
        "left_ring_fingertip",
        "left_pinky_fingertip",
    ),
)

_FLOATING_WUJI_LINKS = _make_links(
    right_palm="right_palm_link",
    left_palm="left_palm_link",
    right_fingertips=tuple(f"right_finger{i}_tip_link" for i in range(1, 6)),
    left_fingertips=tuple(f"left_finger{i}_tip_link" for i in range(1, 6)),
)


BIMANUAL_CONTACT_LINKS_BY_ROBOT_TYPE: dict[str, BimanualContactLinks] = {
    "floating_shadow_bimanual": _SHADOW_LINKS,
    "floating_leap_bimanual": _LEAP_LINKS,
    "bimanual_leap": _LEAP_LINKS,
    "floating_sharpa_bimanual": _FLOATING_SHARPA_LINKS,
    "floating_wuji_bimanual": _FLOATING_WUJI_LINKS,
}


def get_bimanual_contact_links_for_robot_type(robot_type: str) -> BimanualContactLinks:
    """Return the dedicated contact-link tuple for a supported bimanual robot."""

    try:
        return BIMANUAL_CONTACT_LINKS_BY_ROBOT_TYPE[robot_type]
    except KeyError as exc:
        supported = ", ".join(sorted(BIMANUAL_CONTACT_LINKS_BY_ROBOT_TYPE))
        raise ValueError(
            f"No bimanual handover contact links registered for robot_type={robot_type!r}. Supported: {supported}"
        ) from exc


def resolve_bimanual_contact_links(
    *,
    robot_type: str,
    robot_config: Any,
    robot_cfg: Any | None = None,
    include_palms: bool = True,
) -> BimanualContactLinks:
    """Return validated contact body names for the active bimanual robot.

    Args:
        robot_type: The env's selected robot type.
        robot_config: ``DexVerseBaseEnvCfg.robot_config``.
        robot_cfg: Optional ``ArticulationCfg`` used to verify that every name
            resolves to an authored rigid body in the robot USD when USD Python
            bindings are available.
        include_palms: Include left/right palm bodies along with fingertips.

    Raises:
        ValueError: If no contact links are configured, a link cannot be split
            into left/right, or a configured link is not an authored rigid body
            in the robot USD.
    """

    contact_links = get_bimanual_contact_links_for_robot_type(robot_type)
    if not include_palms:
        contact_links = _without_configured_palms(contact_links, robot_config)

    authored_body_names = _authored_rigid_body_names(robot_cfg)
    if authored_body_names is not None:
        missing = [name for name in contact_links.all if name not in authored_body_names]
        if missing:
            raise ValueError(
                f"robot_type={robot_type!r} contact links are not authored rigid bodies in the robot USD: "
                f"{missing}. Available examples: {sorted(authored_body_names)[:20]}"
            )

    _validate_against_robot_config(robot_type=robot_type, contact_links=contact_links, robot_config=robot_config)
    return contact_links


def contact_sensor_names(body_names: tuple[str, ...] | list[str], suffix: str = "_object_s") -> list[str]:
    """Map body names to scene contact-sensor attribute names."""

    return [f"{name}{suffix}" for name in body_names]


def _without_configured_palms(contact_links: BimanualContactLinks, robot_config: Any) -> BimanualContactLinks:
    palm_names = {
        getattr(robot_config, "right_palm_body_name", None),
        getattr(robot_config, "left_palm_body_name", None),
    }
    palm_names.discard(None)
    left = tuple(name for name in contact_links.left if name not in palm_names)
    right = tuple(name for name in contact_links.right if name not in palm_names)
    return BimanualContactLinks(all=(*right, *left), left=left, right=right)


def _validate_against_robot_config(
    *,
    robot_type: str,
    contact_links: BimanualContactLinks,
    robot_config: Any,
) -> None:
    configured = set(getattr(robot_config, "hand_tips_body_names", ()) or ())
    if not configured:
        configured.update(getattr(robot_config, "fingertip_body_names", ()) or ())
        for palm_name in (
            getattr(robot_config, "right_palm_body_name", None),
            getattr(robot_config, "left_palm_body_name", None),
            getattr(robot_config, "palm_body_name", None),
        ):
            if palm_name:
                configured.add(palm_name)
    if not configured:
        return

    missing = [name for name in contact_links.all if name not in configured]
    if missing:
        raise ValueError(
            f"robot_type={robot_type!r} bimanual contact table is inconsistent with robot_config. "
            f"Missing from robot_config fingertip/palm names: {missing}"
        )


def _authored_rigid_body_names(robot_cfg: Any | None) -> set[str] | None:
    """Best-effort USD rigid-body name extraction.

    Returns ``None`` when the USD cannot be inspected in the current Python
    environment. IsaacLab will still validate the body names at scene creation;
    this function just catches mistakes earlier when possible.
    """

    usd_path = _robot_usd_path(robot_cfg)
    if usd_path is None or not usd_path.is_file():
        return None

    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except Exception:
        return None

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return None

    body_names: set[str] = set()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_names.add(prim.GetName())
    return body_names


def _robot_usd_path(robot_cfg: Any | None) -> Path | None:
    if robot_cfg is None:
        return None
    spawn = getattr(robot_cfg, "spawn", None)
    usd_path = getattr(spawn, "usd_path", None)
    if not usd_path:
        return None
    return Path(str(usd_path))
