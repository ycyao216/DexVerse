# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-agent packages (floating-hand release).

The shared :class:`TabletopRobotSetup` bundle is defined here (rather than in a
standalone module) so each hand package imports it via ``from .. import
TabletopRobotSetup``. It must be defined before the per-hand imports below,
since those packages import it during their own initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab.assets import ArticulationCfg

# Retargeting schemes auto-exposed by :func:`dex_retargeting_hand_spec`, in
# preference order. A hand offers whichever ``<side>_<scheme>.yml`` it ships.
DEX_RETARGETING_SCHEMES = ("dexpilot", "vector")


@dataclass(frozen=True)
class TabletopRobotSetup:
    """Bundle the robot-specific pieces needed by tabletop env configs."""

    robot_config_kwargs: dict[str, Any]
    scene_robot: ArticulationCfg
    actions: Any
    controller_mode: str
    teleop_config: dict[str, Any]


def dex_retargeting_hand_spec(
    retarget_dir: Path | str,
    side: str,
    urdf_path: Path | str,
) -> dict[str, Any]:
    """Build one hand's dex-retargeting spec for ``SIMPLE_RELATIVE_DEX_RETARGETING``.

    The retargeting machinery (``simple_relative_retargeting``) is embodiment
    agnostic: it selects a scheme (``dexpilot`` / ``vector`` / ...) from a
    ``config_paths`` dict keyed by scheme name. This helper builds that dict by
    auto-discovering the ``<side>_<scheme>.yml`` files present in ``retarget_dir``
    so each hand automatically offers whatever schemes it ships -- dropping a new
    ``<side>_vector.yml`` exposes it with no code change.
    """
    retarget_dir = Path(retarget_dir)
    config_paths = {
        scheme: str(retarget_dir / f"{side}_{scheme}.yml")
        for scheme in DEX_RETARGETING_SCHEMES
        if (retarget_dir / f"{side}_{scheme}.yml").is_file()
    }
    return {"config_paths": config_paths, "urdf_path": str(urdf_path)}


# Shadow is the only hand shipped in this release. The other embodiments
# (allegro, inspire, leap, sharpa, wuji) are staged under
# source_unreleased/robot_agents/ pending further testing; restoring one is a
# `git mv` back plus re-adding its import/merge lines here and its builder
# entries in tasks/dexverse_base_env_cfg.py.
from .shadow import (
    SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES as _SHADOW_SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES,
)
from .shadow import (
    SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES as _SHADOW_SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES,
)
from .shadow import *

SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES = {
    **_SHADOW_SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES,
}


SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES = {
    **_SHADOW_SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES,
}
