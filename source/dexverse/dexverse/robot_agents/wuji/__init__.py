# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wuji hand robot package (floating-only release)."""

from .floating import *


SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES: dict[str, tuple[str, str]] = {
    "floating_wuji_right": ("dexverse.robot_agents.wuji.floating", "FLOATING_WUJI_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "floating_wuji_left": ("dexverse.robot_agents.wuji.floating", "FLOATING_WUJI_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "floating_wuji_bimanual": ("dexverse.robot_agents.wuji.floating", "FLOATING_WUJI_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
}


SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES: dict[str, str] = {
    "floating_wuji_right": "FLOATING_WUJI_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_wuji_left": "FLOATING_WUJI_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_wuji_bimanual": "FLOATING_WUJI_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
}
