# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Leap hand robot package (floating-only release)."""

from .floating import *


SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES: dict[str, tuple[str, str]] = {
    "floating_leap_right": ("dexverse.robot_agents.leap.floating", "FLOATING_LEAP_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "floating_leap_bimanual": ("dexverse.robot_agents.leap.floating", "FLOATING_LEAP_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "bimanual_leap": ("dexverse.robot_agents.leap.floating", "BIMANUAL_LEAP_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
}


SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES: dict[str, str] = {
    "floating_leap_right": "FLOATING_LEAP_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_leap_bimanual": "FLOATING_LEAP_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
    "bimanual_leap": "BIMANUAL_LEAP_SIMPLE_RELATIVE_DEX_RETARGETING",
}
