# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspire hand robot package (floating-only release)."""

from .floating import *


SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES: dict[str, tuple[str, str]] = {
    "floating_inspire_right": ("dexverse.robot_agents.inspire.floating", "FLOATING_INSPIRE_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "floating_inspire_left": ("dexverse.robot_agents.inspire.floating", "FLOATING_INSPIRE_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "floating_inspire_bimanual": ("dexverse.robot_agents.inspire.floating", "FLOATING_INSPIRE_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
}


SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES: dict[str, str] = {
    "floating_inspire_right": "FLOATING_INSPIRE_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_inspire_left": "FLOATING_INSPIRE_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_inspire_bimanual": "FLOATING_INSPIRE_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
}
