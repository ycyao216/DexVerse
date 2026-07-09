# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow hand robot package (floating-only release)."""

from .floating import *

SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES: dict[str, tuple[str, str]] = {
    "floating_shadow_right": (
        "dexverse.robot_agents.shadow.floating",
        "FLOATING_SHADOW_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    ),
    "floating_shadow_left": (
        "dexverse.robot_agents.shadow.floating",
        "FLOATING_SHADOW_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    ),
    "floating_shadow_bimanual": (
        "dexverse.robot_agents.shadow.floating",
        "FLOATING_SHADOW_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT",
    ),
}


SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES: dict[str, str] = {
    "floating_shadow_right": "FLOATING_SHADOW_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_shadow_left": "FLOATING_SHADOW_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_shadow_bimanual": "FLOATING_SHADOW_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
}
