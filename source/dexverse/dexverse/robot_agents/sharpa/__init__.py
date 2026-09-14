# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sharpa floating-hand robot package."""

from .floating import *


SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES: dict[str, tuple[str, str]] = {
    "floating_sharpa_right": ("dexverse.robot_agents.sharpa.floating", "FLOATING_SHARPA_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "floating_sharpa_left": ("dexverse.robot_agents.sharpa.floating", "FLOATING_SHARPA_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "floating_sharpa_bimanual": ("dexverse.robot_agents.sharpa.floating", "FLOATING_SHARPA_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
}


SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES: dict[str, str] = {
    "floating_sharpa_right": "FLOATING_SHARPA_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_sharpa_left": "FLOATING_SHARPA_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_sharpa_bimanual": "FLOATING_SHARPA_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
}
