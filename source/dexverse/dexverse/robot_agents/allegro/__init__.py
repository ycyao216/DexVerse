# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Allegro hand robot package (floating-only release)."""

from .floating import *


SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES: dict[str, tuple[str, str]] = {
    "floating_allegro_right": ("dexverse.robot_agents.allegro.floating", "FLOATING_ALLEGRO_RIGHT_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "floating_allegro_left": ("dexverse.robot_agents.allegro.floating", "FLOATING_ALLEGRO_LEFT_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
    "floating_allegro_bimanual": ("dexverse.robot_agents.allegro.floating", "FLOATING_ALLEGRO_BIMANUAL_SIMPLE_RELATIVE_RETARGETER_LAYOUT"),
}


SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES: dict[str, str] = {
    "floating_allegro_right": "FLOATING_ALLEGRO_RIGHT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_allegro_left": "FLOATING_ALLEGRO_LEFT_SIMPLE_RELATIVE_DEX_RETARGETING",
    "floating_allegro_bimanual": "FLOATING_ALLEGRO_BIMANUAL_SIMPLE_RELATIVE_DEX_RETARGETING",
}
