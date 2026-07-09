# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplified long-horizon trash-can / drawer scene."""

from ....utils.registration import register_env

register_env(
    __name__,
    "Dexverse-LongHorizon-TrashDrawerSortSimple-v0",
    "trash_drawer_sort_simple_cfg",
    "TrashDrawerSortSimpleEnvCfg",
)
