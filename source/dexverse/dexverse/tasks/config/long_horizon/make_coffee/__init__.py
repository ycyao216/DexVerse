# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-horizon make-coffee task configuration."""

from ....utils.registration import register_env

register_env(
    __name__,
    "Dexverse-LongHorizon-MakeCoffee-v0",
    "long_horizon_make_coffee_cfg",
    "LongHorizonMakeCoffeeEnvFloatingDexHandRightCfg",
)
