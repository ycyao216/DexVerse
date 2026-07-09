# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-horizon cook-foods task configurations."""

from ....utils.registration import register_env

register_env(
    __name__, "Dexverse-LongHorizon-CookFoods-v0", "long_horizon_cook_foods_cfg", "CookFoodsEnvFloatingDexHandRightCfg"
)
