# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual-coordination task configurations (two-hand lifts, handover)."""

from ...utils.registration import register_env

register_env(__name__, "Dexverse-BimanualLiftTray-v0", "lift_tray_cfg", "LiftTrayEnvFloatingShadowBimanualCfg")
register_env(__name__, "Dexverse-BimanualLiftBasket-v0", "lift_basket_cfg", "LiftBasketEnvFloatingShadowBimanualCfg")
register_env(__name__, "Dexverse-BimanualLiftCarton-v0", "lift_carton_cfg", "LiftCartonEnvFloatingShadowBimanualCfg")
register_env(
    __name__, "Dexverse-BimanualLiftDutchOven-v0", "lift_dutch_oven_cfg", "LiftDutchOvenEnvFloatingShadowBimanualCfg"
)
register_env(
    __name__, "Dexverse-BimanualHandover-v0", "bimanual_handover_cfg", "BimanualHandoverEnvFloatingShadowBimanualCfg"
)
