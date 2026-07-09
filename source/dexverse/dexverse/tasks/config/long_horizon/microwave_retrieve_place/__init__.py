# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-horizon microwave-retrieve-place task configurations."""

from ....utils.registration import register_env

register_env(
    __name__,
    "Dexverse-LongHorizon-MicrowaveRetrievePlace-v0",
    "long_horizon_microwave_retrieve_place_cfg",
    "LongHorizonMicrowaveRetrievePlaceEnvFloatingDexHandRightCfg",
)
