# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-rich task configurations (precision insertion / assembly and constrained-space extraction)."""

from ...utils.registration import register_env

register_env(__name__, "Dexverse-NutThread-v0", "nutthread_cfg", "NutThreadEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-InsertPipette-v0", "insertpipette_cfg", "InsertPipetteEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-PlugCharger-v0", "plugcharger_cfg", "PlugChargerEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-InsertPen-v0", "insertpen_cfg", "InsertPenEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-GearMesh-v0", "gearmesh_cfg", "GearMeshEnvFloatingDexHandRightCfg")
register_env(__name__, "Dexverse-InsertPeg-v0", "insertpeg_cfg", "InsertPegEnvFloatingDexHandRightCfg")
register_env(
    __name__, "Dexverse-PickFromClutter-v0", "pick_from_clutter_cfg", "PickFromClutterEnvFloatingDexHandRightCfg"
)
register_env(
    __name__,
    "Dexverse-PickThinObjectFromContainer-v0",
    "pick_thin_object_from_container_cfg",
    "PickThinObjectFromContainerEnvFloatingDexHandRightCfg",
)
