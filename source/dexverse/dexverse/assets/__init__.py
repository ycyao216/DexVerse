# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset path constants, organized by upstream source.

The released assets have appeared in two layouts:

    assets/core_assets/
        partnet_mobility/{bucket,door,fridge,kettle,kitchenpot,knob,
                          microwave,storage_unit,switch}/
        partnet_mobility/storage_furniture_{prismatic,revolute}/<id>/
        synthesis/{shopping basket002,faucet001,Bleach_1,FryingPan,...}/
        dexverse_authored/{insert_peg,plug_charger,push_t,Stick,mug,
                           primitive,prim_rigid,button,nail_board}/
        ycb/{005_tomato_soup_can,035_power_drill_usd,048_hammer_usd}/
        autobio/centrifuge_15ml_screw/
    assets/
        long_horizon_extra/{cooking,table_cleaning}/   (own bundle)
        mani_twin_selected/<category>/<instance>/      (own bundle)
        polyhaven_hdris/polyhaven_hdris/*.hdr
        polyhaven_hdris/debug/monochrome_studio_02_4k.hdr

External datasets (populated at install time by ``scripts/`` helpers,
not redistributed in the repo): ``dexgarmentlab/``,
``sapien_partnet_mobility/``, ``sapien_assets/``.
"""

from __future__ import annotations

from pathlib import Path

_ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"


def _existing_path(*candidates: Path) -> Path:
    """Return the first existing path, falling back to the first candidate."""
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _source_dir(name: str) -> Path:
    return _existing_path(_ASSETS_ROOT / "core_assets" / name, _ASSETS_ROOT / name)


# === In-repo source categories ===
CORE_ASSETS_DIR = _existing_path(_ASSETS_ROOT / "core_assets", _ASSETS_ROOT)

PARTNET_MOBILITY_DIR = _source_dir("partnet_mobility")
PARTNET_MOBILITY_ARTICULATIONS_DIR = _existing_path(
    PARTNET_MOBILITY_DIR / "articulations",
    PARTNET_MOBILITY_DIR,
)
# Multi-asset StorageFurniture pools (output of partnet_mobility_to_usd.py).
STORAGE_FURNITURE_PRISMATIC_DIR = _ASSETS_ROOT / "core_assets" / "partnet_mobility" / "storage_furniture_prismatic"
STORAGE_FURNITURE_REVOLUTE_DIR = _ASSETS_ROOT / "core_assets" / "partnet_mobility" / "storage_furniture_revolute"
# dexverse_authored holds both rigid objects and authored articulations flat;
# the two constants are kept for call-site clarity but resolve to the same dir.
DEXVERSE_AUTHORED_ASSETS_DIR = _ASSETS_ROOT / "core_assets" / "dexverse_authored"
DEXVERSE_AUTHORED_ARTICULATIONS_DIR = DEXVERSE_AUTHORED_ASSETS_DIR
SYNTHESIS_DIR = _ASSETS_ROOT / "core_assets" / "synthesis"
FUNCTIONAL_OBJECTS_DIR = _existing_path(
    _ASSETS_ROOT / "core_assets" / "functional_objects" / "functional",
    _ASSETS_ROOT / "functional_objects" / "functional",
    _ASSETS_ROOT / "core_assets" / "functional_objects",
    _ASSETS_ROOT / "functional_objects",
)
YCB_DIR = _ASSETS_ROOT / "core_assets" / "ycb"
# Root for miscellaneous "shared" asset categories that live directly under the
# assets root (e.g. ``autobio/`` lab-equipment USDs used by unscrew_cap). Kept as
# the assets root itself so ``SHARED_ASSET_DIR / "<category>" / "<asset>"``
# resolves like the other top-level categories.
SHARED_ASSET_DIR = _ASSETS_ROOT

# === long_horizon_extra: large high-quality meshes used only by long-horizon
# tasks (cook_foods, trash_drawer_sort_simple). Shipped as its own opt-in HF
# bundle so the default core_assets.zip stays small.
LONG_HORIZON_EXTRA_DIR = _ASSETS_ROOT / "long_horizon_extra"
LONG_HORIZON_EXTRA_COOKING_DIR = LONG_HORIZON_EXTRA_DIR / "cooking"
LONG_HORIZON_EXTRA_TABLE_CLEANING_DIR = LONG_HORIZON_EXTRA_DIR / "table_cleaning"
POLYHAVEN_HDRI_DIR = _ASSETS_ROOT / "polyhaven_hdris" / "polyhaven_hdris"
DEBUG_HDR_PATH = _ASSETS_ROOT / "polyhaven_hdris" / "debug" / "monochrome_studio_02_4k.hdr"

# === mani_twin_selected: processed ManiTwin-100K object pool used by the
# default grasping / object-pool tasks. Shipped as its own HF bundle
# (mani_twin_selected.zip, fetched by download_assets.py); can also be rebuilt
# from upstream with scripts/asset_tools/download_selected_objects.py.
MANI_TWIN_SELECTED_DIR = _ASSETS_ROOT / "mani_twin_selected"

# === External-dataset roots (downloaded by scripts/* helpers) ===
DEXGARMENTLAB_DIR = _ASSETS_ROOT / "dexgarmentlab"
SAPIEN_PARTNET_MOBILITY_DIR = _ASSETS_ROOT / "sapien_partnet_mobility"
SAPIEN_ASSETS_DIR = _ASSETS_ROOT / "sapien_assets"

# YCB-style debug objects (013_apple etc.) live inside the ManipulationTwin
# pool. Kept as a string for str-formatting call sites.
DEBUG_YCB_OBJ_DIR = str(MANI_TWIN_SELECTED_DIR)
