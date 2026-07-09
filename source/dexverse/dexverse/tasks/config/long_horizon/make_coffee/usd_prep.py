# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-time USD preparation for the make-coffee long-horizon task.

The coffee machine (``synthesis/coffee_machine001/model_coffee_machine_6.usd``)
is a clean single-articulation-root asset (root body ``E_body_3``) with
all-``convexDecomposition`` colliders, so it spawns directly. BUT, like the
PartNet microwave dials, it authors **no mass** on any link
(``physics:mass`` = None everywhere). PhysX then auto-computes a near-zero
rotational inertia for the small control links, and the switch/knob joint will
not turn under fingertip contact (it reads as "not rotatable" no matter how low
the joint friction), and a friction-detent on it is ill-conditioned. See the
``articulation-hinge-armature-limits`` note: the fix is a small real link mass
on the manipulated link *plus* an actuator ``armature`` (set in the task cfg).

:func:`prepare_coffee_machine_usd` caches a copy that gives the listed control
links a small real mass. The result is cached next to the source as
``<stem>__massprep.usd`` and rebuilt only when the source is newer (bump the
suffix when this logic changes so stale caches are not reused). Returns the
source path unchanged on any error / missing ``pxr`` / nothing to do, so it
never blocks task load.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_coffee_machine_usd(
    usd_path: str | Path,
    *,
    # The switch lever (``E_switch_5``, the rotated "knob") is the one we must
    # turn, so it needs a usable inertia. Give the steam wand / portafilter a
    # mass too so their joints hold under the friction detent rather than
    # flopping from a near-zero inertia.
    link_masses: dict[str, float] | None = None,
) -> str:
    """Return a path to a copy of the coffee machine with small link masses set."""
    if link_masses is None:
        link_masses = {
            "E_switch_5": 0.08,  # the knob/lever the task rotates
            "E_steam_wand_2": 0.08,
            "E_portafilter_4": 0.12,
            "E_button_1_6": 0.02,
            "E_button_2_7": 0.02,
            "E_button_3_8": 0.02,
        }

    src = Path(usd_path)
    if not src.is_file():
        return str(src)

    try:
        from pxr import Usd, UsdPhysics  # type: ignore
    except ImportError:
        return str(src)

    cleaned = src.with_name(f"{src.stem}__massprep.usd")
    if cleaned.exists() and cleaned.stat().st_mtime >= src.stat().st_mtime:
        return str(cleaned)

    try:
        stage = Usd.Stage.Open(str(src))
        if stage is None:
            return str(src)
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            return str(src)
        root = default_prim.GetPath().pathString

        applied = 0
        for link_name, mass in link_masses.items():
            prim = stage.GetPrimAtPath(f"{root}/{link_name}")
            if not prim.IsValid():
                continue
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_attr = mass_api.GetMassAttr() or mass_api.CreateMassAttr()
            mass_attr.Set(float(mass))
            applied += 1

        if applied == 0:
            return str(src)

        stage.GetRootLayer().Export(str(cleaned))
        logger.info(
            "Prepared coffee machine %s -> %s (set mass on %d link(s)).",
            src.name,
            cleaned.name,
            applied,
        )
        return str(cleaned)
    except Exception as exc:  # noqa: BLE001 - never break asset loading over this
        logger.warning("Coffee-machine mass preparation failed for %s: %s", src, exc)
        return str(src)
