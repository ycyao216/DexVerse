# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base configuration for contact-rich insertion / assembly tasks.

All insertion tasks (nut-thread, peg/pen/pipette insert, gear-mesh,
plug-charger) share the same teleop-first wiring in ``__post_init__``:

* fingertip → object **contact sensors** plus a ``contact`` observation term
  (via :func:`setup_fingertip_contact_observation`), and
* **disabling the default free-object machinery** (the ``object_pose`` command,
  the object reset / scale / physics-material events, and the
  ``object_out_of_bound`` termination), since each task fixes its own assets and
  defines its own success logic.

Subclasses set :attr:`contact_object_prim` (the prim under ``{ENV_REGEX_NS}``
the fingertip contact sensors filter against) and override scene / observations
/ rewards / terminations / events as needed. Task-specific asset placement and
``episode_length_s`` stay in each task's ``__post_init__`` after ``super()``.
"""

from isaaclab.utils import configclass

from ... import dexverse_base_env_cfg as dexverse_base_env
from ... import mdp


@configclass
class ContactRichEnvCfg(dexverse_base_env.DexVerseBaseEnvCfg):
    """Base config for contact-rich insertion / assembly tasks (robot-agnostic)."""

    # Prim under ``{ENV_REGEX_NS}`` that the fingertip contact sensors filter against.
    contact_object_prim: str = "Object"

    def __post_init__(self):
        super().__post_init__()

        self.commands.object_pose = None
        mdp.setup_fingertip_contact_observation(self, target_prim=self.contact_object_prim)
        self.events.object_physics_material = None
        self.events.object_scale_mass = None
        self.events.reset_object = None
        self.terminations.object_out_of_bound = None
