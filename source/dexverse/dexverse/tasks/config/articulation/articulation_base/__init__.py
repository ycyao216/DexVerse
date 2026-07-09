# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for tabletop articulated-object manipulation tasks.

Provides reusable config classes for tasks where an articulated object sits on
the tabletop and the success condition is that one of its joints reaches a
target value (angle for revolute, distance for prismatic).

Child task modules subclass :class:`ArticulationBaseEnvFloatingDexHandRightCfg`,
override the class attributes (USD path, scale, init pose, joint name, success
threshold), and register the resulting class as a gym environment.

This module does not register any gym environments itself.
"""

from .articulation_base_cfg import (  # noqa: F401
    ArticulationBaseEnvCfg,
    ArticulationBaseEnvFloatingDexHandRightCfg,
    ArticulationBaseEventCfg,
    ArticulationBaseObservationsCfg,
    ArticulationBaseRewardsCfg,
    ArticulationBaseSceneCfg,
    ArticulationBaseTerminationsCfg,
    make_articulation_cfg,
    make_multi_articulation_cfg,
)
from .usd_helpers import (  # noqa: F401
    collect_articulation_usds_from_dir,
    ensure_single_articulation_root,
    is_usd_file_path,
)
