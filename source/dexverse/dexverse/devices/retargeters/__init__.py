# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retargeters for mapping input device data to robot commands."""

from ..wrist_origin import compute_wrist_joint_origin
from .simple_absolute_retargeting import (
    SimpleAbsoluteRetargeter,
    SimpleAbsoluteRetargeterCfg,
)
from .simple_relative_retargeting import (
    SimpleRelativeRetargeter,
    SimpleRelativeRetargeterCfg,
)

__all__ = [
    # Unified retargeters (recommended)
    # Hand-specific retargeters (backward compatibility)
    "SimpleRelativeRetargeter",
    "SimpleRelativeRetargeterCfg",
    "SimpleAbsoluteRetargeter",
    "SimpleAbsoluteRetargeterCfg",
    "compute_wrist_joint_origin",
]
