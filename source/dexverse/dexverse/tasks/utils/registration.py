# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared gym-registration helper for DexVerse task packages.

Every task package registers its environments by calling :func:`register_env`
with its own ``__name__`` so the env-config entry point resolves to
``<package>.<module>:<class>``. This keeps the registration boilerplate (RL
agent entry points, env checker flag, idempotency guard) in one place.
"""

import gymnasium as gym

from ..config import agents


def register_env(pkg: str, env_id: str, module: str, cls: str) -> None:
    """Register a DexVerse env config under ``env_id`` (idempotent).

    ``pkg`` is the caller's ``__name__``; the env-config class ``cls`` must live
    in the ``<pkg>.<module>`` submodule.
    """
    if env_id in gym.registry:
        return
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{pkg}.{module}:{cls}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        },
    )
