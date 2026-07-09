# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for parsing and loading task configurations."""

from __future__ import annotations

import collections
import importlib
import os

import gymnasium as gym
import yaml
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict | object:
    """Load default configuration given its entry point from the gym registry."""
    cfg_entry_point = gym.spec(task_name).kwargs.get(entry_point_key)
    if cfg_entry_point is None:
        agents = collections.defaultdict(list)
        for k in gym.spec(task_name.split(":")[-1]).kwargs:
            if k.endswith("_cfg_entry_point") and k != "env_cfg_entry_point":
                spec = (
                    k.replace("_cfg_entry_point", "")
                    .replace("rl_games", "rl-games")
                    .replace("rsl_rl", "rsl-rl")
                    .split("_")
                )
                agent = spec[0].replace("-", "_")
                algorithms = [item.upper() for item in (spec[1:] if len(spec) > 1 else ["PPO"])]
                agents[agent].extend(algorithms)
        msg = "\nExisting RL library (and algorithms) config entry points: "
        for agent, algorithms in agents.items():
            msg += f"\n  |-- {agent}: {', '.join(algorithms)}"
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'."
            f"\nPlease check that the gym registry has the entry point: '{entry_point_key}'."
            f"{msg if agents else ''}"
        )
    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            config_file = cfg_entry_point
        else:
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            config_file = os.path.join(mod_path, file_name)
        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.full_load(f)
    else:
        if callable(cfg_entry_point):
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)
        else:
            cfg_cls = cfg_entry_point
        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
        cfg = cfg_cls() if callable(cfg_cls) else cfg_cls
    return cfg


def parse_env_cfg(
    task_name: str,
    device: str = "cuda:0",
    num_envs: int | None = None,
    use_fabric: bool | None = None,
    json_path: str | None = None,
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    """Parse configuration for an environment and override based on inputs."""
    if json_path is not None and "Template" in task_name:
        raise ValueError(
            "This checkout accepts `json_path` for CLI compatibility, but JSON-template task support is "
            f"not available for task '{task_name}' because the template config modules are missing."
        )
    cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for the task: '{task_name}' is not a class. Please provide a class.")
    cfg.sim.device = device
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    if getattr(cfg, "requires_gpu_sim", False) and not str(cfg.sim.device).startswith("cuda"):
        raise ValueError(
            f"Task '{task_name}' requires GPU simulation for deformable assets; got sim.device='{cfg.sim.device}'."
        )
    return cfg
