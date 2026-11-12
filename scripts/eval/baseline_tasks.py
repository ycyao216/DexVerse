# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline evaluation suites: env id + natural-language prompt per task.

The prompts are **not** cosmetic. They are copied verbatim from the LeRobot
training set's ``meta/tasks.jsonl`` — the same strings stored in the RLDS
``language_instruction`` field that the OpenVLA-OFT and pi0/pi0.5 checkpoints
were finetuned on. Changing a prompt here silently moves the policy off its
training distribution, so treat this table as part of the benchmark definition
rather than as a label for humans.

``task_index`` is retained for logging and for cross-referencing the training
set only. Both policy servers read the prompt string directly; neither does a
``task_index`` -> prompt lookup.
"""

from __future__ import annotations

from typing import NamedTuple


class BaselineTask(NamedTuple):
    """One row of a baseline suite."""

    task_index: int
    env_id: str
    prompt: str


# 12 single-hand tasks (28-DoF floating Shadow right hand: third_person + wrist cameras).
SINGLE_HAND_TASKS: tuple[BaselineTask, ...] = (
    BaselineTask(0, "Dexverse-FunctionalHammerStrike-v0", "strike the nail with the hammer"),
    BaselineTask(1, "Dexverse-FunctionalPourCan-v0", "pour from the can"),
    BaselineTask(2, "Dexverse-FunctionalPourMug-v0", "pour from the mug"),
    BaselineTask(3, "Dexverse-GraspBleach-v0", "grasp the bleach bottle"),
    BaselineTask(4, "Dexverse-GraspCup-v0", "grasp the cup"),
    BaselineTask(5, "Dexverse-GraspKettle-v0", "grasp the kettle"),
    BaselineTask(6, "Dexverse-GraspPan-v0", "grasp the pan"),
    BaselineTask(7, "Dexverse-OpenFaucet-v0", "open the faucet"),
    BaselineTask(8, "Dexverse-RemoveCupFromRack-v0", "remove the cup from the rack"),
    BaselineTask(9, "Dexverse-InsertPen-v0", "insert the pen into the holder"),
    BaselineTask(
        10,
        "Dexverse-PushSmallSphereObstacleSlope-v0",
        "push the small sphere up the slope around the obstacle",
    ),
    BaselineTask(11, "Dexverse-PushT-v0", "push the T-shaped block to the goal"),
)

# 7 bimanual tasks (56-DoF floating Shadow pair: third_person + left/right wrist cameras).
#
# Only two of these live under ``tasks/config/bimanual/``; the other five are
# registered from ``tasks/config/articulation/`` and their config classes are
# still named ``...EnvFloatingDexHandRightCfg`` for historical reasons even
# though they set ``robot_type = "floating_shadow_bimanual"``. Never infer the
# embodiment from a class name or a package path — read ``robot_type`` off the
# parsed env cfg, as ``eval_policy.py`` does (it cross-checks the result against
# :func:`suite_of` and aborts on a mismatch).
BIMANUAL_TASKS: tuple[BaselineTask, ...] = (
    BaselineTask(0, "Dexverse-SlideUtilityKnife-v0", "slide out the utility knife blade"),
    BaselineTask(1, "Dexverse-SqueezeScissors-v0", "squeeze the scissors"),
    BaselineTask(2, "Dexverse-BimanualLiftCarton-v0", "lift the carton with both hands"),
    BaselineTask(3, "Dexverse-BimanualLiftTray-v0", "lift the tray with both hands"),
    BaselineTask(4, "Dexverse-OpenFlatFolder-v0", "open the flat folder"),
    BaselineTask(5, "Dexverse-OpenLaptop-v0", "open the laptop"),
    BaselineTask(6, "Dexverse-OpenStapler-v0", "open the stapler"),
)

SUITES: dict[str, tuple[BaselineTask, ...]] = {
    "single": SINGLE_HAND_TASKS,
    "bimanual": BIMANUAL_TASKS,
}

SUITE_NAMES: tuple[str, ...] = tuple(SUITES)


def get_suite(name: str) -> tuple[BaselineTask, ...]:
    """Return the baseline suite registered under ``name``."""
    try:
        return SUITES[name]
    except KeyError:
        raise ValueError(f"Unknown suite {name!r}. Available: {sorted(SUITES)}") from None


def find_task(env_id: str) -> BaselineTask:
    """Look up a single task by env id across every suite.

    Raises ``ValueError`` with the full candidate list when ``env_id`` is not a
    baseline task — a task can be registered in DexVerse and still have no
    baseline prompt, in which case there is nothing to send the policy server.
    """
    for suite in SUITES.values():
        for task in suite:
            if task.env_id == env_id:
                return task
    known = [t.env_id for suite in SUITES.values() for t in suite]
    raise ValueError(f"{env_id!r} is not a baseline task. Choose one of: {known}")


def suite_of(env_id: str) -> str:
    """Return the suite name (``single`` / ``bimanual``) that declares ``env_id``."""
    for suite_name, suite in SUITES.items():
        if any(task.env_id == env_id for task in suite):
            return suite_name
    raise ValueError(f"{env_id!r} is not a baseline task.")
