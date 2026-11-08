# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent identity of the task-upgrade development suite.

The original 100 v0 registrations remain unchanged, except for the long-horizon
tasks in ``LONG_HORIZON_SIM_FIX_REVISIONS``, whose physics, teleop and success checks
were corrected in place under a new revision. Twenty baseline families also have a
v1 implementation, selected by an explicit Gym version suffix.
"""

V0_BENCHMARK_REVISION = "official-v0-30cc673"
BENCHMARK_REVISION = "baseline-v1-2026-09-08"
TASK_SOURCE_REVISION = "4f7262f"
V0_SOURCE_REVISION = "30cc673"
LEGACY_UPGRADE_REVISION = "task-upgrades-2026-09-04"
DOOR_LATCH_REVISION = "door-cutaway-auto-latch-v1-2026-09-08"
CUP_RACK_SMALL_DESTINATION_REVISION = "cup-rack-15cm-destination-v1-2026-09-09"
DISPENSER_SMALL_CUP_REVISION = "dispenser-small-cup-v1-2026-09-09"
PUSH_T_RODS_REVISION = "push-t-goal-rods-15cm-v1-2026-09-10"
# Per task, so changing one task does not invalidate another task's recordings.
LONG_HORIZON_SIM_FIX_REVISIONS = {
    "Dexverse-LongHorizon-MakeCoffee-v0": "long-horizon-sim-fixes-v0-2026-11-07",
    "Dexverse-LongHorizon-CookFoods-v0": "long-horizon-cook-foods-fixes-v0-2026-11-07b",
}

BASELINE_TASKS = (
    "Dexverse-GraspBleach-v0",
    "Dexverse-GraspPan-v0",
    "Dexverse-GraspKettle-v0",
    "Dexverse-GraspCup-v0",
    "Dexverse-RemoveCupFromRack-v0",
    "Dexverse-FunctionalPourCan-v0",
    "Dexverse-FunctionalPourMug-v0",
    "Dexverse-FunctionalHammerStrike-v0",
    "Dexverse-OpenFaucet-v0",
    "Dexverse-OpenDoor-v0",
    "Dexverse-OpenLaptop-v0",
    "Dexverse-SqueezeScissors-v0",
    "Dexverse-SlideUtilityKnife-v0",
    "Dexverse-OpenStapler-v0",
    "Dexverse-OpenFlatFolder-v0",
    "Dexverse-BimanualLiftTray-v0",
    "Dexverse-BimanualLiftCarton-v0",
    "Dexverse-InsertPen-v0",
    "Dexverse-PushSmallSphereObstacleSlope-v0",
    "Dexverse-PushT-v0",
)

BASELINE_V0_TASKS = BASELINE_TASKS
BASELINE_V1_TASKS = tuple(task.removesuffix("-v0") + "-v1" for task in BASELINE_V0_TASKS)
UPGRADED_TASKS = BASELINE_V1_TASKS
BASELINE_PAIRS = dict(zip(BASELINE_V0_TASKS, BASELINE_V1_TASKS))

# Descriptive names used by private/pre-versioned recordings, not Gym aliases.
LEGACY_UPGRADE_TASKS = {
    "Dexverse-CloseFolderInsertShelf-v0": "Dexverse-OpenFlatFolder-v1",
    "Dexverse-CutSeamUtilityKnife-v0": "Dexverse-SlideUtilityKnife-v1",
    "Dexverse-CutStripScissors-v0": "Dexverse-SqueezeScissors-v1",
    "Dexverse-ReloadStapler-v0": "Dexverse-OpenStapler-v1",
    "Dexverse-BimanualRetrieveTrayFromRack-v0": "Dexverse-BimanualLiftTray-v1",
    "Dexverse-BimanualCartonRegrasp-v0": "Dexverse-BimanualLiftCarton-v1",
    "Dexverse-PlaceCupUnderDispenser-v0": "Dexverse-GraspCup-v1",
    "Dexverse-OpenTiltedDoor-v0": "Dexverse-OpenDoor-v1",
}

# Ordered with BASELINE_V1_TASKS; strings keep catalogue use simulator-free.
_V1_IMPLEMENTATIONS = (
    ("functional.grasp_bleach_cfg", "GraspBleachEnvFloatingDexHandRightCfg"),
    ("functional.grasp_pan_cfg", "GraspPanEnvFloatingDexHandRightCfg"),
    ("functional.grasp_kettle_cfg", "GraspKettleEnvFloatingDexHandRightCfg"),
    ("functional.place_cup_under_dispenser_cfg", "PlaceCupUnderDispenserEnvFloatingDexHandRightCfg"),
    ("functional.remove_cup_from_rack_cfg", "RemoveCupFromRackEnvFloatingDexHandRightCfg"),
    ("functional.pour_can_cfg", "PourCanEnvFloatingDexHandRightCfg"),
    ("functional.pour_mug_cfg", "PourMugEnvFloatingDexHandRightCfg"),
    ("functional.hammer_strike_cfg", "HammerStrikeEnvFloatingDexHandRightCfg"),
    ("articulation.openfaucet_cfg", "OpenFaucetEnvFloatingDexHandRightCfg"),
    ("articulation.open_cutaway_door_cfg", "OpenCutawayDoorEnvFloatingDexHandRightCfg"),
    ("articulation.open_laptop_cfg", "OpenLaptopEnvFloatingDexHandRightCfg"),
    ("articulation.cut_strip_scissors_cfg", "CutStripScissorsEnvFloatingDexHandRightCfg"),
    ("articulation.cut_seam_utility_knife_cfg", "CutSeamUtilityKnifeEnvFloatingDexHandRightCfg"),
    ("articulation.reload_stapler_cfg", "ReloadStaplerEnvFloatingDexHandRightCfg"),
    ("articulation.close_folder_insert_shelf_cfg", "CloseFolderInsertShelfEnvFloatingDexHandRightCfg"),
    ("bimanual.retrieve_tray_from_rack_cfg", "RetrieveTrayFromRackEnvFloatingShadowBimanualCfg"),
    ("bimanual.carton_regrasp_cfg", "CartonRegraspEnvFloatingShadowBimanualCfg"),
    ("contact_rich.insertpen_cfg", "InsertPenEnvFloatingDexHandRightCfg"),
    ("non_prehensile.push_small_sphere_obstacle_slope_cfg", "PushSmallSphereObstacleSlopeEnvFloatingDexHandRightCfg"),
    ("non_prehensile.pusht_cfg", "PushTEnvFloatingDexHandRightCfg"),
)
V1_CONFIGS = dict(zip(BASELINE_V1_TASKS, _V1_IMPLEMENTATIONS))


def baseline_tasks(version: str = "v0") -> tuple[str, ...]:
    if version == "v0":
        return BASELINE_V0_TASKS
    if version == "v1":
        return BASELINE_V1_TASKS
    if version == "all":
        return tuple(task for pair in BASELINE_PAIRS.items() for task in pair)
    raise ValueError(f"Unknown baseline version: {version}")


def task_identity(task: str) -> dict:
    """Metadata describes the selected environment, not the checkout alone."""
    task = task.split(":")[-1]
    if task == "Dexverse-PushT-v1":
        return {"task_version": 1, "benchmark_revision": PUSH_T_RODS_REVISION,
                "task_source_revision": "local-push-t-goal-rods-15cm-v1"}
    if task == "Dexverse-OpenDoor-v1":
        return {"task_version": 1, "benchmark_revision": DOOR_LATCH_REVISION,
                "task_source_revision": "local-cutaway-door-auto-latch-v1"}
    if task == "Dexverse-RemoveCupFromRack-v1":
        return {"task_version": 1, "benchmark_revision": CUP_RACK_SMALL_DESTINATION_REVISION,
                "task_source_revision": "local-cup-rack-15cm-destination-v1"}
    if task == "Dexverse-GraspCup-v1":
        return {"task_version": 1, "benchmark_revision": DISPENSER_SMALL_CUP_REVISION,
                "task_source_revision": "local-dispenser-small-cup-v1"}
    if task in BASELINE_V1_TASKS:
        return {
            "task_version": 1,
            "benchmark_revision": BENCHMARK_REVISION,
            "task_source_revision": TASK_SOURCE_REVISION,
        }
    if task in LONG_HORIZON_SIM_FIX_REVISIONS:
        return {"task_version": 0, "benchmark_revision": LONG_HORIZON_SIM_FIX_REVISIONS[task],
                "task_source_revision": "local-long-horizon-sim-fixes"}
    if task in LEGACY_UPGRADE_TASKS or not task.startswith("Dexverse-") or not task.endswith("-v0"):
        raise ValueError(f"Unknown canonical task identity: {task}")
    return {"task_version": 0, "benchmark_revision": V0_BENCHMARK_REVISION, "task_source_revision": V0_SOURCE_REVISION}


def validate_replay_identity(payload: dict, target_task: str, explicit_override: bool = False) -> None:
    """Never silently replay a known pre-versioned upgrade as official v0."""
    if explicit_override:
        return  # CLI override is a deliberate provenance decision, recorded separately.
    imported = payload.get("demo_import") or {}
    if imported.get("compatibility") == "needs_review":
        raise ValueError(
            f"Imported recording needs review: {imported.get('review_reasons', [])}. "
            "Resolve the incompatibility before using --task-override explicitly."
        )
    source_task = payload.get("env_name") or payload.get("task")
    revision = payload.get("benchmark_revision")
    if source_task in LEGACY_UPGRADE_TASKS:
        raise ValueError(
            f"Legacy upgraded task {source_task!r} is not a v0 alias. Review provenance and use "
            f"--task-override {LEGACY_UPGRADE_TASKS[source_task]} explicitly."
        )
    expected = task_identity(target_task)["benchmark_revision"]
    if target_task == "Dexverse-PushT-v1" and revision != PUSH_T_RODS_REVISION:
        raise ValueError(
            "Push-T v1 now includes ten low orange rods around the goal with 15 cm minimum gaps, "
            "a rotation-aware verified route and 70% overlap success. "
            "Earlier obstacle-free, three-tall-rod or table-wide 38 cm-gap recordings use a different layout; "
            "review provenance before an explicit task override."
        )
    if target_task == "Dexverse-GraspCup-v1" and revision != DISPENSER_SMALL_CUP_REVISION:
        raise ValueError(
            "Dispenser v1 now uses a 10% smaller cup (scale 0.81) with matching spawn heights, "
            "rim zone and release geometry. Earlier recordings use the larger cup; "
            "review provenance before an explicit task override."
        )
    if target_task == "Dexverse-RemoveCupFromRack-v1" and revision != CUP_RACK_SMALL_DESTINATION_REVISION:
        raise ValueError(
            "Cup-from-rack v1 now uses a 10% smaller cup and a 15 x 15 cm raised destination table. "
            "Earlier recordings use a larger cup, a main-table goal, or a larger destination table; "
            "review provenance before an explicit task override."
        )
    if target_task == "Dexverse-OpenDoor-v1" and revision != DOOR_LATCH_REVISION:
        raise ValueError(
            "Door v1 now uses the narrow cutaway and automatic spring latch. Review provenance; "
            "tilted-door and previous manual-catch recordings are incompatible."
        )
    if source_task in LEGACY_UPGRADE_TASKS or (revision is not None and revision != expected):
        suggestion = LEGACY_UPGRADE_TASKS.get(source_task, BASELINE_PAIRS.get(source_task))
        raise ValueError(
            f"Recording identity {source_task!r} / {revision!r} does not match {target_task!r}. "
            f"Review provenance and use --task-override explicitly (candidate: {suggestion})."
        )


def action_layout(env) -> dict:
    """Describe action terms and joint order without importing the simulator."""
    manager = env.action_manager
    terms = []
    for name in manager.active_terms:
        term = manager.get_term(name)
        terms.append({"name": name, "dimension": int(term.action_dim)})
    robot = env.scene["robot"]
    return {"dimension": int(manager.total_action_dim), "terms": terms, "robot_joint_names": list(robot.joint_names)}
