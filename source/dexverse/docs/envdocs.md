# Dexverse Environment Docs (Table Top Manipulation)

This document lists all non-`top_down_grasp` environments currently registered in this repo, along with task intent, termination conditions, and file locations.

## Dexverse-PourWineGlass-v0
- Task: Grasp a wineglass and tilt it beyond 135° (pouring).
- Termination (success): `lift_and_tilt` when wineglass lift >= configured minimum and tilt exceeds 135°.
- Files:
  - Config: `dexverse/tasks/config/dexterous/pourwineglass/pourwineglass_cfg.py`
  - README: `dexverse/tasks/config/dexterous/pourwineglass/README.md`

## Dexverse-Relocate-v0
- Task: Pick the sphere and relocate to a target position above the table.
- Termination (success): `object_at_goal_position` when object is within the configured distance threshold of target command.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/relocate/relocate_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/relocate/floating_shadow_right_cfg.py`
  - README: `dexverse/tasks/manager_based/table_top_manipulation/config/relocate/README.md`

## Dexverse-PickUpStick-v0
- Task: Grasp a stick, lift it 0.10m above its initial height, and orient it upright (<10° from world +Z).
- Termination (success): `lift_and_tilt` with `tilt_ge=False` when lift >= 0.10m (relative to init) and tilt <= 10°.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/pickup_stick/pickup_stick_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/pickup_stick/floating_shadow_right_cfg.py`

## Dexverse-PickUpCan-v0
- Task: Grasp a can, lift it, and tilt it beyond 120° (pouring-style).
- Termination (success): `lift_and_tilt` when can lift >= configured minimum and tilt exceeds 120°.
- Termination (failure): `object_out_of_bound` when can leaves the table workspace.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/pickup_can/pickup_can_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/pickup_can/floating_shadow_right_cfg.py`

## Dexverse-OpenMicrowave-v0
- Task: Open microwave door beyond a target angle.
- Termination (success): `joint_opened` when max joint angle exceeds `OPEN_ANGLE_RAD`.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/openmicrowave/openmicrowave_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/openmicrowave/floating_shadow_right_cfg.py`
  - README: `dexverse/tasks/manager_based/table_top_manipulation/config/openmicrowave/README.md`

## Dexverse-OpenDrawer-v0
- Task: Pull drawer out beyond a target distance.
- Termination (success): `joint_opened` when max joint position exceeds `OPEN_DISTANCE_M`.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/opendrawer/opendrawer_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/opendrawer/floating_shadow_right_cfg.py`
  - README: `dexverse/tasks/manager_based/table_top_manipulation/config/opendrawer/README.md`

## Dexverse-OpenDoor-v0
- Task: Open door beyond 80° using door hinge joint.
- Termination (success): `joint_opened` on `joint_0` (door hinge), threshold = 80°.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/opendoor/opendoor_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/opendoor/floating_shadow_right_cfg.py`
  - README: `dexverse/tasks/manager_based/table_top_manipulation/config/opendoor/README.md`

## Dexverse-TurnOnSwitch-v0
- Task: Flip a switch on a vertical board to the “on” position.
- Termination (success): `joint_opened` when max joint angle exceeds threshold (~0.5 rad).
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/turnonswitch/turnonswitch_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/turnonswitch/floating_shadow_right_cfg.py`
  - README: `dexverse/tasks/manager_based/table_top_manipulation/config/turnonswitch/README.md`

## Dexverse-RotateKnob-v0
- Task: Rotate a knob from lower limit to upper limit.
- Termination (success): `joint_reached_upper` when knob joint reaches upper limit.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/rotateknob/rotateknob_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/rotateknob/floating_shadow_right_cfg.py`
  - README: `dexverse/tasks/manager_based/table_top_manipulation/config/rotateknob/README.md`

## Dexverse-LiftLid-v0
- Task: Lift kettle lid above a target displacement.
- Termination (success): `joint_opened` when lid joint exceeds `LID_LIFT_DISTANCE_M`.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/liftlid/liftlid_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/liftlid/floating_shadow_right_cfg.py`
  - README: `dexverse/tasks/manager_based/table_top_manipulation/config/liftlid/README.md`

## Dexverse-GraspPot-v0
- Task: Grasp pot and lift above a height threshold.
- Termination (success): `object_lifted` when pot root height exceeds `LIFT_HEIGHT_M`.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/grasppot/grasppot_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/grasppot/floating_shadow_right_cfg.py`
  - README: `dexverse/tasks/manager_based/table_top_manipulation/config/grasppot/README.md`

## Dexverse-GraspBucket-v0
- Task: Move bucket handle to mid-range (from lower limit start). (Note: bucket currently fixed root; success based on handle joint.)
- Termination (success): `joint_relative_move` when bucket handle displacement exceeds configured threshold.
- Files:
  - Config: `dexverse/tasks/manager_based/table_top_manipulation/config/graspbucket/graspbucket_cfg.py`
  - Robot cfg: `dexverse/tasks/manager_based/table_top_manipulation/config/graspbucket/floating_shadow_right_cfg.py`
  - README: `dexverse/tasks/manager_based/table_top_manipulation/config/graspbucket/README.md`
