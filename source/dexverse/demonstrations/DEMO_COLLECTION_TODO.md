# DexVerse — Demonstration Collection Checklist (for intern)

**Goal:** record VR/teleop demonstrations **with recorded per-step states** for the
benchmark tasks that don't have usable ones yet. The benchmark is **56 IL + 5
long-horizon = 61 tasks**.

> ⚠️ **A demo is only usable if it has recorded states** (`record_state`). Many old
> demos were recorded *without* states and don't count. Coverage below is by
> *usable* (has-states) demos: **25 usable, 36 to collect**.
>
> The 39 "multi-goal" combination tasks are **out of scope**.

---

## How to record a demo

Run in the `dexbench` conda env with a VR/OpenXR headset (hand-tracking):

```bash
conda activate dexbench
python scripts/record_demos.py \
    --task <ENV_ID> \
    --dataset_dir source/dexverse/demonstrations/<category> \
    --num_demos 50 \
    --num_success_steps 10
```

- **States are recorded automatically** — `--record_state` is always enabled in the
  current `scripts/record_demos.py` (you do **not** need to pass any flag). Every new
  demo will include the per-step states. ✅
- Saves to `<dataset_dir>/<ENV_ID>/<ENV_ID>_<timestamp>.pkl` (one folder per task).
- `--num_success_steps 10` = a demo is auto-concluded after 10 continuous success steps.
- Defaults `--teleop_device handtracking`, `--teleop_retargeter relative`,
  `--retargeting_scheme dexpilot` — leave as-is unless told otherwise.
- **Target per task:** _____ successful demos (confirm with lead).
- Sanity-check with `python scripts/replay_demos.py --task <ENV_ID> ...`.

### Verify a demo actually has states
```python
import pickle
d = pickle.load(open("<file>.pkl", "rb"))
print(d["record_state"], "states" in d["episodes"][0])   # both should be True
```

---

## ❗ To collect (36)

`(no demos)` = nothing on file. `(re-record: stateless)` = demos exist but lack
states — must be re-recorded with the current script.

### grasping (6)
- [ ] `Dexverse-PickCube-v0` → `grasping/` (no demos)
- [ ] `Dexverse-StackCube-v0` → `grasping/` (no demos)
- [ ] `Dexverse-RelocateSphere-v0` → `grasping/` (no demos)
- [ ] `Dexverse-PickUpStick-v0` → `grasping/` (no demos)
- [ ] `Dexverse-RelocateObject-v0` → `grasping/` (no demos)
- [ ] `Dexverse-GraspTwoItems-v0` → `grasping/` (no demos)

### functional (2)
- [ ] `Dexverse-PourWineGlass-v0` → `functional/` (no demos)
- [ ] `Dexverse-FunctionalDrillApply-v0` → `functional/` (re-record: stateless)

### articulation (15)
- [ ] `Dexverse-OpenCabinet-v0` → `articulation/` (no demos)
- [ ] `Dexverse-LiftLid-v0` → `articulation/` (no demos)
- [ ] `Dexverse-OpenDoor-v0` → `articulation/` (no demos)
- [ ] `Dexverse-RotateKnob-v0` → `articulation/` (no demos)
- [ ] `Dexverse-OpenMicrowave-v0` → `articulation/` (no demos)
- [ ] `Dexverse-OpenDrawer-v0` → `articulation/` (no demos)
- [ ] `Dexverse-TurnOnSwitch-v0` → `articulation/` (no demos)
- [ ] `Dexverse-GraspBucket-v0` → `articulation/` (no demos)
- [ ] `Dexverse-LiftBucket-v0` → `articulation/` (no demos)
- [ ] `Dexverse-GraspPot-v0` → `articulation/` (no demos)
- [ ] `Dexverse-PushButton-v0` → `articulation/` (no demos)
- [ ] `Dexverse-CabinetDoubleDoor-v0` → `articulation/` (no demos, bimanual)
- [ ] `Dexverse-UnscrewCap-v0` → `articulation/` (no demos, bimanual)
- [ ] `Dexverse-OpenGlasses-v0` → `articulation/` (no demos, bimanual)
- [ ] `Dexverse-LiftBasketHandle-v0` → `articulation/` (re-record: stateless, bimanual)

### bimanual (3)
- [ ] `Dexverse-BimanualLiftDutchOven-v0` → `bimanual/` (no demos)
- [ ] `Dexverse-BimanualHandover-v0` → `bimanual/` (no demos)
- [ ] `Dexverse-BimanualLiftBasket-v0` → `bimanual/` (re-record: stateless)

### contact-rich (5)
- [ ] `Dexverse-NutThread-v0` → `contact_rich/` (no demos)
- [ ] `Dexverse-InsertPipette-v0` → `contact_rich/` (no demos)
- [ ] `Dexverse-PickFromClutter-v0` → `contact_rich/` (no demos)
- [ ] `Dexverse-GearMesh-v0` → `contact_rich/` (re-record: stateless)
- [ ] `Dexverse-InsertPeg-v0` → `contact_rich/` (re-record: stateless)

### non-prehensile (1)
- [ ] `Dexverse-PushSphereUpSlope-v0` → `non_prehensile/` (no demos)

### long-horizon (4)
- [ ] `Dexverse-LongHorizon-MakeCoffee-v0` → `long_horizon/` (no demos)
- [ ] `Dexverse-LongHorizon-MicrowaveRetrievePlace-v0` → `long_horizon/` (no demos)
- [ ] `Dexverse-LongHorizon-CookFoods-v0` → `long_horizon/` (no demos)
- [ ] `Dexverse-LongHorizon-OvenBakeSalmon-v0` → `long_horizon/` (no demos)

---

## ✅ Already have usable (has-states) demos (25) — do NOT re-collect

functional: GraspBleach, GraspPan, GraspKettle, GraspCup, RemoveCupFromRack,
FunctionalPourCan, FunctionalPourMug, FunctionalHammerStrike ·
articulation: OpenFaucet, OpenLaptop, SqueezeScissors, SlideUtilityKnife,
OpenStapler, OpenFlatFolder, OpenHuaweiPhone ·
bimanual: BimanualLiftTray, BimanualLiftCarton ·
contact-rich: InsertPen, PlugCharger, PickThinObjectFromContainer ·
non-prehensile: PushSmallSphereObstacleSlope, PushT, PivotLargeCuboidAgainstWall,
TakeBookOffShelf · long-horizon: TrashDrawerSortSimple

(Most usable ones live in `baseline/`. Several have only 1–2 demos — confirm with
lead whether they need topping up to the per-task target.)

---

## Notes
- Existing demos live in **legacy folders** (`articulation/`, `bimanual_articulation/`,
  `dexterous/`, `grasping/`, `rigid/`, `baseline/`) predating the 6-skill taxonomy. New
  recordings should use the skill-category folders above; the old layout will be consolidated.
- The 7 bimanual-articulation tasks recorded under the legacy
  `Dexverse-FixateThenManipulate-<X>-v0` id are **stateless** — they don't count;
  usable copies (where they exist) come from `baseline/` or the current `Dexverse-<X>-v0` id.
- Multi-goal (39) deferred. Audit: gym registry + `demonstrations/` tree + per-pkl state check.
