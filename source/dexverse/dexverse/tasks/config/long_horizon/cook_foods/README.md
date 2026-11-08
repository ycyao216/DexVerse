# Cook Foods

`Dexverse-LongHorizon-CookFoods-v0` · one right hand · revision `long-horizon-cook-foods-fixes-v0-2026-11-07b`

Put the pot on the green burner, pour in the water and the can of soup, tip the meat in
from the bowl, then turn on the stove.

## Rules

The pours, the pot and the meat can be done in any order. Turning on the stove is last.

| Step | Counts when |
|---|---|
| Pour water | bottle lifted at least 10 cm, tipped at least 70°, its spout within 10 cm of the pot's centre, pot upright (20° or less), all held for 1 s |
| Pour soup | the same, with the can |
| Pot on burner | pot within 8 cm of the green burner's centre, sitting on the grate, tilted 15° or less |
| Meat | all 4 pieces inside the pot |
| Turn on stove | green knob turned at least 45° (it stops at 48°), with both pours done, the pot on the burner and all the meat in it |

Success is declared when all of this is true at the same moment, held for
`--num_success_steps` consecutive steps (default 10).

**Fails the attempt:** tipping the pot 50° or more after anything has been poured into it.
Reset and start again.

## What you see

- Green rings mark the burner and the knob that count.
- Liquid appears in the pot once a pour counts: blue for water, red for soup, orange-red
  once both are in. It disappears if the pot is spilled.
- Once the knob is on, blue flames show around the burner and its ring turns orange.
  Success resets the scene a moment later; add `--num_success_steps 60` to see the flame
  for a second first.
- The terminal announces each pour and prints a `Waiting:` line every few seconds. A pour
  that falls short says why, for example `Water pour not counted: tipped to 62 deg, needs 70.`

Carry the pot by its handle. When you set it on the burner, turn the handle away from
the green knob so it does not get in the way later.

## Record

```bash
task=LongHorizon-CookFoods
python scripts/record_demos.py --task Dexverse-${task}-v0 --dataset_dir longhorizon \
    --teleop_device handtracking --enable_pinocchio --enable_cameras --num_demos 2
```

To practise without recording, run `scripts/teleop_agent.py` with the same `--task`,
`--teleop_device`, `--enable_pinocchio` and `--enable_cameras` flags.

The checks are in `checks.py`, with the measurements behind each threshold.
