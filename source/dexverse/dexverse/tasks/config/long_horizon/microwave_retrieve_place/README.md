# Microwave Retrieve Place

`Dexverse-LongHorizon-MicrowaveRetrievePlace-v0` · one right hand · revision `long-horizon-microwave-fixes-v0-2026-11-07b`

Take the dish of stew out of the fridge, shut the fridge, put the dish in the microwave,
shut the microwave and turn a dial.

## Rules

The steps count in this order. An open fridge door blocks the microwave door, so the
fridge has to be shut before the microwave is opened.

| Step | Counts when |
|---|---|
| Open fridge | fridge door at least 45° |
| Shut fridge | fridge door 10° or less |
| Open microwave | microwave door at least 30°, only after the fridge is shut |
| Place dish | dish resting upright (30° or less) on the microwave's floor, its centre at least 3 cm inside the door, for 0.5 s |
| Shut microwave | microwave door 5° or less |
| Turn dial | either dial turned at least 18°, either way (they stop at ±90°) |

Success is declared when the dish is still inside, the door shut and a dial turned at
that moment, held for `--num_success_steps` consecutive steps (default 10).

**Fails the attempt:** tipping the dish 50° or more at any point. Reset and start again.

## What you see

- The microwave's window glows yellow while it is running: door shut and a dial turned.
- The terminal announces each step and prints a `Waiting:` line every few seconds saying
  what the current step still needs, for example `microwave door at 9 deg, shut it to 5`.

Carry the dish by its handle and keep it level; it tips most while being steered around
the microwave door. Put it in with the handle toward the door or sideways, far enough
in for the door to shut.

## Record

```bash
task=LongHorizon-MicrowaveRetrievePlace
python scripts/record_demos.py --task Dexverse-${task}-v0 --dataset_dir longhorizon \
    --teleop_device handtracking --enable_pinocchio --enable_cameras --num_demos 2
```

To practise without recording, run `scripts/teleop_agent.py` with the same `--task`,
`--teleop_device`, `--enable_pinocchio` and `--enable_cameras` flags.

The checks are in `checks.py`, and the dish is built by `dish.py`.
