# Make Coffee

`Dexverse-LongHorizon-MakeCoffee-v0` · one right hand · revision `long-horizon-sim-fixes-v0-2026-11-07`

Take the mug off the cup holder, pour milk into it, set it under the coffee machine's
spouts and turn the machine's switch lever.

## Rules

The steps count in this order.

| Step | Counts when |
|---|---|
| Pour milk | bottle tipped at least 70°, its spout over the mug's mouth (within 4.45 cm of its centre) and 1-12 cm above the rim, mug upright (20° or less), all held for 1 s |
| Place mug | mug within 5 cm of the point under the spouts (the green disc on the tray), resting on the tray, upright (15° or less), still, and your fingers at least 9 cm away, for 0.25 s |
| Turn lever | lever turned at least 35° (it stops at 38°) |

Success is declared when the lever is turned and the mug is still resting on the tray at
that moment, held for `--num_success_steps` consecutive steps (default 10).

**Fails the attempt:** tipping the mug 50° or more after the pour and before it is placed.
Reset and start again.

## What you see

- A green disc on the machine's tray marks where the mug goes.
- The terminal announces each step ("Milk poured. Next: ...") and prints a `Waiting:` line
  every few seconds saying what the current step still needs.

## Record

```bash
task=LongHorizon-MakeCoffee
python scripts/record_demos.py --task Dexverse-${task}-v0 --dataset_dir longhorizon \
    --teleop_device handtracking --enable_pinocchio --enable_cameras --num_demos 2
```

To practise without recording, run `scripts/teleop_agent.py` with the same `--task`,
`--teleop_device`, `--enable_pinocchio` and `--enable_cameras` flags.

The checks are in `checks.py`, with the measurements behind each threshold.
