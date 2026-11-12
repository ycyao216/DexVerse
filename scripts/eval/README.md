# Online policy evaluation

Evaluate a vision-language-action policy on the DexVerse baseline suites by
rolling it out in the simulator and scoring each task's success termination.

Two baselines are wired up:

| `--policy` | Model | Server | Protocol |
| --- | --- | --- | --- |
| `openvla` | [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | `vla-scripts/deploy.py` | HTTP `POST /act`, json_numpy body |
| `pi0` | [openpi](https://github.com/Physical-Intelligence/openpi) π₀ / π₀.₅ | `scripts/serve_policy.py` | WebSocket via `openpi_client` |

The policy always runs **in a separate process**. Neither backend can share an
interpreter with Isaac Sim — the torch / JAX / transformers pins conflict — so
evaluation is a two-terminal workflow.

```
scripts/eval/
├── eval_policy.py      # entry point: runs the env, queries the server, scores
├── policy_clients.py   # the two wire protocols behind one interface
├── baseline_tasks.py   # the suites: env id + training prompt per task
├── run_all_tasks.sh    # sweep a suite, one process per task
├── serve_openvla.sh    # start the OpenVLA-OFT server
└── serve_pi0.sh        # start the openpi server
```

## Checkpoints

The finetuned baseline weights live on the Hugging Face Hub under the
[`dexverse`](https://huggingface.co/dexverse) organization, one repo per
baseline with a `single/` and a `bimanual/` folder inside:

| repo | embodiment folder | size | contains |
| --- | --- | --- | --- |
| [`dexverse/openvla-oft-dexverse`](https://huggingface.co/dexverse/openvla-oft-dexverse) | `single/`, `bimanual/` | 16 / 17 GB | 4-shard merged VLA, action head, proprio projector, `dataset_statistics.json` |
| [`dexverse/pi05-dexverse`](https://huggingface.co/dexverse/pi05-dexverse) | `single/`, `bimanual/` | 7.0 GB each | `model.safetensors`, `metadata.pt`, `assets/.../norm_stats.json` |

Download one embodiment at a time — pulling a whole repo fetches both:

```bash
pip install -U "huggingface_hub[cli]"

# π₀.₅
hf download dexverse/pi05-dexverse --include 'single/*' --local-dir ~/dexverse-ckpts/pi05
hf download dexverse/pi05-dexverse --include 'bimanual/*' --local-dir ~/dexverse-ckpts/pi05

# OpenVLA-OFT
hf download dexverse/openvla-oft-dexverse --include 'single/*' --local-dir ~/dexverse-ckpts/openvla
hf download dexverse/openvla-oft-dexverse --include 'bimanual/*' --local-dir ~/dexverse-ckpts/openvla
```

Both repos are public — no Hugging Face account or token is needed. `CKPT_DIR`
is then the embodiment folder, e.g. `~/dexverse-ckpts/pi05/single`. Downloads
resume if interrupted; re-running a command is a no-op once complete.

Two things not to "clean up" after downloading:

- **π₀.₅ — keep `assets/dexbench-data/...` exactly where it is.** openpi resolves
  normalization statistics by `assets/<asset_id>/norm_stats.json`, where the id
  comes from the training config. Renaming it makes the server start *without*
  norm stats and silently emit unnormalized actions.
- **OpenVLA-OFT — the `unnorm_key` must stay `dexbench_rlds/{single,bimanual}`.**
  Those are the only keys in `dataset_statistics.json`, and the string is also
  what selects the action dimension (see the argv trap below). `dexbench` is the
  benchmark's former name.

## Quick start

**Terminal 1 — policy server.** Every path is a required environment variable;
there are no defaults, and each script validates what it is given before
starting.

```bash
# OpenVLA-OFT
OPENVLA_OFT_ROOT=/path/to/openvla-oft \
PYTHON_BIN=/path/to/envs/openvla-oft/bin/python \
CKPT_DIR=~/dexverse-ckpts/openvla/single \
UNNORM_KEY=dexbench_rlds/single \
    bash scripts/eval/serve_openvla.sh

# openpi
OPENPI_ROOT=/path/to/openpi \
PYTHON_BIN=/path/to/openpi/.venv/bin/python \
CKPT_DIR=~/dexverse-ckpts/pi05/single \
CONFIG_NAME=pi05_dexbench \
    bash scripts/eval/serve_pi0.sh
```

For a bimanual checkpoint point `CKPT_DIR` at the `bimanual/` folder, pass
`EMBODIMENT=bimanual` to `serve_openvla.sh` (it switches
`--num_images_in_input` from 2 to 3), and point `CONFIG_NAME` at the bimanual
config for `serve_pi0.sh`.

**Terminal 2 — evaluation.** Run in the DexVerse / Isaac Lab environment:

```bash
# single task
python scripts/eval/eval_policy.py --policy pi0 --enable_cameras --headless \
    --task Dexverse-GraspCup-v0 --num_episodes 2 --max_steps 200 --save_videos

# whole suite, one process per task (recommended)
POLICY=pi0 SUITE=single bash scripts/eval/run_all_tasks.sh --save_videos
POLICY=openvla UNNORM_KEY=dexbench_rlds/single SUITE=single \
    bash scripts/eval/run_all_tasks.sh
```

`--enable_cameras` is mandatory: the policy is fed rendered RGB.

`run_all_tasks.sh` gives each task its own Python process. `eval_policy.py
--suite` can do the whole sweep in one process, but Isaac Sim's USD stage
teardown can hang after many sequential `gym.make` / `env.close` cycles, and one
crashed task would take the sweep with it.

## Baseline suites

19 tasks, defined in `baseline_tasks.py`:

- **`single`** — 12 tasks, 28-DoF floating Shadow right hand, third-person +
  wrist cameras.
- **`bimanual`** — 7 tasks, 56-DoF floating Shadow pair, third-person + left and
  right wrist cameras.

The prompt strings are copied verbatim from the LeRobot training set's
`meta/tasks.jsonl`, i.e. the exact `language_instruction` values the checkpoints
were finetuned on. Editing one moves the policy off its training distribution;
treat the table as part of the benchmark definition.

Embodiment is read from `env_cfg.robot_type` at runtime, never guessed from the
task or class name — five of the seven bimanual tasks are registered under
`tasks/config/articulation/` with config classes still named
`...EnvFloatingDexHandRightCfg` despite setting
`robot_type = "floating_shadow_bimanual"`. `eval_policy.py` aborts if the
detected embodiment disagrees with the suite the task is listed in.

## The state-width contract

`--observation_preset` sets the proprio history length and therefore how wide
the state vector reaching the server is. **The two baselines need different
values**, because they were trained on differently-built datasets:

| `--policy` | default `--observation_preset` | default `--state_dim` | state sent (single / bimanual) |
| --- | --- | --- | --- |
| `openvla` | `none` | 28 / 56 | `(28,)` / `(56,)` |
| `pi0` | `rgb` | unset (no truncation) | `(84,)` / `(168,)` |

openvla-oft's RLDS transform clips proprio to a single frame; openpi's norm
stats were computed over the 3-frame stack the `rgb` preset produces. Sending
the wrong width does not raise — it silently feeds the model a mismatched
vector — so `eval_policy.py` always prints what it is about to send before the
first request:

```
[pi0-eval] robot_type='floating_shadow_right' -> single embodiment
[pi0-eval] applied observation preset: rgb
[pi0-eval] first request | third_person=(256, 256, 3) wrist=(256, 256, 3) state=(84,) (raw proprio dim=84, sent=84, preset=rgb)
```

Check that line against your checkpoint's norm stats before trusting any
numbers. Both defaults can be overridden explicitly.

## OpenVLA-OFT: the `--unnorm_key` argv trap

openvla-oft chooses its action and proprio dimensions by **inspecting
`sys.argv` at import time** (`prismatic/vla/constants.py::detect_robot_platform`).
It scans the joined argv for `dexbench`, `libero`, `aloha`, `bridge` in that
order and silently falls back to LIBERO — 7-dim actions — when nothing matches.

The only argv token carrying that keyword is `--unnorm_key`. So the key must
contain `dexbench`, and additionally `bimanual` for the two-hand model.
`serve_openvla.sh` refuses to start otherwise; set
`OVERRIDE_ARGV_KEYWORD_CHECK=1` only if you know the platform is selected some
other way. This is also why renaming the released dataset key from `dexbench_*`
to `dexverse_*` requires a matching change in openvla-oft.

## Output

```
runs/<policy>_eval_online/
├── summary.json                       # task -> metrics
└── <task>/
    ├── metrics.json                   # metrics + the full eval config used
    ├── first_frame_*.png              # with --dump_first_frame
    └── videos/episode_XXX_{success,failure}.mp4   # with --save_videos
```

`metrics.json` records `success_rate`, `avg_return`, `avg_ep_len`,
`inference_latency_ms`, alongside the resolved `embodiment`, `robot_type`,
`observation_preset`, `state_dim_sent`, `replan_interval` and `seed` — enough to
reproduce the run.

Success is scored by detaching the task's `success` termination term from the
env config and evaluating it manually each step, so an episode is not cut short
the instant success first fires.

## Notes

- `--disable_bg_randomize` is **on by default**. It nulls the
  `reset_environment_background` and `reset_table_texture` event terms to match
  `create_demo_files_sequential.py --disable-bg-randomize`, which is how the
  training set was generated. Pass `--no_disable_bg_randomize` to evaluate under
  per-episode visual randomization instead. Both terms are already `None` in the
  base `EventCfg`, so on the current task set the flag is a no-op; it is there so
  a task that re-enables them still gets evaluated on the training distribution.
- `--replan_interval` is how many actions of each predicted chunk are executed
  before re-querying. Defaults to 8 for openvla (its full training-time
  `NUM_ACTIONS_CHUNK` open-loop horizon) and 5 for pi0.
- Action chunks are clamped to ±3.14 rad and NaN/Inf-scrubbed before reaching
  `env.step` — PhysX segfaults rather than raising on non-finite input, and
  unconverged checkpoints do emit it.
- `openpi_client` waits for its server **indefinitely**, retrying every 5 s. If
  evaluation seems to hang at startup, check that the server is up and the port
  matches; it is not loading, it is retrying.
- Run the simulator and the policy server on different GPUs.
  `run_all_tasks.sh` pins Isaac Sim to `CUDA_VISIBLE_DEVICES=0` by default.
- Only `openvla` and `pi0` are wired up here. Diffusion-policy and CFM baselines
  are not part of this harness.
