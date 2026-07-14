<p align="center">
  <img src="docs/dexverse_logo_v1.png" alt="DexVerse logo" width="600"/>
</p>

<h1 align="center">DexVerse: A Modular Benchmark for Multi-Task, Multi-Embodiment Dexterous Manipulation</h1>

<p align="center">
  Yunchao Yao<sup>1*</sup>, Zhuxiu Xu<sup>1,2*</sup>, Tianqi Zhang<sup>1</sup>, Zixian Liu<sup>1</sup>, Sikai Li<sup>1</sup>, Zhenyu Wei<sup>1</sup>, Feng Chen<sup>2</sup>, Dihong Huang<sup>1</sup>,<br/>
  Kechang Wan<sup>1</sup>, Chenyang Ma<sup>1</sup>, Shuqi Zhao<sup>3</sup>, Shenghua Gao<sup>2</sup>, Masayoshi Tomizuka<sup>3</sup>, Yi Ma<sup>2</sup>, Mingyu Ding<sup>1†</sup>
</p>

<p align="center">
  <sup>1</sup>UNC-Chapel Hill &nbsp;&nbsp; <sup>2</sup> The University of Hong Kong &nbsp;&nbsp; <sup>3</sup>UC Berkeley<br/>
  <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author
</p>

<p align="center">
  🌐 <a href="https://ycyao216.github.io/DexVerse.site/"><strong>Project Page</strong></a>
</p>

## Release Roadmap

- [x] Initial release: task suite, assets
- [x] Release teleoperation and data-collection tooling and corresponding documentations
- [ ] Baseline environment demonstrations and baseline code
- [ ] Full shadowhand demonstration dataset
- [ ] Cross-embodiment robot assets, instructions, and demonstrations

---

This repository is the official codebase for **DexVerse**, a benchmark for tabletop dexterous
manipulation built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab). 
**Docs:** `source/dexverse/docs/envdocs.md` lists the registered tasks.

## Repository Structure

The repository is organized as an Isaac Lab extension project: the installable Python package lives
under `source/dexverse`, and runnable entry points live under `scripts/`. Large binary files (robot
and object assets, demonstrations) are downloaded from Hugging Face
into this tree during setup (see [Downloading Assets](#downloading-assets) below).

```
DexVerse/
├── source/dexverse/                 # Installable Python package (the Isaac Lab extension)
│   ├── dexverse/                        # Core package
│   │   ├── tasks/                           # Task/environment definitions and configs
│   │   ├── assets/                          # Asset configs (objects, scenes, background HDRIs, ...)
│   │   ├── devices/                         # Teleop input devices (OpenXR, retargeters)
│   │   ├── robot_agents/                    # Per-robot-hand configs
│   │   └── utils/                           # Shared utilities
│   ├── demonstrations/                  # Demonstration data (populated by download_demos.py)
│   ├── docker_utils/                    # Docker Compose patch for IsaacLab. 
│   └── docs/                            # Extension docs; envdocs.md lists all registered tasks
├── scripts/                         # Entry points and tooling (not installed as a package)
│   ├── list_envs.py                     # List registered tasks
│   ├── zero_agent.py / random_agent.py  # Dummy agents for sanity checks
│   ├── teleop_agent.py                  # Interactive VR teleoperation
│   ├── record_demos.py                  # Demonstration recording
│   ├── run_dexverse.py                  # Joint-slider debug UI
│   ├── asset_tools/                     # Asset download utilities
│   └── demo_tools/                      # Demo download / conversion / inspection utilities
├── datastorage/                     # Host-mounted demo output (Docker; gitignored contents)
└── docs/                            # Repo-level docs (observation space, known hand issues)
```



## Installation



### Prerequisites

DexVerse runs on top of [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) and
[Isaac Lab](https://github.com/isaac-sim/IsaacLab). We recommend using `conda` to manage the
python environment, and cloning Isaac Lab and DexVerse side by side in the same parent directory:

```
workspace/
├── IsaacLab/    # simulator framework (Isaac Lab v2.3.2)
└── DexVerse/    # this repository
```

The steps below install Isaac Sim 5.1.0 and Isaac Lab v2.3.2 into a fresh conda environment.
They mainly follow the [official Isaac Lab pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html),
with two adjustments (marked `FIX`) that work around known dependency conflicts in that release:

```bash
# Create and activate a fresh environment (run from the workspace/ directory)
conda create -n dexverse python=3.11
conda activate dexverse
pip install --upgrade pip

# Install Isaac Sim 5.1.0
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install the CUDA 12.8 builds of PyTorch that Isaac Sim 5.1 is built against.
# FIX: we add torchaudio to the official guide's command here. With the exact expected
# torch version already present, Isaac Lab's installer skips its own torch reinstall step,
# which would otherwise uninstall torchaudio without restoring it.
pip install -U torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Temporary fix to counter isaaclab v2.3.2 installation dependency breaks. Inspired by https://github.com/isaac-sim/IsaacLab/issues/4576 
pip install "setuptools==65.0.0"
pip install "flatdict==4.0.1" --no-build-isolation

# Install Isaac Lab v2.3.2
git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.2
cd IsaacLab
sudo apt install cmake build-essential   # Linux system dependencies
./isaaclab.sh --install
```

> **Note**: during `./isaaclab.sh --install`, pip may still print dependency-resolver warnings. We will continue to monitor the effect of these conflicts.



### Install the DexVerse package

Most dependencies will be satisfied after IsaacLab installation. After IsaacLab is fully installed, from this repository root and in the same virtual environment that IsaacLab is installed to, install the extension (optionally in editable mode):

```bash
python -m pip install -e source/dexverse
```

This is all the core benchmark needs. The environments, teleoperation, demo recording and conversion run entirely on packages that ship with Isaac Lab's official environment.

### Optional: extras for the demo inspection tools

A few offline utilities under `scripts/demo_tools/` use packages that are *not* part of Isaac
Lab's official environment. Install them if you use these tools:

```bash
# opencv-python is capped and numpy is constrained on purpose: opencv-python >= 4.12
# requires numpy >= 2, but Isaac Sim 5.1 and Isaac Lab are built against numpy 1.26 and
# will break if numpy is upgraded to 2.x.
python -m pip install matplotlib "opencv-python<4.12" open3d imageio-ffmpeg "numpy<2"
```



## Downloading Assets

The robot hand and object/scene assets are not stored in the git repository. They are hosted on the gated
Hugging Face dataset [`dexverse/DexVerse_release`](https://huggingface.co/datasets/dexverse/DexVerse_release)
and must be downloaded before any environment can run. Log in once and accept the dataset terms:

```bash
pip install huggingface_hub
hf auth login   # and follow the prompt to login to hugging face
# alternatively, you can create hugging face tokens and set the environmetn HF_TOKEN=<your-token> 
```



### Robot hand assets

The robot hand configs (Python/YAML) ship with the repo, but the USD/URDF/mesh files they load are
downloaded separately. Fetch every available hand with:

```bash
python scripts/asset_tools/download_robot_agents.py --all
```

The bundles extract into `source/dexverse/dexverse/robot_agents/`, directly next to the configs
that use them. 

> **Note**: we currently release the Shadow hand. The remaining robot combinations are coming soon. 



### Object and scene assets

The tabletop objects, scenes, HDRI backgrounds, and the ManiTwin-100K object pool used by the
tasks are downloaded the same way. Fetch everything with:

```bash
python scripts/asset_tools/download_assets.py --all
```

The bundles extract into `source/dexverse/dexverse/assets/`. Note that `--all` pulls the full set
(core assets ~410 MB, ManiTwin object pool ~2.2 GB, HDRIs ~1.8 GB, plus long-horizon task meshes),
so expect a few GB of downloads.

With both downloads in place, every registered environment is functional. If you only need a
subset (a single hand, or just the core assets), both scripts support finer-grained flags — run
them with `--help`, or `download_robot_agents.py --list` to see the available hand bundles.

## Quick Start

Verify the installation by listing the registered tasks:

```bash
python scripts/list_envs.py
```

You can also run tasks with a dummy agent. This loads the full environment without needing demonstrations or a trained policy:

```bash
# apply zero actions every step
python scripts/zero_agent.py --task=<TASK_NAME> --enable_cameras --num_envs=1

# apply uniformly sampled random actions
python scripts/random_agent.py --task=<TASK_NAME> --enable_cameras --num_envs=1 
```

If the simulator window opens and the scene steps without errors, the core installation is complete.
Pick any `<TASK_NAME>` from the `list_envs.py` output (the full catalog is documented in
`source/dexverse/docs/envdocs.md`), and use `--num_envs=<N>` to control how many parallel
environments are spawned.

## Teleoperation and Data Collection

Our VR teleoperation and demonstration-recording pipeline is built on top of Isaac Lab's
[CloudXR teleoperation guide](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html). That workflow streams the simulation to an XR headset (e.g. Apple Vision Pro) over NVIDIA CloudXR and sends hand-tracking inputs back to control the robot. It also has visualizations explaining the UI of the teleoperation app. 

**Docker is the currently recommended way to set up the CloudXR runtime and its dependencies.** Isaac Lab
ships Docker Compose patches that run the simulator and CloudXR runtime together. We additionally provide an
additional patch to mount this repository into the container (see
`source/dexverse/docker_utils/README.md` for details).

### Prerequisites

Before starting, you can install Docker following the [official instructions](https://docs.docker.com/engine/install/).
The CloudXR guide also lists [system requirements](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#system-requirements) (GPU, RAM, XR device, network) and firewall rules for the streaming ports. It is also necessary to connect both the Apple Vision Pro to the same wireless network as the one used by the machine running  DexVerse. Also make sure the wireless network setting allows direct connection (can verify using ping). Internet connection is not required for the Apple Vision Pro to communicate with the machine running DexVerse. 

### Start the demonstration collection environment: Docker Compose with CloudXR

The easiest way to start the container is from the Isaac Lab repository root. This starts the Isaac Lab and CloudXR runtime containers together. If prompted, enable X11 forwarding so the Isaac Sim UI is visible on the host.

```bash
cd /path/to/IsaacLab
./docker/container.py start \
    --files docker-compose.cloudxr-runtime.patch.yaml \
            ../../DexVerse/source/dexverse/docker_utils/docker-compose.dexverse.patch.yaml \
    --env-files .env.cloudxr-runtime
```

Enter the Isaac Lab base container:

```bash
./docker/container.py enter base
```

Inside the container, install DexVerse once (the repo is bind-mounted at `/workspace/dexverse`):

```bash
cd /workspace/dexverse
python -m pip install -e source/dexverse
```

In the Isaac Sim UI, open the **AR** panel, set **Selected Output Plugin** to **OpenXR** and
**OpenXR Runtime** to **System OpenXR Runtime**, then click **Start AR**. Connect from your XR client as described in the [Apple Vision Pro section](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#use-apple-vision-pro-for-teleoperation) of the Isaac Lab guide.

When finished, stop the containers from the Isaac Lab root:

```bash
./docker/container.py stop \
    --files docker-compose.cloudxr-runtime.patch.yaml \
            ../../DexVerse/source/dexverse/docker_utils/docker-compose.dexverse.patch.yaml \
    --env-files .env.cloudxr-runtime
```

> **Note**: paths in `--files` are resolved relative to `IsaacLab/docker` and assume DexVerse is  
> cloned next to Isaac Lab under the same parent directory. If your layout differs, you can set  
> `DEXVERSE_PATH=/abs/path/to/DexVerse` before running `container.py`.



### Debug teleoperation (`teleop_agent.py`)

Use `scripts/teleop_agent.py` to test VR teleop, retargeting, and task setup **without** writing demos to disk.  

```bash
cd /workspace/dexverse
./isaaclab.sh -p scripts/teleop_agent.py \
    --task Dexverse-PickUpStick-v0 \
    --teleop_device handtracking \
    --enable_pinocchio
```

Common optional flags:

- `--robot_type`  — override the robot variants
- `--teleop_retargeter relative|absolute` — wrist retargeting mode (default: `relative`). `relative` takes the pose of the operator's wrist when the teleoperation process is started. `absolute` directly take the pose of the operator's wrist in the simulator's frame and match the robot's wrist link to that.  
- `--retargeting_scheme dexpilot|vector` — finger retargeting optimizer (default: `dexpilot`)
- `--enable_debug_vis` — show zone / reference-point markers in the viewport

Use START / STOP / RESET from the XR client to control the session (more details see the official IsaacLab CloudXR guide).

### Record demonstrations (`record_demos.py`)

Once teleop feels good, switch to `scripts/record_demos.py` to save trajectory pickles. It uses the
same VR stack as `teleop_agent.py`, but additionally records per-step actions and scene states and
auto-saves when the task success condition is met.

The Docker patch mounts `datastorage/` into the container and sets `DEXVERSE_DATA_DIR`, so  
recordings land on the host under `DexVerse/datastorage/`. Pass `--dataset_dir` only to choose a subfolder (for example `grasping, or can leave blank`):

```bash
cd /workspace/dexverse
./isaaclab.sh -p scripts/record_demos.py \
    --task Dexverse-PickUpStick-v0 \
    --dataset_dir grasping \
    --teleop_device handtracking \
    --enable_pinocchio \
    --num_demos 50 \
    --num_success_steps 10
```

By default, output path on the host: `DexVerse/datastorage/grasping/Dexverse-PickUpStick-v0/<TASK>_<timestamp>.pkl`. File names can also be specified. See output of `--help` for other argument options. 

Each pickle includes per-step scene states (`record_state` is always on). Use START to begin
recording an episode; a demo is saved after `--num_success_steps` consecutive successful steps. To see other arguments, use `--help`. Smooth teleoperation also depends on CPU, GPU, and network condition. 

### Basic replaying and converting demos (`--set-state`)

Isaac Sim / PhysX dynamics can differ slightly across GPUs and driver versions, so replaying
recorded **actions** step-by-step on another machine may drift from the original trajectory.

When converting pickles to HDF5 for training, it is recommended to use `scripts/demo_tools/create_demo_files_sequential.py` with `--set-state` (the default). This flag directly set the recorded scene state at each timestep instead of using environment steps from actions. This keeps observations consistent across machines.

```bash
python scripts/demo_tools/create_demo_files_sequential.py \
    --file datastorage/grasping/Dexverse-PickUpStick-v0/<TASK>_<timestamp>.pkl \
    --set-state
```

Pass `--no-set-state` only if you explicitly want true action replay. See `create_demo_files_sequential.py --help` for the full set of output and selection options.

#### Observation modes (`--obs-groups`)

You choose which observations end up in the HDF5 with `--obs-groups`. It accepts either a
single **preset** name (which narrows the env's observation space before it is built) or an
explicit **list of group names**. When omitted, every active observation group on the env is
captured, and downstream consumers can select the subset they need.

Available presets:


| Preset                                 | Enabled groups                        | Notes                         |
| -------------------------------------- | ------------------------------------- | ----------------------------- |
| `rgb`                                  | policy, proprio, goal, rgb            | single view, history length 3 |
| `rgb_depth` (alias `rgbd`)             | policy, proprio, goal, rgb, depth     | single view, history length 3 |
| `pointcloud`                           | policy, proprio, goal, pointcloud     | single view, history length 3 |
| `state`                                | policy, proprio, contact, state, goal | no image history              |
| `3view_rgb`                            | policy, proprio, goal, rgb            | three camera views            |
| `3view_rgb_depth` (alias `3view_rgbd`) | policy, proprio, goal, rgb, depth     | three camera views            |
| `3view_pointcloud`                     | policy, proprio, goal, pointcloud     | three camera views            |


```bash
# Preset (narrows the obs space to RGB + proprio/goal):
python scripts/demo_tools/create_demo_files_sequential.py \
    --file datastorage/grasping/Dexverse-PickUpStick-v0/<TASK>_<timestamp>.pkl \
    --obs-groups rgb

# Explicit group list (captures exactly these groups):
python scripts/demo_tools/create_demo_files_sequential.py \
    --file datastorage/grasping/Dexverse-PickUpStick-v0/<TASK>_<timestamp>.pkl \
    --obs-groups proprio rgb depth
```

Image storage dtypes are configurable with `--rgb-dtype` (`uint8` default, or `float32`) and
`--depth-dtype` (`float16` default, or `float32`).

#### Recording a plain camera video for quick debugging during conversion (`--record-video`)

`create_demo_files_sequential.py` can also write one MP4 per episode straight from a scene camera while it replays, which is handy for sanity-checking a conversion run. This is a raw camera render for debug purpose and is not affected by the choice of observation modes.

```bash
python scripts/demo_tools/create_demo_files_sequential.py \
    --file datastorage/grasping/Dexverse-PickUpStick-v0/<TASK>_<timestamp>.pkl \
    --record-video \
    --video-camera third_person_camera \
    --video-fps 30 \
    --video-dir outputs/replay_videos
```

`--video-dir` defaults to a `videos/` sibling of each HDF5 output.

### Rendering debug videos from an H5 (`render_demo_video.py`)

`We also provide scripts/demo_tools/render_demo_video.py` that takes an observation HDF5 file (the output of the converter) and renders a composited and more complete debug MP4 for a trajectory. The layout is: all visual streams (RGB / depth / point cloud) on the left, an action heatmap with a moving step bar on the upper right, and a per-step observation/state text panel on the lower right (showing the file's `obs_groups` / preset and a summary of numeric obs terms).

```bash
python scripts/demo_tools/render_demo_video.py \
    --dataset_file datastorage/grasping/Dexverse-PickUpStick-v0/Dexverse-PickUpStick-v0.demo.h5 \
    --episode 0 \
    --output outputs/demo_videos/pickup_stick_demo0.mp4 \
    --fps 15
```

Use `--episode all` (or `*`) to render every episode into separate MP4s (written under
`--output-dir`, or next to the source H5). Restrict the text panel with `--obs-display-groups`
(e.g. `proprio goal contact`), and add `--include_next_obs` to also plot `next_obs/*` streams.
When `--output` is omitted, the MP4 is written as `<h5-stem>__demo_<idx>.mp4` beside the source
file. See `render_demo_video.py --help` for figure size, DPI, and other options.

See also `source/dexverse/docker_utils/README.md` for Docker mount details.

## Demonstrations

> 🚧 **Data and Instructions Coming soon.** 



## Contact

For questions about the benchmark or this codebase, please don't hesitate to open a GitHub issue or directly reach out to:

- **Yunchao Yao** — [yunchaoy@cs.unc.edu](mailto:yunchaoy@cs.unc.edu)



## Citation

If you find DexVerse useful in your research, please cite:

```bibtex
@article{yao2026dexverse,
  title   = {DexVerse: A Modular Benchmark for Multi-Task, Multi-Embodiment Dexterous Manipulation},
  author  = {Yao, Yunchao and Xu, Zhuxiu and Zhang, Tianqi and Li, Sikai and Wei, Zhenyu and Chen, Feng and Huang, Dihong and Wan, Kechang and Ma, Chenyang and Zhao, Shuqi and Gao, Shenghua and Tomizuka, Masayoshi and Ma, Yi and Ding, Mingyu},
  journal = {arXiv preprint arXiv:2607.08751},
  year    = {2026}
}
```
