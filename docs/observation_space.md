# Observation Space

DexVerse environments follow a manager-based observation design (built on
Isaac Lab's `ObservationGroupCfg` / `ObservationTermCfg` API), in which the
observation space is organized into a small number of **groups**, each
holding a set of individually named **terms**. Rather than exposing a single
monolithic observation vector, DexVerse decomposes observations by *consumer*
(policy, asymmetric critic, perception backbone, visualization) and by
*modality* (proprioception, contact, object/task state, privileged
quantities, goal, RGB, depth, point cloud), so that downstream users can
subscribe to exactly the information their method requires and can reason
precisely about what a trained policy did or did not have access to.

## 1. Observation groups

| Group | Contents | Available to trained policy? |
|---|---|---|
| `policy` | Previous action | Yes |
| `proprio` | Robot joint positions (true proprioception) | Yes |
| `contact` | Per-fingertip contact forces (robot-dependent; only present when the robot's contact sensors are enabled) | Only in the `state` preset |
| `state` | Deployable, observable task state: object/articulation pose, joint positions, derived geometry (e.g., tilt angle). Contains **no velocities**. | Only in the `state` preset |
| `privileged` | Simulation-only quantities: joint/object linear and angular velocities, fingertip and palm pose/velocity. Cheap in sim, unrealistic to obtain on hardware. | No — recorded for logging/analysis and asymmetric-critic use only |
| `goal` | Target/command observations (e.g., target object pose), populated by task-specific command generators | Yes, when the task defines a goal |
| `rgb` | RGB images from one or more cameras | Yes, in vision presets |
| `depth` | Depth images (distance-to-image-plane) from one or more cameras | Yes, in vision presets |
| `pointcloud` | Back-projected 3D point cloud from camera depth | Yes, in point-cloud presets |
| `scene_vis` | Zero-sized terms whose sole purpose is to drive visualization markers (goal spheres, frame axes, zone boundaries); never delivered to a policy | No |
| `debug_vis` | Additional visualization-only terms, opt-in via `enable_debug_vis` (e.g., for teleoperation/inspection tooling) | No (off by default) |

Two design choices are worth calling out explicitly for readers of the
benchmark:

- **`state` vs. `privileged`.** DexVerse distinguishes *state* — task
  quantities a real deployment could plausibly observe (poses, joint
  positions, derived geometric features) — from *privileged* information —
  quantities that are trivial to read in simulation but require additional
  instrumentation on hardware (velocities of the object, fingertips, and
  palm). This split allows evaluating policies under a deployment-realistic
  observation budget on the one hand, and training privileged/asymmetric
  critics or teacher policies with full simulator state on the other.
- **Contact/tactile is confined to the `state` preset.** Contact-sensor
  readings are included only when the state-based observation preset is
  active, reflecting the sim-to-real gap in force-sensing fidelity relative
  to vision.

## 2. Observation presets

Because each task config can populate any subset of the groups above,
DexVerse provides a small number of named **presets** that select a
consistent, ready-to-use observation configuration for a given policy class.
Presets are applied by nulling every group not on their allow-list and by
setting a temporal history length on the `policy`/`proprio` groups (frame
stacking, used by vision-based policies to compensate for not having access
to privileged velocity information):

| Preset | Enabled groups | History length | Cameras |
|---|---|---|---|
| `state` | `policy`, `proprio`, `contact`, `state`, `goal` | 0 | — |
| `rgb` | `policy`, `proprio`, `goal`, `rgb` | 3 | single front view |
| `rgb_depth` (alias `rgbd`) | `policy`, `proprio`, `goal`, `rgb`, `depth` | 3 | single front view |
| `pointcloud` | `policy`, `proprio`, `goal`, `pointcloud` | 3 | single front view |
| `3view_rgb` | same as `rgb` | 3 | 3 views |
| `3view_rgb_depth` (alias `3view_rgbd`) | same as `rgb_depth` | 3 | 3 views |
| `3view_pointcloud` | same as `pointcloud` | 3 | 3 views |

A user selects a preset by name (e.g., `observation_preset="rgb_depth"`) when
constructing the environment configuration; this determines which terms are
retained in the final `ObservationsCfg` without requiring the user to
manually zero out unused groups. Note that only the `state` preset exposes
the `state` and `contact` groups — the vision-based presets rely purely on
raw sensory input plus proprioceptive/action history rather than
privileged or ground-truth task state.

## 3. Frame conventions

DexVerse does not use a single global observation frame; instead, each term
documents the frame it is expressed in, and most low-dimensional state terms
are expressed **relative to the robot's root (base) frame** rather than the
world frame:

- **Robot-root-frame terms** (suffix `_b`): object position/orientation
  (`object_pos_b`, `object_quat_b`), object linear/angular velocity, object
  surface point samples, and hand/fingertip pose and velocity
  (`hand_tips_state_b`) are all expressed relative to the robot's root pose,
  computed as `R_root^{-1} (x_world - p_root)`. This keeps the observation
  invariant to the robot's placement in the scene.
- **Configurable reference-frame terms.** Some terms (e.g., articulated-body
  pose/velocity used by drawer/door-style tasks) take an explicit
  `base_asset_cfg` parameter, so the "base" frame is not hardcoded to the
  robot — articulation tasks report body pose relative to the **table**
  frame rather than the robot frame.
- **World-frame terms.** Camera-derived point clouds (`camera_point_cloud_w`,
  `merged_camera_point_cloud_w`) are returned in the **world frame** by
  default, since they are computed by back-projecting rendered depth and
  cropping to a table-relative bounding box; a camera-frame variant
  (`camera_point_cloud_c`, using the ROS camera convention) is also available
  for methods that expect points expressed relative to the camera.
- **Goal/command frame.** Target poses produced by the pose-command
  generator can be defined in either the robot base frame or the world
  frame, controlled by a `use_world_frame` flag on the command
  configuration (task-dependent; e.g., relocation tasks use world-frame
  targets). Regardless of the sampling frame, target visualization and
  success metrics are always computed in world frame.
- **Orientation convention.** All quaternions are expressed as `(w, x, y, z)`.

## 4. Camera / visual observations

DexVerse provides three camera viewpoints, each independently toggleable:

- **Third-person camera** — a fixed overhead-angled view at world position
  `(-1.5, 0, 1.5)` looking down at the table, `640×480` resolution, RGB +
  depth (`distance_to_image_plane`), pinhole projection with a
  `[0.1, 10.0]` m clipping range.
- **Wrist camera** — mounted on the robot's palm link, `128×128` resolution,
  RGB + depth, ROS camera convention, `[0.01, 2.0]` m clipping range. For
  bimanual robots, separate left/right wrist cameras are used.
- **Side third-person cameras** (optional, `multiview_cameras=True` or a
  `3view_*` preset) — two additional world-frame viewpoints for multi-view
  policies.

RGB observations are stored as raw uint8 (`normalize=False`) rather than
Isaac Lab's default zero-centered normalization, so that recorded
demonstrations remain directly viewable as ordinary images; policies that
expect centered/normalized pixel input should normalize at the model
boundary.

The `pointcloud` group back-projects a camera's depth buffer into 3D using
the camera intrinsics, crops the result to a table-relative bounding box
(configurable XY inset and Z range above the table surface), and
subsamples/pads to a fixed number of points (4096 by default) in world
frame. This differs from a ground-truth object point cloud in that it
reflects only the surfaces actually visible to the camera (i.e., it respects
occlusion), which is important when comparing perception-based methods
against methods with access to privileged object geometry.

## 5. Practical notes for users

- Because `state`, `privileged`, `rgb`, `depth`, and `pointcloud` are
  mutually orthogonal groups that can each be independently populated or
  nulled, users comparing policy classes across modalities (state-based,
  RGB(-D), point-cloud) should use the corresponding named preset rather
  than hand-assembling an observation config, to ensure a fair and
  consistent choice of history length and included groups.
- The `privileged`/`state` split is the key mechanism for restricting a
  policy to deployment-realistic information; any comparison intended to
  reflect real-robot feasibility should use the `state` preset (or a vision
  preset) rather than the full, unfiltered `ObservationsCfg`.
- Because most task-relevant scalar/vector observations are given in the
  robot root frame while point-cloud observations default to the world
  frame, users combining low-dimensional state with point-cloud input in a
  single policy should be aware they are not expressed in the same frame
  and may need to transform one into the other's frame before fusing them.

*(Source: `source/dexverse/dexverse/tasks/dexverse_base_env_cfg.py`,
`ObservationsCfg` and the `_OBSERVATION_PRESETS` table; observation-term
implementations and frame documentation in
`source/dexverse/dexverse/tasks/mdp/observations.py`.)*
