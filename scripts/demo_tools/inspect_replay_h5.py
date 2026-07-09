# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect & visualise a single trajectory in a replay HDF5 file.

This script reads the HDF5 file produced by ``replay_demos.py --save_obs_action_h5``
and, for a chosen episode:

    * prints file-level and episode-level attributes,
    * prints a one-line summary of every dataset (shape, dtype, quick stats),
    * plays the trajectory back live, step-by-step, showing:
        - RGB image streams in OpenCV windows,
        - depth streams as colourised OpenCV windows,
        - point-cloud streams in an Open3D window (or matplotlib fallback).

Detection is name- and shape-based. We look at the full dataset path (e.g.
``obs/policy/rgb``) -- anything matching /rgb|color|image|camera/ is treated as
RGB, /depth|distance/ as depth, /point.?cloud|points|pcd/ (or trailing axis of 3
with a matching name) as a point cloud. Anything else is printed as a compact
numeric summary per step.

Example::

    python scripts/demo_tools/inspect_replay_h5.py \
        --dataset_file merged.replay_obs.h5 \
        --episode 0 \
        --fps 15

Keyboard controls in the OpenCV windows:

    q / ESC  -- quit
    space    -- pause / resume
    n        -- advance one frame while paused
    b        -- rewind one frame while paused
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
import time
from typing import Any

import h5py
import numpy as np

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import open3d as o3d  # type: ignore

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import matplotlib.pyplot as plt  # noqa: E402

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _probe_cv2_highgui() -> bool:
    """Return True iff ``cv2.namedWindow`` actually works on this build."""
    if not HAS_CV2:
        return False
    try:
        cv2.namedWindow("__cv2_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__cv2_probe__")
        return True
    except Exception:
        return False


CV2_GUI_OK = _probe_cv2_highgui()


# ---------------------------------------------------------------------------
# Generic HDF5 walking helpers
# ---------------------------------------------------------------------------


def walk_leaf_datasets(group: h5py.Group, prefix: str = "") -> list[tuple[str, h5py.Dataset]]:
    """Return ``[(relative_path, dataset), ...]`` for every leaf dataset in ``group``."""
    out: list[tuple[str, h5py.Dataset]] = []
    for name, item in group.items():
        full = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Group):
            out.extend(walk_leaf_datasets(item, full))
        elif isinstance(item, h5py.Dataset):
            out.append((full, item))
    return out


def select_episode_group(file: h5py.File, episode: str | int) -> h5py.Group:
    """Pick the episode group from the HDF5 file.

    ``episode`` may be:
        * an integer -> looks up ``/data/demo_<episode>`` first, falls back to the
          N-th group by attribute ``episode_index``, then to the N-th listed.
        * a string   -> treated as a group name under ``/data`` (e.g. ``demo_3``).
    """
    if "data" not in file:
        raise ValueError("HDF5 file has no '/data' group; not a replay pickle output?")
    data = file["data"]
    group_names = list(data.keys())
    if isinstance(episode, str):
        if episode not in data:
            raise KeyError(f"Episode {episode!r} not in /data (available: {group_names}).")
        return data[episode]

    name_by_counter = f"demo_{int(episode)}"
    if name_by_counter in data:
        return data[name_by_counter]

    # Try matching by attr 'episode_index'.
    for name in group_names:
        g = data[name]
        if "episode_index" in g.attrs and int(g.attrs["episode_index"]) == int(episode):
            return g

    # Final fallback: index into the (insertion-ordered) list.
    if 0 <= int(episode) < len(group_names):
        return data[group_names[int(episode)]]

    raise KeyError(f"Cannot resolve episode {episode!r}. Available groups: {group_names}.")


# ---------------------------------------------------------------------------
# Term classification heuristics
# ---------------------------------------------------------------------------


_DEPTH_TOKENS = {"depth", "distance", "disparity"}
_RGB_TOKENS = {"rgb", "color", "colour", "image", "img", "camera"}
_PCD_TOKENS = {"pcd", "pointcloud", "points"}


def _path_tokens(path: str) -> set[str]:
    """Split a dataset path into lowercase alphanumeric tokens.

    Any non-alphanumeric character (``/``, ``_``, ``-``, ``.``) acts as a
    separator, so ``obs/perception/depth_image`` becomes
    ``{'obs', 'perception', 'depth', 'image'}``. This is much more forgiving
    than `/`-bounded regex matches for real-world key names like
    ``depth_image`` or ``rgb_image``.
    """
    return {t for t in re.split(r"[^a-z0-9]+", path.lower()) if t}


def classify_term(path: str, shape: tuple[int, ...], dtype: np.dtype) -> str:
    """Return one of ``rgb``, ``depth``, ``pointcloud``, ``numeric``, ``other``."""
    if len(shape) == 0:
        return "other"

    tokens = _path_tokens(path)
    tail = shape[1:]  # leading axis is (usually) time

    is_pcd_named = bool(tokens & _PCD_TOKENS) or ("point" in tokens and "cloud" in tokens)
    if is_pcd_named:
        if len(tail) >= 2 and tail[-1] == 3:
            return "pointcloud"
        if len(tail) == 1 and tail[0] % 3 == 0:
            return "pointcloud"

    is_depth_named = bool(tokens & _DEPTH_TOKENS)
    if is_depth_named:
        # (H, W), (H, W, 1), or (T, H, W, 1) style tails are all fine -- we squeeze
        # the channel axis at display time.
        if len(tail) in (2, 3):
            return "depth"

    is_rgb_named = bool(tokens & _RGB_TOKENS) and not is_depth_named
    if is_rgb_named:
        if len(tail) == 3 and (tail[-1] == 3 or tail[0] == 3):
            return "rgb"
        if len(tail) == 4 and (tail[-1] == 3 or tail[1] == 3):
            return "rgb"

    # Shape-only fallback for untagged RGB: (T, H, W, 3) float/uint.
    if len(tail) == 3 and tail[-1] == 3 and dtype.kind in ("u", "f"):
        return "rgb"

    return "numeric"


# ---------------------------------------------------------------------------
# Frame conversion helpers
# ---------------------------------------------------------------------------


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert ``(H, W, 3)`` / ``(3, H, W)`` / float arrays to ``HxWx3`` uint8 BGR for cv2."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= 1.0 + 1e-6 and lo >= -1e-6:
            arr = np.clip(arr * 255.0, 0, 255)
        else:
            # Fall back to per-frame normalisation.
            rng = hi - lo if hi > lo else 1.0
            arr = np.clip((arr - lo) * (255.0 / rng), 0, 255)
        arr = arr.astype(np.uint8)
    return arr[..., ::-1]  # RGB -> BGR for cv2


def depth_to_display(frame: np.ndarray) -> np.ndarray:
    """Collapse optional channel axis and colourise a single depth frame."""
    arr = np.asarray(frame).astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    valid = np.isfinite(arr) & (arr > 0)
    if not np.any(valid):
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    lo = float(np.percentile(arr[valid], 2))
    hi = float(np.percentile(arr[valid], 98))
    rng = max(hi - lo, 1e-6)
    norm = np.clip((arr - lo) / rng, 0.0, 1.0)
    norm = (norm * 255.0).astype(np.uint8)
    if HAS_CV2:
        return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    gray = np.stack([norm, norm, norm], axis=-1)
    return gray


def points_of_frame(frame: np.ndarray) -> np.ndarray:
    """Reshape a point-cloud frame down to ``(N, 3)``."""
    arr = np.asarray(frame).astype(np.float32)
    if arr.ndim == 1 and arr.size % 3 == 0:
        return arr.reshape(-1, 3)
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        return arr.reshape(-1, 3)
    # Last-resort trim to multiple of 3.
    flat = arr.reshape(-1)
    n = (flat.size // 3) * 3
    return flat[:n].reshape(-1, 3)


def _concat_frames_horizontal(frames: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of ``(H, W, 3)`` uint8 frames horizontally.

    Frames with mismatched heights are resampled to the max height while
    preserving aspect ratio (cv2 bilinear when available, nearest-neighbor
    numpy fallback otherwise). Returns the input unchanged when there is
    only one frame, so the single-camera case is a no-op.
    """
    if not frames:
        raise ValueError("no frames to concat")
    if len(frames) == 1:
        return frames[0]
    target_h = max(f.shape[0] for f in frames)
    resized: list[np.ndarray] = []
    for f in frames:
        if f.shape[0] == target_h:
            resized.append(f)
            continue
        scale = target_h / f.shape[0]
        new_w = max(1, int(round(f.shape[1] * scale)))
        if HAS_CV2:
            resized.append(cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_LINEAR))
        else:
            ys = np.linspace(0, f.shape[0] - 1, target_h).astype(np.int64)
            xs = np.linspace(0, f.shape[1] - 1, new_w).astype(np.int64)
            resized.append(f[ys[:, None], xs[None, :]])
    return np.concatenate(resized, axis=1)


def _camera_sort_key(path: str) -> tuple[int, str]:
    """Stable left → center → right → wrist ordering for camera term names.

    Used so a horizontal-concat strip reads ``left | center | right | wrist``
    instead of alphabetical-but-mixed. Falls back to leaf name for ties.
    """
    leaf = path.rsplit("/", 1)[-1].lower()
    if leaf.startswith("left_wrist_"):
        return (4, leaf)
    if leaf.startswith("right_wrist_"):
        return (5, leaf)
    if leaf.startswith("wrist_"):
        return (3, leaf)
    if leaf.startswith("left_"):
        return (0, leaf)
    if leaf.startswith("right_"):
        return (2, leaf)
    return (1, leaf)


def _group_by_parent(items: dict[str, h5py.Dataset]) -> dict[str, list[tuple[str, h5py.Dataset]]]:
    """Group ``{full_path: dataset}`` by the parent path so co-located image
    terms (e.g. all of ``obs/rgb/*``) get one merged viewer.

    Returns ``{parent_path: [(full_path, ds), ...]}`` with members sorted by
    :func:`_camera_sort_key`. Paths with no parent (top-level datasets) get
    their own group keyed by the path itself.
    """
    groups: dict[str, list[tuple[str, h5py.Dataset]]] = {}
    for path, ds in items.items():
        parts = path.split("/")
        parent = "/".join(parts[:-1]) if len(parts) > 1 else path
        groups.setdefault(parent, []).append((path, ds))
    for members in groups.values():
        members.sort(key=lambda pair: _camera_sort_key(pair[0]))
    return groups


def numeric_summary(arr: np.ndarray) -> str:
    """Short one-line summary for logging small numeric observations."""
    a = np.asarray(arr)
    if a.size == 0:
        return "<empty>"
    if a.size <= 8 and a.ndim <= 1:
        return np.array2string(a, precision=3, suppress_small=True)
    a_float = a.astype(np.float32, copy=False)
    return (
        f"shape={tuple(a.shape)} dtype={a.dtype} "
        f"min={a_float.min():+.3f} max={a_float.max():+.3f} "
        f"mean={a_float.mean():+.3f}"
    )


# ---------------------------------------------------------------------------
# Point-cloud visualiser (Open3D if available, matplotlib otherwise)
# ---------------------------------------------------------------------------


class PointCloudViewerO3D:
    """Single Open3D window that gets updated every frame for one point-cloud term."""

    def __init__(self, title: str):
        if not HAS_OPEN3D:
            raise RuntimeError("Open3D not available.")
        self._pcd = o3d.geometry.PointCloud()
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name=title, width=720, height=540)
        opt = self._vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.1])
        opt.point_size = 3.0
        self._added = False

    def update(self, points: np.ndarray) -> None:
        if points.size == 0:
            return
        self._pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        # Colour by height so static scenes are still readable.
        z = points[:, 2]
        if z.max() > z.min():
            t = (z - z.min()) / (z.max() - z.min())
        else:
            t = np.zeros_like(z)
        colors = np.stack([t, 1.0 - t, 0.5 + 0.5 * np.sin(4 * t)], axis=-1)
        self._pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
        if not self._added:
            self._vis.add_geometry(self._pcd)
            self._added = True
        else:
            self._vis.update_geometry(self._pcd)
        self._vis.poll_events()
        self._vis.update_renderer()

    def close(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover - best-effort cleanup
            self._vis.destroy_window()


class PointCloudViewerMPL:
    """matplotlib 3D scatter fallback (slower, non-interactive camera)."""

    def __init__(self, title: str):
        if not HAS_MPL:
            raise RuntimeError("matplotlib not available.")
        plt.ion()
        self._fig = plt.figure(title, figsize=(5, 4))
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_title(title)
        self._scatter = None

    def update(self, points: np.ndarray) -> None:
        if points.size == 0:
            return
        if points.shape[0] > 4000:
            idx = np.random.choice(points.shape[0], 4000, replace=False)
            points = points[idx]
        if self._scatter is not None:
            self._scatter.remove()
        self._scatter = self._ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], s=1.5, c=points[:, 2], cmap="viridis"
        )
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_zlabel("z")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            plt.close(self._fig)


def make_pcd_viewer(title: str):
    if HAS_OPEN3D:
        return PointCloudViewerO3D(title)
    if HAS_MPL:
        return PointCloudViewerMPL(title)
    return None


# ---------------------------------------------------------------------------
# Image / depth stream viewers (cv2 HighGUI -> matplotlib -> PNG-on-disk)
# ---------------------------------------------------------------------------


class _ImageStreamCv2:
    """OpenCV HighGUI window (``imshow`` + ``waitKey``)."""

    def __init__(self, title: str):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(self, frame_bgr: np.ndarray) -> None:
        cv2.imshow(self.title, frame_bgr)

    def close(self) -> None:
        with contextlib.suppress(Exception):  # pragma: no cover - best-effort cleanup
            cv2.destroyWindow(self.title)


class _ImageStreamMpl:
    """matplotlib fallback for environments without cv2 HighGUI."""

    def __init__(self, title: str):
        if not HAS_MPL:
            raise RuntimeError("matplotlib not available.")
        plt.ion()
        self._fig = plt.figure(title, figsize=(6, 4.5))
        self._ax = self._fig.add_subplot(111)
        self._ax.set_title(title)
        self._ax.axis("off")
        self._im = None
        self._key_buffer: list[str] = []
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        if event.key is not None:
            self._key_buffer.append(event.key)

    def show(self, frame_bgr: np.ndarray) -> None:
        # frame_bgr is BGR (cv2 convention); matplotlib expects RGB.
        if frame_bgr.ndim == 3 and frame_bgr.shape[-1] == 3:
            frame = frame_bgr[..., ::-1]
        else:
            frame = frame_bgr
        if self._im is None:
            self._im = self._ax.imshow(frame)
        else:
            self._im.set_data(frame)
        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:  # pragma: no cover - mpl backend quirks
            pass

    def poll_key(self) -> str | None:
        if self._key_buffer:
            return self._key_buffer.pop(0)
        return None

    def close(self) -> None:
        with contextlib.suppress(Exception):
            plt.close(self._fig)


class _ImageStreamDisk:
    """Headless PNG dumper: writes one file per frame."""

    def __init__(self, title: str, out_dir: str):
        self.title = title
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")
        self._dir = os.path.join(out_dir, safe)
        os.makedirs(self._dir, exist_ok=True)
        self._frame_idx = 0

    def show(self, frame_bgr: np.ndarray) -> None:
        path = os.path.join(self._dir, f"frame_{self._frame_idx:06d}.png")
        self._frame_idx += 1
        if HAS_CV2:
            cv2.imwrite(path, frame_bgr)
        elif HAS_MPL:
            if frame_bgr.ndim == 3 and frame_bgr.shape[-1] == 3:
                frame = frame_bgr[..., ::-1]
            else:
                frame = frame_bgr
            plt.imsave(path, frame)

    def close(self) -> None:
        pass


def make_image_viewer(title: str, *, prefer: str, save_dir: str | None):
    """Construct the best available image/depth viewer.

    ``prefer`` is one of ``"cv2"``, ``"mpl"``, ``"disk"``.  ``"disk"`` means
    "write PNGs to ``save_dir``"; it disables any live display.
    """
    if prefer == "disk":
        if save_dir is None:
            raise ValueError("save_dir must be provided when prefer='disk'.")
        return _ImageStreamDisk(title, save_dir)
    if prefer == "cv2" and CV2_GUI_OK:
        return _ImageStreamCv2(title)
    if HAS_MPL:
        return _ImageStreamMpl(title)
    if save_dir is not None and (HAS_CV2 or HAS_MPL):
        return _ImageStreamDisk(title, save_dir)
    return None


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _format_attr(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode()
        except Exception:
            return str(value)
    if isinstance(value, np.ndarray):
        return np.array2string(value, max_line_width=120, precision=3)
    return str(value)


def print_file_summary(file: h5py.File) -> None:
    print("=" * 78)
    print(f"HDF5 file : {file.filename}")
    print("=" * 78)
    print("Root attrs:")
    for k in sorted(file.attrs):
        print(f"  {k:<24}: {_format_attr(file.attrs[k])}")
    if "data" in file:
        print(f"Episodes: {len(file['data'])}")
        for name in file["data"]:
            g = file["data"][name]
            ep_idx = g.attrs.get("episode_index", "?")
            env_id = g.attrs.get("env_id", "?")
            T = g.attrs.get("num_samples", "?")
            succ = g.attrs.get("success", "?")
            print(f"  {name:<12} episode_index={ep_idx} env_id={env_id} num_samples={T} success={succ}")


def print_episode_summary(group: h5py.Group) -> None:
    print("-" * 78)
    print(f"Episode group: {group.name}")
    print("-" * 78)
    print("Episode attrs:")
    for k in sorted(group.attrs):
        print(f"  {k:<24}: {_format_attr(group.attrs[k])}")

    leaves = walk_leaf_datasets(group)
    print(f"\nDatasets ({len(leaves)}):")
    for path, ds in leaves:
        shape = tuple(ds.shape)
        try:
            cat = classify_term(path, shape, ds.dtype)
            # Quick stats on a representative frame (t=0) if time-dimension present.
            extra = ""
            if ds.size > 0 and len(shape) >= 1:
                try:
                    sample = ds[0]
                    extra = f"  first-frame {numeric_summary(np.asarray(sample))}"
                except Exception:
                    extra = ""
        except Exception:
            cat = "?"
            extra = ""
        print(f"  [{cat:>11}] {path:<40} shape={shape} dtype={ds.dtype}{extra}")


# ---------------------------------------------------------------------------
# Live playback
# ---------------------------------------------------------------------------


_SNAPSHOT_PREFIXES = ("initial_obs/", "final_obs/")


def build_visuals_and_numerics(
    group: h5py.Group,
    only_terms: list[str] | None,
    *,
    include_next_obs: bool = False,
) -> tuple[
    dict[str, h5py.Dataset],
    dict[str, h5py.Dataset],
    dict[str, h5py.Dataset],
    dict[str, h5py.Dataset],
]:
    """Classify per-step datasets in ``group`` into rgb / depth / pointcloud / numeric.

    ``initial_obs/*`` and ``final_obs/*`` snapshot groups are dropped (they have no
    time axis). ``next_obs/*`` visualisers are dropped by default because they are
    near-duplicates of the corresponding ``obs/*`` streams shifted by one step;
    pass ``include_next_obs=True`` to keep them.

    Only datasets whose leading axis matches the episode length
    (``group.attrs['num_samples']``) are used for live playback.
    """
    rgbs: dict[str, h5py.Dataset] = {}
    depths: dict[str, h5py.Dataset] = {}
    pcds: dict[str, h5py.Dataset] = {}
    numerics: dict[str, h5py.Dataset] = {}

    expected_T: int | None = None
    if "num_samples" in group.attrs:
        expected_T = int(group.attrs["num_samples"])

    for path, ds in walk_leaf_datasets(group):
        if only_terms and not any(tok.lower() in path.lower() for tok in only_terms):
            continue
        if ds.ndim < 1 or ds.shape[0] == 0:
            continue
        # Drop single-frame snapshots.
        if any(path.startswith(pfx) for pfx in _SNAPSHOT_PREFIXES):
            continue
        # Require a per-step leading axis when we know the episode length.
        if expected_T is not None and int(ds.shape[0]) != expected_T:
            continue
        cat = classify_term(path, tuple(ds.shape), ds.dtype)
        drop_duplicate_stream = (
            not include_next_obs and path.startswith("next_obs/") and cat in {"rgb", "depth", "pointcloud"}
        )
        if drop_duplicate_stream:
            continue
        if cat == "rgb":
            rgbs[path] = ds
        elif cat == "depth":
            depths[path] = ds
        elif cat == "pointcloud":
            pcds[path] = ds
        else:
            numerics[path] = ds
    return rgbs, depths, pcds, numerics


def play_episode(  # noqa: C901
    group: h5py.Group,
    fps: float = 15.0,
    only_terms: list[str] | None = None,
    max_numeric_prints: int = 8,
    include_next_obs: bool = False,
    save_frames_dir: str | None = None,
) -> None:
    rgbs, depths, pcds, numerics = build_visuals_and_numerics(group, only_terms, include_next_obs=include_next_obs)
    T = (
        min(ds.shape[0] for ds in {**rgbs, **depths, **pcds, **numerics}.values())
        if (rgbs or depths or pcds or numerics)
        else 0
    )
    if T == 0:
        print("Nothing to play back (no selected per-step datasets found).")
        return

    # Decide the image/depth display backend.
    if save_frames_dir:
        prefer = "disk"
        os.makedirs(save_frames_dir, exist_ok=True)
        image_backend = "disk"
    elif CV2_GUI_OK:
        prefer = "cv2"
        image_backend = "cv2"
    elif HAS_MPL:
        prefer = "mpl"
        image_backend = "matplotlib"
    elif HAS_CV2:
        prefer = "disk"
        # Auto-dump alongside the HDF5 file so the user still gets *something*.
        save_frames_dir = os.path.join(
            os.path.dirname(os.path.abspath(group.file.filename)) or ".",
            f"inspect_frames_{os.path.basename(group.name.strip('/')) or 'demo'}",
        )
        os.makedirs(save_frames_dir, exist_ok=True)
        image_backend = f"disk -> {save_frames_dir}"
    else:
        prefer = "none"
        image_backend = "none"

    # Group multi-camera datasets by their parent path (e.g. ``obs/rgb``,
    # ``next_obs/depth``) so a 3-view preset shows as one horizontally-stitched
    # viewer instead of three separate windows. Single-camera setups end up
    # with a single-member group and behave identically to the old per-term
    # viewer path.
    rgb_groups = _group_by_parent(rgbs)
    depth_groups = _group_by_parent(depths)

    have_image_streams = bool(rgb_groups or depth_groups) and prefer != "none"
    rgb_viewers: dict[str, Any] = {}
    depth_viewers: dict[str, Any] = {}
    if have_image_streams:
        for title in rgb_groups:
            viewer = make_image_viewer(title, prefer=prefer, save_dir=save_frames_dir)
            if viewer is not None:
                rgb_viewers[title] = viewer
        for title in depth_groups:
            viewer = make_image_viewer(title, prefer=prefer, save_dir=save_frames_dir)
            if viewer is not None:
                depth_viewers[title] = viewer
        have_image_streams = bool(rgb_viewers or depth_viewers)

    # Point-cloud viewers.
    pcd_viewers = {path: make_pcd_viewer(path) for path in pcds}
    pcd_backend = "open3d" if HAS_OPEN3D else ("matplotlib" if HAS_MPL else "none")

    def _members_label(members: list[tuple[str, h5py.Dataset]]) -> str:
        return " | ".join(m[0].rsplit("/", 1)[-1] for m in members)

    print("\n" + "=" * 78)
    print(f"Playing back {T} step(s)")
    print(f"  image backend    : {image_backend}")
    if rgb_groups:
        for parent, members in rgb_groups.items():
            print(f"  rgb group        : {parent}  [{_members_label(members)}]")
    else:
        print("  rgb groups       : (none)")
    if depth_groups:
        for parent, members in depth_groups.items():
            print(f"  depth group      : {parent}  [{_members_label(members)}]")
    else:
        print("  depth groups     : (none)")
    print(f"  point-cloud terms: {list(pcds)}  (backend={pcd_backend})")
    print(f"  numeric terms    : {len(numerics)} (first {max_numeric_prints} printed each step)")
    if prefer == "cv2":
        print("  keys (image window): q/ESC=quit, space=pause/resume, n=step fwd, b=step back")
    elif prefer == "mpl":
        print("  keys (matplotlib window): q=quit, space=pause/resume, n=step fwd, b=step back")
    else:
        print("  non-interactive mode; press Ctrl-C to stop.")
    print("=" * 78)

    # Numeric term order: surface actions first, then a sorted preview.
    priority = [p for p in ("actions", "source_actions") if p in numerics]
    extras = sorted(p for p in numerics if p not in priority)
    numeric_paths = (priority + extras)[:max_numeric_prints]

    dt = 1.0 / max(fps, 1e-3)
    t = 0
    paused = False
    last_draw = time.time()

    while 0 <= t < T:
        # Update image/depth viewers — one composited frame per parent group.
        for title, members in rgb_groups.items():
            viewer = rgb_viewers.get(title)
            if viewer is None:
                continue
            try:
                frames = [to_uint8_rgb(ds[t]) for _, ds in members]
                viewer.show(_concat_frames_horizontal(frames))
            except Exception as err:
                print(f"[warn] failed to render RGB group {title!r}: {err}")
        for title, members in depth_groups.items():
            viewer = depth_viewers.get(title)
            if viewer is None:
                continue
            try:
                frames = [depth_to_display(ds[t]) for _, ds in members]
                viewer.show(_concat_frames_horizontal(frames))
            except Exception as err:
                print(f"[warn] failed to render depth term {title!r}: {err}")

        # Update point-cloud viewers.
        for title, ds in pcds.items():
            viewer = pcd_viewers.get(title)
            if viewer is None:
                continue
            try:
                viewer.update(points_of_frame(ds[t]))
            except Exception as err:
                print(f"[warn] failed to render pcd term {title!r}: {err}")

        # Print numeric summary for this step.
        header = f"\n[t={t:4d}/{T - 1}]"
        if paused:
            header += " (paused)"
        sys.stdout.write(header + "\n")
        for path in numeric_paths:
            frame = numerics[path][t]
            sys.stdout.write(f"  {path:<40} {numeric_summary(frame)}\n")
        sys.stdout.flush()

        # Pacing + key polling per backend.
        key_char: str | None = None
        if prefer == "cv2" and have_image_streams:
            wait_ms = 1 if paused else max(1, int(dt * 1000))
            k = cv2.waitKey(wait_ms) & 0xFF
            if k != 255:
                if k == 27:
                    key_char = "q"
                else:
                    key_char = chr(k) if 32 <= k < 127 else None
        elif prefer == "mpl" and (rgb_viewers or depth_viewers):
            plt.pause(0.001 if paused else dt)
            for v in list(rgb_viewers.values()) + list(depth_viewers.values()):
                if isinstance(v, _ImageStreamMpl):
                    k = v.poll_key()
                    if k is not None:
                        key_char = k
                        break
        elif pcds and HAS_MPL and not HAS_OPEN3D:
            plt.pause(0.001 if paused else dt)
        else:
            elapsed = time.time() - last_draw
            if not paused and elapsed < dt:
                time.sleep(dt - elapsed)
        last_draw = time.time()

        if key_char in ("q", "escape"):
            break
        if key_char == " ":
            paused = not paused
            continue
        if paused:
            if key_char == "n":
                t = min(T - 1, t + 1)
                continue
            if key_char == "b":
                t = max(0, t - 1)
                continue
            continue
        t += 1

    for viewer in list(rgb_viewers.values()) + list(depth_viewers.values()):
        with contextlib.suppress(Exception):
            viewer.close()
    if prefer == "cv2":
        with contextlib.suppress(Exception):
            cv2.destroyAllWindows()
    for viewer in pcd_viewers.values():
        if viewer is not None:
            viewer.close()

    if save_frames_dir:
        print(f"\n[inspect] image/depth frames saved under: {save_frames_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and visualise one trajectory from a replay HDF5 file.",
    )
    parser.add_argument("--dataset_file", required=True, type=str, help="Path to replay .h5 / .hdf5 file.")
    parser.add_argument(
        "--episode",
        default="0",
        type=str,
        help=(
            "Episode to inspect. Integer N -> group 'demo_<N>', falls back to the N-th "
            "group by attribute 'episode_index', then to the N-th listed group. "
            "A string (e.g. 'demo_3') is used verbatim."
        ),
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Live playback frames per second.")
    parser.add_argument(
        "--terms",
        nargs="+",
        default=None,
        help="Only show datasets whose path contains any of these (case-insensitive) tokens.",
    )
    parser.add_argument("--list", action="store_true", help="Only list structure; skip live playback.")
    parser.add_argument(
        "--max_numeric_prints",
        type=int,
        default=8,
        help="Cap on numeric terms printed each playback step.",
    )
    parser.add_argument(
        "--include_next_obs",
        action="store_true",
        help=(
            "Also open image/depth/point-cloud viewers for next_obs/* streams. "
            "By default these are skipped because they are near-duplicates of obs/*."
        ),
    )
    parser.add_argument(
        "--save_frames_dir",
        type=str,
        default=None,
        help=(
            "If set, write RGB/depth frames as PNGs into this directory instead of "
            "opening live windows. Useful on headless hosts where cv2 HighGUI is "
            "unavailable (e.g. OpenCV built without GTK support)."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset_file):
        raise FileNotFoundError(args.dataset_file)

    try:
        episode_arg: str | int = int(args.episode)
    except ValueError:
        episode_arg = args.episode

    with h5py.File(args.dataset_file, "r") as file:
        print_file_summary(file)
        group = select_episode_group(file, episode_arg)
        print_episode_summary(group)
        if args.list:
            return
        play_episode(
            group,
            fps=args.fps,
            only_terms=args.terms,
            max_numeric_prints=args.max_numeric_prints,
            include_next_obs=args.include_next_obs,
            save_frames_dir=args.save_frames_dir,
        )


if __name__ == "__main__":
    main()
