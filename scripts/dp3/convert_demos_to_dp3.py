#!/usr/bin/env python3
# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert ``create_demo_files_sequential.py`` HDF5 demos into DP3-ready HDF5 format.

The input HDF5 is produced by ``scripts/demo_tools/create_demo_files_sequential.py`` with at least
the ``pointcloud`` and ``proprio`` observation groups captured (typically via
``--obs-groups pointcloud`` or the ``pointcloud`` / ``3view_pointcloud`` preset).

Expected input layout::

    /                              attrs: task, observation_preset, num_episodes, ...
      data/
        demo_<i>/                  attrs: episode_index, success, num_samples, ...
          actions                  (T, action_dim)  float32
          obs/
            pointcloud/
              camera_point_cloud_w (T, num_points_in, 3)   single-view, or
              merged_point_cloud_w (T, num_points_in, 3)   3-view preset
            proprio/
              <term>               (T, *)                  one or more terms

Processing pipeline per episode:
  1. Extract the point cloud term (single- or multi-view) and FPS-downsample
     each frame from ``num_points_in`` to ``--num_points``.
  2. Concatenate all proprio terms (sorted by key) into a flat ``agent_pos``.
  3. Store point_cloud, agent_pos, and action as ``(T, *)`` arrays per episode.
  4. Compute the workspace AABB over all valid points and store it as
     dataset-level metadata for normalization.

Usage::

    # Step 1: Download teleop demos (HF dataset is gated; log in first):
    python scripts/demo_tools/download_demos.py --task functional/Dexverse-GraspPan-v0

    # Step 2: Replay headlessly and write HDF5 with the pointcloud preset:
    python scripts/demo_tools/create_demo_files_sequential.py \\
        --task functional/Dexverse-GraspPan-v0 \\
        --obs-groups pointcloud \\
        --output-dir outputs/demos_h5

    # Step 3: Convert to DP3 HDF5 (this script):
    python scripts/dp3/convert_demos_to_dp3.py \\
        --input_h5 outputs/demos_h5/functional/Dexverse-GraspPan-v0/Dexverse-GraspPan-v0.pointcloud.seq.demo.h5 \\
        --out_file outputs/dp3_graspPan.hdf5 \\
        --num_points 512
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import h5py
import numpy as np
import torch

from dexverse.recording_compat import recorded_task_name

# Observation paths inside a ``create_demo_files_sequential.py`` HDF5 demo group.
_POINTCLOUD_GROUP = "obs/pointcloud"
_PROPRIO_GROUP = "obs/proprio"
_POINTCLOUD_TERM_PRIORITY = ("camera_point_cloud_w", "merged_point_cloud_w")
_ACTION_KEY = "actions"


def _fps_downsample_np(points: np.ndarray, num_points: int) -> np.ndarray:
    """Farthest-point sample a ``(N, 3)`` numpy array down to ``num_points``."""
    N = points.shape[0]
    if N <= num_points:
        if N == 0:
            return np.zeros((num_points, 3), dtype=points.dtype)
        pad_idx = np.random.choice(N, num_points - N, replace=True)
        return np.concatenate([points, points[pad_idx]], axis=0)

    pts = torch.from_numpy(points).float()
    try:
        import pytorch3d.ops as torch3d_ops

        sampled, _ = torch3d_ops.sample_farthest_points(pts.unsqueeze(0), K=num_points)
        return sampled.squeeze(0).numpy()
    except ImportError:
        pass

    selected = np.zeros(num_points, dtype=np.int64)
    dists = np.full(N, np.inf, dtype=np.float64)
    selected[0] = np.random.randint(N)
    for i in range(1, num_points):
        last = points[selected[i - 1]]
        d = np.sum((points - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = np.argmax(dists)
    return points[selected]


def _resolve_pointcloud_term(demo_group: h5py.Group) -> str:
    """Return the pointcloud term name available in this demo, preferring single-view."""
    if _POINTCLOUD_GROUP not in demo_group:
        raise KeyError(
            f"Demo group has no '{_POINTCLOUD_GROUP}'. "
            "Run create_demo_files_sequential.py with --obs-groups pointcloud (or the "
            "'pointcloud' / '3view_pointcloud' preset) first."
        )
    pcd_group = demo_group[_POINTCLOUD_GROUP]
    for term in _POINTCLOUD_TERM_PRIORITY:
        if term in pcd_group:
            return term
    available = list(pcd_group.keys())
    raise KeyError(
        f"None of {_POINTCLOUD_TERM_PRIORITY} found under '{_POINTCLOUD_GROUP}'. Available terms: {available}"
    )


def _load_pointcloud(demo_group: h5py.Group, term_name: str) -> np.ndarray:
    """Load the per-frame pointcloud array. Reshape to (T, N, 3) regardless of flatten."""
    arr = demo_group[f"{_POINTCLOUD_GROUP}/{term_name}"][...]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 2:
        # (T, N*3) flat → (T, N, 3)
        T = arr.shape[0]
        if arr.shape[1] % 3 != 0:
            raise ValueError(f"Pointcloud term {term_name!r} has shape {arr.shape}; trailing dim not divisible by 3.")
        return arr.reshape(T, -1, 3)
    raise ValueError(f"Unexpected pointcloud shape for term {term_name!r}: {arr.shape}")


def _load_proprio(demo_group: h5py.Group) -> np.ndarray:
    """Concatenate every proprio term (sorted by key) into a flat ``(T, D)`` agent_pos.

    Sorting keeps the concat order deterministic across episodes. With the
    default ``pointcloud`` preset there is only one term (``joint_pos``),
    so ordering is moot — but custom setups can append more terms.
    """
    if _PROPRIO_GROUP not in demo_group:
        raise KeyError(
            f"Demo group has no '{_PROPRIO_GROUP}'. "
            "create_demo_files_sequential.py must be run with the proprio group enabled."
        )
    proprio_group = demo_group[_PROPRIO_GROUP]
    term_names = sorted(proprio_group.keys())
    if not term_names:
        raise ValueError(f"'{_PROPRIO_GROUP}' is empty in this demo.")

    parts = []
    T_ref = None
    for name in term_names:
        arr = np.asarray(proprio_group[name][...], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        else:
            arr = arr.reshape(arr.shape[0], -1)
        if T_ref is None:
            T_ref = arr.shape[0]
        elif arr.shape[0] != T_ref:
            raise ValueError(f"Proprio term {name!r} has T={arr.shape[0]} but expected {T_ref}.")
        parts.append(arr)
    return np.concatenate(parts, axis=1)


def _load_actions(demo_group: h5py.Group) -> np.ndarray:
    if _ACTION_KEY not in demo_group:
        raise KeyError(f"Demo group has no '{_ACTION_KEY}' dataset.")
    return np.asarray(demo_group[_ACTION_KEY][...], dtype=np.float32)


def _compute_bbox(point_clouds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute axis-aligned bbox over all valid points across a list of (T, N, 3) arrays."""
    all_mins: list[np.ndarray] = []
    all_maxs: list[np.ndarray] = []
    for pcd_seq in point_clouds:
        flat = pcd_seq.reshape(-1, 3)
        valid = np.isfinite(flat).all(axis=1) & (np.abs(flat).sum(axis=1) > 1e-6)
        if valid.any():
            v = flat[valid]
            all_mins.append(v.min(axis=0))
            all_maxs.append(v.max(axis=0))
    if not all_mins:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    return (
        np.min(all_mins, axis=0).astype(np.float32),
        np.max(all_maxs, axis=0).astype(np.float32),
    )


def convert(
    input_h5: str,
    out_file: str,
    num_points: int = 512,
    skip_failed: bool = True,
) -> dict:
    """Convert ``create_demo_files_sequential.py`` HDF5 into a DP3-ready HDF5."""
    in_path = Path(input_h5).expanduser().resolve()
    out_path = Path(out_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    pcd_arrays: list[np.ndarray] = []
    point_cloud = agent_pos = actions = None  # for summary

    with h5py.File(in_path, "r") as hf_in, h5py.File(out_path, "w") as hf_out:
        task = recorded_task_name(str(hf_in.attrs.get("task", "unknown")))
        observation_preset = str(hf_in.attrs.get("observation_preset", ""))
        demo_names = sorted(hf_in["data"].keys(), key=lambda n: int(n.split("_")[-1]))

        # Resolve pointcloud term from the first non-empty demo so we keep one
        # consistent term across the whole file.
        pcd_term_name = None
        for name in demo_names:
            group = hf_in["data"][name]
            if int(group.attrs.get("num_samples", 0)) > 0:
                pcd_term_name = _resolve_pointcloud_term(group)
                break
        if pcd_term_name is None:
            raise ValueError(f"No non-empty demos found in {in_path}")

        out_data = hf_out.create_group("data")
        hf_out.attrs["task"] = task
        hf_out.attrs["num_points"] = num_points
        hf_out.attrs["source_h5"] = str(in_path)
        hf_out.attrs["observation_preset"] = observation_preset
        hf_out.attrs["pointcloud_term"] = pcd_term_name

        for name in demo_names:
            group = hf_in["data"][name]
            num_samples = int(group.attrs.get("num_samples", 0))
            success = bool(group.attrs.get("success", False))
            has_success = bool(group.attrs.get("has_success_flag", False))
            if num_samples == 0:
                print(f"[convert] {name}: num_samples=0, skipping")
                continue
            if skip_failed and has_success and not success:
                print(f"[convert] {name}: success=False, skipping")
                continue

            pcd_seq_raw = _load_pointcloud(group, pcd_term_name)  # (T, N, 3)
            agent_pos = _load_proprio(group)  # (T, D)
            actions = _load_actions(group)  # (T, Da)
            T_pcd = pcd_seq_raw.shape[0]
            if not (T_pcd == agent_pos.shape[0] == actions.shape[0]):
                raise ValueError(
                    f"{name}: T mismatch — pcd={T_pcd}, proprio={agent_pos.shape[0]}, actions={actions.shape[0]}"
                )

            pcd_down = np.stack(
                [_fps_downsample_np(pcd_seq_raw[t], num_points) for t in range(T_pcd)],
                axis=0,
            )
            pcd_arrays.append(pcd_down)
            point_cloud = pcd_down

            ep_group = out_data.create_group(f"episode_{written}")
            obs_group = ep_group.create_group("obs")
            obs_group.create_dataset("point_cloud", data=pcd_down, compression="gzip", compression_opts=4)
            obs_group.create_dataset("agent_pos", data=agent_pos, compression="gzip", compression_opts=4)
            ep_group.create_dataset("action", data=actions, compression="gzip", compression_opts=4)
            ep_group.attrs["source_demo_name"] = name
            ep_group.attrs["source_episode_index"] = int(group.attrs.get("episode_index", written))
            ep_group.attrs["success"] = success
            ep_group.attrs["num_steps"] = num_samples
            written += 1

            if (written % 5 == 0) or (name == demo_names[-1]):
                print(
                    f"[convert] {name} -> episode_{written - 1}: "
                    f"T={T_pcd} pcd={pcd_down.shape} agent_pos={agent_pos.shape} action={actions.shape}"
                )

        if written == 0:
            raise ValueError(f"No episodes written from {in_path} (all empty or filtered).")

        bbox_min, bbox_max = _compute_bbox(pcd_arrays)
        hf_out.attrs["bbox_min"] = bbox_min
        hf_out.attrs["bbox_max"] = bbox_max
        hf_out.attrs["num_episodes"] = written

    summary = {
        "task": task,
        "observation_preset": observation_preset or None,
        "pointcloud_term": pcd_term_name,
        "num_episodes": written,
        "num_points": num_points,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "output_file": str(out_path),
        "point_cloud_shape": list(point_cloud.shape) if point_cloud is not None else None,
        "agent_pos_dim": int(agent_pos.shape[-1]) if agent_pos is not None else None,
        "action_dim": int(actions.shape[-1]) if actions is not None else None,
    }
    print(f"\n[convert] Done. Wrote {written} episodes to {out_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert create_demo_files_sequential.py HDF5 demos into DP3-ready HDF5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_h5",
        type=str,
        required=True,
        help="HDF5 produced by scripts/demo_tools/create_demo_files_sequential.py (with pointcloud + proprio groups).",
    )
    parser.add_argument("--out_file", type=str, required=True, help="Output DP3 HDF5 path.")
    parser.add_argument(
        "--num_points",
        type=int,
        default=512,
        help="Per-frame target point count after FPS downsampling (default: 512).",
    )
    parser.add_argument(
        "--include_failed",
        action="store_true",
        default=False,
        help="Include demos with success=False (default: skip them).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        input_h5=args.input_h5,
        out_file=args.out_file,
        num_points=args.num_points,
        skip_failed=not args.include_failed,
    )


if __name__ == "__main__":
    main()
