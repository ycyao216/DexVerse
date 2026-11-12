#!/usr/bin/env python3
"""Convert sequential DexVerse HDF5 demonstrations to a LeRobot Dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Map HDF5 camera names to LeRobot image feature names.
CAMERAS = {
    "rgb_image": "observation.images.rgb_image",
    "left_rgb_image": "observation.images.left_rgb_image",
    "right_rgb_image": "observation.images.right_rgb_image",
    "wrist_rgb_image": "observation.images.wrist_rgb_image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DexVerse HDF5 demos to LeRobot Dataset")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing HDF5 files")
    parser.add_argument("--output-root", type=Path, required=True, help="LeRobot Dataset output directory")
    parser.add_argument("--repo-id", default="Dexverse-PickCube-v0", help="LeRobot Dataset identifier")
    parser.add_argument("--fps", type=int, default=60, help="Dataset frame rate")
    parser.add_argument("--state-dim", type=int, default=28, help="State dimensions retained from proprio history")
    parser.add_argument("--task", default=None, help="Task text; defaults to the HDF5 task attribute")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


def build_features(state_dim: int) -> dict:
    """Define numeric and video features for the output dataset."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (28,),
            "names": [f"action_{i}" for i in range(28)],
        },
    }
    for key in CAMERAS.values():
        features[key] = {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def iter_episodes(input_dir: Path):
    """Iterate over episodes in deterministic file and episode order."""
    for h5_path in sorted(input_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as h5:
            task = h5.attrs.get("task", "Dexverse-PickCube-v0")
            if isinstance(task, bytes):
                task = task.decode()
            for episode_name in sorted(h5["data"].keys(), key=lambda x: int(x.split("_")[-1])):
                yield h5_path, episode_name, str(task)


def main() -> int:
    args = parse_args()
    h5_files = sorted(args.input_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in: {args.input_dir}")
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_root,
        fps=args.fps,
        features=build_features(args.state_dim),
        robot_type="dexverse_shadow_hand",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=1,
        # H.264 provides broad compatibility across players and training tools.
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="medium", g=2),
    )

    episode_count = 0
    frame_count = 0
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as h5:
            task_attr = h5.attrs.get("task", "Dexverse-PickCube-v0")
            if isinstance(task_attr, bytes):
                task_attr = task_attr.decode()
            task = args.task or str(task_attr)
            for episode_name in sorted(h5["data"].keys(), key=lambda x: int(x.split("_")[-1])):
                episode = h5["data"][episode_name]
                actions = np.asarray(episode["actions"], dtype=np.float32)
                proprio = np.asarray(episode["obs"]["proprio"]["joint_pos"], dtype=np.float32)
                length = len(actions)

                # 3view_rgb flattens three 28-dimensional state frames; retain the latest frame.
                state = proprio[:, -args.state_dim :]
                for t in range(length):
                    frame = {
                        "observation.state": state[t],
                        "action": actions[t],
                        "task": task,
                    }
                    for h5_name, lerobot_name in CAMERAS.items():
                        frame[lerobot_name] = np.asarray(episode["obs"]["rgb"][h5_name][t], dtype=np.uint8)
                    dataset.add_frame(frame)
                dataset.save_episode(parallel_encoding=False)
                episode_count += 1
                frame_count += length
                print(f"[episode {episode_count}] {h5_path.name}:{episode_name} ({length} frames)")

    dataset.finalize()
    summary = {
        "episodes": episode_count,
        "frames": frame_count,
        "output": str(args.output_root),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
