# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the same recorded episodes with state restoration and action replay.

Run with the Isaac Lab Python environment. Physics uses --device (default CPU);
MP4 rendering uses the GPU. The source pickle must be trusted. Success is measured from the task's
replayed termination signal, never from the original recording label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODES = {"set_state": "--set-state", "action_replay": "--no-set-state"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def summarize(output: Path, manifest: dict) -> dict:
    """Validate paired outputs and write measured per-episode and aggregate results."""
    rows = []
    totals = {}
    reference_actions = {}
    device = manifest.get("sim_device", "cpu")
    for mode in MODES:
        paths = list((output / mode).rglob("*.h5"))
        if len(paths) != 1:
            raise ValueError(f"Expected one HDF5 for {mode}, found {paths}")
        with h5py.File(paths[0], "r") as h5:
            # Older converters predate these attributes. For those outputs the run
            # manifest records the mode/device; validate H5 metadata when available.
            if "set_state" in h5.attrs and bool(h5.attrs["set_state"]) != (mode == "set_state"):
                raise ValueError(f"Wrong set_state metadata in {paths[0]}")
            # Use the converter's existing metadata; accept earlier comparison files too.
            replay_device = h5.attrs.get("replay_sim_device", h5.attrs.get("sim_device"))
            expected_device = "cuda:0" if device == "cuda" else device
            if replay_device == "cuda":
                replay_device = "cuda:0"
            if ((replay_device is not None and replay_device != expected_device)
                    or h5.attrs["observation_preset"] != "state"):
                raise ValueError(f"Expected {device} physics and state observations in {paths[0]}")
            if "conversion_complete" in h5.attrs and not h5.attrs["conversion_complete"]:
                raise ValueError(f"Conversion incomplete: {paths[0]}")
            groups = sorted(h5["data"].values(), key=lambda g: int(g.attrs["episode_index"]))
            indices = [int(g.attrs["episode_index"]) for g in groups]
            if indices != manifest["episode_indices"]:
                raise ValueError(f"Incomplete or mismatched episode selection for {mode}: {indices}")
            mode_rows = []
            for group in groups:
                index = int(group.attrs["episode_index"])
                actions = group["actions"][()]
                if mode == "set_state":
                    reference_actions[index] = actions
                elif not np.array_equal(actions, reference_actions[index]):
                    raise ValueError(f"Action inputs differ for episode {index}")
                expected_steps = manifest["episode_lengths"][index]
                if "terminations/success" not in group:
                    raise ValueError(f"Replay success unavailable: {mode}, episode {index}")
                flags = group["terminations/success"][()]
                if len(actions) != expected_steps or flags.shape != (expected_steps,):
                    raise ValueError(f"Incomplete trajectory for {mode}, episode {index}")
                run = max_run = 0
                for flag in flags:
                    run = run + 1 if flag else 0
                    max_run = max(max_run, run)
                videos = list((output / mode).rglob(f"episode_{index:05d}.mp4"))
                if len(videos) != 1 or videos[0].stat().st_size == 0:
                    raise ValueError(f"Missing video for {mode}, episode {index}")
                # Count decoded frames, including the initial state, to detect truncated videos.
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                     "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
                     "-of", "json", str(videos[0])],
                    text=True, capture_output=True, check=True,
                )
                stream = json.loads(probe.stdout)["streams"][0]
                frames = int(stream["nb_read_frames"])
                if frames != expected_steps + 1:
                    raise ValueError(f"Video frame count mismatch: {videos[0]}: {frames} != {expected_steps + 1}")
                row = {
                    "mode": mode,
                    "episode_index": index,
                    "steps": len(actions),
                    "recorded_success": bool(group.attrs["success"]),
                    "replay_success": max_run >= manifest["num_success_steps"],
                    "success_any_step": bool(flags.any()),
                    "success_final_step": bool(flags[-1]) if len(flags) else False,
                    "max_consecutive_success_steps": max_run,
                    "video_frames": frames,
                    "video": str(videos[0].relative_to(output)),
                }
                mode_rows.append(row)
                rows.append(row)
            totals[mode] = {
                "episodes": len(mode_rows),
                "successes": sum(r["replay_success"] for r in mode_rows),
                "success_rate": sum(r["replay_success"] for r in mode_rows) / len(mode_rows),
                "any_step_successes": sum(r["success_any_step"] for r in mode_rows),
                "final_step_successes": sum(r["success_final_step"] for r in mode_rows),
                "hdf5": str(paths[0].relative_to(output)),
            }
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"task": manifest["task"], "sim_device": device,
              "num_success_steps": manifest["num_success_steps"], "modes": totals}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"# Replay comparison ({device}): {manifest['task']}", "",
        f"Same {len(manifest['episode_indices'])} episodes (indices {manifest['episode_indices'][0]}–"
        f"{manifest['episode_indices'][-1]}), state observations, one environment, seed {manifest['seed']}.",
        "Both modes restore each recorded initial state. Action replay then steps the recorded actions; "
        "set-state restores each recorded post-step state.", "",
        f"Success requires at least {manifest['num_success_steps']} consecutive successful post-step checks "
        "within the recorded trajectory length. Any-step and final-step counts are secondary metrics.", "",
        "| Replay mode | Success | Rate | Any-step | Final-step |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, data in totals.items():
        lines.append(f"| {mode} | {data['successes']}/{data['episodes']} | {data['success_rate']:.1%} | "
                     f"{data['any_step_successes']} | {data['final_step_successes']} |")
    lines += ["", "## Artifacts", "", "- [Per-episode results](episodes.csv)",
              "- [Machine-readable summary](summary.json)", "- [Provenance and exact commands](manifest.json)"]
    for mode, data in totals.items():
        lines.append(f"- [{mode} state observations and actions]({data['hdf5']})")
    lines += ["", "## Paired videos", "", "| Episode | Set-state | Action replay |", "| ---: | --- | --- |"]
    indexed = {(r["mode"], r["episode_index"]): r for r in rows}
    for index in manifest["episode_indices"]:
        cells = []
        for mode in MODES:
            row = indexed[mode, index]
            label = "success" if row["replay_success"] else "failure"
            cells.append(f"[{label}]({row['video']})")
        lines.append(f"| {index} | {' | '.join(cells)} |")
    lines += ["", "## Interpretation and reproducibility", "",
              "The source metadata does not record the collection physics device, seed, or simulator versions. "
              "CPU collection therefore cannot be confirmed from this pickle. These results compare replay "
              "in the current checkout and environment, using the current task success condition.", "",
              "The HDF5 legacy `success` attribute remains the original recording label. "
              "Report success is calculated from the existing `terminations/success` dataset. "
              "Set-state success checks the recorded path against the current task criterion; "
              "it does not establish that live physics reproduces that path.", "",
              f"MP4s contain one initial frame plus one frame per action. Physics device is {device}; camera rendering "
              f"runs on GPU. Video playback is {manifest['video_fps']} FPS. "
              "Every video was decoded to verify its frame count, and paired action arrays were checked for equality.", ""]
    (output / "report.md").write_text("\n".join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Physics device: cpu, cuda, or cuda:N (for example cuda:0).")
    parser.add_argument("--num-success-steps", type=int, default=10)
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    source, output = args.file.resolve(), args.output_dir.resolve()
    if args.summarize_only:
        print(json.dumps(summarize(output, json.loads((output / "manifest.json").read_text())), indent=2))
        return
    if args.num_episodes < 1 or args.num_success_steps < 1 or args.video_fps < 1:
        parser.error("Episode count, success steps, and video FPS must be positive")
    if (output / "manifest.json").exists():
        parser.error("Output already contains a run; choose a new directory or use --summarize-only")
    with source.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("format") != "dexverse_trajectory":
        raise ValueError("Expected a single-task dexverse_trajectory pickle")
    episodes = payload["episodes"][:args.num_episodes]
    if len(episodes) != args.num_episodes:
        raise ValueError(f"Requested {args.num_episodes} episodes, found {len(episodes)}")
    for index, episode in enumerate(episodes):
        if not len(episode["actions"]) or len(episode.get("states", [])) != len(episode["actions"]) + 1:
            raise ValueError(f"Episode {index} needs nonempty actions and T+1 recorded states")
    output.mkdir(parents=True, exist_ok=True)
    converter = REPO / "scripts/demo_tools/create_demo_files_sequential.py"
    commands = {}
    for mode, flag in MODES.items():
        if (output / mode).exists():
            raise FileExistsError(output / mode)
        commands[mode] = [
            sys.executable, "-u", str(converter), "--file", str(source), "--select-episodes",
            *map(str, range(args.num_episodes)), "--obs-groups", "state", "--device", args.device,
            "--seed", str(args.seed),
            "--record-video", "--video-fps", str(args.video_fps), flag,
            "--output-dir", str(output / mode),
        ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": payload.get("env_name") or payload["task"], "source": str(source),
        "source_sha256": sha256(source), "source_metadata": {k: v for k, v in payload.items() if k != "episodes"},
        "episode_indices": list(range(args.num_episodes)),
        "episode_lengths": [len(e["actions"]) for e in episodes],
        "seed": args.seed, "num_success_steps": args.num_success_steps, "video_fps": args.video_fps,
        "sim_device": args.device,
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "repository": str(REPO), "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "packages": {name: importlib.metadata.version(name) for name in ["isaacsim", "isaaclab", "torch", "h5py", "numpy", "imageio", "imageio-ffmpeg"]},
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "source_code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                               [converter, Path(__file__).resolve(), *sorted((REPO / "source/dexverse/dexverse").rglob("*.py"))]},
        "commands": commands, "pythonpath": str(REPO / "source/dexverse"),
    }
    (output / "working_tree.patch").write_text(command_output(["git", "diff", "HEAD"]))
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "source/dexverse") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for mode, command in commands.items():
        print(f"[{mode}] {shlex.join(command)}", flush=True)
        started = time.monotonic()
        with (output / f"{mode}.log").open("w") as log:
            process = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest.setdefault("runs", {})[mode] = {"returncode": process.returncode, "seconds": time.monotonic() - started}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"{mode} conversion failed ({process.returncode}); see {output / (mode + '.log')}")
        print(f"[{mode}] finished in {manifest['runs'][mode]['seconds']:.1f}s", flush=True)
    print(json.dumps(summarize(output, manifest), indent=2))
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
