#!/usr/bin/env python3
"""Capture open/closed Vision Pro hand samples without launching Isaac Sim."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from avp_stream import VisionProStreamer


FINGERS = {
    "index": (6, 7, 8, 9),
    "middle": (11, 12, 13, 14),
    "ring": (16, 17, 18, 19),
    "little": (21, 22, 23, 24),
}


def wait_for_data(streamer: VisionProStreamer, timeout_s: float = 30.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample = streamer.get_latest()
        if sample is not None:
            return sample
        time.sleep(0.1)
    raise RuntimeError("No Vision Pro data received within 30 seconds. Check that Tracking Streamer is running.")


def countdown(label: str, seconds: int = 5) -> None:
    print(f"\nHold your right hand in the {label} pose.")
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining}...", flush=True)
        time.sleep(1.0)


def capture(streamer: VisionProStreamer, label: str) -> np.ndarray:
    countdown(label)
    # Discard a few cached reads, then take the newest complete world-frame hand.
    sample = None
    for _ in range(5):
        sample = wait_for_data(streamer)
        time.sleep(0.03)
    hand = np.asarray(sample.right, dtype=np.float64)
    if hand.ndim != 3 or hand.shape[0] < 25 or hand.shape[1:] != (4, 4):
        raise RuntimeError(f"Unexpected right-hand data shape: {hand.shape}")
    print(f"Captured: {label}")
    return hand[:25].copy()


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm < 1.0e-8 or second_norm < 1.0e-8:
        return float("nan")
    cosine = np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def describe(hand: np.ndarray, label: str) -> None:
    positions = hand[:, :3, 3]
    wrist = positions[0]
    print(f"\nRight-hand measurements for {label}:")
    for name, indices in FINGERS.items():
        joints = positions[np.asarray(indices)]
        segments = np.diff(joints, axis=0)
        proximal_bend = angle_deg(segments[0], segments[1])
        distal_bend = angle_deg(segments[1], segments[2])
        tip_distance = np.linalg.norm(joints[-1] - wrist)
        print(
            f"  {name:6s}: proximal bend={proximal_bend:6.1f} deg  "
            f"distal bend={distal_bend:6.1f} deg  wrist-to-tip={tip_distance:.4f} m"
        )
    pinch = np.linalg.norm(positions[4] - positions[9])
    print(f"  thumb-index pinch distance: {pinch:.4f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ip",
        default=os.environ.get("VISIONPRO_IP", "192.168.3.37"),
        help="Vision Pro local-network IP (default: VISIONPRO_IP or 192.168.3.37)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("visionpro_hand_samples.npz"),
        help="Output NPZ path",
    )
    args = parser.parse_args()

    print(f"Connecting to Vision Pro at {args.ip}")
    streamer = VisionProStreamer(ip=args.ip, record=False)
    try:
        wait_for_data(streamer)
        print("Tracking data received. Each capture begins after an automatic countdown.")
        open_hand = capture(streamer, "fully open hand")
        fist_hand = capture(streamer, "natural fist")
        np.savez_compressed(args.output, open_hand=open_hand, fist_hand=fist_hand)
        describe(open_hand, "fully open hand")
        describe(fist_hand, "natural fist")
        print(f"\nRaw samples saved to: {args.output}")
    finally:
        streamer.cleanup()


if __name__ == "__main__":
    main()
