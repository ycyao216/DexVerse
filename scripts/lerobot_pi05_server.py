#!/usr/bin/env python3
"""Local inference service for fine-tuned LeRobot PI05 adapters.

The service receives observations and returns PI05 action chunks over a Unix
socket. Socket permissions restrict access to the current user.
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a fine-tuned LeRobot PI05 adapter over a Unix socket")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the LoRA pretrained_model directory")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Training dataset root directory")
    parser.add_argument("--dataset-repo-id", default="Dexverse-PickCube-v0", help="Training dataset identifier")
    parser.add_argument("--socket-path", type=Path, default=Path("/tmp/dexverse_pi05.sock"), help="Unix socket path")
    parser.add_argument("--device", default="cuda", help="PI05 inference device")
    return parser.parse_args()


def recv_exact(conn: socket.socket, size: int) -> bytes:
    """Read exactly ``size`` bytes or raise when the peer disconnects."""
    chunks: list[bytes] = []
    while size:
        chunk = conn.recv(size)
        if not chunk:
            raise ConnectionError("PI05 client disconnected before completing the request.")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def recv_message(conn: socket.socket) -> Any:
    """Receive a message encoded as an 8-byte length and a pickle payload."""
    size = struct.unpack("!Q", recv_exact(conn, 8))[0]
    return pickle.loads(recv_exact(conn, size))


def send_message(conn: socket.socket, payload: Any) -> None:
    """Send a message encoded as an 8-byte length and a pickle payload."""
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!Q", len(body)) + body)


class PI05Service:
    """Load a PI05 LoRA adapter and run action-chunk inference."""

    def __init__(self, args: argparse.Namespace):
        checkpoint = args.checkpoint.expanduser().resolve()
        # config.json describes the policy interface; adapter_config.json identifies the base model.
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.pretrained_path = checkpoint
        config.pretrained_revision = None
        config.use_peft = True
        config.device = args.device

        dataset_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
        self.policy = make_policy(config, ds_meta=dataset_meta)
        # Override the saved processor device so the service can run on any selected device.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        self.policy.eval()

    def reset(self) -> None:
        """Reset policy and processor state for a new episode."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def warmup(self) -> None:
        """Run one inference pass before accepting client connections."""
        observation: dict[str, Any] = {
            "observation.state": np.array([0.5, 0.0, 0.3, *([0.0] * 25)], dtype=np.float32),
            "task": "Dexverse-PickCube-v0",
        }
        for key in ("rgb_image", "left_rgb_image", "right_rgb_image", "wrist_rgb_image"):
            observation[f"observation.images.{key}"] = np.zeros((256, 256, 3), dtype=np.uint8)
        self.reset()
        self.predict_chunk(observation)
        self.reset()

    @torch.inference_mode()
    def predict_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        """Process a LeRobot observation and return an action chunk."""
        # Convert NumPy values before batching so the processor receives tensors
        # with shapes (1, 28) and (1, C, H, W).
        raw_observation: dict[str, Any] = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                # Convert camera images from HWC to the CHW layout expected by the vision encoder.
                if key.startswith("observation.images.") and value.ndim == 3:
                    value = np.moveaxis(value, -1, 0)
                raw_observation[key] = torch.from_numpy(np.ascontiguousarray(value))
            else:
                raw_observation[key] = value
        batch = self.preprocessor(raw_observation)
        actions = self.policy.predict_action_chunk(batch)[:, : self.policy.config.n_action_steps]
        if actions.is_cuda:
            torch.cuda.synchronize(actions.device)
        # The postprocessor maps model outputs back to the dataset action scale.
        result = self.postprocessor(actions)
        result_numpy = result[0].detach().cpu().numpy().astype(np.float32, copy=False)
        return result_numpy


def serve(service: PI05Service, socket_path: Path) -> None:
    """Serve one local client connection at a time."""
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(1)
    print(f"[PI05 server] ready: {socket_path}", flush=True)
    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    while True:
                        request = recv_message(conn)
                        command = request.get("command")
                        if command == "reset":
                            service.reset()
                            send_message(conn, {"ok": True})
                        elif command == "predict_chunk":
                            action_chunk = service.predict_chunk(request["observation"])
                            # Lists remain compatible across client and server NumPy versions.
                            send_message(conn, {"ok": True, "action_chunk": action_chunk.tolist()})
                        elif command == "close":
                            send_message(conn, {"ok": True})
                            return
                        else:
                            raise ValueError(f"Unknown PI05 request: {command!r}")
                except (ConnectionError, EOFError):
                    continue
                except Exception as exc:
                    send_message(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    service = PI05Service(args)
    print(f"[PI05 server] warming up on {args.device}...", flush=True)
    service.warmup()
    print("[PI05 server] warmup complete", flush=True)
    serve(service, args.socket_path.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
