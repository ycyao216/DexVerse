"""Socket policy adapter for PI05 evaluation.

The adapter packages DexVerse observations into the LeRobot field layout and
requests action chunks from a PI05 inference service.
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np


SOCKET_PATH = os.environ.get("LEROBOT_PI05_SOCKET", "/tmp/dexverse_pi05.sock")
TASK_TEXT = os.environ.get("LEROBOT_PI05_TASK", "Dexverse-PickCube-v0")
KEEP_ALIVE_INTERVAL = float(os.environ.get("LEROBOT_PI05_KEEP_ALIVE_INTERVAL", "0.5"))
STATE_DIM = 28
CAMERA_KEYS = ("rgb_image", "left_rgb_image", "right_rgb_image", "wrist_rgb_image")
SCENE_CAMERA_NAMES = {
    "rgb_image": "third_person_camera",
    "left_rgb_image": "third_person_camera_left",
    "right_rgb_image": "third_person_camera_right",
    "wrist_rgb_image": "wrist_camera",
}


def recv_exact(conn: socket.socket, size: int) -> bytes:
    """Read exactly ``size`` bytes or raise when the peer disconnects."""
    chunks: list[bytes] = []
    while size:
        chunk = conn.recv(size)
        if not chunk:
            raise ConnectionError("PI05 server disconnected before completing the response.")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


class PI05SocketPolicy:
    """Cache PI05 action chunks and execute one action per environment step."""

    def __init__(self, checkpoint: str, device: str):
        del checkpoint, device  # The inference service owns model loading.
        self.socket_path = SOCKET_PATH
        self.task = TASK_TEXT
        self.connection: socket.socket | None = None
        self.actions: deque[np.ndarray] = deque()
        self.keep_alive: Any | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-request")
        self.rgb_frames: dict[str, np.ndarray] = {}
        self.state: np.ndarray | None = None

    def set_rgb_frames(self, frames: dict[str, np.ndarray]) -> None:
        """Store the latest HWC RGB frames copied to CPU by the recorder."""
        self.rgb_frames = {name: np.ascontiguousarray(frame) for name, frame in frames.items()}

    def set_state(self, state: np.ndarray) -> None:
        """Store the current 28-dimensional robot joint state."""
        self.state = np.ascontiguousarray(state, dtype=np.float32)

    def _connect(self) -> socket.socket:
        if self.connection is None:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.connect(self.socket_path)
            self.connection = conn
        return self.connection

    def set_keep_alive(self, callback: Any) -> None:
        """Set the callback used to keep the simulator responsive during inference."""
        self.keep_alive = callback

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        conn = self._connect()
        body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        conn.sendall(struct.pack("!Q", len(body)) + body)
        response_size = struct.unpack("!Q", recv_exact(conn, 8))[0]
        response = pickle.loads(recv_exact(conn, response_size))
        if not response.get("ok", False):
            raise RuntimeError(f"PI05 server request failed: {response.get('error', 'unknown error')}")
        return response

    def _to_lerobot_observation(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Build a LeRobot observation from the current state and RGB frames."""
        del observations
        if self.state is None or self.state.shape != (STATE_DIM,):
            raise RuntimeError(f"Expected a valid {STATE_DIM}-dimensional robot state.")
        result: dict[str, Any] = {
            "observation.state": self.state,
            "task": self.task,
        }
        for key in CAMERA_KEYS:
            camera_name = SCENE_CAMERA_NAMES[key]
            if camera_name not in self.rgb_frames:
                raise RuntimeError(f"Missing CPU RGB frame for camera {camera_name!r}.")
            result[f"observation.images.{key}"] = self.rgb_frames[camera_name]
        return result

    def reset(self, observations: Any | None = None) -> None:
        """Clear local actions and reset the remote policy state."""
        del observations
        self.actions.clear()
        self._request({"command": "reset"})

    def act(self, observations: dict[str, Any]) -> np.ndarray:
        """Request a new action chunk when the local cache is empty."""
        if not self.actions:
            observation = self._to_lerobot_observation(observations)
            future = self.executor.submit(
                self._request,
                {"command": "predict_chunk", "observation": observation},
            )
            last_keep_alive = time.monotonic()
            while not future.done():
                now = time.monotonic()
                if callable(self.keep_alive) and now - last_keep_alive >= KEEP_ALIVE_INTERVAL:
                    self.keep_alive()
                    last_keep_alive = now
                time.sleep(0.01)
            reply = future.result()
            self.actions.extend(np.asarray(reply["action_chunk"], dtype=np.float32))
        return self.actions.popleft()

    def close(self) -> None:
        """Close the client connection and stop the request worker."""
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def create_policy(env: Any, checkpoint: str, device: str, args: Any) -> PI05SocketPolicy:
    """Create the policy adapter used by the evaluation entry point."""
    del env, args
    return PI05SocketPolicy(checkpoint, device)
