# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy-server adapters for online evaluation.

Both supported baselines run the policy **out of process** and talk to it over
the network, because neither can share an interpreter with Isaac Sim (torch /
JAX / transformers version skew). This module hides the two wire protocols
behind one :class:`PolicyClient` interface so ``eval_policy.py`` never branches
on the backend inside the rollout loop.

======================  ====================================================
Backend                 Wire protocol
======================  ====================================================
``openvla``             HTTP POST ``/act``, json_numpy-encoded body.
                        Served by ``vla-scripts/deploy.py`` in openvla-oft.
``pi0``                 WebSocket via ``openpi_client``.
                        Served by ``scripts/serve_policy.py`` in openpi.
======================  ====================================================

Backend dependencies (``requests`` + ``json_numpy`` / ``openpi_client``) are
imported when the client is constructed, not at module import, so evaluating
one baseline does not require the other's packages. Build the client *before*
``AppLauncher`` runs — see the note in ``eval_policy.py``.
"""

from __future__ import annotations

import abc
import dataclasses
import time

import numpy as np

# Joint targets are radians; anything outside this is a diverged prediction.
# Isaac Sim's PhysX solver segfaults on NaN/Inf rather than raising, so every
# chunk is sanitized before it reaches ``env.step`` — unconverged checkpoints
# do emit them.
ACTION_LIMIT_RAD = 3.14

_JSON_HEADERS = {"Content-Type": "application/json"}


@dataclasses.dataclass(frozen=True)
class EvalObservation:
    """One timestep of observation, in DexVerse's canonical naming.

    Each client renames these into its own server's expected keys. ``wrist`` is
    set for single-hand embodiments; ``left_wrist`` / ``right_wrist`` for
    bimanual ones.
    """

    third_person: np.ndarray  # (H, W, 3) uint8
    state: np.ndarray  # (D,) float32
    wrist: np.ndarray | None = None
    left_wrist: np.ndarray | None = None
    right_wrist: np.ndarray | None = None

    @property
    def is_bimanual(self) -> bool:
        return self.left_wrist is not None or self.right_wrist is not None

    def shape_summary(self) -> str:
        """Compact one-line description used in the first-request log."""
        parts = [f"third_person={self.third_person.shape}"]
        if self.is_bimanual:
            parts.append(f"left_wrist={None if self.left_wrist is None else self.left_wrist.shape}")
            parts.append(f"right_wrist={None if self.right_wrist is None else self.right_wrist.shape}")
        else:
            parts.append(f"wrist={None if self.wrist is None else self.wrist.shape}")
        parts.append(f"state={self.state.shape}")
        return " ".join(parts)


def sanitize_actions(actions: np.ndarray) -> np.ndarray:
    """Clamp a predicted action chunk into a range PhysX can consume."""
    actions = np.asarray(actions, dtype=np.float32)
    actions = np.nan_to_num(actions, nan=0.0, posinf=ACTION_LIMIT_RAD, neginf=-ACTION_LIMIT_RAD)
    return np.clip(actions, -ACTION_LIMIT_RAD, ACTION_LIMIT_RAD)


class PolicyClient(abc.ABC):
    """Query a remote policy server for an action chunk."""

    name: str = "policy"

    @abc.abstractmethod
    def _request(self, obs: EvalObservation, prompt: str) -> np.ndarray:
        """Backend-specific call returning an unsanitized ``(horizon, action_dim)`` chunk."""

    def infer(self, obs: EvalObservation, prompt: str) -> tuple[np.ndarray, float]:
        """Return ``(sanitized_chunk, latency_ms)``."""
        start = time.perf_counter()
        chunk = self._request(obs, prompt)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return sanitize_actions(chunk), latency_ms

    def describe(self) -> str:
        return self.name


class OpenVLAHttpClient(PolicyClient):
    """Client for openvla-oft's ``vla-scripts/deploy.py`` FastAPI server.

    Payload matches ``experiments.robot.openvla_utils.get_vla_action``. The
    server denormalizes with the ``dataset_statistics.json`` under the
    checkpoint, selected by ``unnorm_key``, and returns a list of
    ``NUM_ACTIONS_CHUNK`` actions.
    """

    name = "openvla"

    def __init__(self, host: str, port: int, unnorm_key: str, timeout_s: float = 120.0) -> None:
        # json_numpy.patch() is deliberately NOT called. It only patches stdlib
        # json, while `requests` encodes bodies with simplejson when that is
        # installed — which still raises "Object of type ndarray is not JSON
        # serializable". Serializing explicitly and posting raw bytes via
        # `data=` bypasses requests' own encoder entirely.
        import json_numpy
        import requests

        self._json_numpy = json_numpy
        self._requests = requests
        self.endpoint = f"http://{host}:{port}/act"
        self.unnorm_key = unnorm_key
        self.timeout_s = timeout_s

    def _request(self, obs: EvalObservation, prompt: str) -> np.ndarray:
        if obs.is_bimanual:
            images = {"left_wrist_image": obs.left_wrist, "right_wrist_image": obs.right_wrist}
        else:
            images = {"wrist_image": obs.wrist}
        payload = {
            "full_image": obs.third_person,
            **images,
            "state": obs.state,
            "instruction": prompt,
            "unnorm_key": self.unnorm_key,
        }
        response = self._requests.post(
            self.endpoint,
            data=self._json_numpy.dumps(payload),
            headers=_JSON_HEADERS,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        result = self._json_numpy.loads(response.text)
        # deploy.py catches its own exceptions and returns the bare string
        # "error"; the traceback is only on the server's stdout.
        if isinstance(result, str):
            raise RuntimeError(
                f"OpenVLA server returned {result!r} instead of an action chunk. "
                "Check the server log for the traceback."
            )
        return np.asarray(result, dtype=np.float32)

    def describe(self) -> str:
        return f"openvla HTTP {self.endpoint} (unnorm_key={self.unnorm_key})"


class OpenPiWebsocketClient(PolicyClient):
    """Client for openpi's ``scripts/serve_policy.py`` websocket server.

    Matches the repack in ``LeRobotDexbenchDataConfig`` /
    ``LeRobotDexbenchBimanualDataConfig``. Those configs set
    ``prompt_from_task=True``, so the client-sent ``prompt`` string reaches the
    model directly and no ``task_index`` lookup happens server-side.
    """

    name = "pi0"

    def __init__(self, host: str, port: int) -> None:
        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.host = host
        self.port = port

    @property
    def server_metadata(self) -> object:
        return self._client.get_server_metadata()

    def _request(self, obs: EvalObservation, prompt: str) -> np.ndarray:
        if obs.is_bimanual:
            images = {"left_wrist": obs.left_wrist, "right_wrist": obs.right_wrist}
        else:
            images = {"wrist": obs.wrist}
        result = self._client.infer({
            "third_person": obs.third_person,
            **images,
            "state": obs.state,
            "prompt": prompt,
        })
        return np.asarray(result["actions"], dtype=np.float32)

    def describe(self) -> str:
        return f"pi0 websocket {self.host}:{self.port}"


#: Default ``--observation_preset`` per backend. These differ because the two baselines
#: were trained on differently-built datasets: openvla-oft's RLDS transform
#: clips proprio to a single frame, while the openpi norm stats were computed
#: over the 3-frame stacked state produced by the ``rgb`` preset. Getting this
#: wrong silently feeds the policy a wrong-width state vector, so
#: ``eval_policy.py`` always prints the resulting state dimension.
DEFAULT_OBS_PRESET = {"openvla": "none", "pi0": "rgb"}

POLICY_NAMES: tuple[str, ...] = ("openvla", "pi0")


def make_client(policy: str, host: str, port: int, unnorm_key: str | None = None) -> PolicyClient:
    """Construct the client for ``policy``, importing its backend deps eagerly."""
    if policy == "openvla":
        if not unnorm_key:
            raise ValueError("--policy openvla requires --unnorm_key (the dataset_statistics.json key).")
        return OpenVLAHttpClient(host=host, port=port, unnorm_key=unnorm_key)
    if policy == "pi0":
        return OpenPiWebsocketClient(host=host, port=port)
    raise ValueError(f"Unknown policy {policy!r}. Available: {list(POLICY_NAMES)}")
