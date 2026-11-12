"""Resolve the host OpenXR exports for the running CloudXR Docker runtime.

Only the simulator child gets these overrides. In particular, do not replace
the desktop or web service's XDG_RUNTIME_DIR with CloudXR's IPC directory.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess


def cloudxr_environment(environment: dict[str, str]) -> dict[str, str]:
    """Return a validated manifest/IPC pair from the same shared runtime root.

    DEXVERSE_CLOUDXR_ROOT supports explicitly configured/non-Docker setups.
    Otherwise inspect the live container, avoiding stale runtime trees elsewhere
    on the desktop. No container is started, stopped, or reconfigured here.
    """
    configured_root = environment.get("DEXVERSE_CLOUDXR_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
    else:
        container = environment.get("DEXVERSE_CLOUDXR_CONTAINER", "cloudxr-runtime")
        try:
            result = subprocess.run(["docker", "inspect", container], check=True,
                                    capture_output=True, text=True, timeout=5)
            info = json.loads(result.stdout)[0]
            if not info["State"]["Running"]:
                raise ValueError("CloudXR container is not running")
            sources = [m["Source"] for m in info["Mounts"] if m["Destination"] == "/openxr"]
            if len(sources) != 1:
                raise ValueError("CloudXR container must expose one /openxr mount")
            root = Path(sources[0]).resolve()
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ValueError("无法读取运行中的 CloudXR 容器。请启动 cloudxr-runtime，或设置 DEXVERSE_CLOUDXR_ROOT 为共享 openxr 目录。") from exc

    manifest = root / "share/openxr/1/openxr_cloudxr.json"
    runtime_dir = root / "run"
    try:
        runtime = json.loads(manifest.read_text())["runtime"]
        library = (manifest.parent / runtime["library_path"]).resolve()
        if not library.is_file():
            raise ValueError("CloudXR runtime library is missing")
        if not runtime_dir.is_dir() or not (runtime_dir / "ipc_cloudxr").is_socket():
            raise ValueError("CloudXR IPC socket is not ready")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"CloudXR 共享目录尚未就绪：{root}。请检查运行库和 run/ipc_cloudxr。") from exc
    return {"XR_RUNTIME_JSON": str(manifest), "XDG_RUNTIME_DIR": str(runtime_dir)}
