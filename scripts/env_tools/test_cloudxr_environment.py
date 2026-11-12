"""CloudXR launch exports, without a simulator or Docker daemon."""
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/dexverse"))
from dexverse.teleop_utils import cloudxr_environment as xr


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "xr"
    manifest = root / "share/openxr/1/openxr_cloudxr.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"runtime": {"library_path": "../../../lib/libopenxr_cloudxr.so"}}))
    (root / "lib").mkdir()
    (root / "lib/libopenxr_cloudxr.so").touch()
    (root / "run").mkdir()
    with socket.socket(socket.AF_UNIX) as ipc:
        ipc.bind(str(root / "run/ipc_cloudxr"))
        yield root


def test_exports_use_active_container_mount_not_desktop_or_stale_runtime(runtime, monkeypatch):
    def inspect(command, **kwargs):
        assert command == ["docker", "inspect", "cloudxr-runtime"]
        assert kwargs["timeout"] == 5
        return SimpleNamespace(stdout=json.dumps([{"State": {"Running": True}, "Mounts": [
            {"Source": str(runtime), "Destination": "/openxr"}]}]))
    monkeypatch.setattr(xr.subprocess, "run", inspect)
    environment = {"XDG_RUNTIME_DIR": "/run/user/1001", "XR_RUNTIME_JSON": "/old/openxr.json"}
    result = xr.cloudxr_environment(environment)
    assert result == {"XDG_RUNTIME_DIR": str(runtime / "run"),
                      "XR_RUNTIME_JSON": str(runtime / "share/openxr/1/openxr_cloudxr.json")}
    assert environment["XDG_RUNTIME_DIR"] == "/run/user/1001"


def test_explicit_runtime_root_checks_library_and_ipc(runtime, monkeypatch):
    monkeypatch.setattr(xr.subprocess, "run", lambda *a, **kw: pytest.fail("explicit root must not inspect Docker"))
    environment = {"DEXVERSE_CLOUDXR_ROOT": str(runtime)}
    assert xr.cloudxr_environment(environment)["XDG_RUNTIME_DIR"] == str(runtime / "run")
    (runtime / "run/ipc_cloudxr").unlink()
    with pytest.raises(ValueError, match="尚未就绪"):
        xr.cloudxr_environment(environment)


@pytest.mark.parametrize("running,mounts", [(False, []), (True, [])])
def test_stopped_or_unmounted_container_is_reported(running, mounts, monkeypatch):
    monkeypatch.setattr(xr.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout=json.dumps([{"State": {"Running": running}, "Mounts": mounts}])))
    with pytest.raises(ValueError, match="无法读取运行中的 CloudXR"):
        xr.cloudxr_environment({})
