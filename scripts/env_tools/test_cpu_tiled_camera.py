"""Exercise the production subclass against real camera/kernel source, without Sim."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import importlib.util
import sys
from typing import Sequence

wp = pytest.importorskip("warp")

@pytest.fixture(scope="module")
def isaac_sources():
    wp.init()
    spec = importlib.util.find_spec("isaaclab")
    if spec is None:
        pytest.skip("Installed IsaacLab source is required, without importing its simulator")
    root = Path(next(iter(spec.submodule_search_locations)))
    path = root / "utils/warp/kernels.py"
    spec = importlib.util.spec_from_file_location("offline_camera_kernels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Compile only the original camera method, avoiding any simulator imports.
    source_path = root / "sensors/camera/tiled_camera.py"
    tree = ast.parse(source_path.read_text())
    camera_class = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "TiledCamera")
    method = next(item for item in camera_class.body if isinstance(item, ast.FunctionDef)
                  and item.name == "_update_buffers_impl")
    namespace = {"np": np, "torch": torch, "wp": wp,
                 "reshape_tiled_image": module.reshape_tiled_image, "Sequence": Sequence}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return module.reshape_tiled_image, namespace["_update_buffers_impl"]

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "source/dexverse/dexverse"


def production_namespace(base):
    path = PACKAGE / "cpu_tiled_camera.py"
    tree = ast.parse(path.read_text())
    # Supply a stand-in TiledCamera base with the real installed update method.
    # All production helper, proxy, and subclass implementations execute intact.
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module == "isaaclab.sensors")]
    scope = {"TiledCamera": base}
    exec(compile(tree, str(path), "exec"), scope)
    return scope


def camera_base(original_update, clipping="none"):
    height, width = 7, 11
    rgba = np.arange(height * width * 4, dtype=np.uint8).reshape(height, width, 4)
    depth = np.linspace(0, 15, height * width, dtype=np.float32).reshape(height, width)
    depth[0, :4] = [np.inf, -np.inf, np.nan, -0.]

    class Camera:
        _update_buffers_impl = original_update

        def __init__(self):
            self._frame = np.zeros(1, dtype=int)
            self.cfg = SimpleNamespace(update_latest_camera_pose=True, height=height, width=width,
                data_types=["rgb", "distance_to_camera", "distance_to_image_plane"],
                depth_clipping_behavior=clipping, spawn=SimpleNamespace(clipping_range=(.1, 10.)))
            self.device = "cpu"
            self._view = SimpleNamespace(count=1)
            self._data = SimpleNamespace(info={}, output={
                "rgba": torch.zeros((1, height, width, 4), dtype=torch.uint8),
                "distance_to_camera": torch.zeros((1, height, width, 1)),
                "distance_to_image_plane": torch.zeros((1, height, width, 1))})
            self._annotators = {"rgba": SimpleNamespace(get_data=lambda: rgba),
                "distance_to_camera": SimpleNamespace(get_data=lambda: depth),
                "distance_to_image_plane": SimpleNamespace(get_data=lambda: depth)}
            self.pose_updates = []

        def _update_poses(self, ids):
            self.pose_updates.append(list(ids))

        def _tiling_grid_shape(self):
            return (1, 1)

    return Camera


@pytest.mark.parametrize("clipping", ["none", "zero", "max"])
def test_production_update_matches_original_every_buffer_and_side_effect(isaac_sources, clipping):
    kernel, original = isaac_sources
    base = camera_base(original, clipping)
    production = production_namespace(base)
    expected, actual = base(), production["CpuTiledCamera"]()
    original_wp = original.__globals__["wp"]
    original_launch = original_wp.launch
    for _ in range(3):
        expected._update_buffers_impl([0])
        actual._update_buffers_impl([0])
        for name, expected_output in expected._data.output.items():
            assert torch.equal(actual._data.output[name].contiguous().view(torch.uint8),
                               expected_output.contiguous().view(torch.uint8))
    assert actual.pose_updates == expected.pose_updates == [[0], [0], [0]]
    assert np.array_equal(actual._frame, expected._frame)
    assert actual._data.output["rgb"].data_ptr() == actual._data.output["rgba"].data_ptr()
    assert original.__globals__["wp"] is original_wp and original_wp.launch is original_launch
    assert base._update_buffers_impl is original
    private = production["_single_tile_update"](original)
    assert private.__code__ is original.__code__
    assert private.__globals__["wp"]._kernel is kernel


@pytest.mark.parametrize("device,count,types", [
    ("cuda:0", 1, ["rgb"]), ("cpu", 2, ["rgb"]), ("cpu", 1, ["normals"]),
])
def test_unsupported_sensor_uses_base_directly(device, count, types):
    class Base:
        def _update_buffers_impl(self, env_ids):
            self.base_call = env_ids
            return "base-result"

    camera = production_namespace(Base)["CpuTiledCamera"]()
    camera.device, camera._view, camera.cfg = device, SimpleNamespace(count=count), SimpleNamespace(data_types=types)
    assert camera._update_buffers_impl([5]) == "base-result"
    assert camera.base_call == [5]


@pytest.mark.parametrize("case", ["noncontiguous_source", "noncontiguous_target", "unsupported_dtype", "two_tiles"])
def test_production_buffer_guard_leaves_unsupported_buffers_untouched(isaac_sources, case):
    _, original = isaac_sources
    wp = original.__globals__["wp"]
    production = production_namespace(camera_base(original))
    source, target = torch.arange(24, dtype=torch.float32), torch.zeros((1, 4, 6, 1))
    dim = (1, 4, 6)
    if case == "noncontiguous_source":
        source = torch.arange(48, dtype=torch.float32)[::2]
    elif case == "noncontiguous_target":
        target = torch.zeros((1, 4, 12, 1))[:, :, ::2, :]
    elif case == "unsupported_dtype":
        source, target = source.to(torch.int32), target.to(torch.int32)
    else:
        dim = (2, 4, 6)
    assert not production["_copy_single_tile_cpu"](
        wp, dim, [wp.from_torch(source), wp.from_torch(target), 4, 6, 1, 1], "cpu")
    assert not target.any()


def test_configuration_uses_private_copies_and_keeps_custom_cameras():
    class Base:
        pass

    class Custom(Base):
        pass

    class Config(SimpleNamespace):
        pass

    production = production_namespace(Base)
    path = PACKAGE / "tasks/config/long_horizon/handling.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "configure_cpu_camera_copies")
    node.body = [item for item in node.body if not isinstance(item, ast.ImportFrom)]
    scope = {"deepcopy": deepcopy, "TiledCamera": Base, "TiledCameraCfg": Config,
             "CpuTiledCamera": production["CpuTiledCamera"]}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    camera = Config(class_type=Base, height=256, width=256, data_types=["rgb", "distance_to_image_plane"],
                    update_period=0., offset={"position": [1, 2, 3]})
    custom = Config(class_type=Custom)
    cfg = SimpleNamespace(scene=SimpleNamespace(camera=camera, custom=custom, disabled=None))
    before = deepcopy(vars(camera))
    scope["configure_cpu_camera_copies"](cfg)
    assert cfg.scene.camera is not camera and vars(camera) == before
    assert cfg.scene.camera.class_type is production["CpuTiledCamera"]
    actual_fields = {k: v for k, v in vars(cfg.scene.camera).items() if k != "class_type"}
    assert actual_fields == {k: v for k, v in before.items() if k != "class_type"}
    assert cfg.scene.custom is custom and cfg.scene.custom.class_type is Custom
    assert cfg.scene.disabled is None
