# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""An opt-in TiledCamera with an exact single-view CPU buffer copy.

IsaacLab's pixel-by-pixel untile kernel is an identity copy for one camera.
The subclass keeps the installed update implementation, including annotator
reads, camera poses, depth clipping and RGB aliases. Since IsaacLab exposes no
copy-operation override, a private function namespace replaces only its Warp
launch reference. Neither the IsaacLab module nor Warp is modified globally.
"""
from __future__ import annotations

from functools import lru_cache, update_wrapper
from types import FunctionType

from isaaclab.sensors import TiledCamera


def _copy_single_tile_cpu(wp, dim, inputs, device):
    """Return whether the existing buffers admit a bit-preserving bulk copy."""
    if not wp.get_device(device).is_cpu or not isinstance(inputs, (tuple, list)) or len(inputs) != 6:
        return False
    source, target, height, width, channels, tiles_x = inputs
    if not isinstance(dim, (tuple, list)) or tuple(dim) != (1, height, width) or tiles_x != 1:
        return False
    if height < 1 or width < 1 or not isinstance(source, wp.array) or not isinstance(target, wp.array):
        return False
    if not source.device.is_cpu or not target.device.is_cpu or source.requires_grad or target.requires_grad:
        return False
    if source.dtype != target.dtype or not (
        (source.dtype == wp.uint8 and channels in (3, 4)) or (source.dtype == wp.float32 and channels == 1)
    ):
        return False
    if source.shape != (height * width * channels,) or target.shape != (1, height, width, channels):
        return False
    if not source.is_contiguous or not target.is_contiguous:
        return False
    wp.to_torch(target).copy_(wp.to_torch(source).reshape(target.shape))
    return True


class _CameraCopyWarpProxy:
    def __init__(self, warp, kernel):
        self._warp = warp
        self._kernel = kernel

    def __getattr__(self, name):
        return getattr(self._warp, name)

    def launch(self, *args, **kwargs):
        if not args and set(kwargs) == {"kernel", "dim", "inputs", "device"}:
            if kwargs["kernel"] is self._kernel and _copy_single_tile_cpu(
                self._warp, kwargs["dim"], kwargs["inputs"], kwargs["device"]
            ):
                return None
        return self._warp.launch(*args, **kwargs)


@lru_cache(maxsize=4)
def _single_tile_update(original):
    """Reuse the installed method code; unsupported future layouts use it intact."""
    namespace = getattr(original, "__globals__", {})
    if "wp" not in namespace or "reshape_tiled_image" not in namespace:
        return original
    private_globals = dict(namespace)
    private_globals["wp"] = _CameraCopyWarpProxy(namespace["wp"], namespace["reshape_tiled_image"])
    private = FunctionType(original.__code__, private_globals, original.__name__,
                           original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    return update_wrapper(private, original)


class CpuTiledCamera(TiledCamera):
    """Preserve complete camera observations with a faster one-view CPU copy."""

    _COPY_DATA_TYPES = frozenset({"rgb", "rgba", "depth", "distance_to_image_plane", "distance_to_camera"})

    def _update_buffers_impl(self, env_ids):
        if (str(self.device) != "cpu" or self._view.count != 1
                or not set(self.cfg.data_types) <= self._COPY_DATA_TYPES):
            return super()._update_buffers_impl(env_ids)
        return _single_tile_update(TiledCamera._update_buffers_impl)(self, env_ids)
