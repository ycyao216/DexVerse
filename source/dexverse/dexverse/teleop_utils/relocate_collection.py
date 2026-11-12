"""Portable object identities for one-object-per-launch human collection."""
from pathlib import Path
import random
import re

RELOCATE_TASK = "Dexverse-RelocateObject-v0"
_ID = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+\Z")


def object_id_from_asset(asset) -> str | None:
    if not isinstance(asset, dict):
        return None
    identity = asset.get("object_id")
    if isinstance(identity, str) and _ID.fullmatch(identity):
        return identity
    parts = str(asset.get("usd_path", "")).replace("\\", "/").split("/")
    if "mani_twin_selected" in parts:
        start = parts.index("mani_twin_selected") + 1
        identity = "/".join(parts[start:start + 2])
        if _ID.fullmatch(identity):
            return identity
    return None


def collection_object_catalog() -> dict[str, str]:
    """Discover the shipped pool without importing Isaac Sim/task registration."""
    from dexverse.assets import MANI_TWIN_SELECTED_DIR
    result = {}
    for directory in sorted(MANI_TWIN_SELECTED_DIR.glob("*/*")):
        if not directory.is_dir():
            continue
        unpacked = directory / "base_rescaled_unpacked/output.usdc"
        path = unpacked if unpacked.is_file() else directory / "base_rescaled.usdz"
        if path.is_file():
            identity = object_id_from_asset({"usd_path": str(path)})
            if identity:
                result[identity] = str(path)
    if not result:
        raise ValueError("RelocateObject 的 100 物体资源未安装。")
    return result


def choose_collection_object(catalog, completed, pending=None, skipped=()):
    """Retry unfinished objects, excluding successes and explicit operator skips."""
    available = sorted(set(catalog) - set(completed) - set(skipped))
    if not available:
        raise ValueError("物体池中没有可采集的物体：其余物体已完成或已跳过。跳过不计入成功数量。")
    return pending if pending in available else random.SystemRandom().choice(available)
