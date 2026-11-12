"""Simulator-free helpers for the category tree and its baseline symlinks."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "source/dexverse/demonstrations"
BASELINE_MANIFEST = DEFAULT_DEST / "baseline_manifest.txt"
CATEGORIES = {"articulation", "bimanual", "contact_rich", "functional", "grasping", "non_prehensile", "long_horizon"}
TASK_ID = re.compile(r"Dexverse-[A-Za-z0-9_-]+-v\d+\Z")
CROSS_HANDS = {"shadow", "allegro", "inspire", "leap", "sharpa", "wuji"}
COLLECTIONS = {"main", "cross_20"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or str(path) != value or any(
        part in {".", "..", ""} for part in value.split("/")
    ):
        raise ValueError(f"Invalid relative path: {value!r}")
    return path


def contained_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    target = root / str(safe_relative(relative))
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes destination through a symlink: {relative}")
    return target


def install_file(source: Path, target: Path, expected_sha256: str | None = None) -> bool:
    """Copy atomically; keep identical files and refuse to replace different data."""
    digest = sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Source checksum mismatch: {source}")
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or sha256(target) != digest:
            raise FileExistsError(f"Different local data exists: {target}. Use a separate --dest.")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".demo-copy-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as stream:
            shutil.copyfileobj(stream, out)
            out.flush()
            os.fsync(out.fileno())
        if sha256(Path(temporary)) != digest:
            raise ValueError(f"Copy checksum mismatch: {target}")
        os.link(temporary, target)  # Fails if another writer installed a file meanwhile.
    finally:
        os.unlink(temporary)
    return True


def task_categories() -> dict[str, str]:
    """Read registrations without importing Isaac Sim; keys are task families."""
    result = {}
    root = REPO_ROOT / "source/dexverse/dexverse/tasks/config"
    for path in sorted(root.rglob("__init__.py")):
        category = path.relative_to(root).parts[0]
        if category not in CATEGORIES:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "register_env" and len(node.args) >= 2):
                continue
            value = node.args[1]
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and TASK_ID.fullmatch(value.value):
                family = re.sub(r"-v\d+$", "", value.value).removeprefix("Dexverse-")
                result[family] = category
    return result


def baseline_entries(version: str = "v0", manifest: Path = BASELINE_MANIFEST) -> list[str]:
    versions = ("v0", "v1") if version == "all" else (version,)
    if any(v not in {"v0", "v1"} for v in versions):
        raise ValueError(f"Unsupported version: {version}")
    entries = []
    for line in manifest.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        parts = safe_relative(entry).parts
        if len(parts) != 2 or parts[0] not in CATEGORIES or not TASK_ID.fullmatch(parts[1]) or not parts[1].endswith("-v0"):
            raise ValueError(f"Invalid baseline manifest entry: {entry}")
        entries.extend(entry[:-2] + v for v in versions)
    if len(entries) != len(set(entries)):
        raise ValueError("Duplicate baseline manifest entries")
    return entries


def sync_baseline_links(dest: Path, entries: list[str] | None = None) -> dict:
    """Expose installed baseline pickles through relative links; never copy data."""
    entries = baseline_entries("all") if entries is None else entries
    baseline = contained_path(dest, "baseline")
    if baseline.is_symlink():
        raise ValueError("The baseline directory itself must not be a symlink")
    baseline.mkdir(parents=True, exist_ok=True)
    links, missing = [], []
    planned = {}
    for entry in entries:
        task_dir = contained_path(dest, entry)
        task = task_dir.name
        files = sorted(task_dir.glob("*.pkl")) if task_dir.is_dir() else []
        if not files:
            missing.append(entry)
        for source in files:
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Canonical recordings must be regular files: {source}")
            name = source.name if source.name.startswith(task) else f"{task}__{source.name}"
            link = baseline / name
            target = os.path.relpath(source, baseline)
            if link in planned and planned[link] != target:
                raise ValueError(f"Baseline link collision: {link}")
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or os.readlink(link) != target:
                    raise FileExistsError(f"Refusing to replace baseline entry: {link}")
            planned[link] = target
    for link, target in planned.items():
        if not link.is_symlink():
            link.symlink_to(target)
        links.append(str(link.relative_to(dest)))
    return {"links": links, "missing": missing}


def local_release_path(remote: str) -> str:
    """Map released files locally, keeping named collections separate."""
    parts = list(safe_relative(remote).parts)
    if not parts or parts.pop(0) != "demonstrations":
        raise ValueError(f"Not a demonstration path: {remote}")
    collection = parts.pop(0) if parts and parts[0] in COLLECTIONS else None
    version = parts.pop(0) if parts and re.fullmatch(r"v\d+", parts[0]) else None
    if len(parts) < 3 or parts[0] not in CATEGORIES or not TASK_ID.fullmatch(parts[1]):
        raise ValueError(f"Expected category/task/file: {remote}")
    if version is not None and not parts[1].endswith("-" + version):
        raise ValueError(f"Version disagrees with task ID: {remote}")
    if collection == "main" and (version is None or len(parts) not in {3, 4} or not parts[-1].endswith(".pkl")
                 or (len(parts) == 4 and parts[2] not in {"train", "validation"})):
        raise ValueError(f"Main demonstrations must be versioned task PKLs: {remote}")
    if collection == "cross_20" and (version is None or len(parts) != 4 or not parts[-1].endswith(".pkl")
                                    or parts[2] not in {f"floating_{hand}_right" for hand in CROSS_HANDS}):
        raise ValueError(f"Cross_20 demonstrations must be versioned task/hand PKLs: {remote}")
    if len(parts) == 3 and parts[2] == "demos.pkl":
        parts[2] = parts[1] + ".pkl"
    return "/".join(([collection] if collection else []) + parts)
