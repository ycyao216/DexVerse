# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for resolving demonstration output paths."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

DEXVERSE_DATA_DIR_ENV = "DEXVERSE_DATA_DIR"


def get_dexverse_data_dir() -> Path | None:
    """Return the configured DexVerse data directory, if any."""
    raw = os.environ.get(DEXVERSE_DATA_DIR_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def _normalize_user_subpath(path: str) -> Path:
    """Normalize a user-supplied path to a relative subpath under the data root."""
    user_path = Path(path)
    if user_path.is_absolute():
        return Path(user_path.name)
    return user_path


def resolve_demo_output_path(
    task_name: str,
    dataset_file: str | None = None,
    dataset_dir: str | None = None,
    *,
    timestamp: str | None = None,
) -> Path:
    """Resolve the output pickle path from CLI options.

  When ``DEXVERSE_DATA_DIR`` is set (e.g. by the Docker Compose patch), all
  recordings are written under that mounted directory regardless of whether the
  user passed an absolute host path.

  Precedence:
  1) ``dataset_file`` (must be a file path; directory names are rejected)
  2) ``dataset_dir`` (directory semantics as
     ``<root>/<subpath>/<task_name>/<task_name>_<time>.pkl``)
  3) default ``<root>/<task_name>/<task_name>_<time>/trajectory.pkl`` when no
     data root is configured, otherwise
     ``<root>/<task_name>/<task_name>_<time>.pkl``
    """
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    auto_name = f"{task_name}_{stamp}.pkl"
    data_root = get_dexverse_data_dir()

    if dataset_file:
        if dataset_file.endswith(("/", "\\")) or Path(dataset_file).is_dir():
            raise ValueError(
                "--dataset_file must be a file path, not a directory. "
                "Use --dataset_dir for directory-based auto naming."
            )
        user_path = Path(dataset_file)
        if data_root is not None:
            subpath = _normalize_user_subpath(dataset_file)
            output_path = data_root / subpath
        else:
            output_path = user_path
    elif dataset_dir:
        if data_root is not None:
            output_path = data_root / _normalize_user_subpath(dataset_dir) / task_name / auto_name
        else:
            output_path = Path(dataset_dir) / task_name / auto_name
    elif data_root is not None:
        output_path = data_root / task_name / auto_name
    else:
        output_path = Path("datasets") / f"{task_name}_{stamp}" / "trajectory.pkl"

    if output_path.suffix.lower() not in {".pkl", ".pickle"}:
        output_path = output_path.with_suffix(".pkl")
    return output_path
