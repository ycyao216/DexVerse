# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a single trajectory from a replay HDF5 file as a composited MP4.

The layout is::

    +---------------------+---------------------+
    |                     |   Actions (heatmap, |
    |   Visual streams    |   T x D with current|
    |   (RGB / depth /    |   step bar)         |
    |    point cloud)     +---------------------+
    |                     |   Obs / state /     |
    |                     |   summary text      |
    +---------------------+---------------------+

The visual half stacks every per-step RGB, depth, and point-cloud stream
found in the chosen episode (point clouds are rendered via matplotlib 3D
scatter, optionally subsampled for speed).  The action heatmap plots
``actions`` (T, D) as an image with a moving vertical bar at the current
step.  The text panel prints the file's ``obs_groups`` / preset, the
current step, action L2 norm, and a compact summary of a configurable
number of numeric observation terms.

Requirements
------------
* matplotlib with an ffmpeg writer (typical install on Linux).
* h5py, numpy.

Usage
-----
::

    python scripts/demo_tools/render_demo_video.py \\
        --dataset_file source/dexverse/demonstrations/grasping/\\
Dexverse-GraspKettle-v0/Dexverse-GraspKettle-v0.demo.h5 \\
        --episode 0 \\
        --output outputs/demo_videos/grasp_kettle_demo0.mp4 \\
        --fps 15

When ``--output`` is omitted, the script writes
``<source-stem>__demo_<idx>.mp4`` next to the source HDF5 file.
"""

from __future__ import annotations

import argparse
import json

# Import classification helpers from the live inspector so the two stay in sync.
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import contextlib

from inspect_replay_h5 import (  # noqa: E402
    _SNAPSHOT_PREFIXES,
    classify_term,
    depth_to_display,
    points_of_frame,
    select_episode_group,
    to_uint8_rgb,
    walk_leaf_datasets,
)

# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _collect_streams(group: h5py.Group, expected_T: int, include_next_obs: bool = False):
    """Return ``(rgbs, depths, pcds, numerics)`` dicts of per-step arrays.

    Each value is a ``(T, ...)`` numpy array loaded eagerly so we can rewind
    and seek freely during animation rendering.  Snapshot groups
    (``initial_obs``/``final_obs``) and (optionally) ``next_obs`` are skipped.
    """
    rgbs: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    pcds: dict[str, np.ndarray] = {}
    numerics: dict[str, np.ndarray] = {}

    for path, ds in walk_leaf_datasets(group):
        if ds.ndim < 1 or ds.shape[0] == 0:
            continue
        if any(path.startswith(pfx) for pfx in _SNAPSHOT_PREFIXES):
            continue
        if not include_next_obs and path.startswith("next_obs/"):
            continue
        if int(ds.shape[0]) != expected_T:
            continue
        cat = classify_term(path, tuple(ds.shape), ds.dtype)
        # ``actions`` and ``source_actions`` are pulled out separately downstream.
        if path in ("actions", "source_actions"):
            continue
        if cat == "rgb":
            rgbs[path] = ds[()]
        elif cat == "depth":
            depths[path] = ds[()]
        elif cat == "pointcloud":
            pcds[path] = ds[()]
        elif cat == "numeric":
            numerics[path] = ds[()]
    return rgbs, depths, pcds, numerics


def _format_root_attrs(file: h5py.File) -> list[str]:
    out: list[str] = []
    for key in ("task", "observation_preset", "obs_groups", "rgb_dtype", "depth_dtype"):
        if key not in file.attrs:
            continue
        raw = file.attrs[key]
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        if key == "obs_groups":
            with contextlib.suppress(Exception):
                raw = ", ".join(json.loads(raw))
        out.append(f"{key}: {raw}")
    return out


def _format_episode_attrs(group: h5py.Group) -> list[str]:
    out: list[str] = []
    for key in ("episode_index", "episode_name", "num_samples", "success"):
        if key not in group.attrs:
            continue
        raw = group.attrs[key]
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        out.append(f"{key}: {raw}")
    return out


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------


def _grid_dims(n: int) -> tuple[int, int]:
    """Pick a roughly square (rows, cols) for ``n`` cells."""
    if n <= 0:
        return (0, 0)
    if n == 1:
        return (1, 1)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _build_figure(
    *,
    rgbs: dict,
    depths: dict,
    pcds: dict,
    numerics: dict,
    actions: np.ndarray | None,
    root_text_lines: list[str],
    episode_text_lines: list[str],
    numeric_summary_terms: list[str],
    grouped_obs_terms: dict | None,
    figsize: tuple[float, float],
    obs_panel_fontsize: int = 8,
):
    """Build the static figure scaffold; return ``(fig, artists)``.

    ``artists`` is a dict holding all mutable matplotlib objects the per-frame
    update function rewrites (images, point-cloud scatter handles, the
    moving action bar, the text artists).
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0], wspace=0.18)

    artists: dict = {}

    # ---- Left: visual streams grid -------------------------------------
    visual_titles = list(rgbs.keys()) + list(depths.keys()) + list(pcds.keys())
    rows, cols = _grid_dims(len(visual_titles))
    if visual_titles:
        left_gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=outer[0, 0], hspace=0.28, wspace=0.12)
        artists["images"] = {}
        artists["pcd_scatters"] = {}
        for idx, path in enumerate(visual_titles):
            r, c = divmod(idx, cols)
            in_rgbs = path in rgbs
            in_depths = path in depths
            in_pcds = path in pcds
            if in_pcds:
                ax = fig.add_subplot(left_gs[r, c], projection="3d")
                ax.set_title(path, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_zlabel("")
                first_pts = points_of_frame(pcds[path][0])
                if first_pts.size:
                    sub = (
                        first_pts
                        if first_pts.shape[0] <= 2000
                        else first_pts[np.random.choice(first_pts.shape[0], 2000, replace=False)]
                    )
                    scatter = ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1.5, c=sub[:, 2], cmap="viridis")
                    artists["pcd_scatters"][path] = (ax, scatter)
                else:
                    scatter = ax.scatter([], [], [], s=1.5)
                    artists["pcd_scatters"][path] = (ax, scatter)
            else:
                ax = fig.add_subplot(left_gs[r, c])
                ax.set_title(path, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if in_rgbs:
                    first = to_uint8_rgb(rgbs[path][0])
                    # to_uint8_rgb returns BGR for cv2 convention; flip to RGB for mpl.
                    img = ax.imshow(first[..., ::-1])
                    artists["images"][path] = ("rgb", img)
                elif in_depths:
                    first = depth_to_display(depths[path][0])  # BGR uint8
                    img = ax.imshow(first[..., ::-1])
                    artists["images"][path] = ("depth", img)
    else:
        # No visuals — just leave the left half blank with a label.
        ax = fig.add_subplot(outer[0, 0])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "(no visual streams in this episode)",
            ha="center",
            va="center",
            color="gray",
            fontsize=11,
        )

    # ---- Right column: actions on top, text panel on bottom ------------
    right_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], height_ratios=[2.2, 1.8], hspace=0.18)

    # ---- Action heatmap ------------------------------------------------
    ax_act = fig.add_subplot(right_gs[0, 0])
    if actions is not None and actions.size > 0:
        T, D = actions.shape
        # Heatmap: rows = action dims, columns = time. Helps visualize how all
        # action channels evolve over the trajectory.
        vmax = float(np.abs(actions).max()) if actions.size else 1.0
        vmax = max(vmax, 1e-3)
        ax_act.imshow(
            actions.T,
            aspect="auto",
            cmap="coolwarm",
            origin="lower",
            extent=[0, T - 1, -0.5, D - 0.5],
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_act.set_title(f"actions (T={T}, D={D})", fontsize=9)
        ax_act.set_xlabel("step")
        ax_act.set_ylabel("action dim")
        # Moving cursor at the current step.
        cursor = ax_act.axvline(0, color="black", linewidth=1.5, alpha=0.85)
        artists["action_cursor"] = cursor
    else:
        ax_act.axis("off")
        ax_act.text(0.5, 0.5, "(no actions)", ha="center", va="center", color="gray")

    # ---- Text panel ----------------------------------------------------
    ax_txt = fig.add_subplot(right_gs[1, 0])
    ax_txt.axis("off")
    static_lines = root_text_lines + [""] + episode_text_lines
    static_block = "\n".join(static_lines)
    ax_txt.text(
        0.0,
        1.0,
        static_block,
        family="monospace",
        fontsize=9,
        va="top",
        ha="left",
        transform=ax_txt.transAxes,
    )
    # Layout below the static block: a single Text artist for the step /
    # action header, then one Text artist per obs group (colour-coded) for
    # the per-step term values. Group artists are stacked top-to-bottom in
    # axis fraction; each gets its own colour so the panel reads at a
    # glance ("green block = goal targets, red block = success flags, …").
    static_line_count = len(static_lines)
    static_block_height_frac = min(0.40, 0.020 * max(static_line_count, 1) * (9 / 12))
    dyn_top = max(0.05, 1.0 - static_block_height_frac - 0.04)

    # Approximate fraction-of-axis height for one monospace line at this
    # font size. Tuned empirically; users can adjust by passing
    # ``--obs-panel-fontsize`` to make more / fewer lines fit.
    line_height = 0.024 * (obs_panel_fontsize / 8.0)

    # Header line (step + action L2 norm). Always default colour.
    header_handle = ax_txt.text(
        0.0,
        dyn_top,
        "",
        family="monospace",
        fontsize=obs_panel_fontsize,
        va="top",
        ha="left",
        color=DEFAULT_GROUP_COLOR,
        transform=ax_txt.transAxes,
    )

    group_artists: dict[str, tuple] = {}
    y = dyn_top - 2 * line_height  # leave room for the header (~2 lines)
    if grouped_obs_terms:
        for group_name, terms in grouped_obs_terms.items():
            colour = GROUP_COLORS.get(group_name, DEFAULT_GROUP_COLOR)
            handle = ax_txt.text(
                0.0,
                y,
                "",
                family="monospace",
                fontsize=obs_panel_fontsize,
                va="top",
                ha="left",
                color=colour,
                transform=ax_txt.transAxes,
            )
            group_artists[group_name] = (handle, terms)
            # Reserve one line for the header + one per term; leave a small
            # gap between groups so they're visually distinct.
            y -= (1 + len(terms)) * line_height + (0.4 * line_height)

    artists["header_text"] = header_handle
    artists["group_text"] = group_artists
    artists["numerics"] = numerics
    artists["numeric_summary_terms"] = numeric_summary_terms
    artists["grouped_obs_terms"] = grouped_obs_terms or {}

    artists["rgbs"] = rgbs
    artists["depths"] = depths
    artists["pcds"] = pcds
    artists["actions"] = actions

    return fig, artists


# ---------------------------------------------------------------------------
# Obs-term grouping (per-step text panel content)
# ---------------------------------------------------------------------------


def _group_obs_terms(numerics: dict[str, np.ndarray], allowed_groups: list[str] | None):
    """Split ``numerics`` (keys like ``obs/<group>/<term>``) into a per-group dict.

    Returns ``{group_name: [(term_label, ds), ...]}`` preserving insertion
    order. Two path shapes are recognised:

    * ``obs/<group>/<term>`` — picked up under ``<group>``.
    * ``terminations/<name>`` — grouped under the synthetic ``terminations``
      group so per-step success / out-of-bound booleans show up next to the
      other obs in the text panel.

    Paths matching neither shape (top-level ``actions``, ``source_actions``)
    are skipped. ``allowed_groups`` filters which groups appear; ``None``
    returns every group.
    """
    grouped: dict[str, list[tuple[str, np.ndarray]]] = {}
    for path in sorted(numerics.keys()):
        parts = path.split("/")
        if len(parts) >= 3 and parts[0] == "obs":
            group_name = parts[1]
            term_label = "/".join(parts[2:])
        elif len(parts) >= 2 and parts[0] == "terminations":
            group_name = "terminations"
            term_label = "/".join(parts[1:])
        else:
            continue
        if allowed_groups is not None and group_name not in allowed_groups:
            continue
        grouped.setdefault(group_name, []).append((term_label, numerics[path]))
    return grouped


# Colour mapping for the per-group sections of the obs text panel.  ``proprio``
# stays near-black so it acts as the "default" group; everything else picks a
# colour that conveys its role (goal = green for "target"; contact = orange
# for "force"; privileged = grey for "sim-only"; scene_vis = dim blue for
# "render-only"; terminations = red for "outcome").
GROUP_COLORS: dict[str, str] = {
    "proprio": "#1a1a1a",
    "goal": "#1a7f4f",
    "contact": "#cc7a00",
    "privileged": "#555555",
    "scene_vis": "#5b6dc2",
    "terminations": "#b8281f",
    "policy": "#1a1a1a",
    "rgb": "#1a1a1a",
    "depth": "#1a1a1a",
    "pointcloud": "#1a1a1a",
}
DEFAULT_GROUP_COLOR = "#1a1a1a"


def _format_value_at(arr: np.ndarray, t: int, max_print: int = 4) -> str:
    """Compact formatting of ``arr[t]`` for the per-step text panel.

    Scalars / very short vectors are shown with full values; longer ones get
    a min/max/mean summary plus the head of the array.
    """
    if t >= arr.shape[0]:
        return "<beyond episode end>"
    sample = np.asarray(arr[t])
    if sample.size == 0:
        return "<empty>"
    if sample.ndim == 0:
        return f"{float(sample):+.3f}"
    flat = sample.reshape(-1)
    if flat.size == 1:
        return f"{float(flat[0]):+.3f}"
    if flat.size <= max_print and sample.ndim == 1:
        return np.array2string(sample, precision=3, suppress_small=True)
    a_float = flat.astype(np.float32, copy=False)
    head = np.array2string(flat[:max_print], precision=3, suppress_small=True)
    return (
        f"{head[:-1]}, ...] "
        f"(N={flat.size}, min={a_float.min():+.3f}, max={a_float.max():+.3f}, mean={a_float.mean():+.3f})"
    )


# ---------------------------------------------------------------------------
# Per-frame update
# ---------------------------------------------------------------------------


def _update_frame(t: int, artists: dict) -> list:
    changed: list = []

    # RGB / depth images.
    for path, (kind, img) in artists.get("images", {}).items():
        if kind == "rgb":
            arr = artists["rgbs"][path][t]
            frame = to_uint8_rgb(arr)[..., ::-1]  # BGR -> RGB
        else:
            arr = artists["depths"][path][t]
            frame = depth_to_display(arr)[..., ::-1]
        img.set_data(frame)
        changed.append(img)

    # Point clouds.
    for path, (ax, scatter) in artists.get("pcd_scatters", {}).items():
        pts = points_of_frame(artists["pcds"][path][t])
        if pts.shape[0] > 2000:
            idx = np.random.choice(pts.shape[0], 2000, replace=False)
            pts = pts[idx]
        if pts.size:
            scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            # Color by z so it's visible even on static cameras.
            scatter.set_array(pts[:, 2])
        changed.append(scatter)

    # Action cursor.
    cursor = artists.get("action_cursor")
    if cursor is not None:
        cursor.set_xdata([t, t])
        changed.append(cursor)

    # Step header (default colour).
    header_lines = [f"step  : {t}"]
    actions = artists.get("actions")
    if actions is not None and actions.size > 0 and t < actions.shape[0]:
        a = actions[t]
        header_lines.append(f"|a|_2 : {float(np.linalg.norm(a)):.3f}")
    header_handle = artists.get("header_text")
    if header_handle is not None:
        header_handle.set_text("\n".join(header_lines))
        changed.append(header_handle)

    # Per-group dynamic text (each in its own colour). The terminations
    # group, when present, uses its own red colour so success flags pop.
    for group_name, (handle, terms) in (artists.get("group_text") or {}).items():
        lines = [f"[{group_name}]"]
        for term_label, arr in terms:
            lines.append(f"  {term_label:<28} : {_format_value_at(arr, t)}")
        handle.set_text("\n".join(lines))
        changed.append(handle)

    return changed


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------


def _default_output_for(dataset_file: Path, episode_label: str) -> Path:
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in episode_label)
    return dataset_file.with_name(f"{dataset_file.stem}__{safe}.mp4")


def _render_one_episode(
    f: h5py.File,
    group: h5py.Group,
    *,
    output: Path,
    fps: int,
    figsize: tuple[float, float],
    dpi: int,
    include_next_obs: bool,
    obs_display_groups: list[str] | None,
    obs_panel_fontsize: int,
    numeric_terms_override: list[str] | None,
    max_numeric_terms: int,
) -> bool:
    """Render a single episode group to ``output``. Returns True on success."""
    root_lines = _format_root_attrs(f)
    ep_lines = _format_episode_attrs(group)
    T = int(group.attrs.get("num_samples", -1))
    if T <= 0:
        print(f"[skip] {group.name}: num_samples={T}; nothing to render.")
        return False

    rgbs, depths, pcds, numerics = _collect_streams(group, expected_T=T, include_next_obs=include_next_obs)
    actions = group["actions"][()] if "actions" in group else None

    # Group obs terms by their parent group (proprio / goal / contact / ...).
    grouped_terms = _group_obs_terms(numerics, allowed_groups=obs_display_groups)

    # ``numeric_terms_override`` is the legacy "top-N picks" mode; if not
    # given, the new group-aware display is used (grouped_terms drives the panel).
    if numeric_terms_override is not None:
        numeric_terms = list(numeric_terms_override)
    else:
        numeric_terms = []  # not used when grouped_terms is present

    print(f"[render] {group.name}")
    for line in ep_lines:
        print(f"    {line}")
    print(
        f"    visuals: {len(rgbs)} rgb, {len(depths)} depth, {len(pcds)} pcd; "
        f"{len(numerics)} numeric terms; actions shape="
        f"{None if actions is None else tuple(actions.shape)}"
    )
    if grouped_terms:
        for g_name, terms in grouped_terms.items():
            print(f"    {g_name}: {len(terms)} term(s)")

    fig, artists = _build_figure(
        rgbs=rgbs,
        depths=depths,
        pcds=pcds,
        numerics=numerics,
        actions=actions,
        root_text_lines=root_lines,
        episode_text_lines=ep_lines,
        numeric_summary_terms=numeric_terms,
        grouped_obs_terms=grouped_terms,
        figsize=figsize,
        obs_panel_fontsize=obs_panel_fontsize,
    )

    anim = FuncAnimation(
        fig,
        lambda t: _update_frame(int(t), artists),
        frames=T,
        interval=int(1000 / max(fps, 1)),
        blit=False,
        cache_frame_data=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"    writing -> {output} (fps={fps}, dpi={dpi}, T={T})")
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=4000)
    anim.save(str(output), writer=writer, dpi=dpi)
    plt.close(fig)
    return True


def _enumerate_episode_groups(f: h5py.File) -> list[h5py.Group]:
    """Return every episode group under ``/data``, sorted by ``episode_index``."""
    if "data" not in f:
        return []
    data = f["data"]
    names = list(data.keys())

    def _key(name: str) -> int:
        attrs = data[name].attrs
        if "episode_index" in attrs:
            with contextlib.suppress(TypeError, ValueError):
                return int(attrs["episode_index"])
        return names.index(name)

    return [data[name] for name in sorted(names, key=_key)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render trajectories from a replay HDF5 file as composited MP4s "
            "(visuals on the left, action heatmap on the upper right, "
            "obs/state text summary grouped by obs-group on the lower right)."
        ),
    )
    parser.add_argument("--dataset_file", required=True, type=Path, help="Path to a .demo.h5 file.")
    parser.add_argument(
        "--episode",
        default="0",
        type=str,
        help=(
            "Episode to render. Integer N -> group 'demo_<N>' (falls back to "
            "the N-th group by attribute 'episode_index', then to the N-th "
            "listed). The literal string 'all' (or '*') renders every "
            "episode in the file into separate MP4s named "
            "<h5-stem>__<group>.mp4 next to the source (or under --output-dir)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output MP4 path. Ignored when --episode is 'all' (per-episode "
            "filenames are auto-generated under --output-dir or beside the H5)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write per-episode MP4s when --episode is 'all'. Defaults to the H5's parent directory.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Output frames-per-second.")
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(16.0, 9.0),
        metavar=("W", "H"),
        help="Figure size in inches (default: 16 9).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Render DPI; larger = sharper but bigger MP4 (default: 100).",
    )
    parser.add_argument(
        "--include_next_obs",
        action="store_true",
        help="Include next_obs/* streams (default: skip; they near-duplicate obs/*).",
    )
    parser.add_argument(
        "--obs-display-groups",
        nargs="+",
        default=None,
        help=(
            "Which obs groups to detail in the text panel (e.g. "
            "`--obs-display-groups proprio goal contact`). Default: every "
            "group present under obs/ in the file. Pass an empty list (no "
            "value after the flag is invalid; instead use `--numeric_terms` "
            "if you want explicit term-by-term control)."
        ),
    )
    parser.add_argument(
        "--obs-panel-fontsize",
        type=int,
        default=8,
        help="Font size for the per-step obs text panel (default: 8).",
    )
    parser.add_argument(
        "--numeric_terms",
        nargs="+",
        default=None,
        help=(
            "Legacy: explicit list of obs paths to print each step. When "
            "provided, suppresses the group-aware display in favour of these "
            "exact terms."
        ),
    )
    parser.add_argument(
        "--max_numeric_terms",
        type=int,
        default=4,
        help="When --numeric_terms is auto, cap on how many to show (legacy).",
    )
    args = parser.parse_args()

    if not args.dataset_file.is_file():
        raise FileNotFoundError(args.dataset_file)

    render_all = args.episode.strip().lower() in ("all", "*")
    episode_arg: str | int = ""
    if not render_all:
        try:
            episode_arg = int(args.episode)
        except ValueError:
            episode_arg = args.episode

    with h5py.File(args.dataset_file, "r") as f:
        print(f"[render] {args.dataset_file}")
        for line in _format_root_attrs(f):
            print(f"  {line}")

        if render_all:
            groups = _enumerate_episode_groups(f)
            if not groups:
                print("[error] no episodes found under /data.")
                return 1
            print(f"[render] all episodes ({len(groups)})")
            out_root = args.output_dir or args.dataset_file.parent
            success_count = 0
            for idx, group in enumerate(groups):
                ep_label = group.name.strip("/").rsplit("/", 1)[-1]
                output = out_root / f"{args.dataset_file.stem}__{ep_label}.mp4"
                ok = _render_one_episode(
                    f,
                    group,
                    output=output,
                    fps=args.fps,
                    figsize=tuple(args.figsize),
                    dpi=args.dpi,
                    include_next_obs=args.include_next_obs,
                    obs_display_groups=args.obs_display_groups,
                    obs_panel_fontsize=args.obs_panel_fontsize,
                    numeric_terms_override=args.numeric_terms,
                    max_numeric_terms=args.max_numeric_terms,
                )
                if ok:
                    success_count += 1
                print(f"[render] progress: {idx + 1}/{len(groups)} ({success_count} ok)")
            print(f"[render] done. {success_count}/{len(groups)} episode(s) rendered.")
        else:
            group = select_episode_group(f, episode_arg)
            ep_label = group.name.strip("/").rsplit("/", 1)[-1]
            output = args.output or _default_output_for(args.dataset_file, ep_label)
            ok = _render_one_episode(
                f,
                group,
                output=output,
                fps=args.fps,
                figsize=tuple(args.figsize),
                dpi=args.dpi,
                include_next_obs=args.include_next_obs,
                obs_display_groups=args.obs_display_groups,
                obs_panel_fontsize=args.obs_panel_fontsize,
                numeric_terms_override=args.numeric_terms,
                max_numeric_terms=args.max_numeric_terms,
            )
            if not ok:
                return 1
            print("[render] done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
