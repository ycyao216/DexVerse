# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for cleaning up env_cfg state after conditional scene modifications.

Currently provides :func:`prune_stale_obs_refs`, which pairs with
:func:`isaaclab.devices.openxr.remove_camera_configs`. That function removes
cameras from ``env_cfg.scene`` when teleoperating with XR, but only scans
``observations.policy`` for now-dangling ``SceneEntityCfg`` references.
Vision / perception groups still point at the removed camera, so env
construction fails with "The scene entity '...' does not exist.". Calling
:func:`prune_stale_obs_refs` right after ``remove_camera_configs`` sweeps
every obs group and nulls out the offending terms. If an entire group
becomes empty as a result, the group itself is dropped so
``ObservationManager`` does not try to concatenate zero tensors (which
raises "The shapes of the terms are: []").
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Non-term attributes defined on ``ObservationGroupCfg``; ignore these when
# deciding whether a group still has any active terms.
_GROUP_META_ATTRS = frozenset({
    "enable_corruption",
    "concatenate_terms",
    "concatenate_dim",
    "history_length",
    "flatten_history_dim",
})


def strip_camera_cfgs(env_cfg: Any) -> Any:
    """Disable every camera sensor on ``env_cfg.scene`` by setting it to ``None``.

    Drop-in replacement for :func:`isaaclab.devices.openxr.remove_camera_configs`
    for XR teleop without camera rendering. That upstream helper removes camera
    fields with ``delattr``, which corrupts the scene configclass: the field
    stays *declared* but has no instance value, so a later
    ``env_cfg.scene.replace(...)`` / dataclass-field iteration -- which IsaacLab
    performs inside ``gym.make`` -- raises ``AttributeError: '...SceneCfg' object
    has no attribute '...'``. Setting the field to ``None`` instead removes the
    camera (``InteractiveScene`` skips ``None`` cfgs) while keeping the
    configclass consistent. Pair with :func:`prune_stale_obs_refs` afterwards to
    drop the now-dangling camera observation terms.
    """
    from isaaclab.sensors import CameraCfg

    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return env_cfg

    for attr_name in [n for n in dir(scene) if not n.startswith("_")]:
        if isinstance(getattr(scene, attr_name, None), CameraCfg):
            setattr(scene, attr_name, None)
            logger.info("Disabled camera scene cfg '%s' (set to None)", attr_name)
    return env_cfg


def prune_stale_obs_refs(env_cfg: Any) -> Any:
    """Null out observation terms that reference missing scene entities.

    Walks every group on ``env_cfg.observations`` (not just ``policy``) and
    sets any term to ``None`` if any of its ``params`` values contains a
    :class:`SceneEntityCfg` whose ``name`` is not an attribute of
    ``env_cfg.scene``. The ``SceneEntityCfg`` may be the param value directly
    or nested inside a list/tuple/set/dict (e.g. the merged-pointcloud term's
    ``sensor_cfgs=[SceneEntityCfg(...), ...]``). If a group has no remaining active terms after
    pruning, the group itself is set to ``None`` on the observations cfg so
    the ``ObservationManager`` skips it entirely (supported upstream: it
    checks ``if group_cfg is None: continue``). Safe no-op if every term
    resolves.
    """
    from isaaclab.managers import SceneEntityCfg

    scene = env_cfg.scene
    # An entity counts as "present" only if it resolves to a non-None cfg. A
    # camera disabled by setting its field to ``None`` (see
    # :func:`strip_camera_cfgs`) is treated as missing here, so obs terms that
    # reference it are pruned -- otherwise the obs manager later fails to
    # resolve the (un-spawned) sensor with "scene entity ... does not exist".
    scene_entities = {n for n in dir(scene) if not n.startswith("_") and getattr(scene, n, None) is not None}

    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return env_cfg

    for group_name in dir(observations):
        if group_name.startswith("_"):
            continue
        group = getattr(observations, group_name, None)
        if group is None or not hasattr(group, "__dict__"):
            continue
        for term_name in list(vars(group)):
            if term_name.startswith("_") or term_name in _GROUP_META_ATTRS:
                continue
            term = getattr(group, term_name, None)
            if term is None or not hasattr(term, "params") or not term.params:
                continue
            stale_name = next(
                (
                    entity.name
                    for value in term.params.values()
                    for entity in _iter_scene_entity_cfgs(value, SceneEntityCfg)
                    if entity.name not in scene_entities
                ),
                None,
            )
            if stale_name is not None:
                setattr(group, term_name, None)
                logger.info(
                    "Pruned stale obs term '%s.%s' (missing scene entity '%s')",
                    group_name,
                    term_name,
                    stale_name,
                )

        # Drop the group entirely if nothing is left. Otherwise the
        # observation manager would try to concatenate an empty list and
        # raise "Unable to concatenate observation terms in group ...".
        if _group_has_no_active_terms(group):
            setattr(observations, group_name, None)
            logger.info(
                "Dropped empty observation group '%s' (all terms referenced missing scene entities)",
                group_name,
            )
    return env_cfg


def _iter_scene_entity_cfgs(value: Any, scene_entity_cls: type):
    """Yield every ``SceneEntityCfg`` nested anywhere in ``value``.

    Recurses into list/tuple/set/dict containers so params whose value is a
    *collection* of ``SceneEntityCfg`` (e.g. the merged-pointcloud term's
    ``sensor_cfgs`` list) are inspected, not just bare ``SceneEntityCfg``
    values.
    """
    if isinstance(value, scene_entity_cls):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_scene_entity_cfgs(item, scene_entity_cls)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_scene_entity_cfgs(item, scene_entity_cls)


def _group_has_no_active_terms(group: Any) -> bool:
    """Return ``True`` if no attribute on ``group`` is an active obs term."""
    for attr_name, attr_val in vars(group).items():
        if attr_name.startswith("_") or attr_name in _GROUP_META_ATTRS:
            continue
        if attr_val is not None:
            return False
    return True
