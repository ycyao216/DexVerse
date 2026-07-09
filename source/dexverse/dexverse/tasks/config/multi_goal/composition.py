# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable composition helpers for articulation + rigid-object combo environments."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg

from ... import mdp

# Order in which we search a source ObservationsCfg for a term name. ``policy``
# is first so leaves that still use the pre-refactor layout (term sitting in
# ``policy``) work. Refactored leaves put object-pose / velocity terms in
# ``proprio`` / ``privileged`` — those are picked up too.
_SOURCE_SEARCH_GROUPS = ("policy", "proprio", "state", "privileged", "goal", "contact", "debug_vis")

# Mapping from "source group where we found a term" to "destination group on
# the host". Refactored sources keep their bucket (proprio stays in proprio,
# state stays in state, privileged stays in privileged, etc.). Pre-refactor
# sources put everything in ``policy`` — those land in the host's ``proprio``
# since the bulk of pre-refactor policy terms are object-pose / orientation / tilt.
_SOURCE_TO_HOST_GROUP = {
    "policy": "proprio",
    "proprio": "proprio",
    "state": "state",
    "privileged": "privileged",
    "goal": "goal",
    "contact": "contact",
    "debug_vis": "debug_vis",
}


@dataclass
class RigidAddonSpec:
    """Specification for adding one rigid object task component to an env.

    ``term_map`` maps ``{host_dst_name: source_src_name}``. Terms are looked
    up in the source ``ObservationsCfg`` across all its groups and routed to
    the host's ``proprio`` group (the natural home for object pose / quat /
    up-axis / tilt etc. under the post-refactor layout).
    """

    scene_name: str
    prim_path: str
    object_cfg: object
    half_height_est: float
    init_rot: tuple[float, float, float, float] | None
    reset_event_name: str
    reset_pose_range: dict[str, list[float]]
    term_map: dict[str, str]
    contact_obs_name: str = "contact_rigid"


def _clone_scene_entity_name(entity_cfg: SceneEntityCfg, new_name: str) -> SceneEntityCfg:
    kwargs = {}
    for attr in ("joint_names", "body_names", "preserve_order"):
        if hasattr(entity_cfg, attr):
            value = getattr(entity_cfg, attr)
            if value is not None:
                kwargs[attr] = value
    return SceneEntityCfg(new_name, **kwargs)


def remap_term_scene_entity(
    term: ObsTerm,
    *,
    old_name: str,
    new_name: str,
) -> ObsTerm:
    """Clone an observation term while remapping SceneEntityCfg name in params."""
    params = dict(term.params) if term.params is not None else {}
    remapped_any = False
    for key, value in params.items():
        if isinstance(value, SceneEntityCfg) and value.name == old_name:
            params[key] = _clone_scene_entity_name(value, new_name)
            remapped_any = True

    # Many object observation terms rely on default args (no explicit params).
    # If no SceneEntityCfg was remapped and we're remapping from "object", inject object_cfg explicitly.
    if not remapped_any and old_name == "object" and "object_cfg" not in params:
        params["object_cfg"] = SceneEntityCfg(new_name)

    return term.replace(params=params)


def _find_term(obs_cfg, term_name: str) -> tuple[ObsTerm, str]:
    """Locate an ``ObsTerm`` by name across the standard groups of ``obs_cfg``.

    Returns ``(term, source_group_name)``. Raises :class:`AttributeError` if
    the term is not present in any known group.
    """
    for group_name in _SOURCE_SEARCH_GROUPS:
        group = getattr(obs_cfg, group_name, None)
        if group is None:
            continue
        term = getattr(group, term_name, None)
        if isinstance(term, ObsTerm):
            return term, group_name
    raise AttributeError(
        f"Observation term {term_name!r} not found in any of {_SOURCE_SEARCH_GROUPS} on {type(obs_cfg).__name__}"
    )


def apply_rigid_addon(
    env_cfg,
    *,
    spec: RigidAddonSpec,
    source_obs_cfg,
    old_entity_name: str = "object",
) -> None:
    """Apply rigid-addon scene/event/observation/contact composition to an env cfg.

    Object pose / orientation observations are routed to ``env_cfg.observations.proprio``
    (matching the post-refactor layout where object positional state lives in
    proprio). Per-fingertip contact-against-the-addon observations go to
    ``env_cfg.observations.contact``.
    """
    object_cfg = spec.object_cfg.replace(prim_path=spec.prim_path)
    setattr(env_cfg.scene, spec.scene_name, object_cfg)

    table_size = env_cfg.scene.table.spawn.size
    table_pos = env_cfg.scene.table.init_state.pos
    table_top_z = table_pos[2] + table_size[2] * 0.5
    object_pos = object_cfg.init_state.pos
    # Treat object init XY as offsets around the table center so combined envs
    # remain consistent even when the table root pose changes.
    object_cfg.init_state.pos = (
        table_pos[0] + object_pos[0],
        table_pos[1] + object_pos[1],
        table_top_z + spec.half_height_est,
    )
    if spec.init_rot is not None:
        object_cfg.init_state.rot = spec.init_rot

    setattr(
        env_cfg.events,
        spec.reset_event_name,
        EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": spec.reset_pose_range,
                "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
                "asset_cfg": SceneEntityCfg(spec.scene_name),
            },
        ),
    )

    # Route each term to the host group that matches its source group, so
    # e.g. positional state lands in ``proprio`` and velocities (if a refactored
    # source exposes them) land in ``privileged``. Hosts already populate their
    # own state — the addon's terms coexist via distinct names (``can_pos_b``
    # alongside ``articulation_pos_b`` etc.). Composing multiple addons works
    # the same way as long as each ``term_map`` uses a unique prefix.
    #
    # Missing source terms are silently skipped — this lets us declare an
    # aspirational term_map (e.g. ``"can_lin_vel_b": "object_lin_vel_b"``) that
    # will start working once the source config is refactored to expose it.
    for dst_name, src_name in spec.term_map.items():
        try:
            source_term, source_group = _find_term(source_obs_cfg, src_name)
        except AttributeError:
            continue
        remapped = remap_term_scene_entity(source_term, old_name=old_entity_name, new_name=spec.scene_name)
        dst_group_name = _SOURCE_TO_HOST_GROUP.get(source_group, "proprio")
        dst_group = getattr(env_cfg.observations, dst_group_name, None)
        if dst_group is None:
            # Host either nulled this group or never declared it. Fall back to
            # proprio (which every articulation host has populated).
            dst_group = env_cfg.observations.proprio
        setattr(dst_group, dst_name, remapped)

    if env_cfg.robot_config.setup_contact_sensors:
        tip_prim_prefix = "{ENV_REGEX_NS}/Robot/"
        finger_tip_body_list = env_cfg.robot_config.fingertip_body_names
        sensor_names = []
        for link_name in finger_tip_body_list:
            sensor_name = f"{link_name}_{spec.scene_name}_s"
            sensor_path = f"{tip_prim_prefix}{link_name}"
            setattr(
                env_cfg.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path=sensor_path,
                    filter_prim_paths_expr=[spec.prim_path],
                ),
            )
            sensor_names.append(sensor_name)

        # ``contact`` group may have been nulled if the host's own contact
        # sensors weren't set up (unlikely for an articulation host, but
        # guard anyway). Reconstruct it inline as a plain ObsGroup.
        if env_cfg.observations.contact is None:
            from isaaclab.managers import ObservationGroupCfg as _ObsGroup

            env_cfg.observations.contact = _ObsGroup()
            env_cfg.observations.contact.enable_corruption = True
            env_cfg.observations.contact.concatenate_terms = True
            env_cfg.observations.contact.history_length = 0

        setattr(
            env_cfg.observations.contact,
            spec.contact_obs_name,
            ObsTerm(
                func=mdp.fingers_contact_force_b,
                params={"contact_sensor_names": sensor_names},
                clip=(-20.0, 20.0),
            ),
        )

    # ``rigid_obs_source`` is a standalone ObservationsCfg *template* we copied
    # terms out of above; it is never a live env, so its ``contact.contact``
    # term is left as the base ``MISSING`` sentinel (only an env's __post_init__
    # fills it). Now that its terms have been routed onto the host, drop the
    # reference so ``cfg.validate()`` (run inside ``gym.make``) does not recurse
    # into the template and fail with "Missing values detected ...
    # rigid_obs_source.contact.contact".
    if getattr(env_cfg, "rigid_obs_source", None) is not None:
        env_cfg.rigid_obs_source = None
