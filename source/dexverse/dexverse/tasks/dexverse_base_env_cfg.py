# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from dexverse.assets import DEBUG_YCB_OBJ_DIR, POLYHAVEN_HDRI_DIR
from dexverse.visual_purpose import set_prim_purpose_guide
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .adr_curriculum import CurriculumCfg

DEFAULT_TABLE_THICKNESS = 0.05
DEFAULT_TABLE_TOP_HEIGHT = 0.6
DEFAULT_TABLE_SIZE = (1.5, 1.5, DEFAULT_TABLE_THICKNESS)
DEFAULT_TABLE_INIT_POS = (0.0, 0.0, DEFAULT_TABLE_TOP_HEIGHT - DEFAULT_TABLE_THICKNESS / 2)
DEFAULT_TABLE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_TABLE_LEG_THICKNESS = 0.08
DEFAULT_TABLE_LEG_INSET = 0.12
DEFAULT_TABLE_LEG_MATERIAL_PATH = f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl"
DEFAULT_CAMERA_POINT_CLOUD_TABLE_XY_INSET = 0.0
DEFAULT_CAMERA_POINT_CLOUD_TABLE_Z_MIN_OFFSET = 0.01
DEFAULT_CAMERA_POINT_CLOUD_TABLE_Z_MAX_OFFSET = 0.60
DEFAULT_WRIST_CAMERA_OFFSET_POS = (0.0, -0.08, -0.08)
DEFAULT_WRIST_CAMERA_OFFSET_ROT = (0.0, 0.0, 0.04997916927067833, 0.9987502603949663)
DEFAULT_WRIST_CAMERA_BODY_SIZE = (0.09, 0.025, 0.025)
DEFAULT_WRIST_CAMERA_BODY_COLOR = (0.25, 0.25, 0.25)
# Extra offset added to the cube position in palm-local coordinates, on top of
# the camera-relative placement. Tune this to bias the cube toward the hand
# without moving the camera. Each component is a meters-offset added to the
# cube's palm-local pose: (x, y, z).
DEFAULT_WRIST_CAMERA_BODY_PALM_OFFSET = (0.0, 0.0, 0.0)


def _collect_polyhaven_hdri_pool() -> list[str]:
    """Glob the local Poly Haven HDRI dir for fully-downloaded `.hdr` files."""
    import glob as _glob
    import os as _os

    if not _os.path.isdir(POLYHAVEN_HDRI_DIR):
        return []
    return sorted(str(p) for p in _glob.glob(_os.path.join(str(POLYHAVEN_HDRI_DIR), "*.hdr")))


DEFAULT_BACKGROUND_HDRI_POOL = _collect_polyhaven_hdri_pool()

# Pool of textures sampled per-episode for the tabletop. Replace these entries
# with any local or Nucleus-accessible texture paths you want to randomize over.
DEFAULT_TABLE_TEXTURE_POOL = [
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
]


def make_obj_cfg_list(usd_parent_dir, object_ids=None):
    import glob
    import os

    from . import object_annotations

    usd_files_all = []
    for ext in ("usd", "usda", "usdc", "usdz"):
        usd_files_all.extend(glob.glob(os.path.join(usd_parent_dir, f"**/*.{ext}"), recursive=True))
    usd_files_all = sorted(set(usd_files_all))
    usd_files_all = object_annotations.prefer_unpacked_usd(usd_files_all)
    usd_files = []
    if object_ids is not None:
        for ids in object_ids:
            for file in usd_files_all:
                if ids in file:
                    usd_files.append(file)
    else:
        usd_files = usd_files_all
    if not usd_files:
        raise ValueError(f"No USD assets found under {usd_parent_dir}")
    return [
        sim_utils.UsdFileCfg(
            func=spawn_usd_with_rigid_properties,
            usd_path=usd_file,
        )
        for usd_file in usd_files
        if "instanceable_meshes" not in usd_file
    ]


def _obj_cfg_list_from_paths(usd_paths):
    """Build a list of ``UsdFileCfg`` objects in the given order."""
    return [sim_utils.UsdFileCfg(func=spawn_usd_with_rigid_properties, usd_path=p) for p in usd_paths]


def _compute_table_leg_size_and_pos(
    table_size: tuple[float, float, float],
    table_pos: tuple[float, float, float],
    x_sign: float,
    y_sign: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    table_bottom_z = table_pos[2] - table_size[2] * 0.5
    leg_height = max(table_bottom_z, DEFAULT_TABLE_LEG_THICKNESS)
    leg_x = table_pos[0] + x_sign * max(table_size[0] * 0.5 - DEFAULT_TABLE_LEG_INSET, 0.0)
    leg_y = table_pos[1] + y_sign * max(table_size[1] * 0.5 - DEFAULT_TABLE_LEG_INSET, 0.0)
    return (
        (DEFAULT_TABLE_LEG_THICKNESS, DEFAULT_TABLE_LEG_THICKNESS, leg_height),
        (leg_x, leg_y, leg_height * 0.5),
    )


def make_table_leg_cfg(name: str, x_sign: float, y_sign: float) -> RigidObjectCfg:
    leg_size, leg_pos = _compute_table_leg_size_and_pos(DEFAULT_TABLE_SIZE, DEFAULT_TABLE_INIT_POS, x_sign, y_sign)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/table_leg_{name}",
        spawn=sim_utils.CuboidCfg(
            size=leg_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=DEFAULT_TABLE_LEG_MATERIAL_PATH,
                project_uvw=True,
                texture_scale=(0.4, 0.4),
            ),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=leg_pos, rot=DEFAULT_TABLE_INIT_ROT),
    )


def _quat_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(v * v for v in q))
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(v / norm for v in q)


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def _rotate_vector_by_quat(
    q: tuple[float, float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    q = _quat_normalize(q)
    rotated = _quat_multiply(_quat_multiply(q, (0.0, *v)), _quat_conjugate(q))
    return rotated[1], rotated[2], rotated[3]


def _camera_offset_rot_to_opengl(
    rot: tuple[float, float, float, float],
    convention: str,
) -> tuple[float, float, float, float]:
    if convention == "opengl":
        return _quat_normalize(rot)
    if convention == "ros":
        # USD cameras use OpenGL convention. Isaac Lab converts ROS camera
        # offsets by composing a 180-degree rotation around the camera x-axis.
        return _quat_normalize(_quat_multiply(rot, (0.0, 1.0, 0.0, 0.0)))
    raise ValueError(f"Unsupported wrist camera offset convention for blocker pose: {convention!r}")


def _wrist_camera_body_pose(
    camera_pos: tuple[float, float, float],
    camera_rot: tuple[float, float, float, float],
    camera_convention: str,
    body_size: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    body_rot = _camera_offset_rot_to_opengl(camera_rot, camera_convention)
    # Move the cuboid behind the camera's optical origin. This keeps the
    # visual/collider from sitting inside the rendered camera frustum.
    local_body_offset = (0.0, 0.0, body_size[2] * 0.5)
    body_offset = _rotate_vector_by_quat(body_rot, local_body_offset)
    body_pos = tuple(camera_pos[i] + body_offset[i] + DEFAULT_WRIST_CAMERA_BODY_PALM_OFFSET[i] for i in range(3))
    return body_pos, body_rot


def make_wrist_camera_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot/palm/wrist_cam") -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=256,
        height=256,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=DEFAULT_WRIST_CAMERA_OFFSET_POS,
            rot=DEFAULT_WRIST_CAMERA_OFFSET_ROT,
            convention="ros",
        ),
    )


@clone
def spawn_cuboid_as_guide(prim_path, cfg, *args, **kwargs):
    """Spawn a cuboid and tag it as USD ``purpose=guide``.

    Guide-tagged prims show in the Kit viewport but are excluded from
    ``TiledCamera`` (RGB/depth) renders, so the teleoperator can see the
    wrist-camera cuboid while policies / recorded videos do not.
    """
    from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid

    prim = spawn_cuboid(prim_path, cfg, *args, **kwargs)
    set_prim_purpose_guide(prim.GetPath().pathString)
    return prim


def make_wrist_camera_body_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/palm/wrist_camera_body",
    collision_enabled: bool = True,
) -> AssetBaseCfg:
    body_pos, body_rot = _wrist_camera_body_pose(
        DEFAULT_WRIST_CAMERA_OFFSET_POS,
        DEFAULT_WRIST_CAMERA_OFFSET_ROT,
        "ros",
        DEFAULT_WRIST_CAMERA_BODY_SIZE,
    )
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            func=spawn_cuboid_as_guide,
            size=DEFAULT_WRIST_CAMERA_BODY_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision_enabled),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=DEFAULT_WRIST_CAMERA_BODY_COLOR,
                roughness=0.5,
            ),
            visible=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=body_pos, rot=body_rot),
    )


def _v3_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v3_scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _v3_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v3_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _v3_norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_v3_dot(a, a))


def _v3_normalize(a: tuple[float, float, float]) -> tuple[float, float, float]:
    n = _v3_norm(a)
    if n < 1e-9:
        return (1.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _quat_from_rot_cols(
    c0: tuple[float, float, float],
    c1: tuple[float, float, float],
    c2: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for rotation matrix with columns c0, c1, c2 (camera axes in world)."""
    m00, m10, m20 = c0[0], c0[1], c0[2]
    m01, m11, m21 = c1[0], c1[1], c1[2]
    m02, m12, m22 = c2[0], c2[1], c2[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return _quat_normalize((w, x, y, z))


def _world_convention_look_at_quat(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    world_up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Quaternion for ``TiledCameraCfg.OffsetCfg(..., convention='world')``.

    Isaac Lab world convention: camera +X is forward, +Z is up. We build columns
    ``[forward, +Y_cam, +Z_cam]`` in world frame so the camera looks from ``eye`` toward ``target``.
    """
    forward = _v3_normalize(_v3_sub(target, eye))
    up_hint = world_up_hint
    if abs(_v3_dot(forward, up_hint)) > 0.99:
        up_hint = (0.0, 1.0, 0.0)
    up = _v3_sub(up_hint, _v3_scale(forward, _v3_dot(up_hint, forward)))
    up = _v3_normalize(up)
    side = _v3_normalize(_v3_cross(up, forward))
    return _quat_from_rot_cols(forward, side, up)


def _make_side_view_camera_cfg(
    prim_path: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]
) -> TiledCameraCfg:
    """Side-view third-person camera factory. Placeholder pose — tune in a subclass."""
    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        width=256,
        height=256,
    )


def make_third_person_camera_left_cfg() -> TiledCameraCfg:
    eye = (0.0, 1.5, 1.5)
    look_at = (0.0, 0.0, DEFAULT_TABLE_TOP_HEIGHT)
    return _make_side_view_camera_cfg(
        prim_path="{ENV_REGEX_NS}/CameraLeft",
        pos=eye,
        rot=_world_convention_look_at_quat(eye, look_at),
    )


def make_third_person_camera_right_cfg() -> TiledCameraCfg:
    eye = (0.0, -1.5, 1.5)
    look_at = (0.0, 0.0, DEFAULT_TABLE_TOP_HEIGHT)
    return _make_side_view_camera_cfg(
        prim_path="{ENV_REGEX_NS}/CameraRight",
        pos=eye,
        rot=_world_convention_look_at_quat(eye, look_at),
    )


def _sync_table_legs_to_table(scene_cfg) -> None:
    table_size = scene_cfg.table.spawn.size
    table_pos = scene_cfg.table.init_state.pos
    table_rot = scene_cfg.table.init_state.rot
    for leg_name, x_sign, y_sign in (
        ("front_left", 1.0, 1.0),
        ("front_right", 1.0, -1.0),
        ("back_left", -1.0, 1.0),
        ("back_right", -1.0, -1.0),
    ):
        leg_cfg = getattr(scene_cfg, f"table_leg_{leg_name}")
        leg_size, leg_pos = _compute_table_leg_size_and_pos(table_size, table_pos, x_sign, y_sign)
        leg_cfg.spawn.size = leg_size
        leg_cfg.init_state.pos = leg_pos
        leg_cfg.init_state.rot = table_rot


def _sync_wrist_camera_attachment(scene_cfg, camera_attr: str, body_attr: str, palm_body_name: str | None) -> None:
    if palm_body_name is None:
        return

    camera_cfg = getattr(scene_cfg, camera_attr, None)
    body_cfg = getattr(scene_cfg, body_attr, None)
    if camera_cfg is None:
        camera_cfg = make_wrist_camera_cfg()
        setattr(scene_cfg, camera_attr, camera_cfg)
    if body_cfg is None:
        body_cfg = make_wrist_camera_body_cfg()
        setattr(scene_cfg, body_attr, body_cfg)

    camera_cfg.prim_path = f"{{ENV_REGEX_NS}}/Robot/{palm_body_name}/wrist_cam"
    body_cfg.prim_path = f"{{ENV_REGEX_NS}}/Robot/{palm_body_name}/wrist_camera_body"

    body_pos, body_rot = _wrist_camera_body_pose(
        camera_cfg.offset.pos,
        camera_cfg.offset.rot,
        camera_cfg.offset.convention,
        body_cfg.spawn.size,
    )
    body_cfg.init_state.pos = body_pos
    body_cfg.init_state.rot = body_rot


def _configure_wrist_camera_attachments(scene_cfg, robot_config) -> None:
    if robot_config.left_palm_body_name is not None and robot_config.right_palm_body_name is not None:
        # Bimanual robots expose explicit scene entities for each hand. The
        # legacy single-camera entity is disabled to avoid spawning duplicates.
        scene_cfg.wrist_camera = None
        scene_cfg.wrist_camera_body = None
        _sync_wrist_camera_attachment(
            scene_cfg, "right_wrist_camera", "right_wrist_camera_body", robot_config.right_palm_body_name
        )
        _sync_wrist_camera_attachment(
            scene_cfg, "left_wrist_camera", "left_wrist_camera_body", robot_config.left_palm_body_name
        )
    else:
        scene_cfg.right_wrist_camera = None
        scene_cfg.left_wrist_camera = None
        scene_cfg.right_wrist_camera_body = None
        scene_cfg.left_wrist_camera_body = None
        _sync_wrist_camera_attachment(scene_cfg, "wrist_camera", "wrist_camera_body", robot_config.palm_body_name)


def make_default_object_cfg(
    usd_parent_dir=DEBUG_YCB_OBJ_DIR,
    object_ids: list[str] | None = None,
    init_pos: tuple[float, float, float] = (0.55, 0.0, 0.14),
    mass: float = 0.3,
    *,
    usd_paths: list[str] | None = None,
    random_choice: bool = True,
) -> RigidObjectCfg:
    """Create a default object config for tabletop tasks.

    When ``usd_paths`` is provided, it is used verbatim (``usd_parent_dir`` and
    ``object_ids`` are ignored) which lets callers control both the concrete
    asset set and the per-env order. With ``random_choice=False`` the
    underlying ``MultiAssetSpawnerCfg`` assigns ``usd_paths[i % N]`` to
    environment ``i``.
    """
    if usd_paths is not None:
        assets_cfg = _obj_cfg_list_from_paths(usd_paths)
    else:
        assets_cfg = make_obj_cfg_list(usd_parent_dir=usd_parent_dir, object_ids=object_ids)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=assets_cfg,
            random_choice=random_choice,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos),
    )


@clone
def spawn_usd_with_rigid_properties(prim_path, cfg, *args, **kwargs):
    """Spawn USD prim and apply rigid/collision/mass properties when present in cfg."""
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, *args, **kwargs)
    prim_path_resolved = prim.GetPath().pathString
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path_resolved, cfg.rigid_props)
    if cfg.collision_props is not None:
        schemas.define_collision_properties(prim_path_resolved, cfg.collision_props)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path_resolved, cfg.mass_props)
    return prim


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg | ArticulationCfg | None = None

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=DEFAULT_TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),  # Dark grey color
                roughness=0.5,
            ),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=DEFAULT_TABLE_INIT_POS, rot=DEFAULT_TABLE_INIT_ROT),
    )
    table_leg_front_left: RigidObjectCfg = make_table_leg_cfg("front_left", 1.0, 1.0)
    table_leg_front_right: RigidObjectCfg = make_table_leg_cfg("front_right", 1.0, -1.0)
    table_leg_back_left: RigidObjectCfg = make_table_leg_cfg("back_left", -1.0, 1.0)
    table_leg_back_right: RigidObjectCfg = make_table_leg_cfg("back_right", -1.0, -1.0)

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(size=(5.0, 5.0)),
        collision_group=-1,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    third_person_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-1.5, 0.0, 1.5),
            rot=(0.985, 0.0, 0.175, 0.0),
            convention="world",
        ),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        width=256,
        height=256,
    )

    # Two extra side-view cameras for multi-view methods (e.g. pi0.5).
    # Placeholder poses — tune per method by overriding in a subclass.
    third_person_camera_left: TiledCameraCfg | None = make_third_person_camera_left_cfg()
    third_person_camera_right: TiledCameraCfg | None = make_third_person_camera_right_cfg()

    wrist_camera: TiledCameraCfg | None = make_wrist_camera_cfg()
    right_wrist_camera: TiledCameraCfg | None = None
    left_wrist_camera: TiledCameraCfg | None = None

    # Toggle collision with this single-hand override:
    # env.scene.wrist_camera_body.spawn.collision_props.collision_enabled=false
    # For bimanual robots, use right_wrist_camera_body / left_wrist_camera_body.
    wrist_camera_body: AssetBaseCfg | None = make_wrist_camera_body_cfg(collision_enabled=True)
    right_wrist_camera_body: AssetBaseCfg | None = None
    left_wrist_camera_body: AssetBaseCfg | None = None


@configclass
class RobotConfig:
    palm_body_name: str = "base"
    """Name of the palm/base body for end-effector control."""
    right_palm_body_name: str | None = None
    """Right palm body name for bimanual wrist-mounted sensors. Defaults to palm_body_name."""
    left_palm_body_name: str | None = None
    """Left palm body name for bimanual wrist-mounted sensors. None means single-hand setup."""
    fingertip_body_names: list[str] = ["thumb_fingertip", "fingertip", "fingertip2", "fingertip3"]
    """List of fingertip body names for observations and rewards."""
    hand_tips_body_names: list[str] | None = None
    """List of body names for hand_tips_state_b observation.
    If None, defaults to [palm_body_name] + fingertip_body_names."""
    wrist_joint_name: str | None = "fr3_hand_joint"
    """Name of the wrist joint for reset_robot_wrist_joint. If None, this reset is disabled."""
    arm_joint_names_expr: str | list[str] | None = ["fr3_joint.*"]
    """Joint name pattern(s) for arm joints used in IK reset functions. If None, IK resets are disabled."""
    setup_contact_sensors: bool = False
    """Whether to set up contact sensors for fingertips. Should be True in robot-specific configs."""

    def __post_init__(self):
        """Set default hand_tips_body_names if not provided."""
        if self.right_palm_body_name is None:
            self.right_palm_body_name = self.palm_body_name
        if self.hand_tips_body_names is None:
            self.hand_tips_body_names = [self.palm_body_name] + self.fingertip_body_names


def _get_tabletop_robot_setup_builders():
    """Build the robot_type -> setup builder registry for tabletop envs.

    Shadow-only release: floating right / left / bimanual variants. The other
    hand families (allegro, inspire, leap, sharpa, wuji) are staged under
    source_unreleased/robot_agents/ pending further testing; restoring one
    means moving its package back and re-adding its import + entries here.
    """
    from dexverse.robot_agents.shadow.floating import (
        build_tabletop_floating_shadow_bimanual_setup,
        build_tabletop_floating_shadow_left_setup,
        build_tabletop_floating_shadow_right_setup,
    )

    return {
        "floating_shadow_right": build_tabletop_floating_shadow_right_setup,
        "floating_shadow_left": build_tabletop_floating_shadow_left_setup,
        "floating_shadow_bimanual": build_tabletop_floating_shadow_bimanual_setup,
    }


def _make_side_camera_image_obs_term(camera_name: str, data_type: str, *, normalize: bool = True) -> ObsTerm:
    return ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg(camera_name),
            "data_type": data_type,
            "normalize": normalize,
        },
    )


def make_left_rgb_obs_term() -> ObsTerm:
    # normalize=False keeps the camera output as raw 0..255 uint8.
    # IsaacLab's mdp.image default centers rgb around zero (RL convention)
    # which makes captured demos look black after the obs-encoder clips
    # the negative half. Policies that want centered input should normalize
    # at the model boundary instead.
    return _make_side_camera_image_obs_term("third_person_camera_left", "rgb", normalize=False)


def make_right_rgb_obs_term() -> ObsTerm:
    return _make_side_camera_image_obs_term("third_person_camera_right", "rgb", normalize=False)


def make_left_depth_obs_term() -> ObsTerm:
    return _make_side_camera_image_obs_term("third_person_camera_left", "distance_to_image_plane")


def make_right_depth_obs_term() -> ObsTerm:
    return _make_side_camera_image_obs_term("third_person_camera_right", "distance_to_image_plane")


def make_camera_point_cloud_obs_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.camera_point_cloud_w,
        params={
            "sensor_cfg": SceneEntityCfg("third_person_camera"),
            "table_cfg": SceneEntityCfg("table"),
            "num_points": 4096,
            "table_xy_inset": DEFAULT_CAMERA_POINT_CLOUD_TABLE_XY_INSET,
            "table_z_min_offset": DEFAULT_CAMERA_POINT_CLOUD_TABLE_Z_MIN_OFFSET,
            "table_z_max_offset": DEFAULT_CAMERA_POINT_CLOUD_TABLE_Z_MAX_OFFSET,
            "flatten": True,
            "visualize": False,
        },
    )


def make_merged_point_cloud_obs_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.merged_camera_point_cloud_w,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("third_person_camera"),
                SceneEntityCfg("third_person_camera_left"),
                SceneEntityCfg("third_person_camera_right"),
            ],
            "table_cfg": SceneEntityCfg("table"),
            "num_points": 4096,
            "table_xy_inset": DEFAULT_CAMERA_POINT_CLOUD_TABLE_XY_INSET,
            "table_z_min_offset": DEFAULT_CAMERA_POINT_CLOUD_TABLE_Z_MIN_OFFSET,
            "table_z_max_offset": DEFAULT_CAMERA_POINT_CLOUD_TABLE_Z_MAX_OFFSET,
            "flatten": True,
            "visualize": False,
        },
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP.

    Base command configuration. Task-specific configs (e.g., TopDownGraspCommandsCfg)
    should extend this to add task-specific commands like object_pose.
    """


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    Groups are intentionally fine-grained so each consumer (policy, asymmetric
    critic, perception backbones) can subscribe to exactly what it needs:

    - ``policy``: previous action only.
    - ``proprio``: robot joint positions (true proprioception).
    - ``contact``: per-robot contact-sensor readings. The concrete ``contact``
      term is ``MISSING`` here and must be filled in by each robot's setup
      (typically gated on ``robot_config.setup_contact_sensors``).
    - ``state``: observable, deployable task state for the IL policy —
      object / articulation poses, joint positions, derived geometry (e.g.
      tilt axis). Contains NO velocities. ``None`` on the root; sub-bases
      that have task state populate it.
    - ``privileged``: velocities (joint / object lin+ang vel) and other
      sim-only quantities — recorded for completeness but NOT fed to the IL
      policy (excluded from ``state_keys``). Cheap to read in sim,
      unrealistic on hardware.
    - ``goal``: command / target observations (non-privileged: real systems
      know what they were asked to do). ``None`` here on the root — sub-bases
      that wire an ``object_pose`` (or similar) command populate this.
    - ``rgb`` / ``depth`` / ``pointcloud``: three separate sensor groups so
      each can be toggled independently.
    - ``debug_vis``: zero-sized visualization-only observations (frame
      markers, goal markers, forbidden zone markers, …). ``None`` by default;
      populated by sub-bases that have visualizers, and gated behind
      ``DexVerseBaseEnvCfg.enable_debug_vis`` so trained policies
      do not see these terms unless explicitly opted-in.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class ProprioObsCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class ContactObsCfg(ObsGroup):
        # Filled in by each robot agent's setup, typically via
        # ``mdp.fingers_contact_force_b`` over per-fingertip ContactSensorCfgs.
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedObsCfg(ObsGroup):
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b_vis,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # Yunchao: below heuristics are from dexsuite's environments.
            # good behaving number for position in m, velocity in m/s, rad/s,
            # and quaternion are unlikely to exceed -2 to 2 range
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg(
                    "robot", body_names=["base", "thumb_fingertip", "fingertip", "fingertip2", "fingertip3"]
                ),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class RgbObsCfg(ObsGroup):
        # normalize=False keeps raw 0..255 uint8. See make_left_rgb_obs_term
        # for the rationale; policies wanting centered input should normalize
        # at the model boundary.
        rgb_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("third_person_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        left_rgb_image = make_left_rgb_obs_term()
        right_rgb_image = make_right_rgb_obs_term()
        # Wrist-camera RGB. ``__post_init__`` on the env nulls whichever of
        # these don't apply for the active robot setup (single-arm robots
        # use ``wrist_rgb_image``; bimanual robots use the right/left pair).
        wrist_rgb_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        right_wrist_rgb_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        left_wrist_rgb_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = False
            self.flatten_history_dim = True
            self.history_length = 0

    @configclass
    class DepthObsCfg(ObsGroup):
        depth_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("third_person_camera"),
                "data_type": "distance_to_image_plane",
            },
        )
        left_depth_image = make_left_depth_obs_term()
        right_depth_image = make_right_depth_obs_term()
        wrist_depth_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "distance_to_image_plane",
            },
        )
        right_wrist_depth_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                "data_type": "distance_to_image_plane",
            },
        )
        left_wrist_depth_image = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                "data_type": "distance_to_image_plane",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = False
            self.flatten_history_dim = True
            self.history_length = 0

    @configclass
    class PointcloudObsCfg(ObsGroup):
        camera_point_cloud_w = make_camera_point_cloud_obs_term()
        # 3-view variant; nulled by default in DexVerseBaseEnvCfg.__post_init__
        # and enabled (with single-source nulled) when ``multiview_cameras=True``.
        merged_point_cloud_w = make_merged_point_cloud_obs_term()

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    contact: ContactObsCfg = ContactObsCfg()
    # ``state``: observable, deployable task state for the imitation-learning
    # policy (object/articulation poses, joint positions, derived geometry).
    # Holds NO velocities — those go in ``privileged``. ``None`` on the root;
    # sub-bases / leaves that have task state populate it (mirrors ``goal``).
    # Treated like ``privileged`` by the preset machinery: only present in the
    # ``state`` preset, never in the vision presets.
    state: ObsGroup | None = None
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    goal: ObsGroup | None = None
    rgb: RgbObsCfg = RgbObsCfg()
    depth: DepthObsCfg = DepthObsCfg()
    pointcloud: PointcloudObsCfg = PointcloudObsCfg()
    debug_vis: ObsGroup | None = None

    @configclass
    class SceneVisObsCfg(ObsGroup):
        """Visualizer-only side-effect terms (sphere goal markers, zone markers, …).

        Conceptually separate from ``debug_vis``: terms here exist purely to
        drive USD visualization markers each step (sphere goal, forbidden-zone
        markers, frame markers, etc.). Trained policies don't subscribe to
        this group, but the obs manager still ticks the terms so their side
        effects fire. **Never nulled by observation presets**, so the scene
        always looks correct in recorded videos regardless of which preset
        the rest of the obs space uses.
        """

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0

    scene_vis: SceneVisObsCfg | None = None


@configclass
class EventCfg:
    """Configuration for randomization."""

    # Disable startup-time physics randomization globally.
    robot_physics_material = None
    joint_stiffness_and_damping = None
    joint_friction = None

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [-0.03, -0.03],
                "roll": [0, 0],
                "pitch": [0, 0],
                "yaw": [-1.8, -1.3],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
        },
    )

    reset_robot_wrist_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="fr3_joint7"),
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
        },
    )

    # Disabled for stable baselines: per-reset HDRI background + table-texture
    # randomization. Keeping the scene visually fixed gives reproducible baseline
    # results. (The texture randomizer additionally crashes on
    # omni.replicator.core>=1.12.16, where rep.functional.get was removed.)
    reset_environment_background = None
    reset_table_texture = None


@configclass
class ActionsCfg:
    pass


@configclass
class RewardsCfg:
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ---------------------------------------------------------------------------
# Observation presets
# ---------------------------------------------------------------------------
#
# Each preset maps to ``(enabled_groups, history_length)``. The preset is
# applied at the END of ``DexVerseBaseEnvCfg.__post_init__`` (after
# every sub-base / leaf has finished populating its obs terms) by:
#   1. Setting every group NOT in ``enabled`` to ``None``.
#   2. Setting ``policy.history_length`` and ``proprio.history_length`` to
#      the preset's value (with ``flatten_history_dim=True`` when > 0) so the
#      policy gets the last ``N`` actions and the last ``N`` non-priv states.
#
# Contact + tactile are routed to the ``state`` preset only (large sim-to-real
# gap on force readings); realistic visual presets compensate for the missing
# privileged info with action / proprio history stacking.
# The ``state`` preset feeds the policy the observable ``state`` group
# (object/articulation poses, joint pos, derived geometry) and intentionally
# EXCLUDES ``privileged`` — the sim-only velocities are not given to the
# (deployable) state policy. ``privileged`` is still produced when no preset
# is applied, for logging / analysis / privileged-teacher use.
# ``debug_vis`` is NOT managed here — it is gated separately by
# ``DexVerseBaseEnvCfg.enable_debug_vis``.
_OBSERVATION_PRESETS: dict[str, dict] = {
    "rgb": {"enabled": {"policy", "proprio", "goal", "rgb"}, "history_length": 3, "multiview": False},
    "rgb_depth": {"enabled": {"policy", "proprio", "goal", "rgb", "depth"}, "history_length": 3, "multiview": False},
    "pointcloud": {"enabled": {"policy", "proprio", "goal", "pointcloud"}, "history_length": 3, "multiview": False},
    "state": {"enabled": {"policy", "proprio", "contact", "state", "goal"}, "history_length": 0, "multiview": False},
    "3view_rgb": {"enabled": {"policy", "proprio", "goal", "rgb"}, "history_length": 3, "multiview": True},
    "3view_rgb_depth": {
        "enabled": {"policy", "proprio", "goal", "rgb", "depth"},
        "history_length": 3,
        "multiview": True,
    },
    "3view_pointcloud": {
        "enabled": {"policy", "proprio", "goal", "pointcloud"},
        "history_length": 3,
        "multiview": True,
    },
}
_OBSERVATION_PRESET_ALIASES: dict[str, str] = {"rgbd": "rgb_depth", "3view_rgbd": "3view_rgb_depth"}
# Groups touched by ``_apply_observation_preset``: every entry not in a
# preset's ``enabled`` set is set to ``None``. ``scene_vis`` and
# ``debug_vis`` are intentionally NOT listed here — they live outside the
# preset taxonomy. ``scene_vis`` always survives because it drives USD
# marker side-effects that should render regardless of which preset the
# trained obs vector uses; ``debug_vis`` is gated separately by
# ``enable_debug_vis``.
_OBSERVATION_PRESET_MANAGED_GROUPS: tuple[str, ...] = (
    "policy",
    "proprio",
    "contact",
    "state",
    "privileged",
    "goal",
    "rgb",
    "depth",
    "pointcloud",
)

OBSERVATION_PRESET_NAMES: tuple[str, ...] = tuple(_OBSERVATION_PRESETS.keys())
"""Public list of canonical preset names (``rgbd`` is also accepted as an alias for ``rgb_depth``)."""


def resolve_observation_preset(name: str) -> str:
    """Return the canonical preset name for ``name``; raise ``ValueError`` if unknown."""
    canonical = _OBSERVATION_PRESET_ALIASES.get(name, name)
    if canonical not in _OBSERVATION_PRESETS:
        valid = sorted(_OBSERVATION_PRESETS.keys())
        aliases = sorted(_OBSERVATION_PRESET_ALIASES.keys())
        raise ValueError(f"Unknown observation preset {name!r}. Available: {valid} (+ aliases {aliases})")
    return canonical


@configclass
class DexVerseBaseEnvCfg(ManagerBasedEnvCfg):
    """Dexsuite reorientation task definition, also the base definition for derivative Lift task and evaluation task"""

    viewer: ViewerCfg = ViewerCfg(eye=(-2.0, 0.0, 1.25), lookat=(0.0, 0.0, 0.45), origin_type="env")
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = None
    robot_config: RobotConfig = RobotConfig()
    robot_type: str | None = None
    supports_object_pose_command: bool = False
    seed: int = 42
    enable_debug_vis: bool = False
    """Central switch for task-specific debug visualization markers (zone
    boundaries, reference points, …). Tasks that have such markers override
    :meth:`configure_debug_vis` to populate ``observations.scene_vis`` when
    this is ``True``; tasks with nothing to visualize inherit the no-op base
    implementation. Defaults to ``False`` so trained policies neither see
    zero-sized terms nor pay the visualizer spawn cost.

    Deliberately does NOT control goal/target markers (Isaac Lab's
    ``CommandTermCfg.debug_vis`` on ``commands.object_pose``) or task
    "progress toward the goal" indicators (e.g. ``pour_show_progress_marker``)
    — those are task-spec visualization, not debug aids, and stay independent.

    Set via a config rebuild (``cfg_cls(enable_debug_vis=True)``), the same
    mechanism CLI launchers already use for ``robot_type`` overrides — see
    ``scripts/record_demos.py``'s ``--enable_debug_vis`` flag."""

    def configure_debug_vis(self) -> None:
        """Populate/depopulate this task's debug-only visualization markers
        (zone boundaries, reference points, ...) in ``observations.scene_vis``
        based on ``self.enable_debug_vis``.

        No-op by default: tasks with nothing to visualize don't need to
        override this. Leaf / sub-base ``__post_init__``s that define zone or
        marker data should call ``self.configure_debug_vis()`` themselves,
        *after* that data is computed — this can't be auto-chained from this
        base ``__post_init__`` because it runs before subclasses add their own
        terms (same ordering constraint documented below for
        ``observation_preset``).
        """
        pass

    observation_preset: str | None = None
    """Optional observation-preset name applied at the end of ``__post_init__``.

    See :data:`OBSERVATION_PRESET_NAMES`. ``None`` (default) leaves the obs
    cfg as declared by sub-base / leaf — every populated group stays active.
    """

    multiview_cameras: bool = False
    """Enable the two side-view third-person cameras (e.g. for pi0.5).

    When ``False`` (default), the side cameras and their rgb/depth/pointcloud
    obs terms are nulled so the env behaves exactly as the single-front-view
    setup. When ``True``, the side cameras stay live, ``left_*``/``right_*``
    rgb+depth terms are kept, and the pointcloud group swaps the single-source
    term for ``merged_point_cloud_w`` over all three third-person cameras.
    Side-camera poses are placeholders — tune them in a subclass.
    """

    def _apply_observation_preset(self, preset: str) -> None:
        """Apply a named observation preset (idempotent).

        Groups not in the preset's ``enabled`` set are set to ``None``; the
        ``policy`` and ``proprio`` groups have their ``history_length`` set
        to the preset's value (with ``flatten_history_dim=True`` when > 0).
        ``debug_vis`` is left untouched (gated by ``enable_debug_vis``).

        If the preset toggles ``multiview``, the scene's side cameras and the
        matching obs terms are (re-)wired before group nulling.

        Can be called manually from a CLI / launcher *after* the env-cfg
        has been constructed; in that case the original ``__post_init__``
        will have already produced a full obs cfg and this method narrows
        it down.
        """
        canonical = resolve_observation_preset(preset)
        spec = _OBSERVATION_PRESETS[canonical]
        self.multiview_cameras = spec["multiview"]
        self._apply_multiview_cameras(self.multiview_cameras)
        for group in _OBSERVATION_PRESET_MANAGED_GROUPS:
            if group in spec["enabled"]:
                continue
            if hasattr(self.observations, group):
                setattr(self.observations, group, None)
        history_length = spec["history_length"]
        for group_name in ("policy", "proprio"):
            group = getattr(self.observations, group_name, None)
            if group is None:
                continue
            group.history_length = history_length
            if history_length > 0:
                group.flatten_history_dim = True

    def _apply_multiview_cameras(self, enable: bool) -> None:
        """Toggle the two side-view third-person cameras and their obs terms.

        Idempotent and bidirectional. When ``enable`` is ``True``, missing
        side cameras and side rgb/depth/merged-pc obs terms are recreated
        from their factories; the single-source pointcloud term is nulled.
        When ``False``, side cameras and side obs terms are nulled and the
        single-source pointcloud term is restored.
        """
        if enable:
            if self.scene.third_person_camera_left is None:
                self.scene.third_person_camera_left = make_third_person_camera_left_cfg()
            if self.scene.third_person_camera_right is None:
                self.scene.third_person_camera_right = make_third_person_camera_right_cfg()
            if self.observations.rgb is not None:
                if getattr(self.observations.rgb, "left_rgb_image", None) is None:
                    self.observations.rgb.left_rgb_image = make_left_rgb_obs_term()
                if getattr(self.observations.rgb, "right_rgb_image", None) is None:
                    self.observations.rgb.right_rgb_image = make_right_rgb_obs_term()
            if self.observations.depth is not None:
                if getattr(self.observations.depth, "left_depth_image", None) is None:
                    self.observations.depth.left_depth_image = make_left_depth_obs_term()
                if getattr(self.observations.depth, "right_depth_image", None) is None:
                    self.observations.depth.right_depth_image = make_right_depth_obs_term()
            if self.observations.pointcloud is not None:
                self.observations.pointcloud.camera_point_cloud_w = None
                if getattr(self.observations.pointcloud, "merged_point_cloud_w", None) is None:
                    self.observations.pointcloud.merged_point_cloud_w = make_merged_point_cloud_obs_term()
        else:
            self.scene.third_person_camera_left = None
            self.scene.third_person_camera_right = None
            if self.observations.rgb is not None:
                self.observations.rgb.left_rgb_image = None
                self.observations.rgb.right_rgb_image = None
            if self.observations.depth is not None:
                self.observations.depth.left_depth_image = None
                self.observations.depth.right_depth_image = None
            if self.observations.pointcloud is not None:
                self.observations.pointcloud.merged_point_cloud_w = None
                if getattr(self.observations.pointcloud, "camera_point_cloud_w", None) is None:
                    self.observations.pointcloud.camera_point_cloud_w = make_camera_point_cloud_obs_term()

    def _apply_robot_setup(self, robot_setup):
        """Apply a robot setup bundle returned by a robot-agent builder."""
        self.robot_config = RobotConfig(**robot_setup.robot_config_kwargs)
        self.scene.robot = robot_setup.scene_robot
        self.actions = robot_setup.actions
        self.controller_mode = robot_setup.controller_mode
        self._teleop_config = dict(robot_setup.teleop_config)

    def _configure_robot_from_type(self):
        """Apply robot/articulation/action setup for unified robot_type configs."""
        setup_builders = _get_tabletop_robot_setup_builders()
        setup_builder = setup_builders.get(self.robot_type)
        self._apply_robot_setup(setup_builder())

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2  # 60 Hz
        self._configure_robot_from_type()
        _sync_table_legs_to_table(self.scene)
        _configure_wrist_camera_attachments(self.scene, self.robot_config)

        # Null wrist-camera obs terms for cameras the active robot setup
        # doesn't populate. ``_configure_wrist_camera_attachments`` leaves the
        # unused camera attrs as ``None`` on the scene; we mirror that into
        # the rgb / depth obs groups so the obs manager doesn't try to read
        # from a missing sensor.
        for _cam_attr, _rgb_term, _depth_term in (
            ("wrist_camera", "wrist_rgb_image", "wrist_depth_image"),
            ("right_wrist_camera", "right_wrist_rgb_image", "right_wrist_depth_image"),
            ("left_wrist_camera", "left_wrist_rgb_image", "left_wrist_depth_image"),
        ):
            if getattr(self.scene, _cam_attr, None) is None:
                if self.observations.rgb is not None:
                    setattr(self.observations.rgb, _rgb_term, None)
                if self.observations.depth is not None:
                    setattr(self.observations.depth, _depth_term, None)

        # Side-view (multi-view) cameras: nulled unless explicitly enabled. When
        # enabled, pointcloud swaps from single-source to merged-3-view. The
        # ``3view_*`` observation presets re-toggle this in
        # ``_apply_observation_preset`` if invoked post-init.
        self._apply_multiview_cameras(self.multiview_cameras)

        if self.robot_config.wrist_joint_name is not None and self.events.reset_robot_wrist_joint is not None:
            self.events.reset_robot_wrist_joint.params["asset_cfg"].joint_names = self.robot_config.wrist_joint_name
        elif self.robot_config.wrist_joint_name is None:
            self.events.reset_robot_wrist_joint = None

        if self.scene.object is not None:
            if hasattr(self.commands, "object_pose"):
                self.commands.object_pose.position_only = False
        else:
            # Disable object-dependent terms when no object is present.
            if hasattr(self.commands, "object_pose"):
                self.commands.object_pose = None
            self.observations.policy.object_quat_b = None
            self.observations.policy.target_object_pose_b = None
            self.observations.pointcloud.object_point_cloud = None
            self.terminations.object_out_of_bound = None
            self.events.object_physics_material = None
            self.rewards.position_tracking = None
            self.rewards.success = None

        self.episode_length_s = 20.0
        self.is_finite_horizon = True

        if self.scene.object is not None and hasattr(self.commands, "object_pose"):
            resample_time = self.episode_length_s + 1.0
            self.commands.object_pose.resampling_time_range = (resample_time, resample_time)

        # ``enable_debug_vis`` is read by sub-base / leaf ``__post_init__``s
        # *before* they populate ``self.observations.debug_vis``. The flag
        # cannot null the group here because sub-bases add terms after
        # ``super().__post_init__()`` returns.
        #
        # ``observation_preset`` has the same ordering constraint and so is
        # NOT auto-applied here either. CLI / launcher code should call
        # ``env_cfg._apply_observation_preset(env_cfg.observation_preset)``
        # *after* ``parse_env_cfg`` (i.e. after the full __post_init__ chain
        # has run on the leaf cfg) and *before* the env is built.

        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15

        # RTX sensors (cameras) need a few renders to latch a newly-written
        # scene after a teleport. The ``--set-state`` demo replay teleports the
        # whole scene every step (``scene.reset_to``) and renders before reading
        # camera obs; with a single render the captured frame can lag the
        # just-set state by ~1 frame / show TAA ghosting. Render a couple of
        # extra times so point cloud / RGB obs reflect the current state. This is
        # read by both demo scripts via ``env.cfg.num_rerenders_on_reset`` and is
        # a no-op when no RTX sensors are present (the replay refresh and the env
        # reset both gate the rerender loop on ``sim.has_rtx_sensors()``).
        self.num_rerenders_on_reset = 2
